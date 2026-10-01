import pytest
import torch

from selas.codecs import (NF4_CODE, decode, encode_int8_row, encode_nf4, encode_raw, nf4_dequant_torch, nf4_import)

from .conftest import needs_cuda


def _ref_nf4(packed, absmax, blocksize, code, shape):
    """Straight-line reference of the bnb NF4 formula."""
    code = torch.tensor(code, dtype=torch.float32)
    hi = (packed >> 4).long()
    lo = (packed & 15).long()
    idx = torch.stack((hi, lo), dim=1).reshape(-1)
    vals = code[idx]
    scale = absmax.float().repeat_interleave(blocksize)
    return (vals * scale).view(shape)


def test_raw_roundtrip_dtypes():
    t = torch.randn(17, 9)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        e = encode_raw("w", t, dt)
        out = decode("raw", e.parts, e.meta, e.shape, dt)
        assert out.dtype == dt and torch.equal(out, t.to(dt))
        assert out.data_ptr() == e.parts["data"].data_ptr()  # same dtype -> zero-copy


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="no float8 in this torch")
def test_raw_fp8_decodes_by_cast():
    t = torch.randn(8, 8).to(torch.float8_e4m3fn)
    e = encode_raw("w", t)
    out = decode("raw", e.parts, e.meta, e.shape, torch.float32)
    assert torch.equal(out, t.float())


def test_int8_row_error_bound():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(64, 256, generator=g)
    t[3] *= 50  # an outlier row must not hurt the others
    e = encode_int8_row("w", t)
    out = decode("int8_row", e.parts, e.meta, e.shape, torch.float32)
    rel = (out - t).norm(dim=1) / t.norm(dim=1)
    assert rel.max() < 0.01


def test_nf4_packing_order_high_nibble_first():
    t = torch.tensor([-1.0, 1.0] + [0.0] * 62)  # first element -> code 0 (-1), second -> code 15 (+1)
    e = encode_nf4("w", t)
    assert e.parts["packed"][0].item() == (0 << 4) | 15
    assert e.parts["absmax"][0].item() == 1.0


def test_nf4_decode_matches_reference_formula():
    g = torch.Generator().manual_seed(1)
    t = torch.randn(96, 128, generator=g)
    e = encode_nf4("w", t)
    ref = _ref_nf4(e.parts["packed"], e.parts["absmax"], 64, NF4_CODE, t.shape)
    for dt in (torch.float32, torch.float16):
        out = nf4_dequant_torch(e.parts["packed"], e.parts["absmax"], 64, NF4_CODE, tuple(t.shape), dt, chunk_bytes=1000)
        assert torch.equal(out, ref.to(dt))
    rel = (ref - t).norm() / t.norm()
    assert rel < 0.12  # NF4 is ~4 bits: ~10 % relative RMS error on Gaussian weights


def test_nf4_encode_picks_nearest_code():
    g = torch.Generator().manual_seed(2)
    t = torch.randn(64 * 10, generator=g)
    e = encode_nf4("w", t)
    deq = _ref_nf4(e.parts["packed"], e.parts["absmax"], 64, NF4_CODE, t.shape)
    code = torch.tensor(NF4_CODE)
    normed = (t.view(-1, 64) / t.view(-1, 64).abs().amax(1, keepdim=True)).reshape(-1)
    best = (normed[:, None] - code[None]).abs().min(dim=1).values
    got = (deq.view(-1, 64) / e.parts["absmax"][:, None]).reshape(-1)
    assert torch.allclose((normed - got).abs(), best, atol=1e-6)


def test_nf4_import_validates_sizes():
    packed = torch.zeros(32, dtype=torch.uint8)
    with pytest.raises(ValueError):
        nf4_import("w", packed, torch.ones(2), (8, 16), 64, NF4_CODE)  # 128 elems need 64 bytes
    e = nf4_import("w", torch.zeros(64, 1, dtype=torch.uint8), torch.ones(2), (8, 16), 64, NF4_CODE)
    assert e.parts["packed"].shape == (64,) and e.parts["absmax"].dtype == torch.float32


@needs_cuda
def test_nf4_cuda_paths_agree_with_cpu():
    g = torch.Generator().manual_seed(3)
    t = torch.randn(192, 64, generator=g)
    e = encode_nf4("w", t)
    cpu = decode("nf4", e.parts, e.meta, e.shape, torch.float16)
    gpu_parts = {k: v.cuda() for k, v in e.parts.items()}
    gpu = decode("nf4", gpu_parts, e.meta, e.shape, torch.float16)  # Triton path if available
    assert torch.equal(gpu.cpu(), cpu)


@needs_cuda
def test_nf4_matches_bitsandbytes_if_installed():
    bnb = pytest.importorskip("bitsandbytes")
    g = torch.Generator().manual_seed(4)
    t = torch.randn(256, 128, generator=g).cuda()
    q, state = bnb.functional.quantize_4bit(t, blocksize=64, quant_type="nf4", compress_statistics=False)
    ref = bnb.functional.dequantize_4bit(q, state).float()
    ours = decode("nf4", {"packed": q.reshape(-1), "absmax": state.absmax.float()}, {"blocksize": 64, "code": state.code.tolist()},
                  tuple(t.shape), torch.float32)
    assert torch.allclose(ours, ref, rtol=0, atol=1e-6)
    e = encode_nf4("w", t.cpu())
    # byte-compatible encoder; bnb uses x * (1/absmax) and decimal thresholds, so allow rare ties
    mismatch = (e.parts["packed"] != q.reshape(-1).cpu()).float().mean().item()
    assert mismatch < 1e-3
