import torch

from selas.codecs import encode_int8_row, encode_nf4, encode_raw
from selas.container import ALIGN, PART_ALIGN, Container, ContainerWriter, part_view


def _write(tmp_path):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(33, 7, generator=g)
    b = torch.randn(64, 64, generator=g)
    c = torch.randn(5, generator=g).to(torch.bfloat16)
    w = ContainerWriter(tmp_path / "c", "test", {"x": 1})
    w.add_unit("u0", [encode_raw("a", a, torch.float16), encode_int8_row("b", b)], group="main", kind="k")
    w.add_unit("u1", [encode_raw("c", c), encode_nf4("d", b)], group="mod")
    w.close()
    return a, b, c


def test_roundtrip_and_alignment(tmp_path):
    a, b, c = _write(tmp_path)
    ct = Container(tmp_path / "c")
    assert ct.component == "test" and ct.config == {"x": 1}
    assert list(ct.units) == ["u0", "u1"]
    for u in ct.units.values():
        assert u.offset % ALIGN == 0 and u.nbytes % ALIGN == 0
        for t in u.tensors.values():
            for p in t.parts.values():
                assert p.offset % PART_ALIGN == 0
    u0 = ct.units["u0"]
    buf = torch.empty(u0.nbytes, dtype=torch.uint8)
    ct.read_unit_into(u0, buf)
    assert torch.equal(part_view(buf, u0.tensors["a"].parts["data"]), a.half())
    u1 = ct.units["u1"]
    buf1 = torch.empty(u1.nbytes, dtype=torch.uint8)
    ct.read_unit_into(u1, buf1)
    assert torch.equal(part_view(buf1, u1.tensors["c"].parts["data"]), c)
    assert u0.group == "main" and u0.kind == "k" and u1.group == "mod"
    assert ct.verify() == []


def test_verify_detects_corruption(tmp_path):
    _write(tmp_path)
    data = tmp_path / "c" / "weights.bin"
    raw = bytearray(data.read_bytes())
    raw[10] ^= 0xFF
    data.write_bytes(bytes(raw))
    assert Container(tmp_path / "c").verify() == ["u0"]
