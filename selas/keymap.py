"""Key mapping between BFL-format FLUX checkpoints and diffusers ``FluxTransformer2DModel``.

``bfl_from_diffusers(n_double, n_single)`` returns, for every BFL key, how to
build it from diffusers keys:

* ``("id", [k])``        — same tensor
* ``("cat", [k1, ...])`` — concatenate along dim 0 (fused qkv / linear1)
* ``("swap", [k])``      — swap the two halves along dim 0 (diffusers' final
                           AdaLayerNormContinuous is (scale, shift); BFL is (shift, scale))
"""

from __future__ import annotations


def bfl_from_diffusers(n_double: int, n_single: int, guidance: bool = True) -> dict[str, tuple[str, list[str]]]:
    m: dict[str, tuple[str, list[str]]] = {}

    def lin(bfl: str, diff: str, op: str = "id", bias: bool = True):
        m[bfl + ".weight"] = (op, [diff + ".weight"] if op != "cat" else [d + ".weight" for d in diff])
        if bias:
            m[bfl + ".bias"] = (op, [diff + ".bias"] if op != "cat" else [d + ".bias" for d in diff])

    lin("img_in", "x_embedder")
    lin("txt_in", "context_embedder")
    lin("time_in.in_layer", "time_text_embed.timestep_embedder.linear_1")
    lin("time_in.out_layer", "time_text_embed.timestep_embedder.linear_2")
    lin("vector_in.in_layer", "time_text_embed.text_embedder.linear_1")
    lin("vector_in.out_layer", "time_text_embed.text_embedder.linear_2")
    if guidance:
        lin("guidance_in.in_layer", "time_text_embed.guidance_embedder.linear_1")
        lin("guidance_in.out_layer", "time_text_embed.guidance_embedder.linear_2")
    lin("final_layer.linear", "proj_out")
    lin("final_layer.adaLN_modulation.1", "norm_out.linear", op="swap")

    for i in range(n_double):
        b, d = f"double_blocks.{i}", f"transformer_blocks.{i}"
        lin(f"{b}.img_mod.lin", f"{d}.norm1.linear")
        lin(f"{b}.txt_mod.lin", f"{d}.norm1_context.linear")
        lin(f"{b}.img_attn.qkv", [f"{d}.attn.to_q", f"{d}.attn.to_k", f"{d}.attn.to_v"], op="cat")
        lin(f"{b}.txt_attn.qkv", [f"{d}.attn.add_q_proj", f"{d}.attn.add_k_proj", f"{d}.attn.add_v_proj"], op="cat")
        lin(f"{b}.img_attn.proj", f"{d}.attn.to_out.0")
        lin(f"{b}.txt_attn.proj", f"{d}.attn.to_add_out")
        lin(f"{b}.img_mlp.0", f"{d}.ff.net.0.proj")
        lin(f"{b}.img_mlp.2", f"{d}.ff.net.2")
        lin(f"{b}.txt_mlp.0", f"{d}.ff_context.net.0.proj")
        lin(f"{b}.txt_mlp.2", f"{d}.ff_context.net.2")
        m[f"{b}.img_attn.norm.query_norm.scale"] = ("id", [f"{d}.attn.norm_q.weight"])
        m[f"{b}.img_attn.norm.key_norm.scale"] = ("id", [f"{d}.attn.norm_k.weight"])
        m[f"{b}.txt_attn.norm.query_norm.scale"] = ("id", [f"{d}.attn.norm_added_q.weight"])
        m[f"{b}.txt_attn.norm.key_norm.scale"] = ("id", [f"{d}.attn.norm_added_k.weight"])

    for i in range(n_single):
        b, d = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        lin(f"{b}.modulation.lin", f"{d}.norm.linear")
        lin(f"{b}.linear1", [f"{d}.attn.to_q", f"{d}.attn.to_k", f"{d}.attn.to_v", f"{d}.proj_mlp"], op="cat")
        lin(f"{b}.linear2", f"{d}.proj_out")
        m[f"{b}.norm.query_norm.scale"] = ("id", [f"{d}.attn.norm_q.weight"])
        m[f"{b}.norm.key_norm.scale"] = ("id", [f"{d}.attn.norm_k.weight"])
    return m
