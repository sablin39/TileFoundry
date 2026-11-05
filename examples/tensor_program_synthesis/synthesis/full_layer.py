"""Incomplete full Qwen3.5 GDN example with supplied analytic VJPs.

Strict 8k BF16 gradient equivalence and GPU lowering remain unresolved.
No autograd or opaque call appears in the authored graph. PyTorch below is used
only for an independent reference. Shapes and the chunk algorithm are supplied;
this frontend does not discover either the model or its differentiation rules.
"""

import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from .graph import Graph
from .multihead import load_official, make_graph, recurrent64


PARAMETERS = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_b.weight",
    "in_proj_a.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)


def load_official_layer():
    """Execute the complete pinned official module, removing dispatch wrappers.

    Function/class bodies remain unchanged. Postponed annotations let the pure
    training path run without importing the entire newer Transformers package.
    This avoids mixing current core math with older surrounding layer semantics.
    """
    from transformers.activations import ACT2FN

    path = Path(__file__).resolve().parents[1] / "references/qwen35/modeling_qwen3_5.py"
    tree = ast.parse(path.read_text())
    wanted = {
        "Qwen3_5RMSNormGated",
        "apply_mask_to_padding_states",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "Qwen3_5GatedDeltaNet",
    }
    selected = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted
    ]
    assert {n.name for n in selected} == wanted
    for root in selected:
        for node in ast.walk(root):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    parsed = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    namespace = {
        **load_official(),
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "ACT2FN": ACT2FN,
    }
    exec(compile(parsed, str(path), "exec"), namespace)
    return namespace["Qwen3_5GatedDeltaNet"]


def reshape(x, *shape):
    return x.graph.op("reshape", x, shape=tuple(shape))


def permute(x, *axes):
    return x.graph.op("permute", x, perm=tuple(axes))


def cut(x, start, end, axis):
    return x.graph.op("slice", x, start=start, end=end, axis=axis)


def silu_prime(x):
    s = x.graph.op("sigmoid", x)
    return s * (x.graph.scalar(1.0) + x * subtract_from_one(s))


def subtract_from_one(x):
    return x.graph.scalar(1.0) - x


def clone_node(graph, node, values):
    recipe = node.recipe
    attrs = dict(recipe.attrs)
    if recipe.op == "scalar":
        result = graph.scalar(attrs["value"])
    elif recipe.op == "constant":
        result = graph.constant(attrs["kind"], attrs["size"])
    elif recipe.op == "input":
        raise ValueError(f"Unbound core input: {node.name}")
    else:
        result = graph.op(recipe.op, *(values[i] for i in recipe.args), **attrs)
        result.named("core." + node.name)
    values[node.id] = result


def full_layer_graph(
    config, length, batch=1, chunk_size=64, variant="qwen_prepared", *, precision="fp32"
):
    """Return Y and VJPs for X plus all nine parameter tensors.

    Full-sequence training, zero initial convolution/recurrent state, immutable
    [B,T,1] padding mask. No mutable decode cache or outer transformer MLP.
    """
    if precision not in {"fp32", "bf16_fp32"}:
        raise ValueError("Unsupported precision policy")
    mixed = precision == "bf16_fp32"
    activation_dtype = "bf16" if mixed else "f32"
    h = config["hidden_size"]
    hk, hv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    kd, vd = config["linear_key_head_dim"], config["linear_value_head_dim"]
    taps, eps = config["linear_conv_kernel_dim"], config["rms_norm_eps"]
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("This supplied layer formula requires SiLU")
    if min(batch, length, h, hk, hv, kd, vd, taps) <= 0 or hv % hk:
        raise ValueError("Positive dimensions and grouped value heads required")
    group, width, m = hv // hk, 2 * hk * kd + hv * vd, batch * length
    graph = Graph("qwen35_full_gdn")
    x = graph.input("hidden_states", (batch, length, h), dtype=activation_dtype)
    mask = graph.input("attention_mask", (batch, length, 1))
    shapes = (
        (width, h),
        (hv * vd, h),
        (hv, h),
        (hv, h),
        (width, 1, taps),
        (hv,),
        (hv,),
        (vd,),
        (h, hv * vd),
    )
    weights = {
        n: graph.input(
            n,
            s,
            dtype=activation_dtype
            if n not in {"A_log", "dt_bias", "norm.weight"}
            else "f32",
        )
        for n, s in zip(PARAMETERS, shapes, strict=True)
    }
    wqkv, wz, wb, wa, wc, alog, dt, wn, wo = weights.values()
    regions, region_start = {}, len(graph.nodes)

    def mark_region(label):
        nonlocal region_start
        regions.update({i: label for i in range(region_start, len(graph.nodes))})
        region_start = len(graph.nodes)

    def fp(value):
        return value.cast("f32") if value.node.dtype == "bf16" else value

    def bf(value):
        return value.cast("bf16") if mixed and value.node.dtype != "bf16" else value

    def mm(lhs, rhs):
        return graph.op(
            "matmul",
            lhs,
            rhs,
            **({"out_dtype": "f32"} if lhs.node.dtype == "bf16" else {}),
        )

    def zeros(*shape, dtype="f32"):
        return graph.op("zeros", shape=tuple(shape), dtype=dtype)

    def pack_value(value, d):
        return permute(reshape(value, batch, length, hk, group, d), 0, 2, 3, 1, 4)

    def unpack_value(value, d):
        return reshape(permute(value, 0, 3, 1, 2, 4), batch, length, hv, d)

    def pack_key(value):
        return reshape(permute(value, 0, 2, 1, 3), batch, hk, 1, length, kd)

    def unpack_key(value):
        return permute(reshape(value, batch, hk, length, kd), 0, 2, 1, 3)

    xf = reshape(bf(fp(x) * mask), m, h).named("layer.masked_input")
    mark_region("fwd.input_mask")
    projected = reshape(bf(mm(xf, wqkv.T)), batch, length, width).named(
        "layer.qkv_projection"
    )
    mark_region("fwd.qkv_projection")
    padded = (
        graph.op(
            "concat",
            zeros(batch, taps - 1, width, dtype=activation_dtype),
            projected,
            axis=1,
        )
        if taps > 1
        else projected
    )
    wc2 = reshape(wc, width, taps)
    windows = [cut(padded, j, j + length, 1) for j in range(taps)]
    tap_weights = [reshape(cut(wc2, j, j + 1, 1), 1, 1, width) for j in range(taps)]
    conv_terms = [fp(u) * fp(w) for u, w in zip(windows, tap_weights, strict=True)]
    conv_pre = conv_terms[0]
    for term in conv_terms[1:]:
        conv_pre = conv_pre + term
    conv_pre = bf(conv_pre).named("layer.conv_pre_activation")
    conv = graph.op("silu", conv_pre).named("layer.conv_activation")
    mark_region("fwd.causal_conv_silu")
    qraw = fp(reshape(cut(conv, 0, hk * kd, 2), batch, length, hk, kd))
    kraw = fp(reshape(cut(conv, hk * kd, 2 * hk * kd, 2), batch, length, hk, kd))
    v = fp(reshape(cut(conv, 2 * hk * kd, width, 2), batch, length, hv, vd))
    qraw.named("layer.q_raw_fp32")
    kraw.named("layer.k_raw_fp32")
    qinv = graph.op("rsqrt", (qraw * qraw).sum(-1) + 1e-6).named("layer.q_inv_norm")
    kinv = graph.op("rsqrt", (kraw * kraw).sum(-1) + 1e-6).named("layer.k_inv_norm")
    q, k = qraw * qinv, kraw * kinv
    mark_region("fwd.qk_normalization")
    z = reshape(bf(mm(xf, wz.T)), batch, length, hv, vd).named("layer.z_projection")
    mark_region("fwd.z_projection")
    b = reshape(bf(mm(xf, wb.T)), batch, length, hv).named("layer.b_projection")
    beta = fp(graph.op("sigmoid", b)).named("layer.beta")
    mark_region("fwd.beta_projection_gate")
    a = reshape(bf(mm(xf, wa.T)), batch, length, hv).named("layer.a_projection")
    aplus = (fp(a) + reshape(dt, 1, 1, hv)).named("layer.a_plus_dt")
    expa = reshape(alog.exp(), 1, 1, hv)
    g = (-expa * graph.op("softplus", aplus)).named("layer.log_decay")
    mark_region("fwd.decay_projection_gate")
    initial_state = zeros(batch, hk, group, kd, vd)

    core = make_graph(length, batch, hk, hv, kd, vd, chunk_size, variant)
    mapped = dict(
        zip(
            core.forward_inputs,
            (
                pack_key(q),
                pack_key(k),
                pack_value(v, vd),
                pack_value(g, 1),
                pack_value(beta, 1),
                initial_state,
            ),
            strict=True,
        )
    )
    mark_region("fwd.core_layout")
    state_dependent = {core.forward_inputs[-1]}
    for node in core.nodes:
        if node.phase == "forward" and node.id not in mapped:
            if any(i in state_dependent for i in node.recipe.args):
                state_dependent.add(node.id)
            clone_node(graph, node, mapped)
            mark_region(
                "fwd.chunk_state_output"
                if node.id in state_dependent
                else "fwd.chunk_prepare"
            )
    o = bf(unpack_value(mapped[core.forward_outputs[0]], vd)).named("layer.core_output")
    mark_region("fwd.core_output_round")
    ofp = fp(o).named("layer.core_output_fp32") if mixed else o
    rinv = graph.op("rsqrt", (ofp * ofp).sum(-1) * (1.0 / vd) + eps).named(
        "layer.rms_inv"
    )
    unit = (ofp * rinv).named("layer.rms_unit")
    unit_storage = bf(unit).named("layer.rms_unit_storage") if mixed else unit
    norm_weight = reshape(wn, 1, 1, 1, vd)
    weighted = fp(unit_storage) * norm_weight
    zgate = graph.op("silu", fp(z))
    gated = bf(weighted * zgate).named("layer.gated_norm_output")
    gated_flat = reshape(gated, m, hv * vd)
    mark_region("fwd.gated_rmsnorm")
    y = reshape(bf(mm(gated_flat, wo.T)), batch, length, h).named("layer.output")
    mark_region("fwd.output_projection")

    graph.begin_backward([y])
    dy = graph.input("d_output", (batch, length, h), dtype=activation_dtype)
    dyf = reshape(dy, m, h)
    dwo = mm(dyf.T, gated_flat)
    dgated = reshape(bf(mm(dyf, wo)), batch, length, hv, vd).named(
        "layer.d_gated_norm_output"
    )
    mark_region("bwd.output_projection")
    dz = bf(fp(dgated) * weighted * silu_prime(fp(z)))
    dweighted = fp(dgated) * zgate
    dwn = reshape((dweighted * fp(unit_storage)).sum(0).sum(1).sum(2), vd)
    dunit = fp(bf(dweighted * norm_weight))
    do = fp(
        bf(
            (rinv * (dunit - unit * ((dunit * unit).sum(-1) * (1.0 / vd)))).named(
                "layer.d_core_output_fp32"
            )
        )
    )
    mark_region("bwd.gated_rmsnorm")

    mapped.update(
        zip(core.backward_inputs, (pack_value(do, vd), initial_state), strict=True)
    )
    mark_region("bwd.core_layout")
    adjoint_nodes = core.ancestors([core.backward_outputs[-1]])
    for node in core.nodes:
        if node.phase == "backward" and node.id not in mapped:
            clone_node(graph, node, mapped)
            mark_region(
                "bwd.state_adjoint"
                if node.id in adjoint_nodes
                else "bwd.chunk_gradients"
            )
    dq, dk, dv, dg, dbeta, _ = [mapped[i] for i in core.backward_outputs]
    dq, dk = unpack_key(dq), unpack_key(dk)
    dqraw = bf((qinv * (dq - q * (dq * q).sum(-1))).named("layer.d_q_raw_fp32"))
    dkraw = bf((kinv * (dk - k * (dk * k).sum(-1))).named("layer.d_k_raw_fp32"))
    dv = bf(unpack_value(dv, vd))
    dg = reshape(unpack_value(dg, 1), batch, length, hv)
    dbeta = reshape(unpack_value(dbeta, 1), batch, length, hv)
    mark_region("bwd.qk_normalization_layout")
    dconv = graph.op(
        "concat",
        reshape(dqraw, batch, length, hk * kd),
        reshape(dkraw, batch, length, hk * kd),
        reshape(dv, batch, length, hv * vd),
        axis=2,
    )
    dpre = bf(fp(dconv) * silu_prime(fp(conv_pre)))
    dw_taps, dpad_terms = [], []
    for j, (window, w) in enumerate(zip(windows, tap_weights, strict=True)):
        dw_taps.append(reshape((fp(dpre) * fp(window)).sum(0).sum(1), width, 1))
        pieces = []
        if j:
            pieces.append(zeros(batch, j, width))
        pieces.append(fp(dpre) * fp(w))
        if taps - 1 - j:
            pieces.append(zeros(batch, taps - 1 - j, width))
        dpad_terms.append(
            graph.op("concat", *pieces, axis=1) if len(pieces) > 1 else pieces[0]
        )
    dwc = reshape(graph.op("concat", *dw_taps, axis=1), width, 1, taps)
    dpad = dpad_terms[0]
    for term in dpad_terms[1:]:
        dpad = dpad + term
    dqkv = reshape(bf(cut(dpad, taps - 1, taps - 1 + length, 1)), m, width).named(
        "layer.d_qkv_projection"
    )
    mark_region("bwd.causal_conv_silu")
    db = reshape(bf(fp(bf(dbeta)) * beta * subtract_from_one(beta)), m, hv)
    softplus_grad = graph.op(
        "where", graph.op("gt", aplus, 20.0), 1.0, graph.op("sigmoid", aplus)
    )
    da = dg * (-expa) * softplus_grad
    dalog = reshape((dg * g).sum(0).sum(1), hv)
    ddt = reshape(da.sum(0).sum(1), hv)
    daf, dzf = reshape(bf(da), m, hv), reshape(dz, m, hv * vd)
    mark_region("bwd.beta_decay_gates")
    dx = bf(
        reshape(
            mm(dqkv, wqkv) + mm(dzf, wz) + mm(db, wb) + mm(daf, wa), batch, length, h
        )
        * mask
    )
    gradients = (
        dx,
        mm(dqkv.T, xf),
        mm(dzf.T, xf),
        mm(db.T, xf),
        mm(daf.T, xf),
        dwc,
        dalog,
        ddt,
        dwn,
        dwo,
    )
    mark_region("bwd.input_projections")
    gradient_names = ("hidden_states", *PARAMETERS)
    graph.finish(
        [
            v.named("d_" + name)
            for name, v in zip(gradient_names, gradients, strict=True)
        ]
    )
    graph.metadata = {
        "scope": "complete Qwen3_5GatedDeltaNet full-sequence training module",
        "batch": batch,
        "length": length,
        "hidden_size": h,
        "key_heads": hk,
        "value_heads": hv,
        "key_dim": kd,
        "value_dim": vd,
        "conv_kernel": taps,
        "chunk_size": chunk_size,
        "variant": variant,
        "gradient_inputs": list(gradient_names),
        "dtype": precision,
        "strict_dtypes": mixed,
        "regions": regions,
        "gradient_precision": "FP32 parameter-gradient outputs; BF16 hidden gradient"
        if mixed
        else "FP32",
        "precision_contract": "BF16 projection/convolution operands and activation boundaries; FP32 matmul/reduction accumulation, shared-head gradient sum, recurrent state, decay parameters and norm weights"
        if mixed
        else "FP32",
        "differentiation": "supplied analytic VJPs; no autograd in candidate",
        "states": "zero initial states; no mutable decode cache",
        "attention_mask": "immutable B,T,1 input mask, matching Qwen input masking",
        "components": [
            "all input projections",
            "causal depthwise convolution",
            "SiLU",
            "Q/K L2 normalization",
            "beta and decay parameterization",
            "grouped multihead chunked GDR",
            "gated RMSNorm",
            "output projection",
        ],
    }
    return graph


def reference_layer(inputs, config):
    """Independent differentiable reference using conv1d and token recurrence.

    Preserves FP64 for auditing equations. Real-shape FP32 validation additionally
    compares to the installed official Qwen module with the pinned core fallback.
    """
    x, mask = inputs["hidden_states"], inputs["attention_mask"]
    bsz, length, _ = x.shape
    hk, hv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    kd, vd = config["linear_key_head_dim"], config["linear_value_head_dim"]
    taps = config["linear_conv_kernel_dim"]
    x = x * mask
    qkv = F.linear(x, inputs["in_proj_qkv.weight"])
    qkv = F.silu(
        F.conv1d(
            qkv.transpose(1, 2),
            inputs["conv1d.weight"],
            padding=taps - 1,
            groups=qkv.shape[-1],
        )[..., :length]
    ).transpose(1, 2)
    q, k, v = qkv.split((hk * kd, hk * kd, hv * vd), dim=-1)
    q, k = [a.reshape(bsz, length, hk, kd) for a in (q, k)]
    v = v.reshape(bsz, length, hv, vd)
    z = F.linear(x, inputs["in_proj_z.weight"]).reshape(bsz, length, hv, vd)
    beta = F.linear(x, inputs["in_proj_b.weight"]).sigmoid()
    a = F.linear(x, inputs["in_proj_a.weight"]) + inputs["dt_bias"]
    g = -inputs["A_log"].exp() * F.softplus(a)
    state = x.new_zeros(bsz, hv, kd, vd)
    o, _ = recurrent64(q, k, v, g, beta, state)
    unit = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    gated = unit * inputs["norm.weight"] * F.silu(z)
    return F.linear(gated.reshape(bsz, length, hv * vd), inputs["out_proj.weight"])


def boundary_policy(graph):
    """One illustrative saved set; deliberately no optimization/search."""
    names = {
        "layer.conv_activation",
        "layer.q_inv_norm",
        "layer.k_inv_norm",
        "layer.core_output",
        "layer.rms_inv",
        "layer.gated_norm_output",
    }
    return [
        i
        for i in graph.candidates
        if graph.nodes[i].name in names or graph.nodes[i].name.endswith(".inverse")
    ]


def synthetic_inputs(graph, *, dtype=torch.float32, seed=13, masked=True):
    gen = torch.Generator().manual_seed(seed)
    values = {}
    for i in graph.forward_inputs:
        node = graph.nodes[i]
        if node.name == "attention_mask":
            value = torch.ones(node.shape, dtype=dtype)
            if masked and node.shape[1] > 2:
                value[0, -2:] = 0
        elif node.name == "norm.weight":
            value = 1 + 0.1 * torch.randn(node.shape, generator=gen, dtype=dtype)
        elif node.name == "A_log":
            value = torch.full(node.shape, -2.0, dtype=dtype)
        elif node.name == "dt_bias":
            value = torch.full(node.shape, -3.0, dtype=dtype)
        else:
            value = torch.randn(node.shape, generator=gen, dtype=dtype)
            if node.name.endswith("weight"):
                value = value / node.shape[-1] ** 0.5
        values[node.name] = value.requires_grad_(node.name != "attention_mask")
    return values
