"""Independent mixed-precision full-layer reference and explicit dtype contracts.

The reference uses FP32 master weights whose compute values lie on the BF16
grid. This makes parameter-gradient accumulation FP32 without making PyTorch's
BF16 leaf-gradient storage an implicit part of the specification.
"""

import torch
import torch.nn.functional as F

from .full_layer import PARAMETERS
from .multihead import load_official, make_graph, pack, unpack_output
from .graph import evaluate_graph

FP32_PARAMETERS = {"A_log", "dt_bias", "norm.weight"}


def mixed_inputs(graph, *, seed=707):
    gen = torch.Generator().manual_seed(seed)
    values = {}
    for i in graph.forward_inputs:
        node = graph.nodes[i]
        if node.name == "attention_mask":
            value = torch.ones(node.shape)
            if node.shape[1] > 2:
                value[0, -2:] = 0
        elif node.name == "A_log":
            value = torch.full(node.shape, -2.0)
        elif node.name == "dt_bias":
            value = torch.full(node.shape, -3.0)
        else:
            value = torch.randn(node.shape, generator=gen)
            if node.name == "norm.weight":
                value = 1 + 0.1 * value
            elif node.name.endswith("weight"):
                # Same scale family as nn.Linear/Conv1d default initialization.
                value = value / (3 * node.shape[-1]) ** 0.5
        if node.dtype == "bf16":
            value = value.bfloat16()
        values[node.name] = value
    return values


def master_reference_inputs(compute_values):
    return {
        name: (value.float() if name in PARAMETERS else value)
        .detach()
        .requires_grad_(name != "attention_mask")
        for name, value in compute_values.items()
    }


def normalized_token_core(q, k, v, g, beta):
    state = q.new_zeros(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        decayed = g[:, t].exp()[..., None, None] * state
        update = beta[:, t, :, None] * (
            v[:, t] - (k[:, t, :, :, None] * decayed).sum(-2)
        )
        state = decayed + k[:, t, :, :, None] * update[..., None, :]
        outputs.append(((q[:, t] / q.shape[-1] ** 0.5)[..., None] * state).sum(-2))
    return torch.stack(outputs, 1)


def reference_mixed_layer(values, config, *, core="official", return_stages=False):
    """BF16 storage boundaries, FP32 contractions and parameter gradients.

    The chunk equations come from the pinned official source. Q/K normalization
    is shared before head expansion so shared-head adjoints accumulate in FP32.
    FP32 parameter leaves already contain exactly the BF16 compute-weight values.
    """
    x, mask = values["hidden_states"], values["attention_mask"]
    bsz, length, _ = x.shape
    hk, hv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    kd, vd = config["linear_key_head_dim"], config["linear_value_head_dim"]
    taps = config["linear_conv_kernel_dim"]
    x32 = x.float() * mask

    def projection(name):
        return F.linear(x32, values[name]).bfloat16()

    qkv = projection("in_proj_qkv.weight")
    if core == "candidate_autograd":
        # Match the forward arithmetic, but let autograd independently derive
        # every gradient. This isolates VJP correctness from BF16 tie crossings.
        padded = F.pad(qkv.float(), (0, 0, taps - 1, 0))
        w = values["conv1d.weight"].reshape(qkv.shape[-1], taps)
        terms = [
            padded[:, j : j + length] * w[:, j].reshape(1, 1, -1) for j in range(taps)
        ]
        pre = terms[0]
        for term in terms[1:]:
            pre = pre + term
        pre = pre.bfloat16()
        conv = F.silu(pre)
    else:
        pre = (
            F.conv1d(
                qkv.transpose(1, 2).float(),
                values["conv1d.weight"],
                padding=taps - 1,
                groups=qkv.shape[-1],
            )[..., :length]
            .bfloat16()
            .transpose(1, 2)
        )
        conv = F.silu(pre)
    q, k, v = conv.split((hk * kd, hk * kd, hv * vd), dim=-1)
    q, k = [a.reshape(bsz, length, hk, kd).float() for a in (q, k)]
    q, k = [a * torch.rsqrt(a.square().sum(-1, keepdim=True) + 1e-6) for a in (q, k)]
    if core != "candidate_autograd":
        q, k = [a.repeat_interleave(hv // hk, dim=2) for a in (q, k)]
    v = v.reshape(bsz, length, hv, vd).float()
    z = projection("in_proj_z.weight").reshape(bsz, length, hv, vd)
    beta = projection("in_proj_b.weight").sigmoid()
    a = projection("in_proj_a.weight")
    g = -values["A_log"].exp() * F.softplus(a.float() + values["dt_bias"])
    if core == "official":
        o, _ = load_official()["torch_chunk_gated_delta_rule"](
            q,
            k,
            v,
            g,
            beta.float(),
            use_qk_l2norm_in_kernel=False,
            output_final_state=False,
        )
    elif core == "token":
        o = normalized_token_core(q, k, v, g, beta.float())
    elif core == "candidate_autograd":
        graph = make_graph(length, bsz, hk, hv, kd, vd, variant="qwen_prepared")
        tensors = pack((q, k, v, g, beta.float(), q.new_zeros(bsz, hv, kd, vd)))
        (o, state), _ = evaluate_graph(
            graph,
            graph.forward_outputs,
            dict(zip(graph.forward_inputs, tensors, strict=True)),
            retain_intermediates=False,
        )
        o = unpack_output(o, state)[0]
    else:
        raise ValueError(core)
    o = o.bfloat16()
    ofp = o.float()
    unit = (
        ofp * torch.rsqrt(ofp.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    ).bfloat16()
    gated = (unit.float() * values["norm.weight"] * F.silu(z.float())).bfloat16()
    y = F.linear(
        gated.reshape(bsz, length, hv * vd).float(), values["out_proj.weight"]
    ).bfloat16()
    stages = {
        "qkv_projection": qkv,
        "conv_pre": pre,
        "conv": conv,
        "z": z,
        "beta": beta,
        "log_decay": g,
        "core_output": o,
        "rms_unit_storage": unit,
        "gated_norm_output": gated,
    }
    return (y, stages) if return_stages else y


def compare_mixed(actual, expected, names, *, relative_l2_max=0.005):
    """Fixed criteria chosen for BF16 rounding; every tensor is checked separately."""
    rows = []
    for name, x, y in zip(names, actual, expected, strict=True):
        xf, yf = x.float(), y.float()
        delta = xf - yf
        rms = yf.square().mean().sqrt().item()
        atol, rtol = 0.002 * max(rms, 1.0), 0.02
        relative = (delta.double().norm() / yf.double().norm().clamp_min(1e-30)).item()
        finite = bool(torch.isfinite(xf).all() and torch.isfinite(yf).all())
        elementwise = bool((delta.abs() <= atol + rtol * yf.abs()).all())
        rows.append(
            dict(
                name=name,
                dtype=str(x.dtype),
                reference_dtype=str(y.dtype),
                passed=finite and elementwise and relative <= relative_l2_max,
                finite=finite,
                elementwise_passed=elementwise,
                relative_l2=relative,
                relative_l2_limit=relative_l2_max,
                max_abs=delta.abs().max().item(),
                reference_rms=rms,
                atol=atol,
                rtol=rtol,
            )
        )
    return rows
