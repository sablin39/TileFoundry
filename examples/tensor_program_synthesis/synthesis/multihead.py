"""Grouped Q/K heads with independent value-head states and explicit core VJPs."""

import ast
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .graph import Graph, constant_tensor, evaluate_graph
from .parallel import scan_gated_delta_rule, prepared_gated_delta_rule
from .problems import chunk_gated_delta_rule

ROOT = Path(__file__).resolve().parents[1]


def lift_grouped(base, batch, key_heads, value_heads):
    """Lift a supplied 2D formula to [B,Hk,G,T,D] using real broadcasting.

    Q and K have G=1; V, gates and states have G=Hv/Hk. Gram matrices therefore
    remain shared, while gate-dependent triangular systems remain head-specific.
    Materialization choices apply uniformly to each entire batched value.
    """
    if min(batch, key_heads, value_heads) <= 0 or value_heads % key_heads:
        raise ValueError("Value heads must be a positive multiple of key heads")
    group = value_heads // key_heads
    graph = Graph(base.name)
    values = {}
    for node in base.nodes:
        if node.phase == "backward" and graph.phase == "forward":
            graph.begin_backward([values[i] for i in base.forward_outputs])
        recipe = node.recipe
        attrs = dict(recipe.attrs)
        if recipe.op == "input":
            prefix = (batch, key_heads, 1 if node.name in {"Q", "K"} else group)
            value = graph.input(node.name, (*prefix, *node.shape))
        elif recipe.op == "scalar":
            value = graph.scalar(attrs["value"])
        elif recipe.op == "constant":
            value = graph.constant(attrs["kind"], attrs["size"])
        else:
            args = [values[i] for i in recipe.args]
            if recipe.op in {"sum", "slice", "concat"}:
                offset = len(args[0].node.shape) - len(base.nodes[recipe.args[0]].shape)
                attrs["axis"] = attrs.get("axis", 0) + offset
            value = graph.op(recipe.op, *args, **attrs).named(node.name)
        values[node.id] = value
    gradients = [values[i] for i in base.backward_outputs]
    # Shared Q/K receive the sum of all value-head contributions, not their mean.
    gradients[:2] = [
        g.sum(2).named(name)
        for g, name in zip(gradients[:2], ("dQ_shared", "dK_shared"), strict=True)
    ]
    graph.finish(gradients)
    graph.metadata = {
        **base.metadata,
        "batch": batch,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "value_heads_per_key_head": group,
        "batch_heads": "executed grouped tensors, not replicated cost estimates",
        "layout": "B,Hk,G,T,D; Q/K use G=1",
        "materialization_granularity": "uniform whole batched tensors",
    }
    return graph


def make_graph(
    length,
    batch=1,
    key_heads=16,
    value_heads=32,
    key_dim=128,
    value_dim=128,
    chunk_size=64,
    variant="sequential",
):
    factory = {
        "sequential": chunk_gated_delta_rule,
        "scan": scan_gated_delta_rule,
        "qwen_prepared": prepared_gated_delta_rule,
    }[variant]
    return lift_grouped(
        factory(length, chunk_size, key_dim, value_dim), batch, key_heads, value_heads
    )


def pack(args):
    q, k, v, g, beta, state = args
    b, t, hk, kd = q.shape
    hv, vd = v.shape[-2:]
    group = hv // hk
    q, k = [x.permute(0, 2, 1, 3).unsqueeze(2) for x in (q, k)]
    v = v.reshape(b, t, hk, group, vd).permute(0, 2, 3, 1, 4)
    g, beta = [x.reshape(b, t, hk, group, 1).permute(0, 2, 3, 1, 4) for x in (g, beta)]
    return q, k, v, g, beta, state.reshape(b, hk, group, kd, vd)


def unpack_output(y, state):
    b, hk, group, t, vd = y.shape
    return y.permute(0, 3, 1, 2, 4).reshape(b, t, hk * group, vd), state.reshape(
        b, hk * group, state.shape[-2], vd
    )


class Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, plan, program, *args):
        values = dict(zip(graph.forward_inputs, args, strict=True))
        roots = [*graph.forward_outputs, *plan.saved]
        if program is None:
            outputs, _ = evaluate_graph(
                graph, roots, values, retain_intermediates=False
            )
        else:
            from tilefoundry.evaluator import evaluate
            from .lower import deep_ir_visitors, function

            values.update(
                {
                    n.id: constant_tensor(n)
                    for n in graph.nodes
                    if n.recipe.op == "constant"
                }
            )
            with deep_ir_visitors():
                outputs = evaluate(
                    function(program.module, "forward"),
                    *(values[i] for i in program.forward.inputs),
                )
        assert sum(
            x.numel() * x.element_size() for x in outputs[2:]
        ) == plan.residual_bytes * (args[0].element_size() // 4)
        needed = set(graph.backward_outputs)
        for i, choice in plan.recipes.items():
            needed.update(graph.nodes[i].recipes[choice].args)
        retained_roots = [i for i in roots if i in plan.saved or i in needed]
        materialized = dict(zip(roots, outputs, strict=True))
        ctx.graph, ctx.plan, ctx.program, ctx.roots = (
            graph,
            plan,
            program,
            retained_roots,
        )
        # Ordinary outputs need not be pinned when backward never consumes them.
        # In particular, avoid retaining an unused FP32 O behind a BF16 cast.
        ctx.save_for_backward(*args, *(materialized[i] for i in retained_roots))
        return outputs[0], outputs[1]

    @staticmethod
    def backward(ctx, dy, ds):
        graph, plan, program = ctx.graph, ctx.plan, ctx.program
        args = ctx.saved_tensors[:6]
        values = dict(zip(graph.forward_inputs, args, strict=True))
        values.update(zip(ctx.roots, ctx.saved_tensors[6:], strict=True))
        values.update(zip(graph.backward_inputs, (dy, ds), strict=True))
        if program is None:
            gradients, _ = evaluate_graph(
                graph,
                graph.backward_outputs,
                values,
                recipes=plan.recipes,
                retain_intermediates=False,
            )
        else:
            from tilefoundry.evaluator import evaluate
            from .lower import deep_ir_visitors, function

            values.update(
                {
                    n.id: constant_tensor(n)
                    for n in graph.nodes
                    if n.recipe.op == "constant"
                }
            )
            with deep_ir_visitors():
                gradients = evaluate(
                    function(program.module, "backward"),
                    *(values[i] for i in program.backward.inputs),
                )
        return None, None, None, *gradients


def candidate(
    graph, plan, q, k, v, g, beta, initial_state, *, program=None, normalize=True
):
    """Qwen-compatible core boundary; normalization AD surrounds supplied VJP."""
    original_dtype = q.dtype
    computation_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, beta = [x.to(computation_dtype) for x in (q, k, v, g, beta)]
    if normalize:
        q, k = [
            x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)
        ]
    tensors = pack((q, k, v, g, beta, initial_state.to(computation_dtype)))
    out, state = Core.apply(graph, plan, program, *tensors)
    out, state = unpack_output(out, state)
    return out.to(original_dtype), state


def load_official():
    """Run the pinned official pure-PyTorch fallbacks, without hub decorators.

    Function bodies are unchanged. Removing dispatch decorators ensures this
    check executes the audited Python source, not a separately downloaded kernel.
    """
    path = ROOT / "references/qwen35/modeling_qwen3_5.py"
    tree = ast.parse(path.read_text())
    wanted = {
        "l2norm",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    }
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body} == wanted
    for node in body:
        node.decorator_list = []
    from transformers.utils import is_torchdynamo_exporting

    namespace = {
        "torch": torch,
        "F": F,
        "is_torchdynamo_exporting": is_torchdynamo_exporting,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def official(implementation, q, k, v, g, beta, initial_state, *, chunk_size=64):
    group = v.shape[2] // q.shape[2]
    q, k = [x.repeat_interleave(group, dim=2) for x in (q, k)]
    return implementation(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def recurrent64(q, k, v, g, beta, initial_state):
    """Independent token recurrence including Q/K normalization and head tying."""
    q, k = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    group = v.shape[2] // q.shape[2]
    q, k = [x.repeat_interleave(group, dim=2) for x in (q, k)]
    state, outputs = initial_state, []
    for t in range(q.shape[1]):
        decayed = g[:, t].exp()[..., None, None] * state
        delta = beta[:, t, :, None] * (
            v[:, t] - (k[:, t, :, :, None] * decayed).sum(-2)
        )
        state = decayed + k[:, t, :, :, None] * delta[..., None, :]
        outputs.append(((q[:, t] / q.shape[-1] ** 0.5)[..., None] * state).sum(-2))
    return torch.stack(outputs, dim=1), state


def sample_inputs(
    batch,
    length,
    hk,
    hv,
    kd=128,
    vd=128,
    *,
    dtype=torch.float32,
    seed=2026,
    case="gated",
    initial_zero=True,
):
    gen = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, dtype=dtype, generator=gen)

    q, k = [randn((batch, length, hk, kd)) for _ in range(2)]
    v = randn((batch, length, hv, vd))
    # Qwen gate parameterization; synthetic logits, not trained activations.
    a, b = [randn((batch, length, hv)) for _ in range(2)]
    a_log = torch.linspace(-2, 1, hv, dtype=dtype)
    gate_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    g = -a_log.to(gate_dtype).exp() * F.softplus(a.to(gate_dtype))
    if case == "weak_decay":
        g = g * 0.001
    if case == "zero_decay":
        g.zero_()
    beta = b.sigmoid()
    state = (
        torch.zeros(batch, hv, kd, vd, dtype=gate_dtype)
        if initial_zero
        else randn((batch, hv, kd, vd)).to(gate_dtype) * 0.1
    )
    args = [x.detach().requires_grad_() for x in (q, k, v, g, beta, state)]
    cot = (
        randn((batch, length, hv, vd)),
        randn((batch, hv, kd, vd)).to(gate_dtype) * 0.1,
    )
    return args, cot


def configuration(name):
    return json.loads((ROOT / "references/qwen35" / f"{name}.json").read_text())[
        "text_config"
    ]
