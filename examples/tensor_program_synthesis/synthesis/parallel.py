"""A supplied affine-scan formulation and dependency analysis for long sequences.

The affine identity is supplied, not discovered by the residual solver. Both
the original and this graph lower through the same TileFoundry HIR path.
"""

from math import ceil, log2, sqrt

from .graph import Graph
from .problems import backward_chunk, finish_chunk, prepare_chunk


def compose(after, before):
    """Composition of S -> A S + B; order is noncommutative."""
    a, b = after
    if before is None:
        return after
    c, d = before
    return a @ c, a @ d + b


def affine_prefix(pairs):
    """Work-efficient balanced exclusive scan, with implicit identity prefixes.

    Upsweep totals and downsweep prefixes share a tree. No power-of-two padding
    is needed. Each level contains independent matrix products.
    """

    def build(lo, hi):
        if hi - lo == 1:
            return lo, hi, pairs[lo], None, None
        middle = (lo + hi) // 2
        left, right = build(lo, middle), build(middle, hi)
        return lo, hi, compose(right[2], left[2]), left, right

    prefixes = [None] * len(pairs)

    def distribute(tree, prefix):
        lo, hi, _, left, right = tree
        if hi - lo == 1:
            prefixes[lo] = prefix
            return
        distribute(left, prefix)
        distribute(right, compose(left[2], prefix))

    tree = build(0, len(pairs))
    distribute(tree, None)
    return prefixes, tree[2]


def apply_affine(pair, initial):
    return initial if pair is None else pair[0] @ initial + pair[1]


def scan_gated_delta_rule(length=512, chunk_size=64, key_dim=64, value_dim=64):
    """Explicit forward and backward affine scans with local analytic VJPs."""
    if min(length, chunk_size, key_dim, value_dim) <= 0:
        raise ValueError("Dimensions must be positive")
    graph = Graph("chunk_gated_delta_rule")
    q = graph.input("Q", (length, key_dim))
    k = graph.input("K", (length, key_dim))
    v = graph.input("V", (length, value_dim))
    g = graph.input("log_decay", (length, 1))
    beta = graph.input("beta", (length, 1))
    initial = graph.input("S0", (key_dim, value_dim))
    scale = 1 / sqrt(key_dim)
    chunks, maps = [], []
    for index, start in enumerate(range(0, length, chunk_size)):
        end = min(length, start + chunk_size)
        ctx = prepare_chunk(
            graph, *(x.rows(start, end) for x in (q, k, v, g, beta)), index, scale
        )
        # U = inv beta V - inv beta D K S = U_base - W S.
        ctx["U_base"] = (ctx["inverse"] @ (ctx["beta"] * ctx["v"])).named(
            f"c{index}.U_base"
        )
        ctx["W"] = (ctx["inverse"] @ (ctx["beta"] * ctx["decay"] * ctx["k"])).named(
            f"c{index}.W"
        )
        ctx["A"] = (
            ctx["end_decay"] * graph.constant("eye", key_dim)
            - ctx["weighted_k"].T @ ctx["W"]
        ).named(f"c{index}.A")
        ctx["B"] = (ctx["weighted_k"].T @ ctx["U_base"]).named(f"c{index}.B")
        # O = Z S + attention U_base. Z also yields the backward forcing.
        ctx["Z"] = (ctx["decay"] * ctx["qs"] - ctx["attention"] @ ctx["W"]).named(
            f"c{index}.Z"
        )
        maps.append((ctx["A"], ctx["B"]))
        chunks.append((start, end, ctx))
    prefixes, total = affine_prefix(maps)
    outputs = []
    for index, ((_, _, ctx), prefix) in enumerate(zip(chunks, prefixes, strict=True)):
        state = (
            apply_affine(prefix, initial).named(f"S_before_chunk_{index}")
            if prefix
            else initial
        )
        output, _, _ = finish_chunk(ctx, state, index, affine=True)
        outputs.append(output)
    final_state = apply_affine(total, initial).named("S_final_scan")
    graph.begin_backward([graph.op("concat", *outputs).named("O"), final_state])
    dy = graph.input("dO", (length, value_dim))
    ds_final = graph.input("dS_final", (key_dim, value_dim))
    backward_maps = [
        (ctx["A"].T, ctx["Z"].T @ dy.rows(start, end))
        for start, end, ctx in reversed(chunks)
    ]
    reverse_prefixes, reverse_total = affine_prefix(backward_maps)
    gradients = [[] for _ in range(5)]
    for (start, end, ctx), prefix in zip(
        reversed(chunks), reverse_prefixes, strict=True
    ):
        ds_out = apply_affine(prefix, ds_final)
        *local, _ = backward_chunk(ctx, dy.rows(start, end), ds_out, scale)
        for values, gradient in zip(gradients, local, strict=True):
            values.insert(0, gradient)
    names = ("dQ", "dK", "dV", "dlog_decay", "dbeta")
    grads = [
        graph.op("concat", *parts).named(name)
        for name, parts in zip(names, gradients, strict=True)
    ]
    graph.finish([*grads, apply_affine(reverse_total, ds_final).named("dS0")])
    graph.metadata = {
        "immutable_inputs": True,
        "length": length,
        "chunk_size": chunk_size,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunks": len(chunks),
        "tail_length": chunks[-1][1] - chunks[-1][0],
        "scale": scale,
        "batch_heads": "one independent sequence/head",
        "state_schedule": "balanced affine prefix and reverse-adjoint scans",
        "scan_depth_upper_bound_per_direction": 2 * ceil(log2(len(chunks))) + 1,
        "discovery_scope": "affine factorization is a supplied alternative algorithm",
        "gate_contract": "scalar log-decay and already activated beta per token",
        "triangular_solve": "finite nilpotent matrix polynomial",
    }
    return graph


def prepared_gated_delta_rule(length=512, chunk_size=64, key_dim=128, value_dim=128):
    """Qwen-style parallel U_base/W preparation with a sequential state loop.

    The local analytic VJP is retained; the state adjoint uses W.T @ dU to avoid
    reconstructing its factorization through J and beta*d*K on the chain.
    """
    if min(length, chunk_size, key_dim, value_dim) <= 0:
        raise ValueError("Dimensions must be positive")
    graph = Graph("chunk_gated_delta_rule")
    q, k = [graph.input(name, (length, key_dim)) for name in ("Q", "K")]
    v = graph.input("V", (length, value_dim))
    g, beta = [graph.input(name, (length, 1)) for name in ("log_decay", "beta")]
    state = graph.input("S0", (key_dim, value_dim))
    scale, chunks, outputs = 1 / sqrt(key_dim), [], []
    for index, start in enumerate(range(0, length, chunk_size)):
        end = min(length, start + chunk_size)
        ctx = prepare_chunk(
            graph, *(x.rows(start, end) for x in (q, k, v, g, beta)), index, scale
        )
        ctx["U_base"] = (ctx["inverse"] @ (ctx["beta"] * ctx["v"])).named(
            f"c{index}.U_base"
        )
        ctx["W"] = (ctx["inverse"] @ (ctx["beta"] * ctx["decay"] * ctx["k"])).named(
            f"c{index}.W"
        )
        chunks.append((start, end, ctx))
    for index, (_, _, ctx) in enumerate(chunks):
        output, state, _ = finish_chunk(ctx, state, index, affine=True)
        outputs.append(output)
    graph.begin_backward([graph.op("concat", *outputs).named("O"), state])
    dy = graph.input("dO", (length, value_dim))
    ds = graph.input("dS_final", (key_dim, value_dim))
    gradients = [[] for _ in range(5)]
    for start, end, ctx in reversed(chunks):
        do = dy.rows(start, end)
        *local, _ = backward_chunk(ctx, do, ds, scale)
        du = ctx["attention"].T @ do + ctx["weighted_k"] @ ds
        ds = ctx["qs"].T @ (ctx["decay"] * do) + ctx["end_decay"] * ds - ctx["W"].T @ du
        for parts, value in zip(gradients, local, strict=True):
            parts.insert(0, value)
    names = ("dQ", "dK", "dV", "dlog_decay", "dbeta")
    graph.finish(
        [
            *(
                graph.op("concat", *parts).named(name)
                for name, parts in zip(names, gradients, strict=True)
            ),
            ds.named("dS0"),
        ]
    )
    graph.metadata = {
        "immutable_inputs": True,
        "length": length,
        "chunk_size": chunk_size,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunks": len(chunks),
        "tail_length": chunks[-1][1] - chunks[-1][0],
        "scale": scale,
        "state_schedule": "Qwen-style U_base/W preparation and sequential state propagation",
        "triangular_solve": "prototype polynomial inverse; not the official row-elimination implementation",
    }
    return graph


def dependency_report(graph, roots, *, available=(), recipes=None):
    """Ideal DAG metrics, not an execution schedule or a GPU timing model.

    Weighted span treats each whole tensor operation as indivisible and weights
    it by elementary work. Matmul depth counts serially dependent matmuls.
    """
    order = graph.schedule(roots, available=available, recipes=recipes)
    weighted, matmul_depth, depth, state_dependent = {}, {}, {}, {}
    total = mm_count = independent = 0
    for i in order:
        n = graph.nodes[i]
        r = n.recipes[0 if recipes is None else recipes.get(i, 0)]
        cost = graph.work(i, r)
        total += cost
        is_mm = int(r.op == "matmul")
        mm_count += is_mm
        weighted[i] = cost + max((weighted.get(j, 0) for j in r.args), default=0)
        matmul_depth[i] = is_mm + max(
            (matmul_depth.get(j, 0) for j in r.args), default=0
        )
        depth[i] = int(cost > 0) + max((depth.get(j, 0) for j in r.args), default=0)
        state_dependent[i] = i == graph.forward_inputs[-1] or any(
            state_dependent.get(j, j == graph.forward_inputs[-1]) for j in r.args
        )
        if not state_dependent[i]:
            independent += cost
    span = max((weighted.get(i, 0) for i in roots), default=0)
    return {
        "work_units": total,
        "weighted_span_units": span,
        "ideal_work_over_span": total / span if span else 0,
        "matmul_count": mm_count,
        "dependent_matmul_depth": max(
            (matmul_depth.get(i, 0) for i in roots), default=0
        ),
        "tensor_op_depth": max((depth.get(i, 0) for i in roots), default=0),
        "work_independent_of_initial_state": independent,
        "node_count": len(order),
    }


def plan_dependencies(graph, plan):
    return {
        "forward": dependency_report(graph, [*graph.forward_outputs, *plan.saved]),
        "backward": dependency_report(
            graph,
            graph.backward_outputs,
            available=graph.external | set(plan.saved),
            recipes=plan.recipes,
        ),
    }
