"""Supplied MLP and chunk gated-delta-rule forward and backward formulas."""

from math import sqrt

import torch

from .graph import Graph


def mlp(batch=8, inputs=16, hidden=32, outputs=8):
    graph = Graph("mlp")
    x = graph.input("X", (batch, inputs))
    w = graph.input("W", (inputs, hidden))
    v = graph.input("V", (hidden, outputs))
    z = (x @ w).named("Z")
    h = z.relu().named("H")
    y = (h @ v).named("Y")
    graph.begin_backward([y])
    dy = graph.input("dY", (batch, outputs))
    dv = h.T @ dy
    dh = dy @ v.T
    mask = z.positive().named("positive_Z")
    dz = dh * mask.cast()
    graph.finish([(dz @ w.T).named("dX"), (x.T @ dz).named("dW"), dv.named("dV")])
    graph.metadata = {
        "immutable_inputs": True,
        "relu_derivative_at_zero": 0,
        "shape": [batch, inputs, hidden, outputs],
    }
    graph.add_rules()
    return graph


def relu_sum(batch=8, inputs=16, hidden=32):
    """A compact-mask witness: backward needs a predicate, not the activation."""
    graph = Graph("relu_sum")
    x = graph.input("X", (batch, inputs))
    w = graph.input("W", (inputs, hidden))
    z = (x @ w).named("Z")
    h = z.relu().named("H")
    graph.begin_backward([h.sum(0).sum(1).named("Y")])
    dy = graph.input("dY", (1, 1))
    mask = z.positive().named("positive_Z")
    dz = mask.cast() * dy
    graph.finish([(dz @ w.T).named("dX"), (x.T @ dz).named("dW")])
    graph.metadata = {
        "immutable_inputs": True,
        "shape": [batch, inputs, hidden],
        "relu_derivative_at_zero": 0,
    }
    graph.add_rules()
    return graph


def inverse_unit_lower(graph, lower):
    """Finite matrix polynomial for (I+L)^-1 with strictly lower triangular L.

    Doubling covers all powers up to C-1; higher powers vanish. This is an
    explicit bounded HIR decomposition, not a tuned triangular-solve kernel.
    """
    size = lower.node.shape[0]
    eye = graph.constant("eye", size)
    power = -lower
    inverse = eye + power
    covered = 2
    while covered < size:
        power = power @ power
        inverse = inverse @ (eye + power)
        covered *= 2
    return inverse


def prepare_chunk(graph, q, k, v, g, beta, index, scale):
    """State-independent preparation; all chunks may execute concurrently."""
    size = q.node.shape[0]
    lower = graph.constant("lower", size)
    strict = graph.constant("strict", size)
    last = graph.constant("last", size)
    prefix = (lower @ g).named(f"c{index}.prefix_log_decay")
    decay = prefix.exp().named(f"c{index}.decay")
    pair_decay = (((prefix - prefix.T) * lower).exp() * lower).named(
        f"c{index}.pair_decay"
    )
    last_log = last.T @ prefix
    end_decay = last_log.exp().named(f"c{index}.end_decay")
    suffix_decay = (last_log - prefix).exp().named(f"c{index}.suffix_decay")
    qs = (q * scale).named(f"c{index}.scaled_q")
    kk = (k @ k.T).named(f"c{index}.KK")
    triangular = (beta * kk * pair_decay * strict).named(f"c{index}.L")
    inverse = inverse_unit_lower(graph, triangular).named(f"c{index}.inverse")
    qk = (qs @ k.T).named(f"c{index}.QK")
    attention = (qk * pair_decay * lower).named(f"c{index}.attention")
    weighted_k = (suffix_decay * k).named(f"c{index}.weighted_K")
    context = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        lower=lower,
        strict=strict,
        last=last,
        decay=decay,
        pair_decay=pair_decay,
        end_decay=end_decay,
        suffix_decay=suffix_decay,
        qs=qs,
        kk=kk,
        inverse=inverse,
        qk=qk,
        attention=attention,
        weighted_k=weighted_k,
    )
    return context


def finish_chunk(context, state, index, *, affine=False):
    """Attach the state-dependent work after preparation or a state scan."""
    k, v, beta, decay = (context[x] for x in ("k", "v", "beta", "decay"))
    ks = (k @ state).named(f"c{index}.KS")
    difference = (v - decay * ks).named(f"c{index}.innovation")
    rhs = (beta * difference).named(f"c{index}.rhs")
    u = (
        context["U_base"] - context["W"] @ state if affine else context["inverse"] @ rhs
    ).named(f"c{index}.U")
    qs_state = (context["qs"] @ state).named(f"c{index}.QS")
    output = (decay * qs_state + context["attention"] @ u).named(f"c{index}.O")
    next_state = (context["end_decay"] * state + context["weighted_k"].T @ u).named(
        f"S_after_chunk_{index}"
    )
    context.update(state=state, ks=ks, difference=difference, u=u, qs_state=qs_state)
    return output, next_state, context


def forward_chunk(graph, q, k, v, g, beta, state, index, scale):
    """Use a chunk triangular system and matrix products to advance the state."""
    context = prepare_chunk(graph, q, k, v, g, beta, index, scale)
    return finish_chunk(context, state, index)


def backward_chunk(ctx, do, ds, scale):
    """Explicit VJP of the chunk equations, including both gate derivatives.

    For (I+L)U=R, dR=(I+L)^-T dU and dL=-tril(dR U^T,-1).
    The backward is supplied algebraically and never generated by autograd.
    """
    k, beta, state = ctx["k"], ctx["beta"], ctx["state"]
    d, pair, e, suffix = (
        ctx["decay"],
        ctx["pair_decay"],
        ctx["end_decay"],
        ctx["suffix_decay"],
    )
    qs, u, lower, strict = ctx["qs"], ctx["u"], ctx["lower"], ctx["strict"]
    du = ctx["attention"].T @ do + ctx["weighted_k"] @ ds
    dr = ctx["inverse"].T @ du
    dl = -(dr @ u.T) * strict
    datt = (do @ u.T) * lower
    dqk = datt * pair
    dpair = datt * ctx["qk"] + dl * beta * ctx["kk"]
    dkk = dl * beta * pair
    dweighted_k = u @ ds.T
    dsuffix = (dweighted_k * k).sum(1)
    de = (ds * state).sum(0).sum(1)
    dd = (do * ctx["qs_state"]).sum(1)
    dinnovation = dr * beta
    dks = -(dinnovation * d)
    dd = dd - (dinnovation * ctx["ks"]).sum(1)
    dq = ((do * d) @ state.T + dqk @ k) * scale
    dk = dqk.T @ qs + dweighted_k * suffix + dks @ state.T + (dkk + dkk.T) @ k
    dv = dinnovation
    dbeta = (dr * ctx["difference"]).sum(1) + (dl * ctx["kk"] * pair).sum(1)
    pair_adjoint = dpair * pair
    suffix_adjoint = dsuffix * suffix
    dc = (
        dd * d
        + pair_adjoint.sum(1)
        - pair_adjoint.sum(0).T
        - suffix_adjoint
        + ctx["last"] * (suffix_adjoint.sum(0) + de * e)
    )
    dg = lower.T @ dc
    dstate = qs.T @ (do * d) + e * ds + k.T @ dks
    return dq, dk, dv, dg, dbeta, dstate


def chunk_gated_delta_rule(length=10, chunk_size=4, key_dim=3, value_dim=2):
    """One head and sequence, several chunks, with an exact short-tail path."""
    if min(length, chunk_size, key_dim, value_dim) <= 0:
        raise ValueError("Dimensions must be positive")
    graph = Graph("chunk_gated_delta_rule")
    q = graph.input("Q", (length, key_dim))
    k = graph.input("K", (length, key_dim))
    v = graph.input("V", (length, value_dim))
    g = graph.input("log_decay", (length, 1))
    beta = graph.input("beta", (length, 1))
    state = graph.input("S0", (key_dim, value_dim))
    scale = 1 / sqrt(key_dim)
    chunks, outputs = [], []
    for index, start in enumerate(range(0, length, chunk_size)):
        end = min(start + chunk_size, length)
        parts = [x.rows(start, end) for x in (q, k, v, g, beta)]
        output, state, context = forward_chunk(graph, *parts, state, index, scale)
        chunks.append((start, end, context))
        outputs.append(output)
    y = graph.op("concat", *outputs).named("O")
    graph.begin_backward([y, state])
    dy = graph.input("dO", (length, value_dim))
    ds = graph.input("dS_final", (key_dim, value_dim))
    gradients = [[] for _ in range(5)]
    for start, end, context in reversed(chunks):
        *local, ds = backward_chunk(context, dy.rows(start, end), ds, scale)
        for values, gradient in zip(gradients, local, strict=True):
            values.insert(0, gradient)
    names = ("dQ", "dK", "dV", "dlog_decay", "dbeta")
    grads = [
        graph.op("concat", *parts).named(name)
        for name, parts in zip(names, gradients, strict=True)
    ]
    graph.finish([*grads, ds.named("dS0")])
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
        "gate_contract": "g is scalar log-decay per token; beta is an already activated scalar",
        "sources": [
            "https://arxiv.org/abs/2412.06464",
            "https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py",
        ],
        "triangular_solve": "finite nilpotent matrix polynomial, explicit fixed-size decomposition",
    }
    return graph


def recurrent_oracle(q, k, v, g, beta, initial_state):
    """Independent token recurrence, differentiable only for validation."""
    state = initial_state
    outputs = []
    scale = 1 / sqrt(q.shape[-1])
    for t in range(q.shape[0]):
        decayed = g[t].exp() * state
        update = beta[t] * (v[t] - k[t] @ decayed)
        state = decayed + k[t, :, None] * update[None, :]
        outputs.append((q[t] * scale) @ state)
    return torch.stack(outputs), state
