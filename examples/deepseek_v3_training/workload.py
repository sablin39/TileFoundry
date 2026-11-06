"""Remote expert route and backbone of the DeepSeek-V3 next-token training example."""

from dataclasses import dataclass

HIDDEN, EXPERT, TOP_K, CONTEXT = 7168, 2048, 8, 1_048_576
BLOCKS = ("dense",) * 3 + ("routed",) * 58


@dataclass(frozen=True)
class Action:
    key: str
    resource: str
    title: str
    detail: str
    sample: str


FORWARD = (
    (
        Action(
            "x_ib",
            "IB",
            "Send activation to the expert node",
            "Source GPU → remote gateway · token row x",
            "ib_out",
        ),
        Action(
            "x_nvl",
            "NVLink",
            "Relay x to the expert owner",
            "Gateway → expert GPU · input tile becomes ready",
            "nvlink_out",
        ),
        Action(
            "gate_up",
            "Compute",
            "Project gate and up",
            "Expert GPU · x × Wgate and x × Wup",
            "gate_up",
        ),
    ),
    (
        Action(
            "down",
            "Compute",
            "Activate and project down",
            "Expert GPU · SwiGLU followed by Wdown",
            "down",
        ),
        Action(
            "y_nvl",
            "NVLink",
            "Return the expert contribution",
            "Expert GPU → gateway · output tile",
            "nvlink_back",
        ),
        Action(
            "y_ib",
            "IB",
            "Return to the token owner",
            "Gateway → source node · ready for weighted sum",
            "ib_back",
        ),
    ),
)
BACKWARD = (
    (
        Action(
            "dy_ib",
            "IB",
            "Send the output gradient",
            "Source GPU → remote gateway · weighted dY",
            "ib_out",
        ),
        Action(
            "dy_nvl",
            "NVLink",
            "Deliver dY to the expert",
            "Gateway → expert GPU · saved forward route",
            "nvlink_out",
        ),
        Action(
            "down_input",
            "Compute",
            "Backpropagate through Wdown",
            "Expert GPU · produce the hidden-state gradient",
            "down_input",
        ),
    ),
    (
        Action(
            "down_weight",
            "Compute",
            "Accumulate the Wdown gradient",
            "Saved hidden rows and incoming dY → dWdown",
            "down_weight",
        ),
    ),
    (
        Action(
            "up_input",
            "Compute",
            "Backpropagate to the input",
            "SwiGLU derivative and gate/up weights → dX",
            "up_input",
        ),
        Action(
            "dx_nvl",
            "NVLink",
            "Return the input gradient",
            "Expert GPU → gateway · one contribution to dX",
            "nvlink_back",
        ),
        Action(
            "dx_ib",
            "IB",
            "Send dX back to its origin",
            "Gateway → source node · sum by original token",
            "ib_back",
        ),
    ),
    (
        Action(
            "up_weight",
            "Compute",
            "Accumulate gate and up gradients",
            "Saved inputs and activation derivatives → dWgate, dWup",
            "up_weight",
        ),
    ),
)
PASSES = {"Forward": FORWARD, "Backward": BACKWARD}


def trace(phases, tiles, slots, policy, samples):
    from tilefoundry.analysis import TileStage, tile_pipeline

    stages = tuple(
        tuple(TileStage(a.key, a.resource, samples[a.sample] if samples else 1) for a in phase)
        for phase in phases
    )
    return tile_pipeline(stages, tiles=tiles, slots=slots, policy=policy)


def cp_payload(ranks):
    """Logical BF16 bytes sent per rank per MLA layer, with one sequence per CP group."""
    heads, q, k, v, shared, latent = 128, 192, 128, 128, 64, 512
    if ranks <= 0 or CONTEXT % ranks or heads % ranks:
        raise ValueError("CP degree must divide both the sequence length and 128 heads")
    block_bytes = 2 * (CONTEXT // ranks)
    # A transpose sends (P-1)/P of each rank's tensor to other ranks.
    inputs = block_bytes * (heads // ranks) * (ranks - 1) * (q + k + v)
    output = block_bytes * (heads // ranks) * (ranks - 1) * v
    shared_key = block_bytes * (ranks - 1) * shared
    compact = block_bytes * (latent + shared)
    return {
        "Ulysses": (inputs, shared_key, output),
        "Ring": ((ranks - 1) * compact, ranks * compact if ranks > 1 else 0),
    }


def expert_ids(token, case):
    """Controlled top-eight choices over 32 toy experts, eight on each rank."""
    source = token % 4
    if case == "colocated":
        owners = [source ^ 1] * 4 + [source ^ 2] * 4
    elif case == "spread":
        owners = [(source + slot) % 4 for slot in range(TOP_K)]
    elif case == "empty":
        owners = [3] * TOP_K
    else:
        raise ValueError(case)
    return [rank * TOP_K + slot for slot, rank in enumerate(owners)]
