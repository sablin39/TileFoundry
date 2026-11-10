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
            "Source GPU → remote gateway · token activation",
            "ib_out",
        ),
        Action(
            "x_nvl",
            "NVLink",
            "Deliver the token activation",
            "Gateway → expert GPU · input tile becomes ready",
            "nvlink_out",
        ),
        Action(
            "gate_up",
            "Compute",
            "Gate/up projection · Expert.w13",
            "Token activation × gate and up weights",
            "gate_up",
        ),
    ),
    (
        Action(
            "down",
            "Compute",
            "SwiGLU + down projection · Expert.w2",
            "Hidden activation → expert output",
            "down",
        ),
        Action(
            "y_nvl",
            "NVLink",
            "Return the expert output",
            "Expert GPU → gateway · expert output tile",
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
            "Source GPU → gateway · weighted output gradient",
            "ib_out",
        ),
        Action(
            "dy_nvl",
            "NVLink",
            "Deliver the output gradient",
            "Gateway → expert GPU · saved forward route",
            "nvlink_out",
        ),
        Action(
            "down_input",
            "Compute",
            "Down projection · input gradient",
            "Output gradient → hidden-activation gradient",
            "down_input",
        ),
    ),
    (
        Action(
            "down_weight",
            "Compute",
            "Down projection · weight gradient",
            "Saved hidden activation × output gradient",
            "down_weight",
        ),
    ),
    (
        Action(
            "up_input",
            "Compute",
            "Gate/up projections · input gradient",
            "SwiGLU derivative → token input gradient",
            "up_input",
        ),
        Action(
            "dx_nvl",
            "NVLink",
            "Return the input gradient",
            "Expert GPU → gateway · token input gradient",
            "nvlink_back",
        ),
        Action(
            "dx_ib",
            "IB",
            "Reduce at the token owner",
            "Gateway → source node · sum by original token",
            "ib_back",
        ),
    ),
    (
        Action(
            "up_weight",
            "Compute",
            "Gate/up projections · weight gradients",
            "Saved inputs × activation gradients",
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
    heads, query_channels, key_channels, value_channels = 128, 192, 128, 128
    positional_channels, latent_channels = 64, 512
    if ranks <= 0 or CONTEXT % ranks or heads % ranks:
        raise ValueError("CP degree must divide both the sequence length and 128 heads")
    block_bytes = 2 * (CONTEXT // ranks)
    # A transpose sends (P-1)/P of each rank's tensor to other ranks.
    inputs = (
        block_bytes
        * (heads // ranks)
        * (ranks - 1)
        * (query_channels + key_channels + value_channels)
    )
    output = block_bytes * (heads // ranks) * (ranks - 1) * value_channels
    shared_key = block_bytes * (ranks - 1) * positional_channels
    compact = block_bytes * (latent_channels + positional_channels)
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
