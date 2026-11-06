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
