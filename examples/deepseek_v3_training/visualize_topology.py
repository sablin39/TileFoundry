"""Draw an expert route across two nodes and its tile schedule on the three resources."""

from colorsys import hls_to_rgb

from visualize import (
    COLOR,
    PROBE_LABELS,
    PROBES,
    SVG,
    arguments,
    observed,
    read_samples,
    simultaneous,
)
from workload import CONTEXT, PASSES, trace

TILES = ("#b75427", "#246d95", "#6d57a0", "#3f8055", "#9b5369", "#537b79", "#8a6c28", "#555d80")
INK, MUTED, RULE = "#20333e", "#48616c", "#d4dcd9"
WIDTH = 1500


def tile_color(tile):
    if tile < len(TILES):
        return TILES[tile]
    red, green, blue = hls_to_rgb((tile * 0.61803398875) % 1, 0.38, 0.6)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def route_map(svg, layer):
    svg.box(34, 116, 382, 357, "#edf2f3", radius=15, stroke="#c2d4db")
    svg.box(752, 116, 714, 357, "#edf3ee", radius=15, stroke="#c3d6c9")
    svg.text(54, 154, "Source node · 16 GPUs", 25, 700)
    svg.text(774, 154, "Expert node · 16 GPUs", 25, 700)
    svg.box(56, 181, 337, 171, "#ffffff", radius=10, stroke="#91b0be")
    svg.text(75, 213, "Source GPU 0", 24, 700, COLOR["NVLink"])
    svg.text(75, 239, "Owns the token activation", 17)
    svg.text(75, 272, "fwd  Weight and sum expert outputs", 16)
    svg.text(75, 299, "bwd  Sum expert input gradients", 16)
    svg.text(75, 329, "Keeps the original token / expert slots", 15, color=MUTED)
    svg.box(775, 181, 238, 171, "#ffffff", radius=10, stroke="#91b0be")
    svg.text(793, 213, "Gateway GPU 0", 23, 700, COLOR["NVLink"])
    svg.text(793, 239, "Receives from InfiniBand", 16)
    svg.text(793, 272, "Stages each incoming tile", 16)
    svg.text(793, 299, "Relays over NVLink", 16)
    svg.text(793, 329, "Returns completed tiles", 16, color=MUTED)
    svg.box(1175, 181, 269, 171, "#ffffff", radius=10, stroke="#8aaf99")
    svg.text(1193, 213, "Expert GPU 1", 24, 700, COLOR["Compute"])
    svg.text(1193, 241, f"Routed block {layer}", 16)
    svg.text(1193, 272, "Gate/up projection", 18, 700)
    svg.text(1193, 299, "SwiGLU → down projection", 17)
    svg.text(1193, 329, "Produces the expert output", 16, color=MUTED)

    svg.text(582, 210, "InfiniBand", 26, 700, COLOR["IB"], "middle")
    svg.text(582, 239, "between nodes", 17, 400, MUTED, "middle")
    svg.arrow([(397, 274), (766, 274)], COLOR["IB"], width=4)
    svg.arrow([(767, 306), (398, 306)], COLOR["IB"], width=3)
    svg.text(582, 339, "fwd activations / bwd output gradients →", 15, 400, COLOR["IB"], "middle")
    svg.text(582, 365, "← expert outputs / input gradients", 15, 400, COLOR["IB"], "middle")
    svg.box(1031, 234, 126, 64, "#dbe9ef", radius=8, stroke="#91b0be")
    svg.text(1094, 260, "NVLink", 20, 700, COLOR["NVLink"], "middle")
    svg.text(1094, 283, "NVSwitch fabric", 13, 400, COLOR["NVLink"], "middle")
    svg.arrow([(1016, 252), (1029, 252)], COLOR["NVLink"], width=3)
    svg.arrow([(1159, 252), (1172, 252)], COLOR["NVLink"], width=3)
    svg.arrow([(1172, 285), (1159, 285)], COLOR["NVLink"], width=3)
    svg.arrow([(1029, 285), (1016, 285)], COLOR["NVLink"], width=3)
    for origin, count, active in ((56, 16, {0}), (775, 16, {0, 1})):
        step = 20 if origin == 56 else 40
        for index in range(count):
            color = COLOR["Compute"] if origin != 56 and index == 1 else COLOR["NVLink"]
            svg.box(
                origin + index * step,
                389,
                step - 5,
                24,
                color if index in active else "#d2dedb",
                radius=3,
            )
    svg.text(56, 443, "Source and 15 other GPUs in this node", 15, color=MUTED)
    svg.text(
        775,
        443,
        "Gateway, expert owner and 14 other GPUs · selected route highlighted",
        15,
        color=MUTED,
    )
    svg.text(
        40,
        507,
        "The selected expert is on another GPU in the remote node. Same-node routes use the local NVLink fabric.",
        17,
        color=MUTED,
    )


def timeline(svg, phase, top, plans, tiles, tolerance, unit):
    actions = {action.key: action for part in PASSES[phase] for action in part}
    bulk, ready = plans
    end = max(event.end for event in ready)
    bulk_end = max(event.end for event in bulk)
    direction = "fwd" if phase == "Forward" else "bwd"
    svg.text(40, top, f"{direction} · tiles moving through the same route", 26, 700)
    svg.text(
        40,
        top + 29,
        "Activation → gate/up → down → expert output"
        if phase == "Forward"
        else "Weighted output gradient → down input gradient → gate/up input gradient → token owner",
        17,
        color=MUTED,
    )
    svg.text(
        1450,
        top,
        f"Tile-ready {end:,.3g} {unit}  ·  bulk {bulk_end:,.3g} {unit}",
        18,
        700,
        anchor="end",
    )
    x0, width, row_height = 272, 1165, 59
    y0 = top + 68
    scale = width / end
    resources = ("IB", "NVLink", "Compute")
    row_y = {name: y0 + index * row_height for index, name in enumerate(resources)}
    windows = simultaneous(ready, tolerance)
    for start, stop in windows:
        svg.box(
            x0 + start * scale, y0 - 13, (stop - start) * scale, row_height * 3, "#f5e8be", radius=3
        )
    for name, label, owner in (
        ("IB", "InfiniBand", "Source node ↔ expert node"),
        ("NVLink", "NVLink", "Gateway GPU ↔ expert GPU"),
        ("Compute", "Expert compute", "Gate/up and down projections"),
    ):
        y = row_y[name]
        svg.box(40, y + 3, 5, 38, COLOR[name], radius=2)
        svg.text(57, y + 19, label, 20, 700, COLOR[name])
        svg.text(57, y + 40, owner, 13, color=MUTED)
        svg.box(x0, y + 6, width, 35, "#e3e9e5", radius=5)
    for tick in range(7):
        time = end * tick / 6
        x = x0 + time * scale
        svg.box(x, y0 - 7, 0.8, row_height * 3 - 5, RULE)
        svg.text(x, y0 + row_height * 3 + 11, f"{time:.3g}", 13, color=MUTED, anchor="middle")
    phase_of = {action.key: index for index, part in enumerate(PASSES[phase]) for action in part}
    selected = sorted((event for event in ready if event.tile == 0), key=lambda event: event.start)
    for before, after in zip(selected, selected[1:]):
        if (
            before.stage.resource != after.stage.resource
            and phase_of[before.stage.name] == phase_of[after.stage.name]
        ):
            svg.arrow(
                [
                    (x0 + before.end * scale, row_y[before.stage.resource] + 23),
                    (x0 + after.start * scale, row_y[after.stage.resource] + 23),
                ],
                tile_color(0),
                width=1.5,
                dashed=True,
            )
    for event in ready:
        x = x0 + event.start * scale
        width = (event.end - event.start) * scale
        y = row_y[event.stage.resource] + 6
        action = actions[event.stage.name]
        svg.items.append("<g>")
        svg.items.append(
            f"<title>{direction} · {action.title} · tile {event.tile + 1} · {event.start:g}–{event.end:g} {unit}</title>"
        )
        svg.box(x + 1, y, max(0.3, width - 2), 35, tile_color(event.tile), radius=4)
        if width > 38:
            svg.text(x + width / 2, y + 23, f"tile {event.tile + 1}", 13, 700, "white", "middle")
        svg.items.append("</g>")
    operations = (
        "Gate/up projection → SwiGLU → down projection"
        if phase == "Forward"
        else "Down input gradient → down weight gradient → gate/up input gradient → gate/up weight gradients"
    )
    svg.text(x0, y0 + row_height * 3 + 40, operations, 15, 700, COLOR["Compute"])
    legend_y = top + 342
    svg.text(40, legend_y, "Tile identity", 17, 700)
    for tile in range(min(tiles, 8)):
        x = 166 + tile * 115
        svg.box(x, legend_y - 19, 101, 27, tile_color(tile), radius=4)
        svg.text(x + 50, legend_y, f"tile {tile + 1}", 15, 700, "white", "middle")
    if tiles > 8:
        svg.text(1115, legend_y, f"{tiles - 8} more tiles", 15, color=MUTED)
    svg.text(
        40,
        top + 384,
        "Each color follows the same tile through transfer and compute. Dashed arrows follow tile 1 across resources.",
        16,
    )
    if windows:
        time = sum(windows[0]) / 2
        active = {event.stage.resource: event for event in ready if event.start <= time < event.end}
        svg.text(
            40, top + 416, f"Inside the first gold interval · {time:.3g} {unit}", 18, 700, "#715019"
        )
        for index, name in enumerate(resources):
            event = active[name]
            x = 40 + index * 478
            svg.box(x, top + 434, 462, 75, "#f8eed2", radius=7)
            svg.box(x + 12, top + 449, 5, 45, tile_color(event.tile), radius=2)
            label = {"IB": "InfiniBand", "NVLink": "NVLink", "Compute": "Expert GPU"}[name]
            svg.text(x + 28, top + 461, f"{label} · tile {event.tile + 1}", 19, 700, COLOR[name])
            svg.text(x + 28, top + 489, actions[event.stage.name].title, 15)
    else:
        svg.text(
            40,
            top + 426,
            "This schedule has no interval with all three resources active together.",
            17,
            color=MUTED,
        )


def main():
    args = arguments(__doc__, "topology.svg")
    samples, probes, rows = read_samples(args.samples)
    captures, tolerance = observed(args.trace, args.tiles, args.slots)
    complete = probes == PROBES.keys()
    unit = "µs" if complete or captures else "ticks"
    svg = SVG(
        WIDTH,
        1700,
        "Where a remote expert tile goes",
        "A two-node route through InfiniBand, a gateway and NVLink, linked to forward and backward timelines generated by TileFoundry.",
    )
    svg.text(40, 51, "Where a remote expert tile goes", 35, 700)
    batch = f"{args.sequences:,} sequences/update" if args.sequences else "training batch open"
    svg.text(
        42,
        82,
        f"B300 study · {args.nodes:,} nodes × 16 GPUs · {CONTEXT:,} tokens/sequence · {batch}",
        17,
        color=MUTED,
    )
    route_map(svg, args.layer)
    for phase, top in (("Forward", 558), ("Backward", 1110)):
        plans = (
            captures[phase]
            if captures
            else [
                trace(PASSES[phase], args.tiles, args.slots, policy, samples if complete else {})
                for policy in ("bulk", "tile_ready")
            ]
        )
        timeline(svg, phase, top, plans, args.tiles, tolerance, unit)
    if captures:
        detail = f"Observed activity. Gold intervals exclude {tolerance:g} µs of clock uncertainty at both ends."
    elif complete:
        detail = (
            "Modeled µs from isolated SM103 medians. Co-running contention requires a device trace."
        )
    elif probes:
        missing = ", ".join(PROBE_LABELS[name] for name in sorted(PROBES.keys() - probes))
        detail = f"Dependency ticks · isolated {rows}-row probes available for {', '.join(PROBE_LABELS[name] for name in sorted(probes))} · waiting for {missing} calibration."
    else:
        detail = "Dependency ticks from TileFoundry. Gold means simultaneous service in the model. Hardware timing awaits calibration."
    svg.text(40, 1662, detail, 16, color=MUTED)
    svg.save(args.out)


if __name__ == "__main__":
    main()
