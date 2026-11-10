"""Draw the complete backbone and the route through one selected expert layer."""

import argparse
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import median

from workload import BLOCKS, CONTEXT, EXPERT, HIDDEN, PASSES, TOP_K, trace

INK, PAPER = "#20333e", "#fbfaf6"
COLOR = {"IB": "#794d96", "NVLink": "#22649a", "Compute": "#207461"}
PROBE_LABELS = {"ib": "InfiniBand", "nvlink": "NVLink", "compute": "expert compute"}
PROBES = {
    "compute": {"gate_up", "down", "down_input", "down_weight", "up_input", "up_weight"},
    "ib": {"ib_out", "ib_back"},
    "nvlink": {"nvlink_out", "nvlink_back"},
}


def read_samples(paths):
    values, probes, rows = {}, set(), None
    for path in paths:
        p = json.loads(path.read_text())
        kind = p["probe"]
        if (p["schema"], p["scope"], p["dtype"], p["hidden"], p["expert"]) != (
            1,
            "isolated",
            "bf16",
            HIDDEN,
            EXPERT,
        ):
            raise ValueError(f"{path}: expected matching BF16 DeepSeek-V3 measurements")
        if kind not in PROBES or kind in probes or set(p["samples_us"]) != PROBES[kind]:
            raise ValueError(f"{path}: unknown, duplicate or incomplete probe")
        expected_devices = 1 if kind == "compute" else 2
        if len(p["devices"]) != expected_devices or any(d["cc"] != [10, 3] for d in p["devices"]):
            raise ValueError(f"{path}: expected SM103 device evidence")
        if kind != "compute":
            channel = "via P2P/" if kind == "nvlink" else "via NET/IB"
            if not any(channel in line for rank in p["evidence"]["nccl_channels"] for line in rank):
                raise ValueError(f"{path}: missing NCCL channel evidence for {kind}")
        if type(p["rows"]) is not int or p["rows"] < 1 or (rows is not None and p["rows"] != rows):
            raise ValueError(f"{path}: match the row count across probes")
        rows = p["rows"]
        for name, v in p["samples_us"].items():
            if not v or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in v
            ):
                raise ValueError(f"{path}: invalid samples for {name}")
            values[name] = median(v)
        probes.add(kind)
    return values, probes, rows


def simultaneous(spans, tolerance=0):
    """Intervals occupied by all three roles on one common clock."""
    changes = defaultdict(Counter)
    for s in spans:
        changes[s.start][s.stage.resource] += 1
        changes[s.end][s.stage.resource] -= 1
    active, result, prev = Counter(), [], None
    for t, change in sorted(changes.items()):
        if prev is not None and all(active[k] for k in COLOR):
            if result and result[-1][1] == prev:
                result[-1] = (result[-1][0], t)
            else:
                result.append((prev, t))
        active.update(change)
        prev = t
    return [(a + tolerance, b - tolerance) for a, b in result if b - a > 2 * tolerance]


def observed(path, tiles, slots):
    """Read paired, per-role completion traces expressed on a common device time base."""
    from tilefoundry.analysis import TileInterval, TileStage

    if path is None:
        return {}, 0
    raw = json.loads(path.read_text())
    tolerance = raw["clock_error_us"]
    if (raw["schema"], raw["cc"], raw["tiles"], raw["slots"]) != (1, [10, 3], tiles, slots):
        raise ValueError("capture geometry must match the requested SM103 trace")
    if (
        not raw["clock_domain"]
        or not raw["source"]
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("capture requires its source, clock domain and finite clock error")
    result = {}
    for name, phases in PASSES.items():
        result[name] = []
        actions = {a.key: a for phase in phases for a in phase}
        for policy in ("bulk", "tile_ready"):
            rows = raw["passes"][name][policy]
            events = {}
            for key, t, s, e in rows:
                if (
                    key not in actions
                    or type(t) is not int
                    or not 0 <= t < tiles
                    or (key, t) in events
                    or not math.isfinite(s)
                    or not math.isfinite(e)
                    or e <= s
                ):
                    raise ValueError("invalid or duplicate tile interval")
                events[key, t] = (s, e)
            if len(events) != len(actions) * tiles:
                raise ValueError("capture requires each stage and tile")
            fence = -math.inf
            for phase in phases:
                for i, a in enumerate(phase):
                    for t in range(tiles):
                        ready = fence
                        if i:
                            producer = phase[i - 1].key
                            ready = max(
                                ready,
                                max(events[producer, k][1] for k in range(tiles))
                                if policy == "bulk"
                                else events[producer, t][1],
                            )
                        if policy == "tile_ready" and i + 1 < len(phase) and t >= slots:
                            ready = max(ready, events[phase[i + 1].key, t - slots][1])
                        if events[a.key, t][0] + 2 * tolerance < ready:
                            raise ValueError(f"{policy}: {a.key} tile {t} precedes readiness")
                fence = max(events[a.key, t][1] for a in phase for t in range(tiles))
            origin = min(s for s, e in events.values())
            result[name].append(
                [
                    TileInterval(
                        TileStage(key, actions[key].resource, e - s), t, s - origin, e - origin
                    )
                    for (key, t), (s, e) in events.items()
                ]
            )
    return result, tolerance


class SVG:
    def __init__(
        self,
        width=1500,
        height=1180,
        title="DeepSeek-V3 training across nodes",
        description="Forward and backward stages of a remote expert route over InfiniBand and NVLink.",
    ):
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
            '<g font-family="Arial, Helvetica, sans-serif">',
        ]

    def text(self, x, y, value, size=16, weight=400, color=INK, anchor="start"):
        self.items.append(
            f'<text x="{x:.2f}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def box(self, x, y, w, h, color, radius=0, stroke="none"):
        style = f' rx="{radius}" stroke="{stroke}"' if radius or stroke != "none" else ""
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{color}"{style}/>'
        )

    def arrow(self, points, color, width=2, dashed=False):
        path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        self.items.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round"{dash}/>'
        )
        (prev_x, prev_y), (x, y) = points[-2:]
        angle = math.atan2(y - prev_y, x - prev_x)
        wings = [
            (x - 10 * math.cos(angle + turn), y - 10 * math.sin(angle + turn))
            for turn in (-0.45, 0.45)
        ]
        vertices = " ".join(f"{a:.2f},{b:.2f}" for a, b in [(x, y), *wings])
        self.items.append(f'<polygon points="{vertices}" fill="{color}"/>')

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.items + ["</g></svg>"]) + "\n")


def backbone(svg, layer):
    svg.box(38, 100, 1424, 135, "#eef1e9")
    svg.text(58, 128, "Next-token training · 61 decoder blocks", 20, 700)
    svg.text(58, 162, "Embedding →", 17, 700)
    left, step = 211, 13.0
    for i, kind in enumerate(BLOCKS):
        x = left + step * i
        svg.items.append(
            f"<g><title>Block {i}: MLA and {'dense FFN' if kind == 'dense' else 'shared and routed experts'}</title>"
        )
        svg.box(x, 144, step - 2, 21, "#637b89" if kind == "dense" else "#548978")
        if i == layer:
            svg.box(x - 1, 139, step, 4, "#a04c2e")
            svg.box(x - 1, 166, step, 4, "#a04c2e")
        svg.items.append("</g>")
    svg.text(1027, 162, "→  Final norm  →  LM head  →  Cross-entropy", 17, 700)
    svg.text(58, 193, "fwd", 15, 700)
    svg.text(
        155,
        193,
        "0–2: MLA + dense FFN     ·     3–60: MLA + one shared expert and 8 of 256 routed experts",
        16,
    )
    svg.text(58, 218, "bwd", 15, 700)
    svg.text(
        155,
        218,
        "Loss gradient → head → blocks 60 to 0 → embedding. Parameter gradients feed the optimizer.",
        16,
    )


def chunk_guide(svg, y, rows):
    """Separate the example's scheduling unit from a GEMM kernel's tiling."""
    svg.box(34, y, 1432, 188, "#edf2f3", radius=12)
    svg.text(
        54, y + 31, "A token chunk = a bundle of routed rows (the scheduler's “tile”)", 22, 700
    )
    svg.text(172, y + 62, f"{HIDDEN:,} input channels →", 15, 700, COLOR["NVLink"], "middle")
    for row in range(4):
        svg.box(55, y + 76 + row * 19, 235, 16, "#6994b3", radius=2)
    svg.text(302, y + 119, "rows", 14, color=COLOR["NVLink"])
    svg.text(55, y + 171, "One row = one routed token activation", 14, color="#4a5e66")
    svg.text(358, y + 69, "Transfer and schedule this row bundle together.", 17, 700)
    svg.text(358, y + 97, "Its ID follows activations, outputs and input gradients.", 16)
    example_rows = rows if rows is not None else 128
    label = "Profile" if rows is not None else "Example size"
    svg.text(
        358,
        y + 125,
        f"{label}: {example_rows:,} BF16 rows = {example_rows * HIDDEN * 2 / 2**20:g} MiB per activation hop.",
        16,
    )
    svg.text(358, y + 153, "Chunk count is separate from the full sequence length.", 16)
    svg.box(922, y + 19, 1, 150, "#bdccd1")
    svg.text(943, y + 32, "GEMM tiling · fwd projection example", 20, 700)
    for row in range(4):
        for col in range(6):
            svg.box(
                945 + col * 24,
                y + 70 + row * 20,
                21,
                17,
                COLOR["Compute"] if row < 2 and col < 2 else "#cedcd7",
                radius=2,
            )
    svg.text(945, y + 171, "Output matrix sketch", 14, color="#4a5e66")
    svg.text(1110, y + 68, "M: token rows · N: output channels", 16, 700)
    svg.text(1110, y + 96, "K: reduction channels", 16, 700)
    svg.text(1110, y + 124, "M / N / K tile sizes are selected", 15)
    svg.text(1110, y + 152, "separately for tensor-core compute.", 15)


def panel(svg, phase, y, layer, tiles, slots, samples, complete, captures, tolerance):
    phases = PASSES[phase]
    actions = [a for part in phases for a in part]
    plans = (
        captures[phase]
        if captures
        else [
            trace(phases, tiles, slots, p, samples if complete else {})
            for p in ("bulk", "tile_ready")
        ]
    )
    ends = [max(s.end for s in spans) for spans in plans]
    unit = "µs" if complete or captures else "ticks"
    direction = "fwd" if phase == "Forward" else "bwd"
    svg.text(44, y, f"{direction} · routed block {layer}", 25, 700)
    xcols, width, row = (498, 985), 445, 40
    for x, name, end in zip(xcols, ("Bulk exchanges", "Chunk-ready pipeline"), ends):
        svg.text(x, y, f"{name}  ·  {end:,.3g} {unit}", 19, 700)
    y0 = y + 18
    for i, a in enumerate(actions):
        yy = y0 + i * row
        svg.box(45, yy + 3, 4, 31, COLOR[a.resource])
        svg.text(60, yy + 16, a.title, 16, 700)
        svg.text(60, yy + 33, a.detail, 13, 400, "#4a5e66")
        if samples:
            value = f"{samples[a.sample]:.2f} µs" if a.sample in samples else "pending"
            svg.text(468, yy + 16, value, 14, 700, COLOR[a.resource], anchor="end")
        for x in xcols:
            svg.box(x, yy + 7, width, 24, "#e6e9e3")
    scale = width / max(ends)
    for x, spans in zip(xcols, plans):
        for start, end in simultaneous(spans, tolerance):
            svg.box(x + start * scale, y0, (end - start) * scale, 5, "#cc9222")
        for t in range(5):
            xx = x + width * t / 4
            svg.box(xx, y0 + 2, 0.8, len(actions) * row, "#cbd2cc")
            svg.text(
                xx, y0 + len(actions) * row + 18, f"{max(ends) * t / 4:.3g}", 12, anchor="middle"
            )
        by_key = {a.key: i for i, a in enumerate(actions)}
        for s in spans:
            xx, bar = x + s.start * scale, (s.end - s.start) * scale
            yy = y0 + by_key[s.stage.name] * row + 7
            svg.items.append(
                f"<g><title>{actions[by_key[s.stage.name]].title} · token chunk {s.tile + 1} · {s.start:g}–{s.end:g} {unit}</title>"
            )
            svg.box(xx, yy, max(0.35, bar - 0.8), 24, COLOR[s.stage.resource])
            if bar > 12:
                svg.text(xx + bar / 2, yy + 17, s.tile + 1, 12, 700, "#ffffff", "middle")
            svg.items.append("</g>")
    windows = simultaneous(plans[1], tolerance)
    if windows:
        when = (
            windows[0][0]
            + min(windows[0][1], min(s.end for s in plans[1] if s.end > windows[0][0]))
        ) / 2
        roles = {s.stage.resource: s.tile + 1 for s in plans[1] if s.start <= when < s.end}
        svg.text(
            498,
            y0 + len(actions) * row + 43,
            f"First overlap · InfiniBand carries chunk {roles['IB']}, NVLink relays chunk {roles['NVLink']}, and compute uses chunk {roles['Compute']}.",
            14,
            700,
            color="#74521b",
        )


def arguments(description=__doc__, default_out="pipeline.svg"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=Path(default_out))
    p.add_argument("--layer", type=int, choices=range(3, len(BLOCKS)), default=3, metavar="3..60")
    p.add_argument(
        "--tiles",
        type=int,
        default=4,
        choices=range(1, 65),
        metavar="1..64",
        help="number of routed token chunks (the scheduler's tiles)",
    )
    p.add_argument("--slots", type=int, default=2)
    evidence = p.add_mutually_exclusive_group()
    evidence.add_argument("--samples", type=Path, nargs="+", default=[])
    evidence.add_argument("--trace", type=Path, help="paired per-chunk intervals on a common clock")
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--sequences", type=int, help="optional global sequences per optimizer update")
    args = p.parse_args()
    if min(args.slots, args.nodes) < 1 or (args.sequences is not None and args.sequences < 1):
        p.error("slots, nodes and sequences must be positive")
    return args


def main():
    args = arguments()
    samples, probes, rows = read_samples(args.samples)
    captures, tolerance = observed(args.trace, args.tiles, args.slots)
    complete = probes == PROBES.keys()
    batch = f"{args.sequences:,} sequences/update" if args.sequences else "training batch open"
    svg = SVG(height=1390)
    svg.text(42, 51, "DeepSeek-V3 training across nodes", 35, 700)
    svg.text(
        44,
        80,
        f"B300 study  ·  {args.nodes:,} × 16 GPUs  ·  {CONTEXT:,} tokens per sequence  ·  {batch}",
        17,
    )
    backbone(svg, args.layer)
    svg.text(44, 272, "One remote expert route", 24, 700)
    svg.text(
        44,
        300,
        "InfiniBand connects the nodes. The remote gateway relays the activation over NVLink, and the expert output returns along the same route.",
        17,
    )
    for x, name, role in [
        (46, "IB", "InfiniBand · between nodes"),
        (390, "NVLink", "NVLink · within one node"),
        (735, "Compute", "Compute · expert GPU"),
    ]:
        svg.box(x, 322, 15, 15, COLOR[name])
        svg.text(x + 24, 335, role, 16, 700)
    svg.text(
        1452, 335, f"{args.tiles} token chunks  /  {args.slots} staging slots", 15, anchor="end"
    )
    chunk_guide(svg, 358, rows)
    panel(
        svg,
        "Forward",
        601,
        args.layer,
        args.tiles,
        args.slots,
        samples,
        complete,
        captures,
        tolerance,
    )
    panel(
        svg,
        "Backward",
        933,
        args.layer,
        args.tiles,
        args.slots,
        samples,
        complete,
        captures,
        tolerance,
    )
    svg.text(
        44,
        1340,
        "Bar numbers identify token chunks. Follow the same number down the rows as its activations, outputs and gradients move.",
        17,
        700,
    )
    if captures:
        foot = f"Observed role intervals on the capture clock. Gold excludes {tolerance:g} µs of uncertainty at each end."
    elif complete:
        foot = "Modeled µs from isolated BF16 compute and one-hop NCCL medians. Co-running workload contention needs a device trace."
    elif samples:
        missing = ", ".join(PROBE_LABELS[name] for name in sorted(PROBES.keys() - probes))
        foot = f"Left: isolated medians for {rows} BF16 rows. Bars: dependency ticks. Pending fabric measurements: {missing}."
    else:
        have = ", ".join(PROBE_LABELS[name] for name in sorted(probes)) or "none"
        missing = ", ".join(PROBE_LABELS[name] for name in sorted(PROBES.keys() - probes))
        foot = f"Unit ticks show dependencies. Isolated B300 probes available: {have}. Pending: {missing}."
    svg.text(44, 1368, foot, 15, color="#4a5e66")
    svg.save(args.out)
    print(
        f"{args.nodes * 16:,} GPUs; {CONTEXT:,} tokens/sequence; {TOP_K * CONTEXT:,} assignments/sequence/routed layer"
    )
    if args.sequences:
        print(f"{args.sequences:,} sequences/update; {args.sequences * CONTEXT:,} tokens/update")


if __name__ == "__main__":
    main()
