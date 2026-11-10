"""Render the standalone lifetime, DualPipe and measured kernel reports as SVG."""

import argparse
import json
from html import escape
from pathlib import Path


class SVG:
    def __init__(self, width, height, title):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{escape(title)}</title>",
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<g font-family="Arial, sans-serif" fill="#14243a">',
        ]
        self.text(30, 37, title, 23)

    def text(self, x, y, text, size=14, fill="#14243a"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}">{escape(str(text))}</text>'
        )

    def rect(self, x, y, width, height, color, title=""):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{max(width, 1)}" height="{height}" rx="3" fill="{color}"><title>{escape(title)}</title></rect>'
        )

    def save(self, path):
        path.write_text("\n".join(self.parts + ["</g></svg>\n"]))


def lifetime(report, path):
    svg = SVG(1540, 675, "Can the weight gradient be deferred safely?")
    svg.text(
        30,
        65,
        "The hidden activation and gate/up activation gradient reuse one scratch range. Time is in dependency ticks.",
    )
    for index, policy in enumerate(("baseline", "unsafe_defer", "preserve_and_defer")):
        y = 105 + index * 180
        case = report[policy]
        label = {
            "baseline": "Baseline: consume hidden activation before reuse",
            "unsafe_defer": "Rejected: the weight-gradient reader still needs the overwritten value",
            "preserve_and_defer": "Legal candidate: preserve the hidden activation in separate storage",
        }[policy]
        svg.text(30, y, label, 18, "#b91c1c" if case["hazards"] else "#166534")
        for row, value in enumerate(case["lifetimes"]):
            top = y + 17 + row * 40
            start = value["producer"]["start"]
            end = max([value["producer"]["end"], *(u["end"] for u in value["consumers"])])
            svg.text(30, top + 13, value["value"].replace("_", " "), 12)
            svg.text(30, top + 27, value["allocation"], 11, "#64748b")
            svg.rect(295 + start * 91, top, (end - start) * 91, 30, "#dbeafe")
            for use, color in [
                (value["producer"], "#60a5fa"),
                *((u, "#fbbf24") for u in value["consumers"]),
            ]:
                svg.rect(295 + use["start"] * 91, top, (use["end"] - use["start"]) * 91, 30, color)
                svg.text(300 + use["start"] * 91, top + 19, use["operation"].replace("_", " "), 11)
        if case["hazards"]:
            svg.text(
                295,
                y + 125,
                "Same physical bytes are overwritten at tick 4; down weight gradient reads them at ticks 7–10.",
                13,
                "#b91c1c",
            )
        for tick in (0, 2, 4, 6, 8, 10, 12, 13):
            x = 295 + tick * 91
            svg.rect(x, y + 138, 1, 5, "#94a3b8")
            svg.text(x - 3, y + 155, tick, 11, "#64748b")
    svg.text(
        30,
        655,
        "Blue: producer / retained value · Gold: consumer · Extra hidden storage: 4 KiB per routed row at expert width 2,048.",
    )
    svg.save(path)


def dualpipe(report, path):
    ranks = report["ranks"]
    slots = report["activation_slots"]
    nslots = slots["slots_per_rank"]["0"]
    svg = SVG(
        1700, 210 + ranks * 100 + nslots * 48, "DualPipe action order and activation ownership"
    )
    svg.text(
        30,
        65,
        f"{ranks}-stage routed FFN example. Numbers in bars identify microbatches; full operation names are in the legend.",
    )
    colors = {"fwd": "#60a5fa", "bwd_input": "#34d399", "bwd_weight": "#fbbf24"}
    for i, (kind, color) in enumerate(colors.items()):
        svg.rect(30 + i * 220, 83, 20, 15, color)
        svg.text(60 + i * 220, 96, kind.replace("_", " "))
    svg.rect(700, 83, 20, 15, "#c084fc")
    svg.text(730, 96, "PP transfer")
    svg.text(1010, 96, "Serial compute within paired groups; dependency ticks", 12)
    extent = max(e["end"] for e in report["events"])
    scale = 1430 / extent
    for rank in range(ranks):
        y = 125 + rank * 100
        svg.text(30, y + 17, f"GPU {rank} compute", 15)
        svg.text(30, y + 50, "PP send", 13)
        svg.rect(215, y, 1430, 28, "#e2e8f0")
        svg.rect(215, y + 37, 1430, 17, "#f1f5f9")
        for event in report["events"]:
            if event["rank"] != rank:
                continue
            kind, direction, batch, stage = event["action"]
            send = kind.startswith("send_")
            label = f"stage {stage}, microbatch {batch}, {kind}, direction {direction}"
            x = 215 + event["start"] * scale
            svg.rect(
                x,
                y + (37 if send else 0),
                (event["end"] - event["start"]) * scale - 1,
                17 if send else 28,
                "#c084fc" if send else colors[kind],
                label,
            )
            if not send:
                svg.text(x + 3, y + 19, batch, 11)
    top = 145 + ranks * 100
    svg.text(
        30,
        top,
        "GPU 0 saved activations: each slot is retained through its matching weight backward",
        17,
    )
    events = {tuple(e["action"]): e for e in report["events"]}
    for slot in range(nslots):
        y = top + 18 + slot * 48
        svg.text(30, y + 22, f"Activation slot {slot}")
        svg.rect(215, y, 1430, 32, "#e2e8f0")
        for item in slots["assignments"]:
            if item["rank"] != 0 or item["slot"] != slot:
                continue
            name = item["value"]
            stage, direction, batch = [
                int(
                    part.removeprefix(
                        "stage" if i == 0 else "direction" if i == 1 else "microbatch"
                    )
                )
                for i, part in enumerate(name.split("."))
            ]
            begin = events["fwd", direction, batch, stage]["start"]
            x = 215 + begin * scale
            svg.rect(
                x,
                y,
                (item["release"] - begin) * scale,
                32,
                "#bfdbfe" if direction == 0 else "#bbf7d0",
            )
            svg.text(x + 5, y + 21, f"stage {stage} / microbatch {batch}", 11)
    svg.text(
        215,
        top + 40 + nslots * 48,
        f"Blue slots: fwd placement GPU 0 → GPU {ranks - 1} · Green slots: fwd placement GPU {ranks - 1} → GPU 0",
        13,
    )
    svg.save(path)


def benchmark(report, path):
    cases = report["cases"]
    svg = SVG(1380, 150 + len(cases) * 145, "SM103: earlier input-gradient delivery and its cost")
    svg.text(
        30,
        65,
        "CUDA-graph medians in microseconds. Both schedules use the same GEMMs and SwiGLU backward; preservation copy is included.",
    )
    svg.text(320, 101, "Input gradient delivered to peer", 17)
    svg.text(810, 101, "All backward outputs ready", 17)
    maximum = max(
        v["median"]
        for c in cases
        for s in c["summary_us"].values()
        for k, v in s.items()
        if k != "joined_graph_completion_us"
    )
    scale = 360 / maximum
    for i, case in enumerate(cases):
        y = 125 + i * 145
        svg.text(30, y + 18, f"{case['rows']:,} routed rows", 17)
        svg.text(30, y + 42, f"Extra hidden storage: {case['extra_live_bytes'] / 2**20:g} MiB", 13)
        svg.text(30, y + 65, case["decision"], 12)
        for col, metric in enumerate(("input_gradient_delivery_us", "backward_completion_us")):
            for row, policy in enumerate(("baseline", "preserve_and_defer")):
                value = case["summary_us"][policy][metric]["median"]
                x, top = 320 + col * 490, y + row * 36
                svg.rect(x, top, value * scale, 26, "#60a5fa" if row == 0 else "#fb923c")
                svg.text(x + value * scale + 8, top + 19, f"{value:.2f} us", 12)
        svg.text(320, y + 102, f"Delivery improvement: {case['delivery_improvement']:.1%}", 14)
        svg.text(810, y + 102, f"Completion change: {case['completion_regression']:+.1%}", 14)
    svg.text(
        30,
        135 + len(cases) * 145,
        "Blue: baseline   Orange: preserve and defer. These are one-expert / one-peer-hop measurements.",
        13,
    )
    svg.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    for name, render in (("lifetime", lifetime), ("dualpipe", dualpipe), ("sm103", benchmark)):
        source = args.out_dir / f"{name}.json"
        if source.exists():
            render(json.loads(source.read_text()), source.with_suffix(".svg"))
