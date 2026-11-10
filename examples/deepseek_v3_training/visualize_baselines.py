"""Render attention and expert ownership using computation names and explicit GPU owners."""

import argparse
from pathlib import Path

from visualize import SVG
from workload import CONTEXT, HIDDEN, TOP_K, cp_payload, expert_ids

OWNER = ("#246984", "#99542f", "#3a755b", "#705392")
PALE = ("#dcebf0", "#f4e5dc", "#dcece2", "#ece4f2")
MUTED, RULE, PANEL = "#465e69", "#ced8d5", "#eff3ef"
WIDTH = 1500


def ownership_grid(svg, x, y, by_head):
    cell = 58
    for index in range(4):
        svg.text(x + index * cell + 28, y - 25, "Heads", 12, 700, MUTED, "middle")
        svg.text(
            x + index * cell + 28,
            y - 8,
            f"{32 * index}–{32 * index + 31}",
            12,
            400,
            MUTED,
            "middle",
        )
        svg.text(x - 12, y + index * cell + 25, "Token", 13, 400, MUTED, "end")
        svg.text(x - 12, y + index * cell + 43, f"block {index + 1}", 13, 700, MUTED, "end")
        for head in range(4):
            owner = head if by_head else index
            svg.box(x + head * cell, y + index * cell, cell - 3, cell - 3, OWNER[owner], radius=4)
            svg.text(
                x + head * cell + 27,
                y + index * cell + 33,
                f"GPU {owner}",
                13,
                700,
                "white",
                "middle",
            )


def context(out):
    svg = SVG(
        WIDTH,
        1196,
        "Attention tensors and their GPU owners",
        "Ulysses exchanges token and head ownership. A compact ring keeps query blocks local and rotates the latent and positional-key blocks.",
    )
    svg.text(40, 51, "Attention tensors and their GPU owners", 34, 700)
    svg.text(
        42,
        82,
        f"{CONTEXT:,} tokens per sequence · 128 heads · four equal token blocks in the ownership diagrams",
        18,
        color=MUTED,
    )
    svg.box(34, 109, 714, 553, PANEL, radius=12)
    svg.box(764, 109, 702, 553, PANEL, radius=12)
    svg.text(54, 148, "Ulysses · move from token owners to head owners", 23, 700)
    svg.text(54, 179, "fwd  Project keys and values, then exchange queries / keys / values.", 16)
    svg.text(220, 224, "Before all-to-all", 18, 700, anchor="middle")
    svg.text(564, 224, "After all-to-all", 18, 700, anchor="middle")
    ownership_grid(svg, 117, 278, False)
    ownership_grid(svg, 459, 278, True)
    svg.arrow([(357, 392), (383, 392)], MUTED, width=3)
    svg.text(54, 550, "A cell names the GPU holding that token block and those heads.", 16, 700)
    svg.text(54, 581, "Each head owner attends over the full sequence.", 17)
    svg.text(54, 610, "fwd  Return the attention output to its token owner.", 16)
    svg.text(54, 638, "bwd  Reverse the exchanges and reduce-scatter positional-key gradients.", 16)

    svg.text(784, 148, "Compact ring · keep each query block local", 23, 700)
    svg.text(784, 179, "KV block = normalized latent + shared positional key.", 16)
    for step in range(4):
        svg.text(
            1010 + step * 119,
            248,
            "Local step" if step == 0 else f"Hop {step}",
            15,
            700,
            MUTED,
            "middle",
        )
    for rank in range(4):
        y = 268 + rank * 59
        svg.box(784, y, 160, 53, OWNER[rank], radius=5)
        svg.text(864, y + 22, f"Query block {rank + 1}", 16, 700, "white", "middle")
        svg.text(864, y + 42, f"stays on GPU {rank}", 13, 400, "white", "middle")
        for step in range(4):
            owner = (rank - step) % 4
            active = owner <= rank
            x = 955 + step * 119
            svg.box(x, y, 110, 53, PALE[owner] if active else "#e3e7e3", radius=5)
            svg.text(
                x + 55,
                y + 22,
                f"KV block {owner + 1}",
                14,
                700,
                OWNER[owner] if active else MUTED,
                "middle",
            )
            svg.text(
                x + 55,
                y + 42,
                "attend" if active else "causal skip",
                13,
                color=MUTED,
                anchor="middle",
            )
    svg.text(
        784, 550, "Future blocks travel onward while their attention math is skipped.", 16, 700
    )
    svg.text(784, 581, "fwd  Expand each arriving block into keys and values.", 16)
    svg.text(784, 610, "fwd  Merge softmax statistics and weighted value sums.", 16)
    svg.text(784, 638, "bwd  Replay blocks and return latent and positional-key gradients.", 16)

    svg.text(40, 714, "Communication per GPU, per attention layer, per sequence", 25, 700)
    svg.text(
        40,
        745,
        "Logical BF16 payload in GiB. All bars use the same scale. Gradient transfers also use BF16 here.",
        17,
        color=MUTED,
    )
    for x, label in [
        (49, "CP GPUs"),
        (226, "Ulysses fwd / bwd"),
        (592, "Ring fwd"),
        (858, "Ring bwd"),
        (1143, "KV expansions in fwd"),
    ]:
        svg.text(x, 789, label, 18, 700)
    svg.text(1143, 813, "Ulysses / Ring", 15, color=MUTED)
    degrees = (2, 4, 8, 16)
    totals = []
    for ranks in degrees:
        payload = cp_payload(ranks)
        split, ring = payload["Ulysses"], payload["Ring"]
        totals.append((sum(split) / 2**30, ring[0] / 2**30, sum(ring) / 2**30))
    maximum = max(max(row) for row in totals)
    for index, (ranks, values) in enumerate(zip(degrees, totals)):
        y = 835 + index * 39
        svg.box(40, y - 8, 1420, 1, RULE)
        svg.text(59, y + 19, ranks, 19, 700)
        for x, gib, color in zip((228, 594, 860), values, ("#366985", "#36806b", "#71578a")):
            bar = 205 * gib / maximum
            svg.box(x, y + 3, bar, 19, color, radius=2)
            svg.text(x + bar + 10, y + 20, f"{gib:.2f}", 18, 700, color)
        svg.text(1150, y + 19, f"{ranks}  /  {ranks * (ranks + 1) // 2}", 18, 700)
    svg.text(
        40,
        1013,
        "Ulysses sends the displayed amount in each pass. KV expansions count whole token-block projections across the CP group.",
        16,
    )
    svg.text(
        40,
        1041,
        "The ring bwd total includes KV replay, accumulated gradients and the final gradient-only hop to the original owner.",
        16,
        color=MUTED,
    )
    svg.box(34, 1066, 1432, 105, PANEL, radius=10)
    same = cp_payload(4)["Ring"][0] / 2**30
    svg.text(
        54,
        1099,
        f"Compact all-gather control · four GPUs · {same:.2f} GiB fwd and {same:.2f} GiB bwd",
        21,
        700,
    )
    svg.text(
        54,
        1128,
        "Each query owner retains all four latent / positional-key blocks and reduce-scatters their gradients.",
        17,
    )
    svg.text(
        54,
        1154,
        "The ring consumes successive blocks and replays them in bwd. These figures compare payload and residency.",
        16,
        color=MUTED,
    )
    svg.save(out)


def row_copies(svg, center, y, count, color):
    """One small horizontal rectangle represents one full tensor row."""
    width, gap = 37, 6
    left = center - (count * (width + gap) - gap) / 2
    for index in range(count):
        svg.box(left + index * (width + gap), y, width, 9, color, radius=3)


def expert_plan(svg, top, method, selected):
    compact, peer = method == "compact", method == "peer"
    owners = tuple(selected)
    count = len(owners) if compact else TOP_K
    kib = count * HIDDEN * 2 // 1024
    title = {
        "naive": "Naive EP · one transfer row per expert",
        "compact": "Compact DeepEP · one transfer row per destination GPU",
        "peer": "Dist-MoE-style peer access · address rows at their source",
    }[method]
    svg.box(34, top, 1432, 335, PANEL, radius=12)
    svg.text(54, top + 34, title, 24, 700)
    payload = (
        f"{TOP_K} input-row reads + {TOP_K} output-slot writes"
        if peer
        else f"Dispatch {kib} KiB + return {kib} KiB"
    )
    svg.text(1446, top + 34, payload, 17, 700, MUTED, "end")
    for x, label in (
        (385, "fwd · read during compute" if peer else "fwd · dispatch"),
        (738, "expert compute"),
        (1100, "fwd · write during compute" if peer else "fwd · return"),
    ):
        svg.text(x, top + 70, label, 17, 700, MUTED, "middle")
    body = top + 93
    center = body + 80
    for x in (54, 1255):
        svg.box(x, body + 4, 191, 154, PALE[0], radius=9, stroke="#a8c4ce")
        svg.text(x + 95, body + 32, "Source GPU 0", 19, 700, OWNER[0], "middle")
    svg.text(
        149, body + 68, "Published input" if peer else "Token activation", 18, 700, anchor="middle"
    )
    svg.text(149, body + 95, "1 full row", 18, anchor="middle")
    row_copies(svg, 149, body + 113, 1, OWNER[0])
    svg.text(
        1350,
        body + 66,
        "Sum owner partials" if compact else "Weight + sum",
        18,
        700,
        anchor="middle",
    )
    svg.text(1350, body + 94, f"{count} rows → 1 row", 18, anchor="middle")
    svg.text(1350, body + 133, "Combined output", 16, 700, OWNER[0], "middle")
    for index, owner in enumerate(owners):
        y = body + 38 + index * 84
        color = OWNER[owner]
        copies = 1 if compact else len(selected[owner])
        outward = (
            f"{copies} addressed reads"
            if peer
            else f"{copies} activation {'row' if compact else 'copies'}"
        )
        inward = (
            f"{copies} output-slot writes"
            if peer
            else ("1 weighted partial sum" if compact else f"{copies} expert outputs")
        )
        svg.arrow([(249, center), (267, center), (267, y), (508, y)], color)
        svg.text(387, y - 12, outward, 17, 700, color, "middle")
        row_copies(svg, 387, y + 12, copies, color)
        svg.box(520, y - 32, 435, 64, PALE[owner], radius=7)
        first, last = selected[owner][0], selected[owner][-1]
        svg.text(537, y - 5, f"GPU {owner} · experts {first}–{last}", 19, 700, color)
        operation = (
            "Fan out → 4 MLPs → weight + sum locally"
            if compact
            else (
                "4 MLPs · read/write source addresses"
                if peer
                else "4 MLPs → 4 separate expert outputs"
            )
        )
        svg.text(537, y + 21, operation, 17)
        svg.arrow([(966, y), (1236, y)], color)
        svg.text(1100, y - 12, inward, 17, 700, color, "middle")
        row_copies(svg, 1100, y + 12, copies, color)
        svg.arrow([(1237, y), (1246, y), (1246, center), (1253, center)], color)
    backward = {
        "naive": (
            "Source sends 8 weighted output-gradient rows. Experts return 8 input-gradient contributions; source sums them.",
            "The source computes each selected-score gradient from its saved expert output.",
        ),
        "compact": (
            "Source sends 2 output-gradient rows. Each owner weights its 4 expert gradients and sums their input gradients.",
            "Owners return 2 input-gradient rows for the source to sum, plus the individual selected-score gradients.",
        ),
        "peer": (
            "Experts read 8 weighted output-gradient rows by address and write 8 input-gradient slots for the source to sum.",
            "The source computes selected-score gradients. Reads and writes are logical accesses; physical traffic depends on the implementation.",
        ),
    }[method]
    svg.text(54, top + 284, "bwd", 18, 700)
    svg.text(110, top + 284, backward[0], 16)
    svg.text(110, top + 312, backward[1], 15, color=MUTED)


def experts(out):
    choice = expert_ids(0, "colocated")
    selected = {owner: [e for e in choice if e // TOP_K == owner] for owner in (1, 2)}
    row_kib = HIDDEN * 2 // 1024
    svg = SVG(
        WIDTH,
        1455,
        "One token, eight experts, one combined output",
        "Three expert-parallel plans follow the same token. Arrows show tensor movement, colors identify expert owners, and the source and local reductions are explicit.",
    )
    svg.text(40, 51, "One token → eight experts → one combined output", 34, 700)
    svg.text(
        42,
        83,
        f"One BF16 row = {HIDDEN:,} channels = {row_kib} KiB. All three plans execute the same eight expert MLPs.",
        18,
        color=MUTED,
    )
    svg.box(34, 110, 310, 142, PALE[0], radius=10)
    svg.text(54, 143, "Source GPU 0 · node 0", 22, 700, OWNER[0])
    svg.text(54, 174, "Owns this token before dispatch", 17)
    svg.text(54, 201, "and after the outputs combine.", 17)
    svg.text(54, 231, "Each small bar below = one full row.", 15, color=MUTED)
    for owner, x in ((1, 367), (2, 928)):
        svg.box(x, 110, 538, 142, PALE[owner], radius=10)
        svg.text(
            x + 20,
            143,
            f"Expert GPU {owner} · node {0 if owner == 1 else 1}",
            22,
            700,
            OWNER[owner],
        )
        for index, expert in enumerate(selected[owner]):
            left = x + 20 + index * 126
            svg.box(left, 161, 117, 39, OWNER[owner], radius=5)
            svg.text(left + 58, 187, f"Expert {expert}", 17, 700, "white", "middle")
        svg.text(x + 20, 231, "Each MLP: gate/up projection → SwiGLU → down projection", 16)
    svg.text(
        42,
        284,
        "Read each fwd path left → right. The left and right blue boxes are the same source GPU. Colors identify the two expert owners.",
        17,
        color=MUTED,
    )
    for top, method in ((304, "naive"), (655, "compact"), (1006, "peer")):
        expert_plan(svg, top, method, selected)
    svg.text(
        40,
        1381,
        "Expert owners accumulate the gate/up and down weight gradients in every plan.",
        18,
        700,
    )
    svg.text(
        40,
        1410,
        "Payloads sum both destination GPUs for this token. Counts cover activation / output data; scores, indices and protocol overhead are additional.",
        16,
        color=MUTED,
    )
    svg.text(
        40,
        1437,
        "The pipeline figures group many routed rows into token chunks; this comparison follows just one token.",
        16,
        color=MUTED,
    )
    svg.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    context(args.out_dir / "context-parallel.svg")
    experts(args.out_dir / "expert-parallel.svg")
