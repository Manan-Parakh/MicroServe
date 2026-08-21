"""Regenerate checkbox counts and the progress bar in docs/bites/README.md.

Run after adding or checking off entries: `python docs/bites/update_progress.py`
Every category heading and the top progress line are derived from the
checkboxes below them — never hand-edit those numbers.
"""

import re
from pathlib import Path

README = Path(__file__).parent / "README.md"
BAR_WIDTH = 40

SECTION_RE = re.compile(r"^(### \S+ .+? — .+? · )`\d+ / \d+`\s*$")
CHECKBOX_RE = re.compile(r"^- \[( |x)\]")


def make_bar(checked: int, total: int) -> str:
    filled = round(BAR_WIDTH * checked / total) if total else 0
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def main() -> None:
    lines = README.read_text(encoding="utf-8").splitlines()

    total_checked = 0
    total_all = 0
    current_section = None
    section_counts: dict[int, list[int]] = {}

    for i, line in enumerate(lines):
        if SECTION_RE.match(line):
            current_section = i
            section_counts[i] = [0, 0]
            continue
        cb = CHECKBOX_RE.match(line.strip())
        if cb and current_section is not None:
            section_counts[current_section][1] += 1
            total_all += 1
            if cb.group(1) == "x":
                section_counts[current_section][0] += 1
                total_checked += 1

    for i, (checked, total) in section_counts.items():
        lines[i] = SECTION_RE.sub(rf"\g<1>`{checked} / {total}`", lines[i])

    pct = round(100 * total_checked / total_all) if total_all else 0
    bar = make_bar(total_checked, total_all)

    out = []
    for line in lines:
        if line.startswith("**Progress:**"):
            out.append(f"**Progress:** `{total_checked} / {total_all}` ({pct}%)")
        elif line.startswith("`[") and line.endswith("]`"):
            out.append(f"`{bar}`")
        else:
            out.append(line)

    README.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"docs/bites/README.md updated: {total_checked} / {total_all} ({pct}%)")


if __name__ == "__main__":
    main()
