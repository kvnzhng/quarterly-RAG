"""Apply text edits to a file one at a time, and report exactly which did not apply.

Why this exists: an all-or-nothing edit helper writes nothing when one anchor is stale,
and a commit message written from intent then describes changes that never landed. That
happened twice in this repository. This helper applies each edit independently, writes
after each success, and prints a PARTIAL line naming every anchor it could not find, so
the person committing can see the truth before writing the message.

Usage from a script:

    import sys; sys.path.insert(0, "scripts")
    from edit_docs import apply
    ok = apply("docs/some.md", [("old text", "new text"), ...])

`apply` returns True only when every edit landed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def apply(path: str, pairs: Sequence[tuple[str, str]]) -> bool:
    failed: list[tuple[int, str]] = []
    for index, (old, new) in enumerate(pairs):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        if old not in text:
            failed.append((index, old[:70]))
            continue
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace(old, new, 1))
    status = "OK " if not failed else "PARTIAL"
    print(f"{status} {path}: {len(pairs) - len(failed)}/{len(pairs)} applied")
    for index, snippet in failed:
        print(f"     FAILED pair {index}: {snippet!r}")
    return not failed


if __name__ == "__main__":
    sys.exit("import this module; it has no command-line form")
