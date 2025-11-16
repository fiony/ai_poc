"""Utility helpers for detecting unresolved Git merge conflicts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


@dataclass(frozen=True)
class Conflict:
    """Record describing a detected merge conflict marker inside a file."""

    path: Path
    line_number: int
    marker: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan files for unresolved Git conflict markers.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=None,
        help="Optional list of files or directories to scan (defaults to current directory)",
    )
    parser.add_argument(
        "--fail-on-found",
        action="store_true",
        help="Exit with a non-zero code when conflicts are detected (default behavior is warning only)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=(".git", ".venv", "__pycache__"),
        help="Directory names to skip during the scan",
    )
    return parser


def find_conflicts(paths: Sequence[Path] | None, exclude: Iterable[str]) -> list[Conflict]:
    """Return a list of conflicts detected under the provided paths."""

    search_roots: list[Path]
    if paths:
        search_roots = [path.resolve() for path in paths]
    else:
        search_roots = [Path.cwd()]

    conflicts: list[Conflict] = []
    for root in search_roots:
        for file_path in _iter_text_files(root, set(exclude)):
            conflicts.extend(_conflicts_in_file(file_path))
    return conflicts


def _iter_text_files(root: Path, exclude: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if any(part in exclude for part in path.parts):
            continue
        if path.is_file():
            yield path


def _conflicts_in_file(path: Path) -> Iterator[Conflict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    for idx, line in enumerate(text.splitlines(), start=1):
        for marker in CONFLICT_MARKERS:
            if marker in line:
                yield Conflict(path=path, line_number=idx, marker=marker)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    conflicts = find_conflicts(args.paths, args.exclude)
    if not conflicts:
        print("No unresolved merge conflict markers found.")
        return 0

    for conflict in conflicts:
        rel_path = conflict.path.relative_to(Path.cwd())
        print(f"{rel_path}:{conflict.line_number}: contains '{conflict.marker}' marker")

    if args.fail_on_found:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
