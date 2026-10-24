#!/usr/bin/env python3
"""Build the library with each optional feature on its own, so a feature no other job builds cannot break unseen

The features are read from Cargo.toml (every key of [features] except `default`), so a new feature is
checked without editing this script. Each one is built with `cargo clippy --lib --features <name> -- -D
warnings`, so a warning only that feature's code raises (a discarded `Result`) fails too.

`--ci` skips the features listed in .github/feature-check-pending.txt (`<feature> <reason>`, one per line,
`#` comments): a feature whose code CI cannot build for real yet (its dependency is still a path sibling
that CI replaces with an empty stub). Each pending line must name a feature of Cargo.toml and give a
reason, and the run prints it as PENDING. Without `--ci` (the local preflight, which has the real
siblings) every feature is built, pending ones included.

Fails when a build fails, when nothing was built, or when the pending list is malformed.

    python3 scripts/feature_check.py [--ci]
    python3 scripts/feature_check.py --self-test
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENDING = Path(".github") / "feature-check-pending.txt"


def features(root: Path) -> list[str]:
    """the optional features of the package at `root`, in Cargo.toml order"""
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return [f for f in manifest.get("features", {}) if f != "default"]


def pending(root: Path, known: list[str]) -> dict[str, str]:
    """feature -> reason from the pending list (an absent file is an empty list)"""
    path = root / PENDING
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, reason = line.partition(" ")
        if name not in known:
            raise ValueError(f"{PENDING}:{no}: {name} is not a feature of Cargo.toml")
        if not reason.strip():
            raise ValueError(f"{PENDING}:{no}: {name} has no reason")
        if name in out:
            raise ValueError(f"{PENDING}:{no}: {name} is listed twice")
        out[name] = reason.strip()
    return out


def cargo_check(root: Path, feature: str) -> bool:
    cmd = ["cargo", "clippy", "--lib", "--features", feature, "--", "-D", "warnings"]
    return subprocess.run(cmd, cwd=root).returncode == 0


def run(root: Path, ci: bool, check=cargo_check) -> int:
    try:
        names = features(root)
        skip = pending(root, names) if ci else {}
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        print(f"feature-check: {e}", file=sys.stderr)
        return 2
    failed, built = [], 0
    for name in names:
        if name in skip:
            print(f"feature-check: PENDING {name}: {skip[name]}")
            continue
        print(f"feature-check: cargo clippy --lib --features {name} -- -D warnings", flush=True)
        built += 1
        if not check(root, name):
            failed.append(name)
    print(f"feature-check: {built} built, {len(skip)} pending, {len(failed)} failed {failed if failed else ''}".rstrip())
    if built == 0:
        print("feature-check: no feature built (compared nothing)", file=sys.stderr)
        return 1
    return 1 if failed else 0


def self_test() -> int:
    cases = []

    def tree(t: str, feats: str, pend: str | None) -> Path:
        root = Path(t)
        (root / "Cargo.toml").write_text(f'[package]\nname = "x"\nversion = "0.0.0"\n\n[features]\n{feats}', encoding="utf-8")
        if pend is not None:
            (root / ".github").mkdir()
            (root / PENDING).write_text(pend, encoding="utf-8")
        return root

    def go(feats: str, pend: str | None, ci: bool, fail: set[str] = frozenset()) -> tuple[int, list[str]]:
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as t:
            root = tree(t, feats, pend)
            code = run(root, ci, lambda _r, f: (seen.append(f), f not in fail)[1])
        return code, seen

    F = 'default = ["std"]\nstd = []\nfoo = []\ntext = ["std"]\n'
    cases.append(("all built", go(F, None, False), (0, ["std", "foo", "text"])))
    cases.append(("default is not a feature to build", go('default = []\na = []\n', None, False), (0, ["a"])))
    cases.append(("pending skipped in ci", go(F, "text needs the real sibling (it moves later)\n", True), (0, ["std", "foo"])))
    cases.append(("pending built locally", go(F, "text needs the real sibling (it moves later)\n", False), (0, ["std", "foo", "text"])))
    cases.append(("a failing feature fails", go(F, None, False, {"foo"}), (1, ["std", "foo", "text"])))
    cases.append(("nothing to build fails", go('default = []\n', None, False), (1, [])))
    cases.append(("everything pending fails", go('a = []\n', "a reason\n", True), (1, [])))
    cases.append(("pending of an unknown feature fails", go(F, "voice reason\n", True), (2, [])))
    cases.append(("pending without a reason fails", go(F, "text\n", True), (2, [])))
    cases.append(("pending twice fails", go(F, "text a\ntext b\n", True), (2, [])))
    cases.append(("comments and blanks", go(F, "# note\n\ntext the reason  # trailing\n", True), (0, ["std", "foo"])))
    bad = [f"{name}: got {got}, want {want}" for name, got, want in cases if got != want]
    if bad:
        print("self-test FAIL", *bad, sep="\n  ")
        return 1
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if argv not in ([], ["--ci"]):
        print(__doc__, file=sys.stderr)
        return 2
    return run(ROOT, argv == ["--ci"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
