#!/usr/bin/env python3
"""Build and test the library with each optional feature on its own and with all of them together, so a
feature no other job builds or tests cannot break unseen

The features are read from Cargo.toml (every key of [features] except `default`), so a new feature is
checked without editing this script. Each one is built with `cargo clippy --lib --features <name> -- -D
warnings` (a warning only that feature's code raises, a discarded `Result`, fails too) and its library
tests are run with `cargo test --lib --features <name>`. Then every feature that is checked is turned on
at once and built and tested the same way (`--all-targets` for clippy), so two features that compile
alone but not together fail too.

`--ci` skips the features listed in .github/feature-check-pending.txt (`<feature> <reason>`, one per line,
`#` comments): a feature whose code CI cannot build for real yet (its dependency is still a path sibling
that CI replaces with an empty stub). Each pending line must name a feature of Cargo.toml and give a
reason, and the run prints it as PENDING. Without `--ci` (the local preflight, which has the real
siblings) every feature is built, pending ones included.

.github/feature-check-apart.txt (same format) lists features that are checked alone but left out of the
build with all features together (one that cannot be combined with another, such as `no_std` beside
`std`); the run prints each as APART.

Fails when a build or a test fails, when nothing was built, or when a list is malformed.

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
APART = Path(".github") / "feature-check-apart.txt"


def features(root: Path) -> list[str]:
    """the optional features of the package at `root`, in Cargo.toml order"""
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return [f for f in manifest.get("features", {}) if f != "default"]


def pending(root: Path, known: list[str], rel: Path = PENDING) -> dict[str, str]:
    """feature -> reason from a list (the pending list by default; an absent file is an empty list)"""
    path = root / rel
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, reason = line.partition(" ")
        if name not in known:
            raise ValueError(f"{rel}:{no}: {name} is not a feature of Cargo.toml")
        if not reason.strip():
            raise ValueError(f"{rel}:{no}: {name} has no reason")
        if name in out:
            raise ValueError(f"{rel}:{no}: {name} is listed twice")
        out[name] = reason.strip()
    return out


def cargo_check(root: Path, feats: list[str], step: str) -> bool:
    """`step` is "clippy" (one feature: the library; several: all targets) or "test" (library tests)"""
    joined = ",".join(feats)
    if step == "clippy":
        targets = ["--lib"] if len(feats) == 1 else ["--all-targets"]
        cmd = ["cargo", "clippy", *targets, "--features", joined, "--", "-D", "warnings"]
    else:
        cmd = ["cargo", "test", "--lib", "--features", joined]
    return subprocess.run(cmd, cwd=root).returncode == 0


def run(root: Path, ci: bool, check=cargo_check) -> int:
    try:
        names = features(root)
        skip = pending(root, names) if ci else {}
        apart = pending(root, names, APART)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        print(f"feature-check: {e}", file=sys.stderr)
        return 2
    failed, built, tested = [], 0, 0
    for name in names:
        if name in skip:
            print(f"feature-check: PENDING {name}: {skip[name]}")
            continue
        print(f"feature-check: cargo clippy --lib --features {name} -- -D warnings", flush=True)
        built += 1
        if not check(root, [name], "clippy"):
            failed.append(name)
            continue
        print(f"feature-check: cargo test --lib --features {name}", flush=True)
        tested += 1
        if not check(root, [name], "test"):
            failed.append(f"{name} (test)")
    together = [n for n in names if n not in skip and n not in apart]
    for name in names:
        if name in apart and name not in skip:
            print(f"feature-check: APART {name}: {apart[name]}")
    if len(together) > 1:
        joined = ",".join(together)
        print(f"feature-check: cargo clippy --all-targets --features {joined} -- -D warnings", flush=True)
        built += 1
        if not check(root, together, "clippy"):
            failed.append("all together")
        else:
            print(f"feature-check: cargo test --lib --features {joined}", flush=True)
            tested += 1
            if not check(root, together, "test"):
                failed.append("all together (test)")
    print(
        f"feature-check: {built} built, {tested} tested, {len(skip)} pending, {len(failed)} failed {failed if failed else ''}".rstrip()
    )
    if built == 0:
        print("feature-check: no feature built (compared nothing)", file=sys.stderr)
        return 1
    return 1 if failed else 0


def self_test() -> int:
    cases = []

    def tree(t: str, feats: str, pend: str | None, apart: str | None) -> Path:
        root = Path(t)
        (root / "Cargo.toml").write_text(f'[package]\nname = "x"\nversion = "0.0.0"\n\n[features]\n{feats}', encoding="utf-8")
        for rel, body in ((PENDING, pend), (APART, apart)):
            if body is not None:
                (root / ".github").mkdir(exist_ok=True)
                (root / rel).write_text(body, encoding="utf-8")
        return root

    def go(
        feats: str, pend: str | None, ci: bool, fail: set[str] = frozenset(), apart: str | None = None
    ) -> tuple[int, list[str]]:
        """the exit code and each check run, as "<step> <features>"; a check fails when its
        "<step> <features>" is in `fail`"""
        seen: list[str] = []

        def check(_r: Path, fs: list[str], step: str) -> bool:
            seen.append(f"{step} {','.join(fs)}")
            return seen[-1] not in fail

        with tempfile.TemporaryDirectory() as t:
            root = tree(t, feats, pend, apart)
            code = run(root, ci, check)
        return code, seen

    def each(*fs: str) -> list[str]:
        return [x for f in fs for x in (f"clippy {f}", f"test {f}")]

    F = 'default = ["std"]\nstd = []\nfoo = []\ntext = ["std"]\n'
    ALL = each("std,foo,text")
    cases.append(("all built and tested, alone and together", go(F, None, False), (0, each("std", "foo", "text") + ALL)))
    cases.append(("default is not a feature to build; one feature has no together step", go('default = []\na = []\n', None, False), (0, each("a"))))
    cases.append(("pending skipped in ci, alone and together", go(F, "text needs the real sibling (it moves later)\n", True), (0, each("std", "foo", "std,foo"))))
    cases.append(("pending built locally", go(F, "text needs the real sibling (it moves later)\n", False), (0, each("std", "foo", "text") + ALL)))
    cases.append(("a failing clippy fails and skips its test", go(F, None, False, {"clippy foo"}), (1, each("std") + ["clippy foo"] + each("text") + ALL)))
    cases.append(("a failing test fails", go(F, None, False, {"test text"}), (1, each("std", "foo", "text") + ALL)))
    cases.append(("a failing combination fails", go(F, None, False, {"clippy std,foo,text"}), (1, each("std", "foo", "text") + ["clippy std,foo,text"])))
    cases.append(("a failing combined test fails", go(F, None, False, {"test std,foo,text"}), (1, each("std", "foo", "text") + ALL)))
    cases.append(("apart: alone but not together", go(F, None, False, apart="foo cannot sit beside std\n"), (0, each("std", "foo", "text", "std,text"))))
    cases.append(("apart of an unknown feature fails", go(F, None, False, apart="voice reason\n"), (2, [])))
    cases.append(("nothing to build fails", go('default = []\n', None, False), (1, [])))
    cases.append(("everything pending fails", go('a = []\n', "a reason\n", True), (1, [])))
    cases.append(("pending of an unknown feature fails", go(F, "voice reason\n", True), (2, [])))
    cases.append(("pending without a reason fails", go(F, "text\n", True), (2, [])))
    cases.append(("pending twice fails", go(F, "text a\ntext b\n", True), (2, [])))
    cases.append(("comments and blanks", go(F, "# note\n\ntext the reason  # trailing\n", True), (0, each("std", "foo", "std,foo"))))
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
