import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmark"


def discover_models():
    models = {}
    for path in sorted(BENCHMARK_DIR.iterdir()):
        if not path.is_dir():
            continue
        main_file = path / "main.py"
        if main_file.exists():
            models[path.name.lower()] = path.name
    return models


MODELS = discover_models()


def supported_flags(entry_script: Path) -> set[str]:
    text = entry_script.read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\([\"']--([A-Za-z0-9_]+)[\"']", text))


def resolve_model(name: str) -> str:
    key = name.lower()
    if key in MODELS:
        return MODELS[key]

    # Accept common aliases like dmon -> DMoN.
    for normalized, actual in MODELS.items():
        if normalized == key.replace("-", "").replace("_", ""):
            return actual
        compact_actual = actual.lower().replace("-", "").replace("_", "")
        if compact_actual == key.replace("-", "").replace("_", ""):
            return actual

    raise SystemExit(
        f"Unknown model: {name}. Available models: {', '.join(sorted(MODELS.values()))}"
    )


def build_child_argv(args: argparse.Namespace, known_flags: set[str]) -> list[str]:
    entry_script = BENCHMARK_DIR / args.model / "main.py"
    argv = [str(entry_script)]

    scalar_args = {
        "seed": args.seed,
        "device": args.device,
        "root": args.root,
        "dataset": args.dataset,
        "log_dir": args.log_dir,
        "ckpt_dir": args.ckpt_dir,
        "runs": args.runs,
        "cache_dir": args.cache_dir,
    }
    for key, value in scalar_args.items():
        if value is not None and key in known_flags:
            argv.extend([f"--{key}", str(value)])

    flag_args = {
        "load_ckpt": args.load_ckpt,
        "resume": args.resume,
        "cpu_embedding": args.cpu_embedding,
    }
    for key, enabled in flag_args.items():
        if enabled and key in known_flags:
            argv.append(f"--{key}")

    if args.extra:
        extra = args.extra[1:] if args.extra[0] == "--" else args.extra
        argv.extend(extra)

    return argv


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark launcher for PyAGC."
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Benchmark model to run, for example DGI, DAEGC, KMeans, NS4GC.",
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Dataset name passed through to the selected benchmark.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", default="../data")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--ckpt_dir", default="ckpts")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--load_ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu_embedding", action="store_true")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available benchmark models and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated command without executing it.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the selected benchmark after '--'.",
    )
    args, unknown = parser.parse_known_args()

    if args.list_models:
        for name in sorted(MODELS.values()):
            print(name)
        return

    if not args.model:
        parser.error("--model is required unless --list-models is used.")

    args.model = resolve_model(args.model)
    entry_script = BENCHMARK_DIR / args.model / "main.py"
    known_flags = supported_flags(entry_script)
    if unknown:
        args.extra = list(unknown) + list(args.extra)
    child_argv = build_child_argv(args, known_flags)

    if args.dry_run:
        print(f"Working directory: {BENCHMARK_DIR / args.model}")
        print(f"Entry script: {entry_script}")
        print("Command:", " ".join(child_argv))
        return

    result = subprocess.run(
        [sys.executable, *child_argv],
        cwd=BENCHMARK_DIR / args.model,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
