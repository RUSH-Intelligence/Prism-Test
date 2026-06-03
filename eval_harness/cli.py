from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

from .config import EvalConfig, load_yaml_config
from .runner import EvalRunner


class CliEntryPoint:
    def run(self, config_file: Optional[str] = "./evaluate_config.yaml", **overrides: Any) -> Dict[str, Any]:
        final_cfg = asdict(EvalConfig())
        valid_keys = {f.name for f in fields(EvalConfig)}
        if config_file:
            file_cfg = {k: v for k, v in load_yaml_config(config_file).items() if k in valid_keys}
            final_cfg.update(file_cfg)

        final_cfg.update({k: v for k, v in overrides.items() if v is not None and k in valid_keys})

        config = EvalConfig(**final_cfg)
        runner = EvalRunner(config)
        run_dir = runner.run()

        return {
            "run_dir": str(run_dir),
            "benchmark": config.benchmark,
            "model": config.model,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prism-Test standalone vLLM eval harness")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one benchmark evaluation")
    run_parser.add_argument("--config_file", default="./evaluate_config.yaml")
    run_parser.add_argument("--benchmark", default=None)
    run_parser.add_argument("--subsets", default=None)
    run_parser.add_argument("--backend", default=None)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--max_new_tokens", type=int, default=None)
    run_parser.add_argument("--system_prompt", default=None)
    run_parser.add_argument("--max_requests", type=int, default=None)
    run_parser.add_argument("--fraction", type=float, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    overrides: Dict[str, Any] = {
        "benchmark": args.benchmark,
        "subsets": args.subsets,
        "backend": args.backend,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": args.system_prompt,
        "max_requests": args.max_requests,
        "fraction": args.fraction,
    }

    result = CliEntryPoint().run(config_file=args.config_file, **overrides)
    print(result)


if __name__ == "__main__":
    main()
