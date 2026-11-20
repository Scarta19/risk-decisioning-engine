"""Command line entry point: ``rde run`` (offline pipeline) and ``rde serve`` (API)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rde.config import Config, load_config


def _cfg(path: str | None) -> Config:
    return load_config(path) if path else Config()


def cmd_run(args: argparse.Namespace) -> None:
    from rde.pipeline import run_pipeline, save_outputs

    cfg = _cfg(args.config)
    result = run_pipeline(cfg)
    save_outputs(result, Path(args.artifacts))
    if args.events_csv:
        from rde.data.synthetic import generate_events

        generate_events(cfg.simulation).to_csv(args.events_csv, index=False)
    if not args.no_report:
        from rde.evaluation.report import write_report

        path = write_report(result.metrics, result.test_frame, Path(args.reports))
        logging.info("Report written to %s", path)
    logging.info("Model bundle written to %s", Path(args.artifacts) / "model_bundle.joblib")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    os.environ.setdefault("RDE_MODEL_PATH", args.model)
    uvicorn.run("rde.serving.app:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="rde", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate, train, calibrate, evaluate and save a bundle")
    run.add_argument("--config", default=None, help="YAML config (defaults built in)")
    run.add_argument("--artifacts", default="artifacts")
    run.add_argument("--reports", default="reports")
    run.add_argument("--events-csv", default=None, help="also write the simulated events")
    run.add_argument("--no-report", action="store_true", help="skip figures and Markdown")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the decision API")
    serve.add_argument("--model", default="artifacts/model_bundle.joblib")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
