"""Command line entry point: ``rde run``, ``rde serve``, ``rde triage-eval``, ``rde monitor``."""

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


def cmd_triage_eval(args: argparse.Namespace) -> None:
    from rde.pipeline import run_pipeline
    from rde.triage import evaluate as E
    from rde.triage.agent import TriageAgent
    from rde.triage.cases import CaseStore
    from rde.triage.llm import AnthropicLLM, ScriptedLLM, heuristic_script

    store = CaseStore.from_pipeline(run_pipeline(_cfg(args.config)))
    if args.backend == "heuristic":
        llm, workers = ScriptedLLM(heuristic_script(store)), 1
    else:
        llm, workers = AnthropicLLM(args.model, args.effort), args.workers
    agent = TriageAgent(store, llm)
    ids = E.sample_cases(store, args.n, args.seed)
    total = len(store.case_ids())
    logging.info("Triaging %d of %d review cases with %s", len(ids), total, llm.model)
    results = E.run(store, agent, ids, workers=workers)
    summary = E.summarise(store, results, llm.model)
    path = E.write_report(summary, results, Path(args.out))
    logging.info("Triage report written to %s", path)


def cmd_monitor(args: argparse.Namespace) -> None:
    from rde.monitoring import study as S

    cfg = S.load_study_config(args.config)
    seeds = args.seeds or [cfg.base.simulation.seed]
    results = [S.run_study(S.with_seed(cfg, seed)) for seed in seeds]
    path = S.write_report(results[0], Path(args.out), results[1:])
    logging.info("Monitoring report written to %s", path)


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

    tri = sub.add_parser("triage-eval", help="run the LLM triage agent on review cases")
    tri.add_argument("--config", default=None)
    tri.add_argument("--backend", choices=["anthropic", "heuristic"], default="anthropic")
    tri.add_argument("--model", default="claude-opus-5-5")
    efforts = ["low", "medium", "high", "xhigh", "max"]
    tri.add_argument("--effort", choices=efforts, default="medium")
    tri.add_argument("--n", type=int, default=60, help="cases to sample (0 = all)")
    tri.add_argument("--seed", type=int, default=0)
    tri.add_argument("--workers", type=int, default=4)
    tri.add_argument("--out", default="reports/triage")
    tri.set_defaults(func=cmd_triage_eval)

    mon = sub.add_parser("monitor", help="monitoring study: drift scenarios with delayed labels")
    mon.add_argument("--config", default="configs/monitoring.yaml")
    mon.add_argument("--seeds", type=int, nargs="*", help="simulated worlds (first one reported)")
    mon.add_argument("--out", default="reports/monitoring")
    mon.set_defaults(func=cmd_monitor)

    args = parser.parse_args(argv)
    if getattr(args, "n", None) == 0:
        args.n = None
    args.func(args)


if __name__ == "__main__":
    main()
