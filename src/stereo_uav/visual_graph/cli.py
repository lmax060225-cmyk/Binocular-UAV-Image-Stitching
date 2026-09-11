"""Cli for the visual graph algorithm."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2

from .features import (
    VisualVerifier,
    load_stereo_pairs,
    point_policy,
)
from .graph import (
    build_incremental_pair_graph,
)
from .models import (
    LEFT_IN,
    OUTPUT_DIR,
    RIGHT_IN,
    Config,
)
from .pipeline import (
    run_registration_and_mosaics,
)
from .reporting import (
    load_saved_graph,
    save_graph_diagnostics,
    save_json,
)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=LEFT_IN)
    parser.add_argument("--right", type=Path, default=RIGHT_IN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--resume-graph",
        action="store_true",
        help="Replay this program's final saved graph after input/config validation",
    )
    parser.add_argument("--opencv-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    from .selftest import run_self_tests

    args, cfg = parse_arguments(), Config()
    if (
        not args.self_test
        and not args.resume_graph
        and (args.output / "data" / "image_optimization_graph.csv").exists()
    ):
        print("ERROR: Results already exist. Use --resume-graph or a new --output directory.")
        return 1
    cv2.setNumThreads(args.opencv_threads)
    cv2.setRNGSeed(cfg.RANDOM_SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        with (args.output / "self_test.log").open("w", encoding="utf-8") as logfile:
            previous = sys.stdout
            sys.stdout = Tee(previous, logfile)
            try:
                return 0 if run_self_tests(args.output) else 1
            finally:
                sys.stdout = previous
    for name in ("data", "mosaics"):
        (args.output / name).mkdir(exist_ok=True)
    logname = "run_resume.log" if args.resume_graph else "run.log"
    with (args.output / logname).open("w", encoding="utf-8") as logfile:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        started = time.perf_counter()
        try:
            print(
                f"Python = {sys.executable}\nOpenCV = {cv2.__version__}\n"
                f"Output = {args.output.resolve()}",
                flush=True,
            )
            pairs, paths = load_stereo_pairs(args.left, args.right, cfg)
            print(
                f"Synchronized pairs = {len(pairs)}, images = {len(paths)}, "
                f"P/grid = {point_policy(len(paths))}",
                flush=True,
            )
            if args.resume_graph:
                edges, shapes, stats = load_saved_graph(args.output, paths, cfg)
                print("Replaying validated FINAL image optimization graph", flush=True)
            else:
                verifier = VisualVerifier(paths, len(paths), cfg)
                graph = build_incremental_pair_graph(pairs, verifier, cfg)
                stats = save_graph_diagnostics(args.output, pairs, paths, graph, verifier, cfg)
                edges = list(graph.optimization_edges_by_pair.values())
                shapes = {key: record.shape for key, record in verifier.feature_cache.items()}
                assert set(shapes) == set(paths)
                assert all(count == 1 for count in verifier.extraction_counts.values())
                del verifier, graph
            metrics, optimizer, mosaics = run_registration_and_mosaics(
                paths, edges, shapes, args.output, cfg
            )
            summary = dict(
                status="completed",
                graph=stats,
                metrics=metrics,
                optimizer=optimizer,
                mosaics=mosaics,
                runtime=time.perf_counter() - started,
            )
            save_json(args.output / "run_summary.json", summary)
            print(f"RUN_COMPLETED in {summary['runtime']:.2f} seconds", flush=True)
            return 0
        except Exception:
            traceback.print_exc()
            save_json(
                args.output / "run_failure.json",
                dict(
                    status="failed",
                    traceback=traceback.format_exc(),
                    runtime=time.perf_counter() - started,
                ),
            )
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr
