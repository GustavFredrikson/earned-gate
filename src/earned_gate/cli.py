"""Command line: inspect a ledger without writing code.

    earned-gate verify LEDGER.db
    earned-gate rollout LEDGER.db --feature F --model-version V \
        --routes ignore,persist,ask,defer,agent --skippable ignore,persist
    earned-gate label LEDGER.db EVAL_ID ROUTE [--source agent]

``rollout`` prints a JSON report and exits 0 when the gate is open, 1 when it
is closed, so scripts can only ever act on an explicit "open". It validates
``--routes``/``--skippable`` like ``GateConfig`` does, and it always uses the
default ``RolloutPolicy`` thresholds (only the window is configurable), so a
gate with a custom policy can disagree with it. ``verify`` exits 1 when the
hash chain does not verify. Every command needs an existing ledger: a missing
path or a file that is not an earned-gate ledger exits 2 and creates nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .errors import EarnedGateError, LedgerIntegrityError
from .ledger import Ledger
from .rollout import RolloutPolicy, assess, check_routes


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="earned-gate", description="Inspect an earned-gate ledger."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="verify the ledger's hash chain")
    p_verify.add_argument("ledger")

    p_roll = sub.add_parser(
        "rollout",
        help="assess the rollout gate with the default thresholds (exit 0 = open)",
    )
    p_roll.add_argument("ledger")
    p_roll.add_argument("--feature", required=True)
    p_roll.add_argument("--model-version", required=True)
    p_roll.add_argument("--routes", required=True, type=_csv, help="cheapest first")
    p_roll.add_argument("--skippable", required=True, type=_csv)
    p_roll.add_argument("--window-days", type=float, default=30.0)

    p_label = sub.add_parser("label", help="record the route an evaluation actually needed")
    p_label.add_argument("ledger")
    p_label.add_argument("eval_id")
    p_label.add_argument("route")
    p_label.add_argument("--source", default="agent")

    args = parser.parse_args(argv)
    try:
        with Ledger(args.ledger, create=False) as ledger:
            if args.cmd == "verify":
                print(json.dumps({"ok": True, "events": ledger.verify()}))
                return 0
            if args.cmd == "rollout":
                check_routes(args.routes, args.skippable)
                report = assess(
                    ledger,
                    feature=args.feature,
                    model_version=args.model_version,
                    routes=args.routes,
                    skippable=args.skippable,
                    policy=RolloutPolicy(window_s=args.window_days * 86400.0),
                )
                print(json.dumps(report.to_json(), indent=2))
                return 0 if report.open else 1
            ledger.label(args.eval_id, args.route, source=args.source)
            print(json.dumps({"ok": True}))
            return 0
    except EarnedGateError as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1 if isinstance(exc, LedgerIntegrityError) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
