"""``weatherquant`` console-script entry point — the 4-branch subcommand dispatch."""

from __future__ import annotations

import logging

from ._args import build_parser
from .calibrate import run_calibrate
from .ingest import run_ingest
from .paper import run_paper
from .pricing import run_price
from .verify import run_verify


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``[project.scripts] weatherquant = weatherquant.cli:main``).

    Parses args, dispatches the ``ingest`` subcommand to :func:`run_ingest`, and returns a
    process exit code (0 on success). Argument validation errors (unknown city, malformed
    date) are raised by argparse BEFORE any ingest call (ASVS V5).
    """
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        totals = run_ingest(args)
        inserted = sum(totals.values())
        print(f"ingest complete: {inserted} row(s) inserted ({totals})")
        return 0
    if args.command == "calibrate":
        persisted = run_calibrate(args)
        total = sum(persisted.values())
        print(f"calibrate complete: {total} stratum row(s) persisted ({persisted})")
        return 0
    if args.command == "price":
        run_price(args)  # prints the blend/bucket/EV/Kelly smoke line(s) itself (D-16)
        return 0
    if args.command == "verify":
        # Unlike the count-dict branches above, propagate run_verify's int exit code so a drift
        # breach (--monitor) yields a NON-ZERO process exit (SYS-02 / D-12).
        return run_verify(args)
    run_paper(args)  # prints the real-midpoint loop-closure smoke line itself (D-08/D-16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
