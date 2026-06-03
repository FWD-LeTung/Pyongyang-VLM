from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_understanding.llm_parser import QueryParser  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo runner."""

    parser = argparse.ArgumentParser(
        description="Demo query_understanding and print the typed JSON response.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Pedestrian search query in Vietnamese or English.",
    )
    return parser.parse_args()


def silence_console_logs() -> None:
    """Keep terminal output reserved for the final JSON response."""

    for logger_name in ("src.query_understanding.llm_parser", "src.utils.timer"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]


def main() -> int:
    """Run query understanding once and print a JSON response."""

    load_dotenv(PROJECT_ROOT / ".env")
    silence_console_logs()

    args = parse_args()
    raw_query = " ".join(args.query).strip()
    if not raw_query:
        raw_query = input("Query: ").strip()

    response = QueryParser().parse(raw_query)
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
