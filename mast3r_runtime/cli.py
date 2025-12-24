"""CLI for MASt3R runtime.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.config import ModelVariant
from .utils.downloader import (
    DownloadError,
    download_all_models,
    download_model,
    print_status,
)


def cmd_download(args: argparse.Namespace) -> int:
    """Download model checkpoints."""
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    try:
        if args.model == "all":
            download_all_models(
                cache_dir=cache_dir,
                force=args.force,
                quiet=args.quiet,
            )
        else:
            variant = ModelVariant(args.model)
            download_model(
                variant,
                cache_dir=cache_dir,
                force=args.force,
                quiet=args.quiet,
            )
        return 0

    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Invalid model: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show download status."""
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    print_status(cache_dir)
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mast3r-runtime",
        description="MASt3R embedded inference runtime",
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        help="Cache directory (default: ~/.cache/mast3r_runtime)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download model checkpoints")
    dl_parser.add_argument(
        "model",
        choices=["all"] + [v.value for v in ModelVariant],
        help="Model to download (or 'all')",
    )
    dl_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download",
    )
    dl_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )
    dl_parser.set_defaults(func=cmd_download)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show download status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
