#!/usr/bin/env python3
"""
LLVD — Lattice Layer Vehicle Detection
Entry point script.

Usage:
    python run.py                          # Run advanced pipeline with default config
    python run.py --config config/default.json
    python run.py --config config/default.json --pipeline base
    python run.py --help
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="LLVD — Lattice Layer Vehicle Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
      Run advanced pipeline with config/default.json

  python run.py --config config/sample_video.json
      Run with a custom config file

  python run.py --pipeline base
      Run the base pipeline (DBSCAN clustering, no tracking)

  python run.py --pipeline advanced
      Run the advanced pipeline (tracking, counting, speed estimation)
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.json",
        help="Path to the JSON config file (default: config/default.json)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["advanced", "base"],
        default="advanced",
        help="Which pipeline to run (default: advanced)",
    )
    args = parser.parse_args()

    # Verify config exists
    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("  Available configs:")
        config_dir = "config"
        if os.path.isdir(config_dir):
            for f in sorted(os.listdir(config_dir)):
                if f.endswith(".json"):
                    print(f"    config/{f}")
        sys.exit(1)

    # Copy the config to user_input_data.json (expected by the pipeline scripts)
    import shutil
    shutil.copy2(args.config, "user_input_data.json")
    print(f"Using config: {args.config}")

    if args.pipeline == "advanced":
        print("Starting Advanced Pipeline (tracking + counting + speed)...")
        print("=" * 70)
        # Import and run the advanced pipeline
        from src.advanced_pipeline import main as run_advanced
        run_advanced()

        # After main completes, print profiling results
        from src.advanced_pipeline import timing_stats, merge_profile_logs, print_profiling_results
        execution_time = run_advanced.__wrapped__() if hasattr(run_advanced, '__wrapped__') else None
        if execution_time is None:
            # The main() already printed results and returned elapsed time
            # Merge and print profiling
            merged_stats = merge_profile_logs(main_stats=timing_stats)
            if merged_stats:
                # Already printed inside main, skip double-print
                pass

    elif args.pipeline == "base":
        print("Starting Base Pipeline (DBSCAN clustering)...")
        print("=" * 70)
        from src.base_pipeline import main as run_base
        run_base()

    # Cleanup the temporary config copy
    if os.path.isfile("user_input_data.json"):
        os.remove("user_input_data.json")

    print("\nDone! Check the output/ directory for results.")


if __name__ == "__main__":
    main()
