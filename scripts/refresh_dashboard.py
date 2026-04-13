import argparse

from financial_analytics_pipeline.config import PipelineConfig
from financial_analytics_pipeline.orchestrator import refresh_dashboard_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun the project and refresh the dashboard files."
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=PipelineConfig().refresh_interval_minutes,
        help="Minutes to wait between refresh cycles.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one refresh cycle and exit.",
    )
    arguments = parser.parse_args()
    if arguments.once:
        print("Running one dashboard refresh...")
    else:
        print(f"Refreshing dashboard every {arguments.interval_minutes} minutes...")
    refresh_dashboard_loop(PipelineConfig(), once=arguments.once, interval_minutes=arguments.interval_minutes)
