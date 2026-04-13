from financial_analytics_pipeline.orchestrator import run_data_pipeline


if __name__ == "__main__":
    artifacts = run_data_pipeline()
    print("Step 1 complete: data pipeline finished.")
    print(f"Source used: {artifacts.source_name}")
    print(f"Raw rows: {len(artifacts.raw_frame)}")
    print(f"Cleaned rows: {len(artifacts.cleaned_frame)}")
    print(f"Feature rows: {len(artifacts.feature_frame)}")
    print("Saved files:")
    print("  - data/raw/creditcard_snapshot.csv")
    print("  - data/processed/creditcard_cleaned.csv")
    print("  - data/processed/creditcard_features.csv")
    print("  - data/processed/financial_pipeline.sqlite")
