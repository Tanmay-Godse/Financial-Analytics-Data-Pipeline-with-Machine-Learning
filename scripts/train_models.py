from financial_analytics_pipeline.orchestrator import run_training_pipeline


if __name__ == "__main__":
    summary = run_training_pipeline()
    best = summary["best_classifier"]
    metrics = best["test"]
    print("Full project run complete.")
    print(f"Best classifier: {best['model_name']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print("Generated outputs:")
    for name, path in summary["visualizations"].items():
        print(f"  - {name}: {path}")
    print("Saved summary: artifacts/model_results.json")
