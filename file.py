from pathlib import Path

PROJECT_NAME = "cloud-cost-optimizer"

STRUCTURE = {
    "data/raw": [],
    "data/processed": [],
    "config": ["config.py"],
    "preprocessing": [
        "__init__.py",
        "data_loader.py",
        "cleaner.py",
        "aggregator.py",
        "feature_engineering.py",
    ],
    "clustering": [
        "__init__.py",
        "kmeans_clustering.py",
        "dbscan_clustering.py",
        "cluster_analyzer.py",
    ],
    "prediction": [
        "__init__.py",
        "arima_predictor.py",
        "lstm_predictor.py",
        "hybrid_predictor.py",
    ],
    "scaling": [
        "__init__.py",
        "scaling_policy.py",
        "cost_model.py",
    ],
    "evaluation": [
        "__init__.py",
        "baselines.py",
        "metrics.py",
        "experiments.py",
    ],
    "visualization": [
        "__init__.py",
        "plots.py",
    ],
    "": ["main.py", "requirements.txt", "README.md"],
}

PLACEHOLDER_CONTENT = {
    "__init__.py": "",
    ".py": "# TODO: implement\n",
    "README.md": "# Cloud Cost Optimizer\n\nWork in progress.\n",
    "requirements.txt": "numpy\npandas\nscikit-learn\nmatplotlib\ntorch\nstatsmodels\n",
}


def create_project(root: Path):
    for folder, files in STRUCTURE.items():
        dir_path = root / folder
        dir_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_path = dir_path / file
            if not file_path.exists():
                content = ""
                for key, value in PLACEHOLDER_CONTENT.items():
                    if file.endswith(key):
                        content = value
                file_path.write_text(content)


if __name__ == "__main__":
    root = Path(PROJECT_NAME)
    root.mkdir(exist_ok=True)
    create_project(root)
    print(f"Project structure '{PROJECT_NAME}' created.")
