"""Offline tests for preprocessing, training, persistence, and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.data_loader import COLUMNS, process_local_archive, validate_dataset
from src.predict import load_model_bundle, predict_from_values
from src.preprocessing import FEATURE_COLUMNS, make_pipeline, split_features_target
from src.train import train_models


def sample_data(rows: int = 60) -> pd.DataFrame:
    """Create deterministic schema-valid test rows without using network data."""
    rng = np.random.default_rng(17)
    index = np.arange(rows)
    data = pd.DataFrame(
        {
            "age": rng.integers(35, 75, rows),
            "sex": index % 2,
            "cp": index % 4 + 1,
            "trestbps": rng.integers(95, 180, rows),
            "chol": rng.integers(150, 340, rows),
            "fbs": index % 2,
            "restecg": index % 3,
            "thalach": rng.integers(80, 200, rows),
            "exang": index % 2,
            "oldpeak": rng.uniform(0, 4, rows).round(1),
            "slope": index % 3 + 1,
            "ca": index % 4,
            "thal": np.choose(index % 3, [3, 6, 7]).astype(float),
            "num": np.where(index % 3 == 0, 2, 0),
        },
        columns=COLUMNS,
    )
    data.loc[0, "ca"] = np.nan
    data.loc[1, "thal"] = np.nan
    return data


class PipelineTests(unittest.TestCase):
    def test_validation_rejects_unknown_target_values(self) -> None:
        data = sample_data()
        data.loc[0, "num"] = 5
        with self.assertRaisesRegex(ValueError, "Unexpected raw target labels"):
            validate_dataset(data)

    def test_validation_rejects_unexpected_feature_text(self) -> None:
        data = sample_data()
        data["chol"] = data["chol"].astype(object)
        data.loc[2, "chol"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected non-numeric values"):
            validate_dataset(data)

    def test_local_archive_fallback_builds_processed_csv(self) -> None:
        data = sample_data()
        content = data.to_csv(index=False, header=False, na_rep="?")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.zip"
            output_path = root / "processed" / "cleveland.csv"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("processed.cleveland.data", content)

            loaded = process_local_archive(archive_path, output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(loaded.shape, (60, 14))
            self.assertEqual(int(loaded["ca"].isna().sum()), 1)
            self.assertEqual(int(loaded["thal"].isna().sum()), 1)

    def test_preprocessing_imputes_without_changing_feature_order(self) -> None:
        features, target = split_features_target(sample_data())
        pipeline = make_pipeline("Logistic Regression")
        transformed = pipeline.named_steps["preprocessor"].fit_transform(features)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(len(target), len(features))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(pipeline.fit(features, target).predict(features).shape, (60,))

    def test_training_artifact_loads_and_predicts(self) -> None:
        data = sample_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            results_path = root / "results.json"
            figures_path = root / "figures"
            results = train_models(
                data=data,
                model_path=model_path,
                results_path=results_path,
                figures_dir=figures_path,
                cv_splits=2,
            )

            self.assertTrue(model_path.is_file())
            self.assertTrue(results_path.is_file())
            self.assertIn(
                results["selected_model"], results["cross_validation"]["models"]
            )
            self.assertEqual(set(results["held_out_test_metrics"]), {
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
            })
            self.assertGreater(len(list(figures_path.glob("*.png"))), 5)

            bundle = load_model_bundle(model_path)
            self.assertEqual(bundle["feature_columns"], FEATURE_COLUMNS)
            row = data.loc[2, FEATURE_COLUMNS].to_dict()
            outcome = predict_from_values(row, bundle)
            self.assertIn(outcome["prediction"], [0, 1])
            self.assertGreaterEqual(outcome["probability"], 0.0)
            self.assertLessEqual(outcome["probability"], 1.0)

            row.pop("age")
            with self.assertRaisesRegex(ValueError, "Input fields mismatch"):
                predict_from_values(row, bundle)


if __name__ == "__main__":
    unittest.main()
