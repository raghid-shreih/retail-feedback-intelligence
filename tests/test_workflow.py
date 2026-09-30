import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from retail_feedback.cli import main
from retail_feedback.client import ModelAPIError, OpenAIClient
from retail_feedback.data import load_reviews, sample_reviews
from retail_feedback.evaluation import compare, evaluate_recommendations, parse_recommendation

FIXTURE = Path(__file__).parent / "fixtures" / "reviews.csv"


class WorkflowTests(unittest.TestCase):
    def test_load_and_seeded_sample(self):
        frame = load_reviews(FIXTURE)
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(sample_reviews(frame, 3).index), list(sample_reviews(frame, 3).index))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.csv"
            path.write_text('Review.Text;Recommended.IND\n"Loved it";1\n"";0\n', encoding="utf-8")
            exported = load_reviews(path)
            self.assertEqual(exported["review_text"].tolist(), ["Loved it"])

    def test_invalid_prediction_abstains_and_counts_as_failed_attempt(self):
        frame = load_reviews(FIXTURE)
        responses = iter([
            '{"Recommended_IND":1,"Reason":"Great fit"}',
            '{"Recommended_IND":"0","Reason":"Broken"}',
            '{"Recommended_IND":1,"Reason":"Comfortable"}',
            '{"Recommended_IND":0,"Reason":"Returned"}',
        ])
        report = evaluate_recommendations(frame, lambda _: next(responses))
        self.assertEqual(report["valid"], 3)
        self.assertEqual(report["accuracy_all_attempts"], 0.75)
        self.assertEqual(report["accuracy_valid_only"], 1.0)
        self.assertEqual(report["confusion_matrix_valid_only"], {"tn": 1, "fp": 0, "fn": 0, "tp": 2})
        self.assertIsNone(report["rows"][1]["predicted"])
        with self.assertRaises(ValueError):
            parse_recommendation('{"Recommended_IND":false,"Reason":"No"}')

    def test_comparison_tracks_schema_failures(self):
        frame = load_reviews(FIXTURE).head(1)
        valid = {"Category": "Fit/Sizing", "Sentiment": "Positive", "Summary": "Good fit",
                 "Personalized_Message": "Thank you", "Retail_Insight": "Review sizing"}
        responses = iter([json.dumps(valid), "{}"])
        report = compare(frame, lambda _: next(responses), ["zero_v1", "few_v1"])
        self.assertEqual(report["summary"]["zero_v1"]["valid"], 1)
        self.assertEqual(report["summary"]["few_v1"]["valid"], 0)

    def test_offline_cli(self):
        self.assertEqual(main(["summary", str(FIXTURE)]), 0)

    def test_credit_exhaustion_stops_instead_of_counting_invalid_prediction(self):
        class FakeStatusError(Exception):
            status_code = 429
            code = "credit_balance_exhausted"

        class FakeConnectionError(Exception):
            pass

        fake_openai = ModuleType("openai")
        fake_openai.APIStatusError = FakeStatusError
        fake_openai.APIConnectionError = FakeConnectionError
        client = OpenAIClient.__new__(OpenAIClient)
        client.model = "test-model"

        def fail(**kwargs):
            raise FakeStatusError("no credits")

        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaisesRegex(ModelAPIError, "credits are exhausted"):
                evaluate_recommendations(load_reviews(FIXTURE).head(1), client.complete)


if __name__ == "__main__":
    unittest.main()
