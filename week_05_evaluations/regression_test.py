import json
from datetime import datetime
from pathlib import Path

class RegressionTracker:
    """Track evaluation results over time."""

    def __init__(self, results_file: str = "eval_history.json"):
        self.results_file = Path(results_file)
        self.history = self._load_history()

    def _load_history(self) -> list:
        if self.results_file.exists():
            return json.loads(self.results_file.read_text())
        return []

    def save_result(self, scores: dict, version: str = None):
        """Save evaluation result."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "version": version or "unknown",
            "scores": scores
        }

        self.history.append(result)
        self.results_file.write_text(json.dumps(self.history, indent=2))

        # Check for regression
        if len(self.history) > 1:
            prev = self.history[-2]["scores"]
            curr = scores

            for metric, value in curr.items():
                if metric in prev and value < prev[metric] - 0.05:
                    print(f"⚠️  REGRESSION: {metric} dropped from {prev[metric]:.2%} to {value:.2%}")

    def show_history(self):
        """Print history."""
        print("\nEvaluation History:")
        for result in self.history[-5:]:
            print(f"  {result['timestamp']}: {result['scores'].get('overall', 'N/A'):.2%}")

# Usage
if __name__ == "__main__":
    tracker = RegressionTracker()

    # Save a result
    tracker.save_result({
        "retrieval": 0.85,
        "faithfulness": 0.90,
        "correctness": 0.88,
        "overall": 0.88
    }, version="v1.0")

    tracker.show_history()
