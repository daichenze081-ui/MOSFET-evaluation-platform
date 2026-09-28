"""Public workflow entry points."""

from mosfet_platform.workflows.evaluate import EvaluationResult, run_evaluation, run_metric_evaluation
from mosfet_platform.workflows.fit import FitResult, run_fit
from mosfet_platform.workflows.predict import PredictionResult, run_prediction
from mosfet_platform.workflows.compare import ComparisonResult, run_comparison
from mosfet_platform.workflows.update import UpdateResult, run_update
from mosfet_platform.workflows.diagnose import run_diagnosis

__all__ = [
    "run_diagnosis",
    "UpdateResult",
    "run_update",
    "ComparisonResult",
    "run_comparison",
    "EvaluationResult",
    "FitResult",
    "PredictionResult",
    "run_evaluation",
    "run_metric_evaluation",
    "run_fit",
    "run_prediction",
]
