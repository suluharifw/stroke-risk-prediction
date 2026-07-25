"""Stroke prediction — an end-to-end ML package.

Public API::

    from stroke_prediction import StrokePredictor, PatientRecord

    predictor = StrokePredictor()
    result = predictor.predict(PatientRecord(
        gender="Male", age=67, hypertension=0, heart_disease=1,
        ever_married="Yes", work_type="Private",
        avg_glucose_level=228.0, bmi=36.6, smoking_status="formerly smoked",
    ))
    print(result.as_dict())
"""
from .inference import Prediction, StrokePredictor
from .schema import PatientRecord

__version__ = "2.0.0"
__all__ = ["StrokePredictor", "Prediction", "PatientRecord", "__version__"]
