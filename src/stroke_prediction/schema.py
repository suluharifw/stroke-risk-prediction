"""Input schema and validation for a single prediction request.

Keeping request parsing in one typed place means the web layer, the CLI,
and the tests all validate inputs the same way and fail with clear messages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Gender = Literal["Male", "Female"]
YesNo = Literal["Yes", "No"]
WorkType = Literal["Govt_job", "Never_worked", "Private", "Self-employed", "children"]
SmokingStatus = Literal["formerly smoked", "never smoked", "smokes"]

_WORK_TYPES = ("Govt_job", "Never_worked", "Private", "Self-employed", "children")
_SMOKING = ("formerly smoked", "never smoked", "smokes")


@dataclass
class PatientRecord:
    """A validated patient record ready for feature encoding.

    Raises:
        ValueError: if any field is outside its allowed range or set.
    """

    gender: Gender
    age: float
    hypertension: int
    heart_disease: int
    ever_married: YesNo
    work_type: WorkType
    avg_glucose_level: float
    bmi: float
    smoking_status: SmokingStatus

    def __post_init__(self) -> None:
        if self.gender not in ("Male", "Female"):
            raise ValueError(f"gender must be Male/Female, got {self.gender!r}")
        if not 0 < self.age <= 120:
            raise ValueError(f"age must be in (0, 120], got {self.age}")
        if self.hypertension not in (0, 1):
            raise ValueError("hypertension must be 0 or 1")
        if self.heart_disease not in (0, 1):
            raise ValueError("heart_disease must be 0 or 1")
        if self.ever_married not in ("Yes", "No"):
            raise ValueError("ever_married must be Yes/No")
        if self.work_type not in _WORK_TYPES:
            raise ValueError(f"work_type must be one of {_WORK_TYPES}")
        if not 0 < self.avg_glucose_level < 600:
            raise ValueError("avg_glucose_level out of plausible range")
        if not 0 < self.bmi < 100:
            raise ValueError("bmi out of plausible range")
        if self.smoking_status not in _SMOKING:
            raise ValueError(f"smoking_status must be one of {_SMOKING}")

    @classmethod
    def from_dict(cls, payload: dict) -> "PatientRecord":
        """Build a record from a raw request dict, coercing numeric fields."""
        return cls(
            gender=payload["gender"],
            age=float(payload["age"]),
            hypertension=int(payload["hypertension"]),
            heart_disease=int(payload["heart_disease"]),
            ever_married=payload["ever_married"],
            work_type=payload["work_type"],
            avg_glucose_level=float(payload["avg_glucose_level"]),
            bmi=float(payload["bmi"]),
            smoking_status=payload["smoking_status"],
        )
