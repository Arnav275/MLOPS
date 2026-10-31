"""Request / response contracts for the inference API (validated by Pydantic)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YN = Literal["yes", "no"]
Job = Literal["teacher", "health", "services", "at_home", "other"]


class StudentRecord(BaseModel):
    """One student at prediction time. Field names/ranges follow the UCI data dictionary.

    Required: subject, g1, absences (g2 is required by the mid-term model).  Everything else
    is optional - missing values are imputed with training medians/modes and reported back.
    """
    model_config = ConfigDict(extra="forbid")

    student_id: Optional[str] = Field(None, max_length=40, description="Your identifier; echoed back and used to attach outcomes via /feedback")
    subject: Literal["math", "portuguese"]
    g1: int = Field(..., ge=0, le=20, description="Period-1 grade")
    g2: Optional[int] = Field(None, ge=0, le=20, description="Period-2 grade (required for mid_term model)")
    absences: int = Field(..., ge=0, le=100)
    failures: Optional[int] = Field(None, ge=0, le=4)
    studytime: Optional[int] = Field(None, ge=1, le=4)
    traveltime: Optional[int] = Field(None, ge=1, le=4)
    age: Optional[int] = Field(None, ge=10, le=30)
    medu: Optional[int] = Field(None, ge=0, le=4)
    fedu: Optional[int] = Field(None, ge=0, le=4)
    famrel: Optional[int] = Field(None, ge=1, le=5)
    freetime: Optional[int] = Field(None, ge=1, le=5)
    goout: Optional[int] = Field(None, ge=1, le=5)
    dalc: Optional[int] = Field(None, ge=1, le=5)
    walc: Optional[int] = Field(None, ge=1, le=5)
    health: Optional[int] = Field(None, ge=1, le=5)
    school: Optional[Literal["GP", "MS"]] = None
    sex: Optional[Literal["F", "M"]] = None
    address: Optional[Literal["U", "R"]] = None
    famsize: Optional[Literal["LE3", "GT3"]] = None
    pstatus: Optional[Literal["T", "A"]] = None
    mjob: Optional[Job] = None
    fjob: Optional[Job] = None
    reason: Optional[Literal["home", "reputation", "course", "other"]] = None
    guardian: Optional[Literal["mother", "father", "other"]] = None
    schoolsup: Optional[YN] = None
    famsup: Optional[YN] = None
    paid: Optional[YN] = None
    activities: Optional[YN] = None
    nursery: Optional[YN] = None
    higher: Optional[YN] = None
    internet: Optional[YN] = None
    romantic: Optional[YN] = None


class Prediction(BaseModel):
    request_id: str
    student_id: Optional[str]
    fail_probability: float = Field(..., ge=0, le=1, description="Model score in [0,1]. NOT calibrated: class-weighted training overstates mid/high scores (see docs/MODEL_CARD.md). Rank with it, act on risk_band / predicted_fail.")
    predicted_fail: bool
    risk_band: Literal["Low", "Medium", "High"]
    recommended_action: str
    threshold: float
    model_version: str
    model_scenario: str
    imputed_fields: list[str]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[StudentRecord] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]


class Feedback(BaseModel):
    """Ground truth that becomes available later (end-of-term grade)."""
    model_config = ConfigDict(extra="forbid")
    request_id: Optional[str] = None
    student_id: Optional[str] = None
    actual_final_grade: Optional[float] = Field(None, ge=0, le=20)
    actual_fail: Optional[bool] = None


class FeedbackResponse(BaseModel):
    recorded: bool
    request_id: str
    actual_fail: bool
