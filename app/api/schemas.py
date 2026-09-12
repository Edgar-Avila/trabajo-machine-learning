"""Pydantic request and response contracts."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class CustomerInput(BaseModel):
    """Validated placeholder feature contract for the churn pipeline."""
    model_config = ConfigDict(populate_by_name=True)
    customer_id: str = Field(min_length=1, max_length=128, alias="CustomerID")
    age: int = Field(ge=18, le=120, alias="Age")
    gender: Gender = Field(alias="Gender")
    tenure: int = Field(ge=0, le=1200, alias="Tenure")
    usage_frequency: int = Field(ge=0, le=10000, alias="Usage Frequency")
    support_calls: int = Field(ge=0, le=10000, alias="Support Calls")
    payment_delay: int = Field(ge=0, le=3650, alias="Payment Delay")
    subscription_type: str = Field(min_length=1, max_length=100, alias="Subscription Type")
    contract_length: str = Field(min_length=1, max_length=100, alias="Contract Length")
    total_spend: float = Field(ge=0, le=10_000_000, alias="Total Spend")
    last_interaction: int = Field(ge=0, le=3650, alias="Last Interaction")

    @field_validator("subscription_type", "contract_length")
    @classmethod
    def non_blank(cls, value: str) -> str:
        """Normalize and reject blank categorical values."""
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

class PredictionResponse(BaseModel):
    """Prediction returned by the API."""
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel

class HealthResponse(BaseModel):
    """Readiness response."""
    status: str
    model_loaded: bool
    model_source: str
    model_version: str | None = None

class CustomerChurnRow(BaseModel):
    """One customer row with its stored churn prediction."""
    model_config = ConfigDict(populate_by_name=True)
    customer_id: str = Field(alias="CustomerID")
    age: int = Field(alias="Age")
    gender: str = Field(alias="Gender")
    tenure: int = Field(alias="Tenure")
    usage_frequency: int = Field(alias="Usage Frequency")
    support_calls: int = Field(alias="Support Calls")
    payment_delay: int = Field(alias="Payment Delay")
    subscription_type: str = Field(alias="Subscription Type")
    contract_length: str = Field(alias="Contract Length")
    total_spend: float = Field(alias="Total Spend")
    last_interaction: int = Field(alias="Last Interaction")
    churn_probability: float = Field(ge=0, le=1, alias="ChurnProbability")
    model_version: int = Field(alias="ModelVersion")
    risk_level: RiskLevel = Field(alias="RiskLevel")

class CustomerListResponse(BaseModel):
    """Paginated customer churn table."""
    items: list[CustomerChurnRow]
    page: int
    page_size: int
    total: int
    total_pages: int

class RiskCount(BaseModel):
    """Number of customers in one risk bucket."""
    level: RiskLevel
    count: int
    percentage: float

class GroupStat(BaseModel):
    """Churn average for a categorical group."""
    name: str
    customers: int
    avg_churn_probability: float

class DashboardSummary(BaseModel):
    """Aggregates shown in the dashboard below the customer table."""
    total_customers: int
    avg_churn_probability: float
    risk_counts: list[RiskCount]
    by_subscription: list[GroupStat]
    by_contract: list[GroupStat]

class VersionPrediction(BaseModel):
    """One churn prediction for a given model version."""
    version: int
    churn_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    predicted_at: str
    delta: float | None = None

class CustomerHistoryResponse(BaseModel):
    """Churn evolution of one customer across model versions."""
    customer_id: str
    churn_label: int | None = None
    history: list[VersionPrediction]

