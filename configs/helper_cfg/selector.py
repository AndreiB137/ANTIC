from typing import Annotated, Literal
from pydantic import BaseModel, Field, ConfigDict

class CommonSelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class NoneSelectorConfig(CommonSelectorConfig):
    type: Literal["none"] = "none"

class KDVSelectorConfig(CommonSelectorConfig):
    type: Literal["kdv"] = "kdv"
    high_quantile: float = 0.95
    low_quantile: float = 0.05
    window_size: int = 10

class KolmogorovSelectorConfig(CommonSelectorConfig):
    type: Literal["kolmogorov"] = "kolmogorov"
    metric: str = "max_abs"
    corr_threshold: float = 0.90
    window_size: int = 10

class BSSNSelectorConfig(CommonSelectorConfig):
    type: Literal["bssn"] = "bssn"
    surge_factor: float = 1.55
    history_len: int = 60
    patience_factor: int = 5
    window_size: int = 10

SelectorConfig = Annotated[
    NoneSelectorConfig | KDVSelectorConfig | KolmogorovSelectorConfig | BSSNSelectorConfig,
    Field(discriminator="type"),
]