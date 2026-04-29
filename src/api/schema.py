# for automatic validation of api response
from pydantic import BaseModel, Field

class LoanRequest(BaseModel):
    annual_inc: float = Field(..., gt=0)
    loan_amnt: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)