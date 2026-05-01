from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource
from feast import ValueType

borrower = Entity(
    name="borrower_id",
    join_keys=["borrower_id"],
    value_type=ValueType.INT64
)

data_source = FileSource(
    path="../data/data/processed/cleaned_data.parquet",
    #path="/home/kaarvin/projects/credit-risk-platform/data/processed/cleaned_data.parquet",
    event_timestamp_column="event_timestamp"
)

loan_features = FeatureView(
    name="loan_features",
    entities=[borrower],
    schema=[
        Field(name="annual_inc", dtype=Float32),
        Field(name="loan_amnt", dtype=Float32),
        Field(name="dti", dtype=Float32),
        Field(name="loan_to_income", dtype=Float32),
        Field(name="dti_ratio", dtype=Float32),
        Field(name="log_income", dtype=Float32),
        Field(name="log_loan", dtype=Float32),
        Field(name="income_bucket", dtype=Int64),
        Field(name="loan_bucket", dtype=Int64),
        Field(name="income_x_dti", dtype=Float32),
        Field(name="loan_x_dti", dtype=Float32),
        Field(name="high_dti_flag", dtype=Int64),
        Field(name="low_income_flag", dtype=Int64),
        Field(name="high_loan_flag", dtype=Int64),
    ],
    source=data_source,
)









