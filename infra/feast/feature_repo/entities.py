"""
Entity definitions for the feature store.
"""
from feast import Entity, ValueType

# Define the symbol entity
symbol = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Stock ticker symbol",
    join_keys=["symbol"],
)
