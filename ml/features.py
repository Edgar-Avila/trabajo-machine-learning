"""Shared feature definitions for the churn pipeline."""
FEATURES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

NUMERIC = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]

TARGET = "Churn"