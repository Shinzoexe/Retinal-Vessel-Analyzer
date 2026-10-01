
import numpy as np


FEATURE_COLUMNS = [
    "vessel_density",
    "vessel_length",
    "branch_points",
    "end_points",
    "tortuosity"
]


def classify_risk(score):
    """
    Classify stroke risk based on normalized score.
    """
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    else:
        return "High"


def calculate_risk(features):

    density = features["vessel_density"]
    length = features["vessel_length"] / 5000
    branch = features["branch_points"] / 200
    end = features["end_points"] / 200
    tortuosity = features["tortuosity"] / 5

    score = (
        0.30 * density +
        0.20 * length +
        0.20 * branch +
        0.10 * end +
        0.20 * tortuosity
    )

    score = min(score, 1.0)

    category = classify_risk(score)

    return score, category
