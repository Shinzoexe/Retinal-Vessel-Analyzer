

import numpy as np
from skimage.morphology import skeletonize


def vessel_density(mask):
    """
    Percentage of vessel pixels.
    """
    return np.sum(mask > 0) / mask.size


def vessel_length(skeleton):
    """
    Vessel length using skeleton.
    """
    return np.sum(skeleton)


def count_branch_end_points(skeleton):

    branch_points = 0
    end_points = 0

    rows, cols = skeleton.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            if skeleton[i, j]:

                neighbors = np.sum(
                    skeleton[i-1:i+2, j-1:j+2]
                ) - 1

                if neighbors == 1:
                    end_points += 1

                elif neighbors > 2:
                    branch_points += 1

    return branch_points, end_points


def vessel_tortuosity(skeleton):

    coords = np.column_stack(
        np.where(skeleton > 0)
    )

    if len(coords) < 2:
        return 0

    vessel_length = len(coords)

    start = coords[0]
    end = coords[-1]

    straight_distance = np.linalg.norm(
        start - end
    )

    if straight_distance == 0:
        return 0

    return float(vessel_length / straight_distance)


def extract_features(mask):

    skeleton = skeletonize(mask > 0)

    density = vessel_density(mask)

    length = vessel_length(skeleton)

    branch_points, end_points = count_branch_end_points(
        skeleton
    )

    tortuosity = vessel_tortuosity(
        skeleton
    )

    return {
        "vessel_density": density,
        "vessel_length": length,
        "branch_points": branch_points,
        "end_points": end_points,
        "tortuosity": tortuosity
    }
