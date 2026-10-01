
import cv2
import numpy as np


def preprocess_fundus(image):
    """
    Preprocess retinal image for vessel segmentation.
    """

    image = cv2.resize(image, (512, 512))

    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(green)

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return enhanced


def load_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load {image_path}")

    return image


def preprocess_image(image_path):

    image = load_image(image_path)

    processed = preprocess_fundus(image)

    return processed
