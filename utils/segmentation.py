
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):

    if not os.path.exists(model_path):
     raise FileNotFoundError(f"Model not found: {model_path}")
    """
    Load the trained U-Net vessel segmentation model.
    """

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


def prepare_tensor(image):
    """
    Convert a preprocessed grayscale image to a PyTorch tensor.
    """

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    tensor = torch.from_numpy(image)

    return tensor.to(DEVICE)


def segment_vessels(image, model):
    """
    Segment retinal vessels from a preprocessed image.
    """

    tensor = prepare_tensor(image)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    mask = (mask * 255).astype(np.uint8)

    return mask
