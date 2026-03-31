"""
preprocessing.py
-----------------
This module contains functions for image preprocessing steps:
  - Grayscale conversion
  - Noise removal (Gaussian Blur)
  - Adaptive thresholding
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""

import cv2
import numpy as np


def convert_to_grayscale(image):
    """
    Convert a BGR color image to grayscale.

    Parameters:
        image (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Blur to remove noise from the image.

    A larger kernel size results in more blurring. The default (5, 5)
    provides a good balance between noise removal and detail preservation.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        kernel_size (tuple): Size of the Gaussian kernel. Must be odd numbers.

    Returns:
        numpy.ndarray: Blurred image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    enhance the contrast of the grayscale image. This helps in making
    cracks more visible against the road surface.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the grid for histogram equalization.

    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def apply_adaptive_threshold(image, max_value=255, block_size=11, constant=2):
    """
    Apply adaptive thresholding to create a binary image that highlights
    potential crack regions. Unlike global thresholding, adaptive
    thresholding calculates different thresholds for different areas,
    which works better for images with varying lighting conditions.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        max_value (int): Maximum value for thresholding.
        block_size (int): Size of the neighbourhood area. Must be odd.
        constant (int): Constant subtracted from the mean.

    Returns:
        numpy.ndarray: Binary (thresholded) image.
    """
    # Invert so that cracks (dark regions) become white
    thresh = cv2.adaptiveThreshold(
        image,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        constant
    )
    return thresh
