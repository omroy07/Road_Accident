"""
detection.py
-------------
This module contains functions for crack detection:
  - Canny Edge Detection
  - Morphological operations (dilation, erosion, opening, closing)
  - Contour detection and filtering
  - Crack severity classification
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  Edge Detection
# ---------------------------------------------------------------------------

def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny Edge Detection to find edges in the image.

    Canny edge detection uses two thresholds:
      - Pixels with gradient above high_threshold are strong edges.
      - Pixels between low and high are kept only if connected to strong edges.
      - Pixels below low_threshold are discarded.

    Parameters:
        image (numpy.ndarray): Input grayscale/blurred image.
        low_threshold (int): Lower threshold for edge detection.
        high_threshold (int): Upper threshold for edge detection.

    Returns:
        numpy.ndarray: Edge-detected binary image.
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# ---------------------------------------------------------------------------
#  Morphological Operations
# ---------------------------------------------------------------------------

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    """
    Apply dilation to make detected edges (cracks) thicker and more visible.
    Dilation expands white (foreground) regions.

    Parameters:
        image (numpy.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element.
        iterations (int): Number of times dilation is applied.

    Returns:
        numpy.ndarray: Dilated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated


def apply_erosion(image, kernel_size=(3, 3), iterations=1):
    """
    Apply erosion to remove small noise from the binary image.
    Erosion shrinks white (foreground) regions.

    Parameters:
        image (numpy.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element.
        iterations (int): Number of times erosion is applied.

    Returns:
        numpy.ndarray: Eroded image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded


def apply_morphological_opening(image, kernel_size=(3, 3)):
    """
    Apply morphological opening (erosion followed by dilation).
    This removes small noise while preserving the shape of larger objects
    like cracks.

    Parameters:
        image (numpy.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element.

    Returns:
        numpy.ndarray: Image after opening operation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


def apply_morphological_closing(image, kernel_size=(3, 3)):
    """
    Apply morphological closing (dilation followed by erosion).
    This fills small gaps in detected crack lines, making them
    more continuous.

    Parameters:
        image (numpy.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element.

    Returns:
        numpy.ndarray: Image after closing operation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------------------------------------------------------
#  Full Morphological Pipeline
# ---------------------------------------------------------------------------

def apply_morphological_pipeline(image):
    """
    Apply a sequence of morphological operations optimised for crack
    detection:
      1. Closing  – fill small gaps in crack lines
      2. Opening  – remove isolated noise dots
      3. Dilation – make crack lines slightly thicker for visibility

    Parameters:
        image (numpy.ndarray): Input binary/edge image.

    Returns:
        numpy.ndarray: Cleaned binary image ready for contour detection.
    """
    # Step 1: Close small gaps in detected cracks
    closed = apply_morphological_closing(image, kernel_size=(3, 3))

    # Step 2: Remove isolated noise pixels
    opened = apply_morphological_opening(closed, kernel_size=(3, 3))

    # Step 3: Slightly dilate to enhance remaining crack lines
    result = apply_dilation(opened, kernel_size=(3, 3), iterations=1)

    return result


# ---------------------------------------------------------------------------
#  Contour Detection
# ---------------------------------------------------------------------------

def detect_contours(binary_image, min_area=100):
    """
    Find contours in the binary image and filter out very small ones
    that are likely noise rather than actual cracks.

    Parameters:
        binary_image (numpy.ndarray): Binary image (edges / threshold).
        min_area (int): Minimum contour area to keep. Contours with area
                        smaller than this are discarded as noise.

    Returns:
        list: Filtered list of contours (each contour is a numpy array).
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep only contours larger than the minimum area threshold
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return filtered_contours


def draw_cracks_on_image(original_image, contours, color=(0, 0, 255),
                         thickness=2):
    """
    Draw detected crack contours on the original image.

    Parameters:
        original_image (numpy.ndarray): Original BGR image (a copy is drawn on).
        contours (list): List of contours to draw.
        color (tuple): BGR color for the contour lines. Default is red.
        thickness (int): Thickness of contour lines.

    Returns:
        numpy.ndarray: Image with cracks highlighted.
    """
    output = original_image.copy()
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


# ---------------------------------------------------------------------------
#  Crack Severity Classification
# ---------------------------------------------------------------------------

def classify_crack_severity(contours, image_shape):
    """
    Classify the severity of cracks based on the total area covered
    by detected contours relative to the total image area.

    Severity Levels:
      - LOW    : crack area < 1 % of image
      - MEDIUM : crack area between 1 % and 5 %
      - HIGH   : crack area > 5 % of image

    Parameters:
        contours (list): Detected crack contours.
        image_shape (tuple): Shape of the image (height, width, ...).

    Returns:
        tuple: (severity_label, crack_percentage)
            severity_label (str): "LOW", "MEDIUM", or "HIGH"
            crack_percentage (float): Percentage of image covered by cracks.
    """
    # Calculate total area of all detected contours
    total_crack_area = sum(cv2.contourArea(c) for c in contours)

    # Total image area (height × width)
    image_area = image_shape[0] * image_shape[1]

    # Crack percentage
    crack_percentage = (total_crack_area / image_area) * 100 if image_area > 0 else 0

    # Classify severity
    if crack_percentage < 1.0:
        severity = "LOW"
    elif crack_percentage < 5.0:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return severity, round(crack_percentage, 2)
