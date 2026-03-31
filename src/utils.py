"""
utils.py
---------
Utility functions for the road crack detection project:
  - Loading images
  - Saving images
  - Displaying intermediate results using Matplotlib
"""

import os
import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from disk.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded BGR image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If OpenCV fails to read the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    print(f"[INFO] Loaded image: {image_path}  "
          f"(Size: {image.shape[1]}x{image.shape[0]})")
    return image


def save_image(image, output_path):
    """
    Save an image to disk. Creates parent directories if they don't exist.

    Parameters:
        image (numpy.ndarray): Image to save.
        output_path (str): Destination file path.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[INFO] Saved image: {output_path}")


def display_results(images_dict, save_path=None):
    """
    Display multiple images side-by-side using Matplotlib.
    Optionally save the composite figure to disk.

    Parameters:
        images_dict (dict): Ordered dictionary where:
            key   = title string to display above the image
            value = image (numpy.ndarray, either BGR or grayscale)
        save_path (str or None): If provided, save the figure to this path.
    """
    num_images = len(images_dict)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Handle the case when there is only one image
    if num_images == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images_dict.items()):
        # Convert BGR to RGB for colour images, keep grayscale as-is
        if len(image.shape) == 3:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(display_img)
        else:
            ax.imshow(image, cmap="gray")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved results figure: {save_path}")

    plt.show()


def display_detailed_results(stages_dict, save_path=None):
    """
    Display intermediate processing stages in a 2-row grid layout.
    Use this for showing all pipeline stages at once.

    Parameters:
        stages_dict (dict): Dictionary of stage_name -> image pairs.
        save_path (str or None): If provided, save the figure to this path.
    """
    num = len(stages_dict)
    cols = min(num, 4)                     # max 4 columns
    rows = (num + cols - 1) // cols        # enough rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (title, image) in enumerate(stages_dict.items()):
        if len(image.shape) == 3:
            axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[idx].imshow(image, cmap="gray")
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].axis("off")

    # Hide any unused subplot slots
    for idx in range(num, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved detailed results: {save_path}")

    plt.show()


def print_severity_report(severity, percentage, num_contours):
    """
    Print a formatted severity report to the console.

    Parameters:
        severity (str): Severity label ("LOW", "MEDIUM", or "HIGH").
        percentage (float): Percentage of image covered by cracks.
        num_contours (int): Number of crack contours detected.
    """
    print("\n" + "=" * 50)
    print("        CRACK DETECTION REPORT")
    print("=" * 50)
    print(f"  Crack contours detected : {num_contours}")
    print(f"  Crack area coverage     : {percentage:.2f} %")
    print(f"  Severity level          : {severity}")
    print("=" * 50)

    if severity == "LOW":
        print("  -> Minor surface cracks. Road is mostly safe.")
    elif severity == "MEDIUM":
        print("  -> Moderate damage. Maintenance recommended.")
    else:
        print("  -> Severe cracking. Immediate repair needed!")
    print("=" * 50 + "\n")
