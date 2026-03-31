"""
main.py
--------
Entry point for the Smart Road Crack Detection system.

Usage:
    python src/main.py                      # Process all images in data/
    python src/main.py data/sample1.jpg     # Process a single image

Pipeline Overview:
    1. Load input image
    2. Convert to grayscale
    3. Enhance contrast (CLAHE)
    4. Apply Gaussian Blur for noise removal
    5. Detect edges using Canny Edge Detection
    6. Apply adaptive thresholding
    7. Combine edge and threshold results
    8. Clean up with morphological operations
    9. Detect and filter contours
   10. Draw cracks on original image and classify severity
"""

import os
import sys
import glob

# ── Make sure the src/ package is importable ─────────────────────────────
# When running as  `python src/main.py` from the project root, the current
# working directory is the project root, so we need to add "src/" to the
# module search path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocessing import (
    convert_to_grayscale,
    apply_gaussian_blur,
    apply_clahe,
    apply_adaptive_threshold,
)
from detection import (
    apply_canny_edge_detection,
    apply_morphological_pipeline,
    detect_contours,
    draw_cracks_on_image,
    classify_crack_severity,
)
from utils import (
    load_image,
    save_image,
    display_detailed_results,
    print_severity_report,
)


# =========================================================================
#  Core Processing Pipeline
# =========================================================================

def process_image(image_path, output_dir="outputs"):
    """
    Run the complete crack detection pipeline on a single image.

    Parameters:
        image_path (str): Path to the input road image.
        output_dir (str): Directory where output images will be saved.

    Returns:
        tuple: (result_image, severity, percentage, num_contours)
    """
    # ── Step 1: Load the image ───────────────────────────────────────────
    original = load_image(image_path)

    # ── Step 2: Convert to grayscale ─────────────────────────────────────
    gray = convert_to_grayscale(original)

    # ── Step 3: Enhance contrast using CLAHE ─────────────────────────────
    enhanced = apply_clahe(gray)

    # ── Step 4: Apply Gaussian Blur to remove noise ──────────────────────
    blurred = apply_gaussian_blur(enhanced, kernel_size=(5, 5))

    # ── Step 5: Apply Canny Edge Detection ───────────────────────────────
    edges = apply_canny_edge_detection(blurred, low_threshold=50,
                                        high_threshold=150)

    # ── Step 6: Apply adaptive thresholding ──────────────────────────────
    thresh = apply_adaptive_threshold(blurred, block_size=11, constant=2)

    # ── Step 7: Combine edges + threshold for robust detection ───────────
    #     Bitwise OR merges both results so we capture cracks detected
    #     by either method.
    import cv2
    combined = cv2.bitwise_or(edges, thresh)

    # ── Step 8: Clean up with morphological operations ───────────────────
    cleaned = apply_morphological_pipeline(combined)

    # ── Step 9: Detect contours of cracks ────────────────────────────────
    contours = detect_contours(cleaned, min_area=100)

    # ── Step 10: Draw cracks on the original image (red) ─────────────────
    result = draw_cracks_on_image(original, contours,
                                   color=(0, 0, 255), thickness=2)

    # ── Step 11: Classify severity ───────────────────────────────────────
    severity, percentage = classify_crack_severity(contours, original.shape)

    # ── Print the report to console ──────────────────────────────────────
    print_severity_report(severity, percentage, len(contours))

    # ── Save intermediate and final outputs ──────────────────────────────
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_image(gray,    os.path.join(output_dir, f"{basename}_1_grayscale.jpg"))
    save_image(blurred, os.path.join(output_dir, f"{basename}_2_blurred.jpg"))
    save_image(edges,   os.path.join(output_dir, f"{basename}_3_edges.jpg"))
    save_image(thresh,  os.path.join(output_dir, f"{basename}_4_threshold.jpg"))
    save_image(cleaned, os.path.join(output_dir, f"{basename}_5_morphology.jpg"))
    save_image(result,  os.path.join(output_dir, f"{basename}_6_result.jpg"))

    # ── Display all stages ───────────────────────────────────────────────
    stages = {
        "1. Original":        original,
        "2. Grayscale":       gray,
        "3. Blurred":         blurred,
        "4. Canny Edges":     edges,
        "5. Threshold":       thresh,
        "6. Morphology":      cleaned,
        "7. Detected Cracks": result,
    }
    display_detailed_results(
        stages,
        save_path=os.path.join(output_dir, f"{basename}_pipeline.png")
    )

    return result, severity, percentage, len(contours)


# =========================================================================
#  Entry Point
# =========================================================================

def main():
    """
    Main function — determines which images to process based on
    command-line arguments:
      • No arguments  → process all .jpg / .png images in data/
      • One argument  → process that single image
    """
    # Determine the project root (one level up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, "data")
    output_dir   = os.path.join(project_root, "outputs")

    if len(sys.argv) > 1:
        # ── Process a single image passed as argument ────────────────────
        image_path = sys.argv[1]
        if not os.path.isabs(image_path):
            image_path = os.path.join(project_root, image_path)
        process_image(image_path, output_dir)
    else:
        # ── Process all images in the data/ folder ───────────────────────
        image_files = sorted(
            glob.glob(os.path.join(data_dir, "*.jpg")) +
            glob.glob(os.path.join(data_dir, "*.jpeg")) +
            glob.glob(os.path.join(data_dir, "*.png"))
        )

        if not image_files:
            print(f"[WARNING] No images found in '{data_dir}/'.")
            print("  Place .jpg or .png road images in the data/ folder and retry.")
            sys.exit(1)

        print(f"\n[INFO] Found {len(image_files)} image(s) to process.\n")

        for idx, img_path in enumerate(image_files, start=1):
            print(f"\n{'-' * 60}")
            print(f"  Processing image {idx}/{len(image_files)}: "
                  f"{os.path.basename(img_path)}")
            print(f"{'-' * 60}")
            process_image(img_path, output_dir)

        print("\n[DONE] All images processed. Results saved to 'outputs/' folder.\n")


if __name__ == "__main__":
    main()
