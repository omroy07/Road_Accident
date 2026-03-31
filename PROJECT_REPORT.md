# Smart Road Crack Detection — Project Report

---

## 1. Introduction

Road infrastructure is a critical component of urban and rural transportation systems. Over time, road surfaces deteriorate due to traffic load, weather conditions, and poor construction quality. **Cracks** are among the earliest and most common signs of road degradation. Early detection of cracks allows municipal authorities to schedule timely repairs, reducing long-term costs and preventing accidents.

This project presents an **automated road crack detection system** built using classical computer vision techniques. The system processes images of road surfaces, identifies crack regions, highlights them visually, and classifies the damage severity — all without relying on deep learning or GPU computation.

---

## 2. Problem Statement

Manual inspection of roads for cracks and surface damage is:

- **Time-consuming** — inspectors must physically visit each road segment
- **Subjective** — different inspectors may classify the same damage differently
- **Expensive** — requires trained personnel and repeated field visits
- **Incomplete** — large portions of road networks remain uninspected

There is a need for an **automated, image-based system** that can quickly and consistently detect cracks in road surfaces using standard photographs.

---

## 3. Motivation

### Why This Problem Matters

India maintains approximately **6.4 million kilometres** of roads, making it one of the largest road networks globally. Despite this, road quality remains a persistent challenge:

- **Pothole-related accidents** cause thousands of injuries and fatalities annually
- Municipal bodies often lack the resources for regular road inspections
- **Climate conditions** (monsoons, extreme heat) accelerate road deterioration
- Delayed repairs lead to exponentially higher maintenance costs

An automated crack detection system can:

1. **Reduce inspection time** from days to minutes
2. **Provide objective, consistent assessments** across different road segments
3. **Enable proactive maintenance** by identifying damage before it worsens
4. **Lower costs** by prioritising repairs based on severity

### Academic Relevance

This project demonstrates practical application of core computer vision concepts:
- Image filtering and enhancement
- Edge detection algorithms
- Morphological image processing
- Contour analysis and feature extraction

---

## 4. Methodology

The crack detection pipeline consists of the following sequential steps:

### Step 1: Image Loading
The system loads road surface images from a specified folder or file path. It supports common formats (JPEG, PNG).

### Step 2: Grayscale Conversion
Colour images are converted to single-channel grayscale images. This simplifies subsequent processing by reducing the data from three channels (BGR) to one intensity channel.

### Step 3: Contrast Enhancement (CLAHE)
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied to the grayscale image. Unlike global histogram equalization, CLAHE operates on small regions (tiles) of the image, preventing over-amplification of noise while improving local contrast. This makes cracks more distinguishable from the road surface.

### Step 4: Noise Removal (Gaussian Blur)
A **Gaussian Blur** filter with a 5×5 kernel is applied to smooth out high-frequency noise that could be mistakenly detected as edges. The Gaussian function naturally weights nearby pixels more than distant ones, preserving genuine edge structures.

### Step 5: Edge Detection (Canny)
The **Canny Edge Detector** is applied to find intensity gradients that correspond to crack boundaries. It uses two thresholds:
- **Low threshold (50)**: Minimum gradient to be considered a potential edge
- **High threshold (150)**: Gradient required for a "strong" edge

Pixels between the two thresholds are kept only if they connect to strong edges (hysteresis thresholding).

### Step 6: Adaptive Thresholding
In parallel with edge detection, **adaptive thresholding** is applied. This computes a local threshold for each pixel based on the mean intensity of its neighbourhood, producing a binary image where crack regions (which are darker than their surroundings) appear white.

### Step 7: Result Combination
The edge detection and thresholding results are combined using a **bitwise OR** operation. This merges information from both techniques, capturing cracks that either method alone might miss.

### Step 8: Morphological Operations
A sequence of morphological operations cleans the combined binary image:
1. **Closing** (dilation → erosion): Fills small gaps in crack lines
2. **Opening** (erosion → dilation): Removes isolated noise pixels
3. **Dilation**: Slightly thickens remaining crack lines for visibility

### Step 9: Contour Detection
**Contour detection** (`cv2.findContours`) identifies connected regions in the cleaned binary image. Contours with area smaller than a minimum threshold (100 pixels) are discarded as noise.

### Step 10: Visualisation and Classification
Detected crack contours are drawn in **red** on the original image. The total crack area is computed as a percentage of the image area and used to classify severity:

| Level | Crack Area | Interpretation |
|-------|-----------|----------------|
| LOW | < 1% | Minor surface cracks |
| MEDIUM | 1% – 5% | Moderate damage |
| HIGH | > 5% | Severe cracking |

---

## 5. Algorithms Used

### 5.1 Canny Edge Detection

The Canny edge detector (developed by John Canny in 1986) is a multi-stage algorithm:

1. **Gaussian smoothing** to reduce noise
2. **Gradient computation** using Sobel operators in both X and Y directions
3. **Non-maximum suppression** to thin edges to single-pixel width
4. **Hysteresis thresholding** using two thresholds to connect weak edges to strong ones

**Why Canny?** It provides clean, well-localised edges with good noise suppression — ideal for detecting thin crack lines against textured road surfaces.

### 5.2 Morphological Operations

Morphological operations use a **structuring element** (kernel) to process binary images:

| Operation | Effect | Use in This Project |
|-----------|--------|-------------------|
| **Dilation** | Expands white regions | Makes crack lines thicker |
| **Erosion** | Shrinks white regions | Removes small noise dots |
| **Opening** | Erosion → Dilation | Cleans noise while preserving shapes |
| **Closing** | Dilation → Erosion | Fills small gaps in crack lines |

### 5.3 Contour Detection

OpenCV's `findContours` function traces the boundaries of connected white regions in a binary image. Each contour is represented as a sequence of (x, y) coordinates. By filtering contours based on their area, we discard noise and retain only significant crack regions.

### 5.4 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE divides the image into small tiles and applies histogram equalization independently to each tile. A clip limit prevents over-amplification of noise in homogeneous regions. The result is a contrast-enhanced image where cracks stand out more clearly against the road background.

### 5.5 Adaptive Thresholding

Unlike global thresholding (which uses a single threshold for the entire image), adaptive thresholding computes a threshold for each pixel based on the **mean intensity of its local neighbourhood**. This handles images with uneven illumination — a common scenario in outdoor road photographs.

---

## 6. Results and Observations

### Key Observations

1. **CLAHE significantly improves detection accuracy** — cracks that are barely visible in the original image become clearly defined after contrast enhancement.

2. **Combining Canny edges with adaptive thresholding** produces better results than either technique alone. Canny captures fine crack edges, while thresholding captures broader dark regions.

3. **Morphological closing is critical** — without it, cracks often appear as disconnected fragments rather than continuous lines.

4. **The minimum contour area filter** (100 pixels) effectively eliminates most noise without losing genuine crack contours.

5. **Severity classification** provides a quick, quantitative assessment that can be used for prioritising road repairs.

### Limitations

- Performance depends on image quality and lighting conditions
- Very fine hairline cracks may not be detected at the default threshold settings
- The system cannot distinguish between crack types (longitudinal, transverse, alligator)
- Shadows, stains, and road markings may occasionally be misidentified as cracks

---

## 7. Challenges Faced

| # | Challenge | Solution |
|---|-----------|----------|
| 1 | Road textures creating false edges | Gaussian blur + minimum contour area filtering |
| 2 | Uneven lighting in outdoor photos | CLAHE + Adaptive thresholding |
| 3 | Disconnected crack fragments | Morphological closing to bridge gaps |
| 4 | Balancing noise removal vs. detail | Tuning Canny thresholds and blur kernel size |
| 5 | Different crack scales/severities | Area-based severity classification |

---

## 8. Future Improvements

1. **Deep Learning Integration** — Use convolutional neural networks (CNNs) like U-Net for semantic segmentation of cracks, improving accuracy on diverse road surfaces.

2. **Real-time Processing** — Implement video-based crack detection for vehicle-mounted cameras or drones.

3. **Crack Type Classification** — Distinguish between longitudinal, transverse, block, and alligator cracking patterns.

4. **GPS Integration** — Tag detected cracks with geographic coordinates for mapping damaged road sections.

5. **Web/Mobile Application** — Build a user-friendly interface where citizens can upload road images and receive instant damage assessments.

6. **Multi-Scale Detection** — Apply the pipeline at multiple resolutions to detect both fine hairline cracks and large-scale damage.

7. **Historical Tracking** — Compare images of the same road section over time to monitor deterioration rates.

---

## 9. Conclusion

This project demonstrates that **classical computer vision techniques** can effectively detect and highlight cracks in road surface images. By combining edge detection (Canny), adaptive thresholding, and morphological operations, the system identifies crack regions and classifies their severity without any deep learning or GPU requirements.

The modular code structure makes it easy to understand, modify, and extend. While deep learning approaches may offer higher accuracy on diverse datasets, the classical approach presented here is:

- **Lightweight** — runs on any CPU
- **Transparent** — each processing step is interpretable
- **Educational** — illustrates fundamental CV concepts
- **Practical** — provides useful results with minimal setup

This project serves as a strong foundation that can be enhanced with more advanced techniques in the future.

---

## References

1. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
2. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV*. O'Reilly Media.
3. OpenCV Documentation — https://docs.opencv.org/
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
5. Ministry of Road Transport and Highways, Government of India — https://morth.nic.in/

---

*Submitted as part of the Computer Vision course — Winter Semester 2025–2026*
