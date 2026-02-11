# Coffee Bean Counter – Program Overview

This program counts coffee beans in an image using classical computer vision techniques. It is designed to be resolution-independent and robust to lighting variation, surface texture, and beans that are touching or partially overlapping.

---

## How the Program Works

### 1. Image Loading and Resolution Standardisation

The program begins by loading the input image. To ensure consistent behaviour across different image sizes, it resizes the image to a fixed width of **1500 pixels** while preserving the original aspect ratio.

This step is important because many image processing operations (such as blur kernels and morphological filters) depend on pixel size. Standardising the width ensures that these operations correspond to approximately the same real-world scale regardless of the original resolution.

---

### 2. Preprocessing

The resized image undergoes several preprocessing steps:

* **Grayscale conversion** to simplify analysis
* **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance local contrast and reduce the effect of uneven lighting
* **Gaussian blur** to reduce small-scale noise and surface grain

These steps improve separation between beans and background before segmentation.

---

### 3. Binary Segmentation

The program then isolates beans from the background using:

* **Adaptive Gaussian thresholding** (with inversion) to create a binary image
* **Morphological opening** to remove small noise artefacts
* **Morphological closing** to fill gaps and solidify bean shapes

This results in a cleaner binary representation of the bean regions.

---

### 4. Distance Transform and Peak Detection

To handle beans that are touching, the program uses a **distance transform**, which computes the distance from each foreground pixel to the nearest background pixel.

* The distance map is blurred to merge internal cracks within beans
* Local maxima are detected in the blurred distance map
* Only peaks above 20% of the maximum intensity are retained
* A minimum separation distance ensures nearby peaks are treated as separate beans

These peaks represent likely bean centres.

---

### 5. Watershed Segmentation

The detected peaks are used as markers for **watershed segmentation**, which separates connected regions into individual objects.

This allows the program to distinguish between beans that are touching or slightly overlapping.

---

### 6. Filtering, Counting, and Visualisation

After segmentation:

* Very small regions are discarded using an area threshold to remove noise
* Each valid bean is outlined with a contour
* The centre of each bean is marked
* The total number of detected beans is counted

The final count is rounded to the nearest ten and printed to the console, along with the total processing time.

Intermediate images from each major stage are saved to disk for debugging and inspection.

---

## Usage

To use the program:

1. Place your image in the same directory as the script.
2. Rename "coffee_beans.jpg" in the script to the filename of your image.
3. Run the program.

The script will output intermediate processing images and print the final object count to the console.

---

## Extending to Other Objects

Although designed for coffee beans, the algorithm can be adapted to count other similar objects. However:

* It has not been extensively tested on other image types
* Results may vary depending on lighting, object shape, object density, and background complexity
* Parameter tuning (thresholds, kernel sizes, area filters) may be required for different use cases

---

## License

This program is open source and may be freely used, modified, and extended.
