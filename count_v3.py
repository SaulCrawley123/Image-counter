#!/usr/bin/env python3
"""
coffee bean counter final
counts coffee beans using geometric peak detection.
standardizes resolution to width=1500px before processing.
"""

import cv2
import numpy as np
import time

def count_coffee_beans(image_path: str) -> int:
    """
    counts coffee beans using geometric peak detection.
    standardizes resolution to width=1500px before processing.
    """
    start_time = time.time()
    
    # 1. load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not load image: {image_path}")
    
    original_h, original_w = img.shape[:2]
    print(f"original size: {original_w}x{original_h}")
    
    # 2. smart resizing (resolution independence)
    # targets a width of 1500px to ensure kernel sizes (21px, etc)
    # represent the same physical size on the bean regardless of input resolution.
    target_width = 1500
    scale = target_width / original_w
    target_height = int(original_h * scale)
    
    # choose the best interpolation method based on scaling
    if scale < 1:
        # shrinking: inter_area is best for avoiding moire patterns
        interpolation = cv2.INTER_AREA
    else:
        # enlarging: inter_cubic is best for sharpness
        interpolation = cv2.INTER_CUBIC
        
    img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
    print(f"processing size: {target_width}x{target_height}")
    
    # copy for final drawing
    result_img = img.copy()
    
    # 3. preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # clahe for local contrast (handles shadows between beans)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # moderate blur to remove grain
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 4. adaptive thresholding
    # block size 25, C=3 provides good separation
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 3
    )
    cv2.imwrite("output_01_binary.jpg", binary)
    
    # 5. morphology
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # remove noise
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # close gaps to solidify beans (crucial for peak detection)
    sure_bg = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    cv2.imwrite("output_02_morph.jpg", sure_bg)
    
    # 6. distance transform
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    
    # 7. blur the distance map
    # (21, 21) kernel merges internal bean cracks into single peaks
    # while keeping touching neighbors separate.
    dist_blurred = cv2.GaussianBlur(dist_transform, (21, 21), 0)
    
    # normalize for visualization
    dist_viz = cv2.normalize(dist_blurred, None, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imwrite("output_03_dist_blurred.jpg", (dist_viz * 255).astype(np.uint8))
    
    # 8. local maxima detection
    # min_distance = 25px allows beans to be close/touching
    min_distance = 25
    kernel_size = min_distance * 2 + 1
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    dist_dilated = cv2.dilate(dist_blurred, kernel_dilate)
    
    # find peaks: local max AND > 20% intensity
    peak_threshold = 0.2 * dist_blurred.max()
    peaks = (dist_blurred == dist_dilated) & (dist_blurred > peak_threshold)
    
    # convert peaks to markers
    peak_markers = np.zeros_like(gray, dtype=np.uint8)
    peak_markers[peaks] = 255
    
    # dilate markers slightly for visibility
    peak_markers = cv2.dilate(peak_markers, kernel_small, iterations=1)
    cv2.imwrite("output_04_peaks.jpg", peak_markers)
    
    # 9. watershed
    _, markers = cv2.connectedComponents(peak_markers)
    markers = markers + 1
    
    unknown = cv2.subtract(sure_bg, peak_markers)
    markers[unknown == 255] = 0
    markers[sure_bg == 0] = 1 # enforce background
    
    cv2.watershed(img, markers)
    
    # 10. count and visualize
    unique_markers = np.unique(markers)
    valid_beans = 0
    
    for mk in unique_markers:
        if mk <= 1: continue # skip boundary and background
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == mk] = 255
        
        # area filter: 180px removes tiny noise fragments
        area = cv2.countNonZero(mask)
        if area > 180:
            valid_beans += 1
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                # draw green contour
                cv2.drawContours(result_img, cnts, -1, (0, 255, 0), 2)
                
                # draw red center dot
                M = cv2.moments(cnts[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imwrite("output_05_detected_beans.jpg", result_img)
    
    # round to nearest 10
    rounded_count = round(valid_beans / 10) * 10
    
    end_time = time.time()
    print(f"processing time: {end_time - start_time:.2f}s")
    print(f"raw count: {valid_beans}")
    print(f"final count (rounded): {rounded_count}")
    
    return rounded_count

def main():
    # input variable using the system-generated filename
    input_0_image = "coffee_beans.jpg"
    # call task function and handle return value
    count = count_coffee_beans(input_0_image)
    
    print("-" * 30)
    print(f"Task Complete. Final Bean Count: {count}")
    print("-" * 30)

if __name__ == "__main__":
    main()