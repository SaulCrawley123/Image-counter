#!/usr/bin/env python3
"""
Coffee Bean Counter
Counts whole coffee beans using contour detection.
Filters out small fragments and noise.
"""

import cv2
import numpy as np


def count_coffee_beans(image_path: str) -> int:
    """
    Count coffee beans using contour detection.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        int: Count of whole coffee beans rounded to nearest 10
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    print(f"Original image size: {width}x{height}")
    
    # Resize to standard resolution for consistent results
    std_w, std_h = 1500, 1000
    img = cv2.resize(img, (std_w, std_h), interpolation=cv2.INTER_AREA if width > std_w else cv2.INTER_CUBIC)
    print(f"Resized to: {std_w}x{std_h}")
    
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_01_grayscale.jpg", gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("output_02_blurred.jpg", blurred)
    
    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("output_03_binary.jpg", binary)
    
    # Morphological closing to bridge crevices
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cv2.imwrite("output_04_closed.jpg", closed)
    
    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate areas - filter out tiny noise
    areas = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100:
            areas.append((i, area))
    
    if not areas:
        print("No contours found")
        return 0
    
    areas_only = np.array([a[1] for a in areas])
    
    # Use percentile-based statistics to be robust to outliers
    p25 = np.percentile(areas_only, 25)
    p75 = np.percentile(areas_only, 75)
    
    # Estimate typical single bean area from the middle 50% of contours
    typical_bean_areas = areas_only[(areas_only >= p25) & (areas_only <= p75)]
    if len(typical_bean_areas) > 0:
        estimated_bean_area = np.mean(typical_bean_areas)
    else:
        estimated_bean_area = np.median(areas_only)
    
    print(f"Contours found: {len(areas)}")
    print(f"Estimated single bean area: {estimated_bean_area:.0f} pixels")
    
    # Filter thresholds
    min_area = estimated_bean_area * 0.25
    max_area = estimated_bean_area * 2.9
    
    bean_count = 0
    single_beans = 0
    merged_regions = 0
    fragments = 0
    
    result_img = original.copy()
    marker_radius = 15
    
    for idx, area in areas:
        cnt = contours[idx]
        
        if area < min_area:
            fragments += 1
            continue
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        if area > max_area:
            # Merged beans - estimate count
            estimated = max(2, round(area / estimated_bean_area))
            bean_count += estimated
            merged_regions += 1
            # Mark merged beans using orange circle
            cv2.circle(result_img, (cx, cy), marker_radius + 2, (0, 165, 255), -1)
        else:
            bean_count += 1
            single_beans += 1
            # Mark single bean using green circle
            cv2.circle(result_img, (cx, cy), marker_radius, (0, 255, 0), -1)
    
    cv2.imwrite("output_05_detected_beans.jpg", result_img)
    
    print(f"Fragments filtered: {fragments}")
    print(f"Single beans: {single_beans}")
    print(f"Merged regions: {merged_regions}")
    print(f"Raw count: {bean_count}")
    
    rounded_count = round(bean_count / 10) * 10
    print(f"Rounded to nearest 10: {rounded_count}")
    
    return rounded_count


def main():
    input_0_image = "coffee_beans.jpg"
    count_coffee_beans(input_0_image)


if __name__ == "__main__":
    main()