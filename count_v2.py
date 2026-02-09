#!/usr/bin/env python3
"""
coffee bean counter
counts whole coffee beans using contour detection.
filters out small fragments and noise.
"""

import cv2
import numpy as np


def count_coffee_beans(image_path: str) -> int:
    """
    count coffee beans using contour detection.
    
    args:
        image_path: path to the input image
        
    returns:
        int: count of whole coffee beans rounded to nearest 10
    """
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not load image: {image_path}")
    
    height, width = img.shape[:2]
    print(f"original image size: {width}x{height}")
    
    # resize to standard resolution for consistent results
    std_w, std_h = 1500, 1000
    img = cv2.resize(img, (std_w, std_h), interpolation=cv2.INTER_AREA if width > std_w else cv2.INTER_CUBIC)
    print(f"resized to: {std_w}x{std_h}")
    
    original = img.copy()
    
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_01_grayscale.jpg", gray)
    
    # apply clahe (contrast limited adaptive histogram equalization)
    # this improves contrast in dark areas without adding massive noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    cv2.imwrite("output_01b_equalized.jpg", equalized)
    
    # apply gaussian blur
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    cv2.imwrite("output_02_blurred.jpg", blurred)
    
    # use otsu thresholding (stable) on the contrast-enhanced image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("output_03_binary.jpg", binary)
    
    # morphological closing to bridge crevices
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # morphological opening to separate touching beans
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    cv2.imwrite("output_04_processed.jpg", opened)
    
    # find external contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # calculate areas - filter out tiny noise
    areas = []
    
    # sanity check: calculate max possible area for a valid bean cluster
    # if a contour is > 5% of the total image, it's likely a background artifact
    total_image_area = std_w * std_h
    max_sane_area = total_image_area * 0.05
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if 100 < area < max_sane_area:
            areas.append((i, area))
    
    if not areas:
        print("no contours found")
        return 0
    
    areas_only = np.array([a[1] for a in areas])
    
    # use percentile-based statistics to be robust to outliers
    p25 = np.percentile(areas_only, 25)
    p75 = np.percentile(areas_only, 75)
    
    # estimate typical single bean area from the middle 50% of contours
    typical_bean_areas = areas_only[(areas_only >= p25) & (areas_only <= p75)]
    if len(typical_bean_areas) > 0:
        estimated_bean_area = np.mean(typical_bean_areas)
    else:
        estimated_bean_area = np.median(areas_only)
    
    print(f"contours found: {len(areas)}")
    print(f"estimated single bean area: {estimated_bean_area:.0f} pixels")
    
    # filter thresholds
    min_area = estimated_bean_area * 0.20
    max_area = estimated_bean_area * 2.5
    
    bean_count = 0
    single_beans = 0
    merged_regions = 0
    fragments = 0
    
    result_img = original.copy()
    overlay = original.copy()
    
    for idx, area in areas:
        cnt = contours[idx]
        
        if area < min_area:
            fragments += 1
            continue
        
        # get centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # calculate radius based on the contour area
        dynamic_radius = int(np.sqrt(area / np.pi))
        
        if area > max_area:
            # merged beans - estimate count
            estimated = max(2, round(area / estimated_bean_area))
            bean_count += estimated
            merged_regions += 1
            
            # draw transparent orange circle on overlay
            cv2.circle(overlay, (cx, cy), dynamic_radius, (0, 165, 255), -1)
            # draw solid orange outline on result_img for visibility
            cv2.circle(result_img, (cx, cy), dynamic_radius, (0, 165, 255), 2)
        else:
            bean_count += 1
            single_beans += 1
            # mark single bean using green circle
            cv2.circle(result_img, (cx, cy), dynamic_radius, (0, 255, 0), -1)

    # blend the overlay with the result image (0.4 alpha for transparency)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)
    
    cv2.imwrite("output_05_detected_beans.jpg", result_img)
    
    print(f"fragments filtered: {fragments}")
    print(f"single beans: {single_beans}")
    print(f"merged regions: {merged_regions}")
    print(f"raw count: {bean_count}")
    
    rounded_count = round(bean_count / 10) * 10
    print(f"rounded to nearest 10: {rounded_count}")
    
    return rounded_count


def main():
    input_0_image = "coffee_beans.jpg"
    count_coffee_beans(input_0_image)


if __name__ == "__main__":
    main()