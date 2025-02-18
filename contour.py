import cv2
import numpy as np
import os

# Global variables
reference_color_lab = None
reference_color_hsv = None
image = None
lab_image = None
hsv_image = None
gray_image = None
mask = None

# Contour Lists
contours = []
deselected_contours = []

# Line boundaries
left_line = None
right_line = None
drawing_line = False

# Region Boundaries

# Tolerance for LAB and HSV color matching
lab_tolerance = 20  # Initial tolerance for LAB
hsv_tolerance = (15, 50, 50)  # (Hue, Saturation, Value) tolerance for HSV

# Size for holds
min_contour_area = 200  # Minimum area for valid hold
max_contour_area = 5000  # Maximum area to exclude large regions like walls

# Morphological kernel
morph_kernel = np.ones((7, 7), np.uint8)  # Kernel for closing and dilation

def normalize_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized_lab = cv2.merge((l, a, b))
    print("LAB image normalized.")
    return normalized_lab

def filter_outliers(region):
    region_reshaped = region.reshape(-1, 3)
    mean_color = np.mean(region_reshaped, axis=0)
    diff = np.linalg.norm(region_reshaped - mean_color, axis=1)
    filtered_pixels = region_reshaped[diff < np.std(diff) * 2]
    if filtered_pixels.size > 0:
        print("Outliers filtered from region.")
        return np.mean(filtered_pixels, axis=0)
    else:
        return mean_color

def click_event(event, x, y, flags, param):
    global reference_color_lab, reference_color_hsv, image, lab_image, hsv_image, mask
    global left_line, right_line, drawing_line

    if left_line is None or right_line is None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if left_line is None:
                left_line = x
                print(f"Left line set at {left_line}")
            elif right_line is None:
                right_line = x
                print(f"Right line set at {right_line}")
                draw_lines()
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            min_x, max_x = max(0, x - 5), min(image.shape[1], x + 5)
            min_y, max_y = max(0, y - 5), min(image.shape[0], y + 5)
            region_lab = lab_image[min_y:max_y, min_x:max_x]
            region_hsv = hsv_image[min_y:max_y, min_x:max_x]
            filtered_lab = filter_outliers(region_lab)
            filtered_hsv = filter_outliers(region_hsv)
            reference_color_lab = filtered_lab
            reference_color_hsv = filtered_hsv
            print(f"Reference LAB color: {reference_color_lab}")
            print(f"Reference HSV color: {reference_color_hsv}")
            find_matching_pixels()

def draw_lines():
    global image, left_line, right_line
    if left_line is not None:
        cv2.line(image, (left_line, 0), (left_line, image.shape[0]), (0, 0, 255), 2)
    if right_line is not None:
        cv2.line(image, (right_line, 0), (right_line, image.shape[0]), (0, 255, 0), 2)
    cv2.imshow("Image", image)
    print("Lines drawn on image.")

def apply_line_filter(mask):
    global left_line, right_line
    if left_line is None or right_line is None:
        return mask
    left = min(left_line, right_line)
    right = max(left_line, right_line)
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    line_mask[:, left:right] = 255
    filtered_mask = cv2.bitwise_and(mask, line_mask)
    print("Line filter applied to mask.")
    return filtered_mask

def find_matching_pixels():
    global image, lab_image, hsv_image, gray_image, mask, reference_color_lab, reference_color_hsv
    if reference_color_lab is None or reference_color_hsv is None:
        return
    diff_lab = np.linalg.norm(lab_image - reference_color_lab, axis=2)
    mask_lab = (diff_lab < lab_tolerance).astype(np.uint8) * 255
    hue_diff = np.abs(hsv_image[:, :, 0] - reference_color_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    diff_s = np.abs(hsv_image[:, :, 1] - reference_color_hsv[1])
    diff_v = np.abs(hsv_image[:, :, 2] - reference_color_hsv[2])
    mask_hsv = ((hue_diff < hsv_tolerance[0]) & 
                (diff_s < hsv_tolerance[1]) & 
                (diff_v < hsv_tolerance[2])).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask_lab, mask_hsv)
    mask = apply_line_filter(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    mask = cv2.dilate(mask, morph_kernel, iterations=3)
    print("Matching pixels found.")
    contour_matching_regions()

def contour_matching_regions():
    global image, mask, contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    print(f"Found {len(contours)} contours.")
    output_image = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            adjusted_radius = max(10, min(radius, 100))
            cv2.circle(output_image, center, adjusted_radius, (255, 0, 0), 2)
    cv2.imshow("Select Holds", output_image)
    cv2.setMouseCallback("Select Holds", contour_selection_event)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 32:
            cv2.destroyWindow("Select Holds")
            break
    contours = [contour for i, contour in enumerate(contours) if i not in deselected_contours]
    final_output_image = image.copy()
    for contour in contours:
        cv2.drawContours(final_output_image, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("Final Matched Holds", final_output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Contours filtered and drawn.")
    FindContours()

def preprocess_image_for_bright_areas(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    print("Image preprocessed for bright areas.")
    return blurred_image

def contour_selection_event(event, x, y, flags, param):
    global contours, deselected_contours, reference_color_lab, reference_color_hsv, lab_image, hsv_image, gray_image
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, contour in enumerate(contours):
            if i == 0:
                continue
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                deselected_contours.append(i)
                redraw_contours()
                return
    elif event == cv2.EVENT_RBUTTONDOWN:
        region_size = 20
        region_lab = lab_image[max(0, y-region_size):min(lab_image.shape[0], y+region_size),
                               max(0, x-region_size):min(lab_image.shape[1], x+region_size)]
        region_hsv = hsv_image[max(0, y-region_size):min(hsv_image.shape[0], y+region_size),
                               max(0, x-region_size):min(hsv_image.shape[1], x+region_size)]
        region_gray = gray_image[max(0, y-region_size):min(gray_image.shape[0], y+region_size),
                                 max(0, x-region_size):min(gray_image.shape[1], x+region_size)]
        filtered_lab = filter_outliers(region_lab)
        filtered_hsv = filter_outliers(region_hsv)
        clicked_lab = filtered_lab
        clicked_hsv = filtered_hsv
        print("Clicked Lab = ", clicked_lab)
        print("Clicked HSV = ", clicked_hsv)
        region_brightness = np.mean(region_lab[:, :, 0])
        lab_tolerance_increase = 60 if region_brightness < 120 else 40
        hsv_tolerance_increase = (80, 150, 150) if region_brightness < 120 else (60, 100, 100)
        lab_diff = np.linalg.norm(reference_color_lab - clicked_lab)
        hsv_diff = np.linalg.norm(reference_color_hsv - clicked_hsv)
        if lab_diff < (lab_tolerance + lab_tolerance_increase) and \
           hsv_diff < np.mean(hsv_tolerance_increase):
            new_contour = np.array([[[x-region_size, y-region_size]], 
                                    [[x+region_size, y-region_size]], 
                                    [[x+region_size, y+region_size]], 
                                    [[x-region_size, y+region_size]]], dtype=np.int32)
            contour_area = cv2.contourArea(new_contour)
            if min_contour_area <= contour_area <= max_contour_area:
                if np.mean(region_gray) < 200:
                    contours.append(new_contour)
                    redraw_contours()
                else:
                    print("Region too bright, likely a wall.")
            else:
                print("Contour area out of valid range, likely a wall.")
        else:
            print("No matching hold found at this location.")

def redraw_contours():
    global image, contours, deselected_contours
    output_image = image.copy()
    for i, contour in enumerate(contours):
        if i not in deselected_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output_image, center, radius, (255, 0, 0), 2)
    cv2.imshow("Select Holds", output_image)
    print("Contours redrawn.")

def initialize_images(input_image):
    global image, lab_image, hsv_image, gray_image
    image = input_image
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Images initialized.")

def FindContours():
    global image, original_image, output_image, contours, min_contour_area, max_contour_area
    image_path = 'wall2.jpg'
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        exit()
    output_image = original_image.copy()
    if not contours:
        print("Error: No contours found.")
        return
    min_contour_area = 0
    max_contour_area = 5000
    if not os.path.exists('contour_images'):
        os.makedirs('contour_images')
    for i, contour in enumerate(contours):
        print(f"Processing Contour {i}")
        return_contours(contours)
    print("Contours processed and saved.")

def return_contours(contours):
    print("Returning contours.")
    return contours

def main():
    global image, lab_image, hsv_image, gray_image
    image_path = "wall2.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    initialize_images(image)
    lab_image = normalize_lab(image)
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    print("Image loaded and initialized.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 32:
            cv2.destroyAllWindows()
            break
    print("Contours selection completed.")

if __name__ == "__main__":
    main()