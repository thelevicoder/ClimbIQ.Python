import cv2
import numpy as np

# Global variables
reference_color_lab = None
reference_color_hsv = None
image = None
lab_image = None
hsv_image = None
mask = None

# Line boundaries
left_line = None
right_line = None
drawing_line = False

# Tolerance for LAB and HSV color matching
lab_tolerance = 20  # Initial tolerance for LAB
hsv_tolerance = (15, 50, 50)  # (Hue, Saturation, Value) tolerance1 for HSV

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
    return normalized_lab


def filter_outliers(region):
    region_reshaped = region.reshape(-1, 3)
    mean_color = np.mean(region_reshaped, axis=0)
    diff = np.linalg.norm(region_reshaped - mean_color, axis=1)
    filtered_pixels = region_reshaped[diff < np.std(diff) * 1.5]  # Include only similar pixels
    if filtered_pixels.size > 0:
        return np.mean(filtered_pixels, axis=0)
    else:
        return mean_color  # Fallback to mean if filtering removes all


def click_event(event, x, y, flags, param):
    global reference_color_lab, reference_color_hsv, image, lab_image, hsv_image, mask
    global left_line, right_line, drawing_line

    if left_line is None or right_line is None:
        # Define left and right lines
        if event == cv2.EVENT_LBUTTONDOWN:
            if left_line is None:
                left_line = x
                print(f"Left line set at {left_line}")
            elif right_line is None:
                right_line = x
                print(f"Right line set at {right_line}")
                # Display the lines on the image
                draw_lines()
    else:
        # Proceed to color selection and contour detection
        if event == cv2.EVENT_LBUTTONDOWN:
            # 10-pixel average
            min_x, max_x = max(0, x - 5), min(image.shape[1], x + 5)
            min_y, max_y = max(0, y - 5), min(image.shape[0], y + 5)

            # Compute the average LAB and HSV colors of the filtered region
            region_lab = lab_image[min_y:max_y, min_x:max_x]
            region_hsv = hsv_image[min_y:max_y, min_x:max_x]

            filtered_lab = filter_outliers(region_lab)
            filtered_hsv = filter_outliers(region_hsv)

            reference_color_lab = filtered_lab
            reference_color_hsv = filtered_hsv

            print(f"Reference LAB color: {reference_color_lab}")
            print(f"Reference HSV color: {reference_color_hsv}")

            # Find matching pixels
            find_matching_pixels()


def draw_lines():
    """Draw the left and right lines on the image."""
    global image, left_line, right_line

    if left_line is not None:
        cv2.line(image, (left_line, 0), (left_line, image.shape[0]), (0, 0, 255), 2)
    if right_line is not None:
        cv2.line(image, (right_line, 0), (right_line, image.shape[0]), (0, 255, 0), 2)
    cv2.imshow("Image", image)


def apply_line_filter(mask):
    """Remove pixels outside the region defined by the left and right lines."""
    global left_line, right_line

    if left_line is None or right_line is None:
        return mask

    # Ensure left_line is smaller than right_line
    left = min(left_line, right_line)
    right = max(left_line, right_line)

    # Create a mask for the region between the lines
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    line_mask[:, left:right] = 255

    # Combine the masks
    filtered_mask = cv2.bitwise_and(mask, line_mask)
    return filtered_mask


def find_matching_pixels():
    global image, lab_image, hsv_image, mask, reference_color_lab, reference_color_hsv

    if reference_color_lab is None or reference_color_hsv is None:
        return

    # LAB color matching
    diff_lab = np.linalg.norm(lab_image - reference_color_lab, axis=2)
    mask_lab = (diff_lab < lab_tolerance).astype(np.uint8) * 255

    # HSV color matching (handle red hue wrap-around)
    hue_diff = np.abs(hsv_image[:, :, 0] - reference_color_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # Handle circular hue space
    diff_s = np.abs(hsv_image[:, :, 1] - reference_color_hsv[1])
    diff_v = np.abs(hsv_image[:, :, 2] - reference_color_hsv[2])
    mask_hsv = ((hue_diff < hsv_tolerance[0]) & 
                (diff_s < hsv_tolerance[1]) & 
                (diff_v < hsv_tolerance[2])).astype(np.uint8) * 255

    # Combine LAB and HSV masks
    mask = cv2.bitwise_and(mask_lab, mask_hsv)

    # Apply the line filter
    mask = apply_line_filter(mask)

    # Morphological operations to refine the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)  # Fill small gaps
    mask = cv2.dilate(mask, morph_kernel, iterations=3)  # Expand regions

    # Group and contour matching regions
    contour_matching_regions()


def contour_matching_regions():
    global image, mask

    # Find connected components in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the original image for displaying results
    output_image = image.copy()

    # Draw contours for each valid region
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            # Draw the contour on the output image
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Matched Holds", output_image)
    cv2.imshow("Mask", mask)


def main():
    global image, lab_image, hsv_image

    # Load the image
    image_path = "wall2.jpg"  # Path to the uploaded image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Normalize the LAB color space
    lab_image = normalize_lab(image)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Show the original image and set mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    # Wait for user to close the windows
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 32:  # Press space to close
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
