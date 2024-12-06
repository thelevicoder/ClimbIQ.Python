import cv2
import numpy as np

# Global variables
reference_color = None
image = None
lab_image = None
mask = None

# Tolerance for LAB color similarity
lab_tolerance = 25  # Adjust for stricter or looser matching


def click_event(event, x, y, flags, param):
    global reference_color, image, lab_image, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        # Extract a 10x10 region around the clicked point
        min_x, max_x = max(0, x - 5), min(image.shape[1], x + 5)
        min_y, max_y = max(0, y - 5), min(image.shape[0], y + 5)

        region = lab_image[min_y:max_y, min_x:max_x]
        # Compute the average LAB color of the region
        reference_color = np.mean(region.reshape(-1, 3), axis=0)
        print(f"Reference LAB color: {reference_color}")

        # Find matching pixels
        find_matching_pixels()


def find_matching_pixels():
    global image, lab_image, mask, reference_color

    if reference_color is None:
        return

    # Compute the distance of each pixel to the reference color
    diff = np.linalg.norm(lab_image - reference_color, axis=2)

    # Create a binary mask of matching pixels
    mask = (diff < lab_tolerance).astype(np.uint8) * 255

    # Group and contour matching regions
    contour_matching_regions()


def contour_matching_regions():
    global image, mask

    # Find connected components in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Copy the original image for displaying results
    output_image = image.copy()

    # Draw contours for each connected component
    for label in range(1, num_labels):  # Skip the background label (0)
        # Create a mask for the current label
        label_mask = (labels == label).astype(np.uint8) * 255

        # Find contours of the labeled region
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the output image
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Matched Holds", output_image)
    cv2.imshow("Mask", mask)


def main():
    global image, lab_image

    # Load the image
    image_path = "wall2.jpg"  # Path to your uploaded image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Show the original image and set mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    # Wait for user to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
