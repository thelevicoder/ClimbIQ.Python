import cv2
import numpy as np


# Function to find pixels matching the clicked color and highlight them
def find_and_highlight_color(b, g, r, tolerance=50):
    # Create a mask for all pixels within the tolerance range
    colors = []
    colors.append(b)
    colors.append(g)
    colors.append(r)
    big_color = max(b, g, r)
    count = -1
    for color in colors:
        count += 1
        if big_color == color:
            print(color, " -  ", count)

    # bspecl = b - tolerance
    # bspech = b + tolerance
    # gspecl = g - tolerance
    # gspech = g + tolerance
    # rspecl = r - tolerance
    # rspech = r + tolerance

    # if count == 0:
    #     bspecl = b - 30
    #     bspech = 255

    # if count == 1:
    #     gspecl = g - 30
    #     gspech = 255

    # if count == 2:
    #     rspecl = r - 30
    #     rspech = 255
 
    lower_bound = np.array([b - tolerance , g - tolerance, r - tolerance])
    upper_bound = np.array([b + tolerance , g + tolerance, r + tolerance])
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # Find contours of the masked regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around the detected regions
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_img, center, 50, (0, 255, 0), 2)  # Green circle, thickness=2
    
    print(f"Total holds detected: {len(contours)}")

# Mouse callback function
def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img[y, x]
        print(f"Color at ({x}, {y}): B={b}, G={g}, R={r}")
        find_and_highlight_color(b, g, r, tolerance=5)
        cv2.imshow('Highlighted Image', output_img)

# Load image in color mode
img = cv2.imread('wall1.jpg', 1)  # BGR mode


if img is None:
    print("Error: Image not found!")
    exit()

# brightness = 10
# contrast = 1
# img2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness) # this is brightness


# img2 = cv2.Laplacian(img, cv2.CV_64F) # this si laplacian

# img2 = cv2.GaussianBlur(img, (7, 7), 0) # this is blur

img2 = 255 - img # this is inverse color



# Create a copy of the image to draw on
output_img = img.copy()

# Display the image and set mouse callback
cv2.imshow('Image', img)
# cv2.imshow('Modified', img2)
cv2.setMouseCallback('Image', get_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
