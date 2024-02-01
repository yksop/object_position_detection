import cv2
import numpy as np

def find_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    return largest_contour

def draw_largest_contour(image, largest_contour):
    result = image.copy()
    cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
    return result

def get_contour_points(contour):
    # Create an image to draw the filled contour
    filled_contour = np.zeros_like(image)
    cv2.drawContours(filled_contour, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Find the coordinates of the filled region
    points = np.column_stack(np.where(filled_contour[:,:,0] > 0))

    return points

image = cv2.imread('Pictures/test_real-grayscale.jpeg')

largest_contour = find_objects(image)

# Get the points of the largest contour
contour_points = get_contour_points(largest_contour)

# Display the points
for point in contour_points:
    print(f"Point: {point}")

# Draw the largest contour on the image
result_image = draw_largest_contour(image, largest_contour)

cv2.imshow('Original Image', image)
cv2.imshow('Result Image with Largest Contour', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
