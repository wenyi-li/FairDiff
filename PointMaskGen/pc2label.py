import numpy as np
import cv2
import os
from tqdm import tqdm
import open3d as o3d

def read_point_cloud(file_path):
    """Read point cloud file"""
    return np.loadtxt(file_path)
def point_cloud_to_image(points, width, height):
    """Convert point cloud to image_blue and image_red"""
    image_blue = np.full((height, width, 3), 255, np.uint8)
    image_red = np.full((height, width, 3), 255, np.uint8)

    for point in points:
        x, y, z = point.astype(int)
        if 0 <= x < height and 0 <= y < width:
            if z >= 5:
                color = [255, 0, 0]
                image_blue[x, y] = color
            elif z <= -5:
                color = [0, 0, 255]
                image_red[x, y] = color
            else:
                color = [255, 255, 255]
                image_red[x, y] = color
                image_blue[x, y] = color

    return image_blue, image_red
def morphological_operations(mask, kernel_size, iterations):
    # Perform morphological operations to close the gaps between blue or red points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)
    mask_closed = cv2.erode(mask_dilated, kernel, iterations=iterations)
    return mask_closed
def fill_color(image, color):
    black = 255 - image
    # Color image to grayscale image
    gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    # Convert to binary image
    t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find all the Outlines and record every point of the outline
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    # Contour index
    max_idx = np.argsort(np.array(area))
    mask = image.copy()
    if color == 'blue':
        color_array = (255, 0, 0)
    elif color == 'red':
        color_array = (0, 0, 255)
    # Fill color by contour index
    for idx in max_idx:
        # 填充轮廓
        mask = cv2.drawContours(mask, contours, idx, color_array, cv2.FILLED)
    return mask
def blue_closed_operations(image, kernel_size, iterations):
    # Define the range for red color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_closed = morphological_operations(mask, kernel_size, iterations)

    # Find contours from the closed image
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a clean image
    image_with_contours = np.ones_like(image) * 255
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 2)
    # Fill color
    mask = fill_color(image_with_contours, 'blue')
    return mask
def red_closed_operations(image, kernel_size, iterations):
    # Define the range for red color in HSV
    # Red has two ranges because it wraps around the color wheel.
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only red colors
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_closed_red = morphological_operations(mask_red, kernel_size, iterations)

    # Find contours from the closed image
    contours_red, _ = cv2.findContours(mask_closed_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a clean white image
    image_with_red_contours = np.ones_like(image) * 255
    cv2.drawContours(image_with_red_contours, contours_red, -1, (0, 0, 255), 2)
    # Fill color
    mask = fill_color(image_with_red_contours, 'red')
    return mask
def combine(blue_img, red_img):
    # Convert images to HSV color space
    blue_hsv = cv2.cvtColor(blue_img, cv2.COLOR_BGR2HSV)
    red_hsv = cv2.cvtColor(red_img, cv2.COLOR_BGR2HSV)
    # Define HSV color range for blue color
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])
    # Define BGR color for red
    red_bgr = np.array([0, 0, 255], dtype=np.uint8)
    # Create a mask for blue and red colors
    blue_mask = cv2.inRange(blue_hsv, blue_lower, blue_upper)
    red_mask = cv2.inRange(red_img, red_bgr, red_bgr)
    # Find contours of the red region
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask to draw the filled red contours
    mask_red_filled = np.zeros_like(red_mask)
    # Draw the filled red contours on the mask
    for contour in contours_red:
        cv2.drawContours(mask_red_filled, [contour], -1, 255, thickness=cv2.FILLED)
    # Erode the red mask to ensure we replace inside the red region without touching the edges
    kernel = np.ones((3, 3), np.uint8)
    mask_red_eroded = cv2.erode(mask_red_filled, kernel, iterations=2)
    # Create mask for the blue region that should replace the red region
    mask_blue_replace_red = cv2.bitwise_and(blue_mask, mask_red_eroded)
    # Prepare the blue region for replacement
    blue_region = cv2.bitwise_and(blue_img, blue_img, mask=mask_blue_replace_red)
    # Prepare the red image by clearing the area where the blue will replace
    mask_inverse = cv2.bitwise_not(mask_blue_replace_red)
    red_cleared = cv2.bitwise_and(red_img, red_img, mask=mask_inverse)
    # Add the blue region to the cleared red image
    result_img = cv2.add(red_cleared, blue_region)
    return result_img
# Read point cloud file

if __name__ == "__main__":
    width, height = 798, 664
    dir = '/data22/Medical-Image/gender512/result/Full/noise2pc' 
    savedir = '/DATA_EDS2/datasets/MedicalImage/synthetic-label/Full'
    files = [f for f in os.listdir(dir) if f.endswith('.xyz')]
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(dir, file_name)
        points = read_point_cloud(file_path)
        # 1. Convert point cloud to image_blue and image_redW
        image_blue, image_red = point_cloud_to_image(points, width, height)
        # 2. Corrosion expands to fill the blue and red areas
        mask_blue = blue_closed_operations(image_blue, kernel_size=15, iterations=20)
        mask_red = red_closed_operations(image_red, kernel_size=15, iterations=20)
        # 3. Combine blue and red areas
        combined_img = combine(mask_blue, mask_red)
        basename = file_name.split('.')[0] + '.png'
        output_path = os.path.join(savedir, basename)
        cv2.imwrite(output_path, combined_img)