import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import open3d as o3d
import argparse

def down_sample(original_pc, sample):
    down_sample_pc = o3d.geometry.PointCloud.farthest_point_down_sample(original_pc, sample)
    return down_sample_pc

def outline2pc(original_image):
    # Define color ranges in BGR (OpenCV uses BGR instead of RGB)
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    RED = np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR
    BLUE = np.array([255, 0, 0], dtype=np.uint8)  # Blue in BGR
    # Create masks for red, blue, and white areas
    red_mask = cv2.inRange(original_image, RED, RED)
    blue_mask = cv2.inRange(original_image, BLUE, BLUE)
    white_mask = cv2.inRange(original_image, WHITE, WHITE)
    # Create an edge mask for the red and blue areas using Canny edge detection
    red_edges = cv2.Canny(white_mask, 50, 200, L2gradient=True)
    blue_edges = cv2.Canny(blue_mask, 50, 200, L2gradient=True)
    # Find contours for the red edges and blue edges
    red_contours, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize an empty image to draw the contours on for red and blue
    red_contour_img = np.zeros_like(original_image)
    blue_contour_img = np.zeros_like(original_image)
    # Draw the red contours
    for contour in red_contours:
        cv2.drawContours(red_contour_img, [contour], -1, (0, 255, 0), 1)
    # Draw the blue contours
    for contour in blue_contours:
        cv2.drawContours(blue_contour_img, [contour], -1, (0, 255, 0), 1)
    # Convert img to single channel
    blue_contour_img = cv2.cvtColor(blue_contour_img, cv2.COLOR_BGR2GRAY)
    red_contour_img = cv2.cvtColor(red_contour_img, cv2.COLOR_BGR2GRAY)
    # Generate blue_contour_points and red_contour_points_3d
    blue_contour_points = np.argwhere(blue_contour_img)
    blue_contour_points_3d = np.hstack((blue_contour_points, np.full((len(blue_contour_points), 1), 5.0)))
    red_contour_points = np.argwhere(red_contour_img)
    red_contour_points_3d = np.hstack((red_contour_points, np.full((len(red_contour_points), 1), -5.0)))
    # Combine the red and blue contour points
    contour_points_3d = np.vstack((red_contour_points_3d, blue_contour_points_3d))
    return contour_points_3d
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--category_path', type=str, default='/datasets/MedicalImage/txt/gender',
                    help='txt stores image paths with different attributes')
args = parser.parse_args()
category_path = args.category_path
attributes_txt = glob.glob(category_path + '/*.txt')
attributes_name = [path.split('/')[-1].split('.')[0] for path in attributes_txt]
base_folder = os.getcwd()

for txtpath in tqdm(attributes_txt, desc="Processing different attributes"):
    pc_path = os.path.join(base_folder, category_path.split('/')[-1], txtpath.split('/')[-1].split('.')[0])
    allpc_path = os.path.join(base_folder, 'Full')
    with open(txtpath, 'r') as file:
        lines = [line.strip() for line in file]
    files = [line.replace("images", "labels") for line in lines]
    target_num_points = 512
    for file_path in tqdm(files, desc="Processing files", unit="file"):
        label = cv2.imread(file_path)
        basename = file_path.split('/')[-1].split('.')[0]
        point_cloud = outline2pc(label)
        np.savetxt('./node.xyz', point_cloud, fmt='%d %d %.2f')
        pcd = o3d.io.read_point_cloud('./node.xyz')   
        if len(pcd.points) >= target_num_points:
            pcd_new = o3d.geometry.PointCloud.farthest_point_down_sample(pcd, target_num_points)
            if len(pcd_new.points) == target_num_points:
                point_cloud_file_path = os.path.join(pc_path, basename + '.xyz')
                point_cloud_full = os.path.join(allpc_path, basename + '.xyz')
                o3d.io.write_point_cloud(point_cloud_file_path, pcd_new)
                o3d.io.write_point_cloud(point_cloud_full, pcd_new)
        os.remove('./node.xyz')