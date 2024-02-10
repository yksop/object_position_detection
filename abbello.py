#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose

import numpy as np
import torch
import tf
import os

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from collections import Counter

import math

import warnings

warnings.filterwarnings("ignore")

bricks_informations = []

table_high = 0.88
brick_error = 0.001
value_error = 0.001

minimum_distance = 0.005
lenght_error = 0.005

PI = 3.14

brick_value = {
    'X1-Y1-Z2': [0.03, 0.03, 0.02],
    'X1-Y2-Z1': [0.03, 0.06, 0.01],
    'X1-Y2-Z2-CHAMFER': [0.03, 0.06, 0.02],
    'X1-Y2-Z2-TWINFILLET': [0.03, 0.06, 0.02],
    'X1-Y2-Z2': [0.03, 0.06, 0.02],
    'X1-Y3-Z2-FILLET': [0.01, 0.09, 0.02],
    'X1-Y3-Z2': [0.03, 0.09, 0.02],
    'X1-Y4-Z1': [0.03, 0.12, 0.01],
    'X1-Y4-Z2': [0.03, 0.12, 0.02],
}

color_ranges = {
    'red': [(0, 50, 50), (10, 255, 255)],  # Hue range: 0-10
    'green': [(36, 50, 50), (70, 255, 255)],  # Hue range: 36-70
    'blue': [(90, 50, 50), (130, 255, 255)],  # Hue range: 90-130
    'yellow': [(20, 50, 50), (35, 255, 255)],  # Hue range: 20-35
    'fuchsia': [(145, 50, 50), (175, 255, 255)],  # Hue range: 145-175
    'orange': [(11, 50, 50), (25, 255, 255)]  # Hue range: 11-25
}


def norm(p1,p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_brick_value(name):
    value = np.array(brick_value[name])
    return value[0], value[1], value[2]


def correct_model_error_value(color):
    if color == 'red': return 0.03
    if color == 'green': return 0.06
    if color == 'blue': return 0.09
    if color == 'yellow': return 0.12


def detect_color_block(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = image_hsv.reshape(image_hsv.shape[0] * image_hsv.shape[1], 3)
    clf = KMeans(n_clusters=5)
    clf.fit_predict(image_hsv)
    center_colors = clf.cluster_centers_

    for color, (lower, upper) in color_ranges.items():
        if lower[0] <= center_colors[0][0] <= upper[0]:
            return color


def find_object(image):
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


def discriminate_brick_points(image, offset):
    largest_contour = find_object(image)
    contoured_image = draw_largest_contour(image, largest_contour)
    filled_contour = np.zeros_like(contoured_image)
    cv2.drawContours(filled_contour, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    points = np.column_stack(np.where(filled_contour[:, :, 0] > 0))
    points_reversed = points[:, ::-1]

    result = []
    for point in points_reversed:
        x, y = point
        result.append([x + int(offset[1]), y + int(offset[2])])

    return result


def object_detection(image_msg: Image, point_cloud2_msg: PointCloud2, input) -> None:
    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")

    cv2.imshow("Original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    brick_list = []
    for bbox in input:
        name = bbox['ID']
        width_of_bb = bbox["w"]
        heigth_of_bb = bbox["h"]
        s_side, l_side, height = get_brick_value(name)
        x1 = bbox["xc"] - width_of_bb / 2
        y1 = bbox["yc"] - heigth_of_bb / 2
        x2 = x1 + width_of_bb
        y2 = y1 + heigth_of_bb

        brick_list.append((name, int(x1), int(y1), int(x2), int(y2), s_side, l_side, height))

    # iteration for each brick
    for tuple in brick_list:

        # cropping image box
        sliceBox = slice(tuple[2] - 10, tuple[4] + 10), slice(tuple[1] - 10, tuple[3] + 10)
        image = img[sliceBox]

        cv2.imshow("Cropped image:", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # correct the long side of brick, using color detection
        color = detect_color_block(image)
        brick_long_side = correct_model_error_value(color)

        # filtering background
        points_2D = []
        points_2D = discriminate_brick_points(image, tuple)

        print("Points 2D:", points_2D)

        points_2D_np = np.array(points_2D)
        plt.scatter(points_2D_np[:, 0], points_2D_np[:, 1], c='r', marker='o')
        plt.title('2D Points of the Brick')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # from a list of tuples to a list of lists
        zed_points = []
        for point in list(points_2D):
            zed_points.append([int(coordinate) for coordinate in point])

        # transforming 2D points in 3D points (of the boundary box)
        points_3d = point_cloud2.read_points(point_cloud2_msg, field_names=['x', 'y', 'z'], skip_nans=False,
                                             uvs=zed_points)

        # from zed frame to world frame
        rotational_matrix = np.array([[0., -0.49948, 0.86632],
                                      [-1., 0., 0.],
                                      [-0., -0.86632, -0.49948]])

        # zed position from world frame
        pos_zed = np.array([-0.4, 0.59, 1.4])

        # selection of informations from point cloud
        zed_points = []
        for point in points_3d:
            zed_points.append(point[:3])

        # trasforming each point in world frame
        data_world = []
        for point in zed_points:
            point = rotational_matrix.dot(point) + pos_zed
            point = np.array(point)
            data_world.append(point)

        min_y, min_x, max_y = get_three_points(data_world)

        print("Three points:", min_y, min_x, max_y)

        alpha = brick_pose_detection(min_y, min_x, max_y, tuple[5])
        print(alpha)
        x, y, z = find_centroid(data_world)
        ros_preprocessing_data(tuple[0], x, y, z, alpha)


def ros_preprocessing_data(name, x, y, z, alpha):
    global bricks_informations
    bricks_informations.append([name, x, y, z, alpha])


def get_three_points(points):
    my_points = np.array(points)

    min_x_index = np.argmin(my_points[:, 0])

    y_min = 100000
    y_max = 0

    for point in points:
        actual_y = point[1]
        if actual_y < y_min:
            y_min = actual_y
        if actual_y > y_max:
            y_max = actual_y

    # getting the set of points around min y and around max y
    y_min_points = my_points[abs(my_points[:, 1] - y_min) <= value_error]
    y_max_points = my_points[abs(my_points[:, 1] - y_max) <= value_error]

    # get the index of of the points with lowest x value
    y_min_index = np.argmin(y_min_points[:, 0])
    y_max_index = np.argmin(y_max_points[:, 0])

    # get the actual three points value
    min_x_final_point = my_points[min_x_index]
    min_y_final_point = y_min_points[y_min_index]
    max_y_final_point = y_max_points[y_max_index]

    return min_y_final_point, min_x_final_point, max_y_final_point


def brick_pose_detection(min_y, min_x, max_y, brick_short_side):
    # calculate the distances of: min_y,min_x and min_x,max_y
    distance1 = math.dist(min_y, min_x)
    distance2 = math.dist(min_x, max_y)

    # if the distance is long enough calculate the hypothetical angular value
    if (distance1 > minimum_distance):
        alpha1 = abs(math.atan2(min_y[0] - min_x[0], min_x[1] - min_y[1]))
    else:
        alpha1 = -1
    if (distance2 > minimum_distance):
        alpha2 = abs(math.atan2(max_y[0] - min_x[0], max_y[1] - min_x[1]))
    else:
        alpha2 = -1

    # actual angular value calculation
    if (alpha1 == -1 and alpha2 == -1):
        print("TANGENT ERROR")
        return 0
    elif (alpha2 == -1 or (alpha1 != -1 and alpha1 < alpha2)):
        if abs(distance1 - brick_short_side) <= lenght_error:
            return alpha1
        else:
            return alpha1 - PI / 2
    else:
        if abs(distance2 - brick_short_side) <= lenght_error:
            return -alpha2
        else:
            return PI / 2 - alpha2


def find_centroid(points_3d):
    points_array = np.array(points_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2])

    centroid = np.mean(points_array, axis=0)

    ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='X', s=100, label='Centroid')
    ax.legend()

    plt.show()

    print("Centroid Coordinates (X, Y, Z) in World Frame:", centroid)

    return centroid[0], centroid[1], centroid[2]


if __name__ == '__main__':
    rospy.init_node('vision_node')

    # Waiting image from zed node
    image_msg = rospy.wait_for_message("/ur5/zed_node/left_raw/image_raw_color", Image)

    # Waiting point cloud from zed node
    point_cloud2_msg = rospy.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    input_data = [
        {'ID': 'X1-Y1-Z2', 'xc': 954, 'yc': 465, 'w': 27, 'h': 44},
        {'ID': 'X1-Y2-Z2-TWINFILLET', 'xc': 1145, 'yc': 463, 'w': 67, 'h': 51}
    ]

    object_detection(image_msg, point_cloud2_msg, input_data)

    # s = rospy.Service('obtain_brick_pose', ObtainBrickPose, handle_obtain_bricks_informations)

    print("the vision is ready to deliver the block position")
    rospy.spin()