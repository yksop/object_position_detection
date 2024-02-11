#! /usr/bin/env python3

"""
    Write something...
"""

import math
import warnings

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Pose
from matplotlib import pyplot as plt
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from tf2_geometry_msgs import tf2_geometry_msgs

warnings.filterwarnings("ignore")

bricks_information = []

table_height = 0.88
brick_error = 0.001
value_error = 0.001

minimum_distance = 0.005
length_error = 0.005

CROPPING_OFFSET = 10

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


##
#   This function computes the norm of two points.
#
#   Inputs:
#   - p1: first point
#   - p2: second point
#
#   Outputs:
#   - math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2): distance between the points.
##


def norm(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


##
#   This function returns three values of a brick: its short and long sides and its height.
#
#   Inputs:
#   - name: the ID that is coming from YOLO recognition
#
#   Outputs:
#   - value[0]: the short side of the brick
#   - value[1]: the long side of the brick
#   - value[2]: height of the brick
##


def get_brick_value(name):
    value = np.array(brick_value[name])
    return value[0], value[1], value[2]


##
#   This function computes the largest contour of an image.
#
#   Inputs:
#   - image: the image on which the computation is performed
#
#   Outputs:
#   - largest_contour: the largest contour of the image based on the area
##


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


##
#   This function draws a contour on an image.
#
#   Inputs:
#   - image: the image on which the contour is drawn
#
#   Outputs:
#   - result: the contour laid on the image
##


def draw_contour(image, contour):
    result = image.copy()
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    return result


##
#   The function computes every single point contained in an object exploiting find_object and draw_contour.
#   The points are found using the largest contour of the image. Hence points that are part of the contour and of the
#   inside of the contour are part of the result.
#
#   Inputs:
#   - image: the image in which the desired object is
#   - offset: this input allows to re-scale a previously cropped image in order to return the real values of the points in the image
#
#   Outputs:
#   - result: all the points that are part of the object
##


def discriminate_brick_points(image, offset):
    largest_contour = find_object(image)
    contoured_image = draw_contour(image, largest_contour)
    cv2.imshow("Contoured image:", contoured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    filled_contour = np.zeros_like(contoured_image)
    cv2.drawContours(filled_contour, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    points = np.column_stack(np.where(filled_contour[:, :, 0] > 0))
    points_reversed = points[:, ::-1]

    result = []
    for point in points_reversed:
        x, y = point
        result.append([x + int(offset[1]), y + int(offset[2])])

    return result


##
#   The function finds the position of the object with respect to the world frame
#
#   Inputs:
#   - image_msg: image published by the zed camera in Rviz
#   - point_cloud2_msg: data published by the PointCloud2
#   - recognized_bricks: information coming from the YOLO model
#
#   Outputs:
#   - centre
#   - orientation
##


def object_detection(image_msg: Image, point_cloud2_msg: PointCloud2, recognized_bricks) -> None:
    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")

    cv2.imshow("Original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    brick_list = []
    for bbox in recognized_bricks:
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
    for brick in brick_list:

        # cropping image box
        sliceBox = slice(brick[2] - CROPPING_OFFSET, brick[4] + CROPPING_OFFSET), slice(brick[1] - CROPPING_OFFSET, brick[3] + CROPPING_OFFSET)
        image = img[sliceBox]

        cv2.imshow("Cropped image:", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # filtering background
        points_2D = []
        points_2D = discriminate_brick_points(image, brick)

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

        # # trasforming each point in world frame
        data_world = []
        for point in zed_points:
            point = rotational_matrix.dot(point) + pos_zed
            point = np.array(point)
            data_world.append(point)

        # for point in zed_points:
        #     pose = Pose()
        #     pose.position.x = point[0]
        #     pose.position.y = point[1]
        #     pose.position.z = point[2]
        #     pose.orientation.w = 1
        #     pose.orientation.x = 0
        #     pose.orientation.y = 0
        #     pose.orientation.z = 0
        #
        #     pose_in_world_frame = convert_coordinates(pose, 'zed2_left_camera_frame', 'world')
        #     transformed_point = [pose_in_world_frame.position.x, pose_in_world_frame.position.y,
        #                          pose_in_world_frame.position.z]
        #     data_world.append(transformed_point)

        min_y, min_x, max_y = get_three_points(data_world)

        print("Three points:", min_y, min_x, max_y)

        centroid = find_centroid(data_world)
        alpha = find_orientation(min_y, min_x, max_y)
        print("Orientation:", alpha)
        # ros_preprocessing_data(brick[0], centroid, alpha)


def ros_preprocessing_data(name, x, y, z, alpha):
    global bricks_information
    bricks_information.append([name, x, y, z, alpha])


##
#   This function finds the leftmost point, the rightmost point and the lowest point at the bottom.
#
#   Inputs:
#   - points: the 3D points that are part of the object
#
#   Outputs:
#   - leftmost_point
#   - rightmost_point
#   - lowest_point
##


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
    lowest_point = my_points[min_x_index]
    rightmost_point = y_min_points[y_min_index]
    leftmost_point = y_max_points[y_max_index]

    return rightmost_point, lowest_point, leftmost_point


##
#   This function finds the orientation of the brick.
#
#   Inputs:
#   - rightmost_point
#   - lowest_point
#   - leftmost_point
#
#   Outputs:
#   - rot: rotation of the brick with reference to the world frame
##


def find_orientation(rightmost_point, lowest_point, leftmost_point):
    orientation = [0, 0, 0]

    d1 = norm(rightmost_point, lowest_point)
    d2 = norm(lowest_point, leftmost_point)

    orientation[0] = PI / 2

    if lowest_point[0] - rightmost_point[0] != 0:
        alpha = math.atan((rightmost_point[1] - lowest_point[1]) / (rightmost_point[0] - lowest_point[0]))
    else:
        alpha = 0

    if d1 > d2:
        alpha = alpha + PI / 2

    orientation[2] = alpha

    return orientation


##
#   This function finds the centroid (centre of mass) of the brick.
#
#   Inputs:
#   - points: the 3D points that are part of the object
#
#   Outputs:
#   - centroid
##


def find_centroid(points):
    points_array = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2])

    centroid = np.mean(points_array, axis=0)

    ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='X', s=100, label='Centroid')
    ax.legend()

    plt.show()

    print("Centroid Coordinates (X, Y, Z) in World Frame:", centroid)

    return centroid


##
#   This function allows to automatically convert a coordinate expressed in a certain frame to another frame.
#   This is done by listening to the frames published by the ROS master.
#
#   Inputs:
#   - coordinates: the coordinates that are needed to be converted
#   - from_frame: initial frame
#   - to_frame: final frame
#
#   Outputs:
#   - output_pose_stamped.pose: pose of the resulting transformed coordinate
##


def convert_coordinates(coordinates, from_frame, to_frame):
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = coordinates
    pose_stamped.header.frame_id = from_frame
    pose_stamped.header.stamp = rospy.Time.now()

    try:
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(0))

        return output_pose_stamped.pose

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise


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
