import warnings

import cv2
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose

from pose_finder.srv import BrickPoseObtainer, BrickPoseObtainerResponse
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

input = []

warnings.filterwarnings("ignore")

bricks_information = []

height_of_table = 0.88
error_of_brick = 0.001
error_of_value = 0.001

minimum_distance = 0.005
error_of_length = 0.005

PI = 3.14

OFFSET = 10

brick_values = {
    'X1-Y1-Z2': [0.03, 0.03, 0.02],
    'X1-Y -Z1': [0.03, 0.06, 0.01],
    'X1-Y2-Z2-CHAMFER': [0.03, 0.06, 0.02],
    'X1-Y2-Z2-TWINFILLET': [0.03, 0.06, 0.02],
    'X1-Y2-Z2': [0.03, 0.06, 0.02],
    'X1-Y3-Z2-FILLET': [0.01, 0.09, 0.02],
    'X1-Y3-Z2': [0.03, 0.09, 0.02],
    'X1-Y4-Z1': [0.03, 0.12, 0.01],
    'X1-Y4-Z2': [0.03, 0.12, 0.02],
}

colors = {
    'red': [(0, 50, 50), (10, 255, 255)],
    'green': [(36, 50, 50), (70, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'yellow': [(20, 50, 50), (35, 255, 255)],
    'fuchsia': [(145, 50, 50), (175, 255, 255)],
    'orange': [(11, 50, 50), (25, 255, 255)]
}


class Input:
    ID: str
    xc: float
    yc: float
    w: float
    h: float


inputs = []


class Outputs:
    ID: str
    xc: float
    yc: float
    zc: float
    roll: float
    pitch: float
    yawn: float


def brick_value(names):
    value = np.array(brick_values[names])
    return value[0], value[1], value[2]


def color_errors(color):
    if color == "red":
        return 0.03
    elif color == "green":
        return 0.06
    elif color == "blue":
        return 0.09
    elif color == "yellow":
        return 0.12


def return_color_of_block(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for key, value in colors.items():
        lower = np.array(value[0])
        upper = np.array(value[1])
        mask = cv2.inRange(hsv, lower, upper)
        if mask.any():
            return key
    return None


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

    points_reversed = points_reversed + offset

    return points_reversed


def detection(image_msgs: Image, point_cloud2_msg: PointCloud2, input):
    global bricks
    centres = []
    orientations = []
    img = CvBridge().imgmsg_to_cv2(image_msgs, 'bgr8')
    bricks = []
    for given_data in input:
        name = given_data['ID']
        width_of_bb = given_data["w"]
        heigth_of_bb = given_data["h"]
        s_side, l_side, height = brick_value(name)
        x1 = given_data["xc"] - width_of_bb / 2 - OFFSET
        y1 = given_data["yc"] - heigth_of_bb / 2 - OFFSET
        x2 = x1 + width_of_bb + OFFSET
        y2 = y1 + heigth_of_bb + OFFSET
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        bricks.append((name, x1, y1, x2, y2, s_side, l_side, height))

    for attr in bricks:
        print(f"attr[1]: {attr[1]}, attr[2]: {attr[2]}, attr[3]: {attr[3]}, attr[4]: {attr[4]}")
        slc = slice(int(attr[2]), int(attr[4] + OFFSET*2)), slice(int(attr[1]), int(attr[3] + OFFSET*2)), slice(None)
        print(slc)
        # cv2.imshow("zed's image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Sliced image pixel values:")
        print(img[slc])
        image = img[slc]
        print("Minimum pixel value:", np.min(image))
        print("Maximum pixel value:", np.max(image))
        print("Image shape:", image.shape)

        # Resize the sliced image for inspection (e.g., resize it to double the original size)
        resized_image = cv2.resize(image, None, fx=2.0, fy=2.0)

        # Save the resized image for inspection
        cv2.imwrite("Pictures/resized_image.png", resized_image)

        # Display the resized image
        # cv2.imshow("resized image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow("sliced image", image)

        # c = return_color_of_block(image)
        # l_side = color_errors(c)

        largest_contour = find_object(image)

        # Check if largest_contour is None
        if largest_contour is None:
            print("the largest contour is NONE")

        uv_points_sliced = discriminate_brick_points(image, np.array([int(attr[1]), int(attr[2])]))

        uv_points_original = uv_points_sliced + np.array([int(attr[1]), int(attr[2])])

        print("UV Points:", uv_points_original)

        # Convert uv_points to np.uint8
        uv_points_display = np.zeros_like(image, dtype=np.uint8)

        # Add a check for valid UV points range
        valid_indices = np.where(
            (uv_points_original[:, 0] >= 0) & (uv_points_original[:, 0] < uv_points_display.shape[1]) &
            (uv_points_original[:, 1] >= 0) & (uv_points_original[:, 1] < uv_points_display.shape[0])
        )

        # Update uv_points_display using only valid indices
        uv_points_display[uv_points_original[valid_indices][:, 1], uv_points_original[valid_indices][:, 0]] = [255, 255,
                                                                                                               255]
        # Display the original image
        cv2.imshow("Original Image", img)

        # Create a larger canvas for displaying both images
        larger_canvas = np.zeros((max(img.shape[0], uv_points_display.shape[0]),
                                  img.shape[1] + uv_points_display.shape[1], 3), dtype=np.uint8)

        # Place the original image on the canvas
        larger_canvas[:img.shape[0], :img.shape[1]] = img

        # Place the UV points image on the canvas next to the original image
        larger_canvas[:uv_points_display.shape[0], img.shape[1]:] = uv_points_display

        # Display the larger canvas
        cv2.imshow("Original Image and UV POINTS", larger_canvas)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        points_from_zed = []
        for point in list(uv_points_original):
            points_from_zed.append([int(coordinate) for coordinate in point])

        three_dimensional_points = pc2.read_points(point_cloud2_msg, field_names=['x', 'y', 'z'],
                                                           skip_nans=False, uvs=points_from_zed)

        # Add your processing for three_dimensional_points here
        three_dimensional_points_array = np.array(list(three_dimensional_points))

        cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
        for uv_point in uv_points_original:
            cv2.circle(img, tuple(uv_point), 5, (255, 0, 0), -1)
        cv2.imshow("UV Points on Original Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Visualize 3D points in a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(three_dimensional_points_array[:, 0], three_dimensional_points_array[:, 1],
                   three_dimensional_points_array[:, 2], label='3D Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title("Plotted points in 3D")
        plt.show()
        print(three_dimensional_points_array)

        if three_dimensional_points_array is not None and len(three_dimensional_points_array) > 0:
            print("Valid 3D Points:")
            print(three_dimensional_points_array)

            # Assuming three_dimensional_points_array has columns x, y, z
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(three_dimensional_points_array[:, 0], three_dimensional_points_array[:, 1],
                       three_dimensional_points_array[:, 2])

            plt.title("Plotted points in 3D")
            plt.show()
        else:
            print("No valid 3D points found.")

        camera_frame_points = []
        for point in three_dimensional_points:
            camera_frame_points.append(point[:3])

    points_in_world_frame = []

    for point in three_dimensional_points:
        print("HELLOOOOOOO")
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]
        pose.orientation.w = 1
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0

        pose_in_world_frame = convert_coordinates(pose, 'zed2_left_camera_frame', 'world')
        transformed_point = [pose_in_world_frame.position.x, pose_in_world_frame.position.y,
                             pose_in_world_frame.position.z]
        points_in_world_frame.append(transformed_point)

    # Add these prints after transforming points to the world frame
    for point in points_in_world_frame:
        print("Transformed Point in World Frame:", point)
    # Convert points_in_world_frame to a NumPy array for easier manipulation
    points_in_world_frame_array = np.array(points_in_world_frame)

    # Check if points_in_world_frame_array has the expected structure
    if len(points_in_world_frame_array.shape) == 2 and points_in_world_frame_array.shape[1] == 3:
        print("Valid 3D Points:")
        print(points_in_world_frame_array)

        # Plot the points in the world frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_in_world_frame_array[:, 0], points_in_world_frame_array[:, 1],
                   points_in_world_frame_array[:, 2], label='Points in World Frame')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title("Points in World Frame")
        plt.show()
    else:
        print("No valid 3D points found.")

    centre = find_centroid(points_in_world_frame)
    position = find_orientation(points_in_world_frame)
    centres.append(centre)
    orientations.append(position)
    #
    # return centres, orientations



def convert_coordinates(coordinates, from_frame, to_frame):
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = coordinates
    pose_stamped.header.frame_id = from_frame
    pose_stamped.header.stamp = rospy.Time.now()

    try:
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(1))
        return output_pose_stamped.pose

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise


def find_centroid(points_3d):
    points_array = np.array(points_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2])

    centroid = np.mean(points_array, axis=0)

    ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='X', s=100, label='Centroid')
    ax.legend()

    plt.show()

    return centroid[0], centroid[1], centroid[2]


def find_orientation(points_3d):
    points_array = np.array(points_3d)

    pca = PCA(n_components=3)
    pca.fit(points_array)

    principal_axes = pca.components_
    eigenvalues = pca.explained_variance_

    roll = np.arctan2(principal_axes[1, 0], principal_axes[0, 0])
    pitch = np.arctan2(-principal_axes[2, 0], np.sqrt(principal_axes[2, 1] ** 2 + principal_axes[2, 2] ** 2))
    yaw = np.arctan2(principal_axes[2, 1], principal_axes[2, 2])

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg


if __name__ == '__main__':
    rospy.init_node('vision_node')

    # Waiting image from zed node
    image_msg = rospy.wait_for_message("/ur5/zed_node/left/image_rect_color", Image)
    # cv2.imwrite("Debug/zed_sample_to_debug", image_msg)

    # Waiting point cloud from zed node
    point_cloud2_msg = rospy.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    input_data = [
        {'ID': 'X1-Y1-Z2', 'xc': 954, 'yc': 465, 'w': 27, 'h': 44},
        {'ID': 'X1-Y2-Z2-TWINFILLET', 'xc': 1145, 'yc': 463, 'w': 67, 'h': 51}
    ]

    # centres, orientations = \
    detection(image_msg, point_cloud2_msg, input_data)

    # print(centres, orientations)

    # s = rospy.Service('obtain_brick_pose', BrickPoseObtainer, get_info_of_bricks_in_world_frame)

    print("the vision is ready to deliver the block position")
    rospy.spin()
