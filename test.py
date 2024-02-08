import warnings

import cv2
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from script import BrickPoseObtainer, BrickPoseObtainerResponse
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

brick_values = {
    'X1-Y1-Z2': [0.03, 0.03, 0.02],
    'X1-Y1-Z1': [0.03, 0.06, 0.01],
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


class Inputs:
    ID: str
    xc: float
    yc: float
    w: float
    h: float


class Outputs:
    ID: str
    xc: float
    yc: float
    zc: float
    roll: float
    pitch: float
    yawn: float


class Block:
    def __init__(self, id, xc, yc, w, h):
        self.id = id
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h


def brick_value(names):
    value = np.array(brick_values[names])
    return value[0], value[1], value[1]


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
    if largest_contour is not None:
        cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
    return result


def discriminate_brick_points(image):
    largest_contour = find_object(image)
    contoured_image = draw_largest_contour(image, largest_contour)
    filled_contour = np.zeros_like(contoured_image)
    cv2.drawContours(filled_contour, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    points = np.column_stack(np.where(filled_contour[:, :, 0] > 0))
    points_reversed = points[:, ::-1]

    return points_reversed


def detection(image_msgs: Image, point_cloud2_msg: PointCloud2, input):
    print("beginning")
    if image_msgs is None:
        print("Image message is empty or None.")
        return
    global bricks
    print("image_msgs:", image_msgs)
    img = CvBridge().imgmsg_to_cv2(image_msgs, 'bgr8')
    if img is None or img.size == 0:
        print("Image is empty or None.")
        return
    print("Before displaying the image")
    cv2.imshow("zed image", img)
    cv2.imwrite("Pictures/USE_THIS.jpg", img)
    print("After displaying the image")

    key = cv2.waitKey(0)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        cv2.destroyAllWindows()

    # Process input and create Block objects
    blocks = []
    for given_data in input:
        block = Block(given_data['ID'], given_data['xc'], given_data['yc'], given_data['w'], given_data['h'])
        blocks.append(block)

    for block in blocks:
        slc = slice(int(block.yc - block.h / 2), int(block.yc + block.h / 2)), \
              slice(int(block.xc - block.w / 2), int(block.xc + block.w / 2))
        if slc is None:
            print("Failed to slice the Cv2 image.")
            return

        image = img[slc]

        cv2.imshow("sliced image", image)

        if image is None or image.size == 0:
            print("Failed slice image.")
            return

        # Assume the color of the block is known
        # color = return_color_of_block(image)  # You can replace this with your color detection logic
        # length_error = color_errors(color)

        uv_points = discriminate_brick_points(image)

        points_from_zed = []
        for point in list(uv_points):
            points_from_zed.append([int(coordinate) for coordinate in point])

        three_dimensional_points = pc2.read_points(point_cloud2_msg, field_names=['x', 'y', 'z'],
                                                           skip_nans=False, uvs=points_from_zed)

        camera_frame_points = []
        for point in three_dimensional_points:
            camera_frame_points.append(point[:3])

        points_in_world_frame = []
        for point in camera_frame_points:
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

        # Find and print the centroid
        centre = find_centroid(points_in_world_frame)
        print(f"Block {block.id} Centroid:", centre)

        # Find and print the orientation
        orientation = find_orientation(points_in_world_frame)
        print(f"Block {block.id} Orientation (Roll, Pitch, Yaw):", orientation)


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

    # Waiting point cloud from zed node
    point_cloud2_msg = rospy.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    # Example input data
    input_data = [{'ID': 'X1-Y1-Z2', 'xc': 954, 'yc': 465, 'w': 27, 'h': 44}]

    detection(image_msg, point_cloud2_msg, input_data)

    print("The vision is ready to deliver the block position")
    rospy.spin()
