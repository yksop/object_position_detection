#! /usr/bin/env python3

"""
    Write something...
"""
import copy
import math
import warnings

import cv2
import numpy as np
import rospy
import open3d as o3d
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from matplotlib import pyplot as plt
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from tf2_geometry_msgs import tf2_geometry_msgs

warnings.filterwarnings("ignore")

bricks_information = []

CROPPING_OFFSET = 10

threshold = 0.01

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
#   - image_msg: image published by the zed camera in RViz
#   - point_cloud2_msg: data published by the PointCloud2
#   - recognized_bricks: information coming from the YOLO model
#
#   Outputs:
#   - centre
#   - orientation
##


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

    return np.mean(points_array, axis=0)


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


def icp_registration(source_pointcloud, target_pointcloud, threshold, starting_transformation):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pointcloud, target_pointcloud, threshold, starting_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    aligned_source_pointcloud = copy.deepcopy(source_pointcloud)
    aligned_source_pointcloud.transform(reg_p2p.transformation)

    return aligned_source_pointcloud, reg_p2p


##
#   This function converts a rotation matrix into euler angles (RPY).
#
#   Inputs:
#   - R: the rotation matrix
#
#   Outputs:
#   - x: roll angle
#   - y: pitch angle
#   - z: yaw angle
##


def rotation_matrix_to_euler_angles(R):
    # Convert a 3x3 rotation matrix to Roll, Pitch, Yaw angles
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return x, y, z


##
#   This function downsamples the point cloud, estimates normals, then computes a FPFH feature for each point.
#   The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point.
#
#   Inputs:
#   - pcd: the pointcloud
#   - voxel_size: the size at which we desire to downsample
#
#   Outputs:
#   - pcd_down: the downsampled pointcloud
#   - pcd_fpfh: FPFH feature for the point
##


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


##
#   This function simply prepares all the pointclouds (the source one and the target one).
#   The previously declared 'prepare_dataset' is used for each.
#
#   Inputs:
#   - source: the source pointcloud
#   - target: the target pointcloud
#   - voxel_size: the size at which we desire to downsample
#
#   Outputs:
#   - source: the source pointcloud
#   - target: the target pointcloud
#   - source_down: the prepared (downsampled) source pointcloud
#   - target_down: the prepared (downsampled) target pointcloud
#   - source_fpfh: FPFH feature for the points of the source pointcloud
#   - target_fpfh: FPFH feature for the points of the target pointcloud
#
##


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


##
#   This executes the actual global registration.
#
##


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def object_detection(image_msg: Image, point_cloud2_msg: PointCloud2, input) -> None:
    outputs = []
    img = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")

    bricks = []
    for bbox in input:
        name = bbox['ID']
        width_of_bb = bbox["w"]
        heigth_of_bb = bbox["h"]
        x1 = bbox["xc"] - width_of_bb / 2
        y1 = bbox["yc"] - heigth_of_bb / 2
        x2 = x1 + width_of_bb
        y2 = y1 + heigth_of_bb

        bricks.append((name, int(x1), int(y1), int(x2), int(y2)))

    for brick in bricks:

        slicing = slice(brick[2] - CROPPING_OFFSET, brick[4] + CROPPING_OFFSET), slice(brick[1] - CROPPING_OFFSET, brick[3] + CROPPING_OFFSET)
        sliced_image = img[slicing]

        uv_points = []
        uv_points = discriminate_brick_points(sliced_image, brick)

        uv_points_zed = []
        for point in list(uv_points):
            uv_points_zed.append([int(coordinate) for coordinate in point])

        points_3d = point_cloud2.read_points(point_cloud2_msg, field_names=['x', 'y', 'z'], skip_nans=False,
                                             uvs=uv_points_zed)

        rotational_matrix = np.array([[0., -0.49948, 0.86632],
                                      [-1., 0., 0.],
                                      [-0., -0.86632, -0.49948]])

        pos_zed = np.array([-0.4, 0.59, 1.4])

        three_d_points_zed = []
        for point in points_3d:
            three_d_points_zed.append(point[:3])

        three_d_points_world = []
        for point in three_d_points_zed:
            point = rotational_matrix.dot(point) + pos_zed
            point = np.array(point)
            three_d_points_world.append(point)

        # for point in uv_points_zed:
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
        #     three_d_points_world.append(transformed_point)

        points_3d_np = np.array(three_d_points_world)

        source_pointcloud = o3d.geometry.PointCloud()
        source_pointcloud.points = o3d.utility.Vector3dVector(points_3d_np)

        stl_file_path = f"Models/{brick[0]}/mesh/{brick[0]}.stl"

        mesh = o3d.io.read_triangle_mesh(stl_file_path)

        target_pointcloud = mesh.sample_points_uniformly(number_of_points=2000)

        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_pointcloud, target_pointcloud, voxel_size=0.05)

        result_fast = execute_fast_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size=0.05)

        aligned_source_pointcloud, reg_p2p = icp_registration(source_pointcloud, target_pointcloud, threshold, result_fast.transformation)

        # Extract transformation matrix
        transformation_matrix = np.array(reg_p2p.transformation)

        # Extract rotation matrix
        rotation_matrix = transformation_matrix[:3, :3]

        # Convert rotation matrix to Euler angles
        euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

        centroid = find_centroid(three_d_points_world)

        out = Outputs()
        out.ID = brick[0]
        out.xc = centroid[0]
        out.yc = centroid[1]
        out.zc = centroid[2]
        out.roll = euler_angles[0]
        out.roll = euler_angles[1]
        out.roll = euler_angles[2]

        outputs.append(out)

    return outputs
