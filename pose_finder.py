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
from sensor_msgs.msg import point_cloud2

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


def get_info_of_bricks_in_world_frame(r):
    global bricks_information

    p = []
    names = []
    for b in bricks_information:
        names.append(b[0])
        ps = Pose()
        ps.position.x = b[1]
        ps.position.y = b[2]
        ps.position.z = b[3]
        ps.orientation.z = b[4]
        ps.append(p)

    r = BrickPoseObtainerResponse()
    r.p = p
    r.names = names
    r.length = len(bricks_information)

    print('Bricks locations...')
    return r


def brick_value(names):
    value = np.array(brick_values[names])
    return value[0], value[1], value[1]


def color_errors(color):
    if colors == "red":
        return 0.03
    elif colors == "green":
        return 0.06
    elif colors == "blue":
        return 0.09
    elif colors == "yellow":
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


def discriminate_brick_points(image):
    largest_contour = find_object(image)
    contoured_image = draw_largest_contour(image, largest_contour)
    filled_contour = np.zeros_like(contoured_image)
    cv2.drawContours(filled_contour, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    points = np.column_stack(np.where(filled_contour[:, :, 0] > 0))
    points_reversed = points[:, ::-1]

    return points_reversed


def detection(image_msgs: Image, point_cloud2_msg: PointCloud2, model) -> None:
    global bricks
    img = CvBridge().imgmsg_to_cv2(image_msgs, 'bgr8')
    result = model(img)
    bounding_boxes = result.pandas().xyxy[0].to_dict(orient="records")
    for bounding_box in bounding_boxes:
        name = bounding_box['name']
        s_side, l_side, height = get_info_of_bricks_in_world_frame(name)
        confidence = bounding_box['confidence']
        x1 = bounding_box['xmin']
        y1 = bounding_box['ymin']
        x2 = bounding_box['xmax']
        y2 = bounding_box['ymax']

        bricks = [(name, confidence, x1, y1, x2, y2, s_side, l_side, height)]

    for attr in bricks:
        sliceBox = slice(attr[3], attr[5]), slice(attr[2], attr[4])
        image = img[sliceBox]

        c = return_color_of_block(image)
        l_side = color_errors(c)

        uv_points = discriminate_brick_points(img)

        points_from_zed = []
        for point in list(uv_points):
            points_from_zed.append([int(coordinate) for coordinate in point])

        three_dimensional_points = point_cloud2.read_points(point_cloud2_msg, field_names=['x', 'y', 'z'],
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
                                 pose_in_world_frame.position.z, pose_in_world_frame.orientation.w,
                                 pose_in_world_frame.orientation.x, pose_in_world_frame.orientation.y,
                                 pose_in_world_frame.orientation.z]
            points_in_world_frame.append(transformed_point)


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

# import numpy as np
# from sensor_msgs.msg import PointCloud2, PointField
# import ros_numpy
# import rospy

# fixed_z_value = 0.0

# class Alberto_Inputs: # this is the data coming from the recognition module
#     def __init__(self, ID: int, xc: float, yc: float, w: float, h: float):
#         self.ID = ID
#         self.xc = xc
#         self.yc = yc
#         self.w = w
#         self.h = h

# list_of_recognized_objects = [
#     Alberto_Inputs(ID=1, xc=10.5, yc=20.3, w=30.0, h=40.0),
#     Alberto_Inputs(ID=2, xc=15.2, yc=25.1, w=35.0, h=45.0),
# ]

# xc_and_yc_tuples = [(obj.xc, obj.yc) for obj in list_of_recognized_objects]

# coordinates_array = np.array([(x, y, fixed_z_value) for x, y in xc_and_yc_tuples], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# point_cloud_msg = ros_numpy.msgify(PointCloud2, {
#     'header': {'stamp': rospy.Time.now(), 'frame_id': 'base_link'},
#     'height': 1,
#     'width': len(xc_and_yc_tuples),
#     'fields': [
#         PointField(name='x', offset=0, datatype=7, count=1),
#         PointField(name='y', offset=4, datatype=7, count=1),
#         PointField(name='z', offset=8, datatype=7, count=1),
#     ],
#     'is_bigendian': False,
#     'point_step': 12,  # 4 bytes per float * 3 coordinate
#     'row_step': 12 * len(xc_and_yc_tuples),
#     'data': coordinates_array.tobytes(),
#     'is_dense': True,
# })

# uvs = ros_numpy.point_cloud2.read_points(point_cloud_msg, field_names=('x', 'y', 'z'), skip_nans=False)

# for uvz in uvs:
#     print(uvz)
