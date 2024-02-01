#include <cv_bridge/cv_bridge.h> // This is needed to convert the image data to something that OpenCV can use
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_transport/image_transport.h> // This is needed to get the camera data. This package is quite similar to subscribing to the sensor_msg/Image message
#include "opencv_node/box_and_target_position.h"
#include <sensor_msgs/PointCloud2.h>

using namespace cv;
using namespace std;
using namespace tf2_ros;
using namespace tf2_geometry_msgs;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace image_transport;
using namespace cv_bridge;
using namespace ros;
using namespace std_msgs;

static const string IMAGE_TOPIC = "/camera1/rgb/image_raw"; // The topic where the camera data is published

tf2_ros::Buffer tf_buffer;

// Main function
int main(int argc, char **argv) {
  // The name of the node
  ros::init(argc, argv, "opencv_services");

  // Default handler for nodes in ROS
  ros::NodeHandle nh("");

  // Used to publish and subscribe to images
  image_transport::ImageTransport it(nh);

  // Subscribe to the /camera raw image topic
  image_transport::Subscriber sub = it.subscribe(IMAGE_TOPIC, 1, from_msgs_image_to_cv_mat);

  ros::Subscriber point_cloud_sub = nh.subscribe(POINT_CLOUD2_TOPIC, 1, point_cloud_cb);

  tf2_ros::TransformListener listener(tf_buffer);

  ros::ServiceServer service = nh.advertiseService("box_and_target_position",  get_box_and_target_position);
}

// This function is called every time a new image is published, converts type
// sensor_msgs/Image to cv::Mat
void from_msgs_image_to_cv_mat(const sensor_msgs::ImageConstPtr &msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  float image_size_y = cv_ptr->image.rows;
  float image_size_x = cv_ptr->image.cols;

  cv::Mat canny_output = get_edges(cv_ptr->image);

  std::vector<cv::Point2f> centroids = extract_centroids(canny_output);

  cout << "Centroids: " << centroids << endl;
}

cv::Mat get_edges(cv::Mat camera_image) {
  // Convert the image to grayscale
  cv::Mat gray_image;
  cv::cvtColor(cv_ptr->image, img_gray, cv::CV_BGR2GRAY);

  // Apply the Canny edge detector
  cv::Mat canny_output;
  cv::Canny(img_gray, canny_output, 10, 350);

  return canny_output;
}

std::vector<cv::Point2f> extract_centroids(cv::Mat canny_output) {
  // detect contours on the binary image
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_output, contours, hierarchy, CV_RETR_TREE,
                   CV_CHAIN_APPROX_NONE);

  std::vector<cv::Moments> mu(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    mu[i] = cv::moments(contours[i], false);
  }

  // what we do here is the core of finding the centroids
  // we use the previously computed moments M to compute
  // the center of the shape (called centroid) making
  // the following: cx = M10/M00 and cy = M01/M00
  std::vector<cv::Point2f> mc(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    float mc_x = mu[i].m10 / mu[i].m00;
    float mc_y = mu[i].m01 / mu[i].m00;
    mc[i] = cv::Point2f(mc_x, mc_y);
  }

  // draw contours and centroids
  cv::Mat drawing(canny_output.size(), CV_8UC3, cv::Scalar(255, 255, 255));

  for (int i = 0; i < contours.size(); i++) {
    cv::Scalar color = cv::Scalar(167, 151, 0); // B G R values
    cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0,
                     cv::Point());
    cv::circle(drawing, mc[i], 4, color, -1, 8, 0);
  }

  cv::nameWindow("Contours", CV_WINDOW_AUTOSIZE);
  cv::imshow("Extracted centroids", drawing);
  cv::waitKey(3);
}

void point_cloud_cb(const sensor_msg::PointCloud2 pCloud) {
    gemoetry_msgs::Point box_position_camera_frame;
    pixel_to_3d_point(pCloud, box_centroid.x, box_centroid.y, box_position_camera_frame)

    geometry_msgs::Point target_position_camera_frame;
    pixel_to_3d_point(pCloud, target_centroid.x, target_centroid.y, target_position_camera_frame)

    box_position_base_frame = transform_between_frames(box_position_camera_frame, from_frame, to_frame);
    target_position_base_frame = transform_between_frames(target_position_camera_frame, from_frame, to_frame);

    ROS_INFO_STREAM("3d box position base frame: x " << box_position_base_frame.x << " y " << box_position_base_frame.y << " z " << box_position_base_frame.z);
    ROS_INFO_STREAM("3d target position base frame: x " << target_position_base_frame.x << " y " << target_position_base_frame.y << " z " << target_position_base_frame.z);
}

// void pixel_to_3d_point(const sensor_msg::PointCloud2 pCloud, const int u, const int v, geometry_msg::Point &p) {
//     // get width and height of 2D point cloud data
//     int width = pCloud.width;
//     int height = pCloud.height;

//     // Convert from u (column / width), v (row/height) to position in array
//     // where X,Y,Z data starts
//     // int arrayPosition = v*pCloud.row_step + u*pCloud.point_step;
//     int arrayPosition = v * pCloud.row_step + u * pCloud.point_step;

//     // compute position in array where x,y,z data start
//     int arrayPosX = arrayPosition + pCloud.fields[0].offset; // X has an offset of 0
//     int arrayPosY = arrayPosition + pCloud.fields[1].offset; // Y has an offset of 4
//     int arrayPosZ = arrayPosition + pCloud.fields[2].offset; // Z has an offset of 8

//     float X = 0.0;
//     float Y = 0.0;
//     float Z = 0.0;

//     memcpy(&X, &pCloud.data[arrayPosX], sizeof(float));
//     memcpy(&Y, &pCloud.data[arrayPosY], sizeof(float));
//     memcpy(&Z, &pCloud.data[arrayPosZ], sizeof(float));

//     p.x = X;
//     p.y = Y;
//     p.z = Z;
// }

// geometry_msgs::Point transform_between_frames(geometry_msgs::Point p, const std::string from_frame, const std::string_to_frame) {
//     geometry_msgs::PoseStamped input_pose_stamped;       // we build a pose of type PoseStamped,
//     input_pose_stamped.pose.position = p;                // which is a point object with additional information.
//     input_pose_stamped.header.frame_id = from_frame;     // Information is 'injected' in the PoseStamped object
//     input_pose_stamped.header.stamp = ros::Time::now();  // with 3D point's position, original frame and timestamp

//     geometry_msgs::PoseStamped output_pose_stamped = tf_buffer.transform(input_pose_stamped, to_frame, ros::Duration(1)); // transformation method, note that it has a timeout (last parameter)
//     return output_pose_stamped.pose.position; // return position of transformed point
// }

// bool get_box_and_target_position(opencv_node::box_and_target_position::Request  &req, opencv_node::box_and_target_position::Response &res) {
//     res.box_position = box_position_base_frame;
//     res.target_position = target_position_base_frame;
//     return true;
// }
         