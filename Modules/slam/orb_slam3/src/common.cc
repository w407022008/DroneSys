/**
* 
* Common functions and variables across all modes (mono/stereo, with or w/o imu)
*
*/

#include "common.h"

#define DEBUG false

ros::Publisher pose_pub;
ros::Publisher map_points_pub;
image_transport::Publisher rendered_image_pub;

std::string map_frame_id, pose_frame_id;

// Coordinate transformation matrix from orb coordinate system to ros coordinate systemm
tf::Matrix3x3 tf_orb_to_ros(1, 0, 0,
                            0, 1, 0,
                            0, 0, 1);

bool interpolation;
float interpolation_rate;
float delay;
int interpolation_sample_num;
int interpolation_order;

ros::Time time_ref;
ros::Time time_stamp_header;
Eigen::MatrixXf A(interpolation_sample_num,interpolation_order+1);
Eigen::MatrixXf b(interpolation_sample_num,7);
Eigen::MatrixXf X(interpolation_order+1,7);

void setup_ros_publishers(ros::NodeHandle &node_handler, image_transport::ImageTransport &image_transport)
{
    pose_pub = node_handler.advertise<geometry_msgs::PoseStamped> ("/orb_slam3_ros/camera", 1);

    map_points_pub = node_handler.advertise<sensor_msgs::PointCloud2>("orb_slam3_ros/map_points", 1);

    rendered_image_pub = image_transport.advertise("orb_slam3_ros/tracking_image", 1);
}

void setup_tf_orb_to_ros(ORB_SLAM3::System::eSensor sensor_type)
{
    // The conversion depends on whether IMU is involved:
    //  z is aligned with camera's z axis = without IMU
    //  z is aligned with gravity = with IMU
    if (sensor_type == ORB_SLAM3::System::MONOCULAR || sensor_type == ORB_SLAM3::System::STEREO || sensor_type == ORB_SLAM3::System::RGBD)
    {
        tf_orb_to_ros.setValue(
             0,  0,  1,
            -1,  0,  0,
             0, -1,  0);
    }
    else if (sensor_type == ORB_SLAM3::System::IMU_MONOCULAR || sensor_type == ORB_SLAM3::System::IMU_STEREO)
    {
        tf_orb_to_ros.setValue(
             0,  1,  0,
            -1,  0,  0,
             0,  0,  1);
    }
    else
    {
        tf_orb_to_ros.setIdentity();
    }
}

void setup_interpolation()
{
    A.resize(interpolation_sample_num,interpolation_order+1);
    b.resize(interpolation_sample_num,7);
    X.resize(interpolation_order+1,7);
    A.setZero();
    b.setZero();
    X.setZero();
}

void publish_ros_tracking_img(cv::Mat image, ros::Time current_frame_time)
{
    std_msgs::Header header;

    header.stamp = current_frame_time;

    header.frame_id = map_frame_id;

    const sensor_msgs::ImagePtr rendered_image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

    rendered_image_pub.publish(rendered_image_msg);
}

void publish_ros_tracking_mappoints(std::vector<ORB_SLAM3::MapPoint*> map_points, ros::Time current_frame_time)
{
    sensor_msgs::PointCloud2 cloud = tracked_mappoints_to_pointcloud(map_points, current_frame_time);
    
    map_points_pub.publish(cloud);
}

void publish_ros_poseStamped(std::deque<geometry_msgs::PoseStamped> pose_msgs, bool & updated)
{
    if(!interpolation)
    {
        if(updated)
        {
            updated = false;
            geometry_msgs::PoseStamped vision = pose_msgs.back();
//    cout << "image: "<< (ros::Time::now() - vision.header.stamp).toSec() << endl;;
            vision.header.stamp += ros::Duration(delay);
            pose_pub.publish(vision);
            pose_msgs.clear();
        }
    }
    else
    {
        if(pose_msgs.size() == interpolation_sample_num)
        {
            if(updated)
            {
                updated = false;
                // Fitting
                time_ref = pose_msgs.front().header.stamp;
                std::deque<geometry_msgs::PoseStamped>::iterator it = pose_msgs.begin();
                int idx=0;
                while(it != pose_msgs.end()){
                    float t = (it->header.stamp-time_ref).toSec();

                    A(idx,0) = 1;
                    for(int i=1;i<=interpolation_order;i++)
                    {
                        A(idx,i) = t*A(idx,i-1);
                    }
                    b(idx,0) = it->pose.position.x;
                    b(idx,1) = it->pose.position.y;
                    b(idx,2) = it->pose.position.z;
                    b(idx,3) = it->pose.orientation.x;
                    b(idx,4) = it->pose.orientation.y;
                    b(idx,5) = it->pose.orientation.z;
                    b(idx,6) = it->pose.orientation.w;
                    it++;
                    idx++;
                }
                auto res = A.transpose() * A;
                auto val = A.transpose() * b;
                // auto  inv = res.inverse();
                X = res.llt().solve(val);
            }

            while(pose_msgs.back().header.stamp >= time_stamp_header - ros::Duration(0.005)){
                if(DEBUG)cout << "[DEBUG] delta time pub: " << (time_stamp_header- pose_msgs.back().header.stamp).toSec() << endl;
                // Interpolation
                geometry_msgs::PoseStamped vision;
                vision.header.stamp = time_stamp_header + ros::Duration(delay);// default delay
                vision.header.frame_id = map_frame_id;
                float t = (time_stamp_header-time_ref).toSec();

                Eigen::VectorXf T(interpolation_order+1);
                T(0) = 1;
                for(int i=1;i<=interpolation_order;i++)
                {
                    T(i) = t*T(i-1);
                }
        
                // Publish
                Eigen::VectorXf state(7);
                state = X.transpose()*T;

                vision.pose.position.x = state[0];
                vision.pose.position.y = state[1];
                vision.pose.position.z = state[2];

                Eigen::Quaternion<double> q = {state[6],state[3],state[4],state[5]};
                q.normalize();
                vision.pose.orientation.x = q.x();
                vision.pose.orientation.y = q.y();
                vision.pose.orientation.z = q.z();
                vision.pose.orientation.w = q.w();
                
                pose_pub.publish(vision);

                time_stamp_header += ros::Duration(1.0/interpolation_rate);
            }
        }
        else if(!pose_msgs.empty())
        {
            time_stamp_header = pose_msgs.back().header.stamp;
        }
    }
}

tf::Transform from_orb_to_ros_tf_transform(cv::Mat transformation_mat)
{
    cv::Mat orb_rotation(3, 3, CV_32F);
    cv::Mat orb_translation(3, 1, CV_32F);

    orb_rotation    = transformation_mat.rowRange(0, 3).colRange(0, 3);
    orb_translation = transformation_mat.rowRange(0, 3).col(3);

    tf::Matrix3x3 tf_camera_rotation(
        orb_rotation.at<float> (0, 0), orb_rotation.at<float> (0, 1), orb_rotation.at<float> (0, 2),
        orb_rotation.at<float> (1, 0), orb_rotation.at<float> (1, 1), orb_rotation.at<float> (1, 2),
        orb_rotation.at<float> (2, 0), orb_rotation.at<float> (2, 1), orb_rotation.at<float> (2, 2)
    );

    tf::Vector3 tf_camera_translation(orb_translation.at<float> (0), orb_translation.at<float> (1), orb_translation.at<float> (2));

    // cout << setprecision(9) << "Rotation: " << endl << orb_rotation << endl;
    // cout << setprecision(9) << "Translation xyz: " << orb_translation.at<float> (0) << " " << orb_translation.at<float> (1) << " " << orb_translation.at<float> (2) << endl;

    // Transform from orb coordinate system to ros coordinate system on camera coordinates
    tf_camera_rotation    = tf_orb_to_ros * tf_camera_rotation;
    tf_camera_translation = tf_orb_to_ros * tf_camera_translation;

    // Inverse matrix
    tf_camera_rotation    = tf_camera_rotation.transpose();
    tf_camera_translation = -(tf_camera_rotation * tf_camera_translation);

    // Transform from orb coordinate system to ros coordinate system on map coordinates
    tf_camera_rotation    = tf_orb_to_ros * tf_camera_rotation;
    tf_camera_translation = tf_orb_to_ros * tf_camera_translation;

    return tf::Transform(tf_camera_rotation, tf_camera_translation);
}

sensor_msgs::PointCloud2 tracked_mappoints_to_pointcloud(std::vector<ORB_SLAM3::MapPoint*> map_points, ros::Time current_frame_time)
{
   const int num_channels = 3; // x y z

   if (map_points.size() == 0)
   {
       std::cout << "Map point vector is empty!" << std::endl;
   }

   sensor_msgs::PointCloud2 cloud;

   cloud.header.stamp = current_frame_time;
   cloud.header.frame_id = map_frame_id;
   cloud.height = 1;
   cloud.width = map_points.size();
   cloud.is_bigendian = false;
   cloud.is_dense = true;
   cloud.point_step = num_channels * sizeof(float);
   cloud.row_step = cloud.point_step * cloud.width;
   cloud.fields.resize(num_channels);

   std::string channel_id[] = { "x", "y", "z"};

   for (int i = 0; i < num_channels; i++)
   {
       cloud.fields[i].name = channel_id[i];
       cloud.fields[i].offset = i * sizeof(float);
       cloud.fields[i].count = 1;
       cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
   }

   cloud.data.resize(cloud.row_step * cloud.height);

   unsigned char *cloud_data_ptr = &(cloud.data[0]);


   for (unsigned int i = 0; i < cloud.width; i++)
   {
       if (map_points[i])
       {

           tf::Vector3 point_translation(map_points[i]->GetWorldPos()(0), map_points[i]->GetWorldPos()(1), map_points[i]->GetWorldPos()(2));

           point_translation = tf_orb_to_ros * point_translation;

           float data_array[num_channels] = {point_translation.x(), point_translation.y(), point_translation.z()};

           memcpy(cloud_data_ptr+(i*cloud.point_step), data_array, num_channels*sizeof(float));
       }
   }
   return cloud;
}
