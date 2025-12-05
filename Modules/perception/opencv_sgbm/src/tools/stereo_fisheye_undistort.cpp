#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <chrono>

class StereoFisheyeUndistortNode {
public:
    StereoFisheyeUndistortNode(): nh("~") {
        // 获取参数
        nh.param<std::string>("left_input_topic", left_input_topic, "/stereo/left/image_raw");
        nh.param<std::string>("right_input_topic", right_input_topic, "/stereo/right/image_raw");
        nh.param<std::string>("left_output_topic", left_output_topic, "/stereo/left/image_undistorted");
        nh.param<std::string>("right_output_topic", right_output_topic, "/stereo/right/image_undistorted");
        nh.param<double>("timestamp_tolerance", timestamp_tolerance, 0.01);
        
        // 获取左右相机内参矩阵和畸变系数
        loadCameraParameters();
        
        // 设置图像订阅者，使用message_filters实现同步
        left_image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, left_input_topic, 10));
        right_image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, right_input_topic, 10));
        
        // 使用近似时间同步策略
        sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *left_image_sub, *right_image_sub));
        sync->registerCallback(boost::bind(&StereoFisheyeUndistortNode::stereoImageCallback, this, _1, _2));
        
        // 创建发布者
        left_undistorted_pub = nh.advertise<sensor_msgs::Image>(left_output_topic, 10);
        right_undistorted_pub = nh.advertise<sensor_msgs::Image>(right_output_topic, 10);
        
        ROS_INFO("StereoFisheyeUndistortNode initialized:");
        ROS_INFO("  Left input topic: %s", left_input_topic.c_str());
        ROS_INFO("  Right input topic: %s", right_input_topic.c_str());
        ROS_INFO("  Left output topic: %s", left_output_topic.c_str());
        ROS_INFO("  Right output topic: %s", right_output_topic.c_str());
    }

    void stereoImageCallback(const sensor_msgs::ImageConstPtr& left_msg, 
                            const sensor_msgs::ImageConstPtr& right_msg) {
        // 检查时间戳差异
        double time_diff = abs((left_msg->header.stamp - right_msg->header.stamp).toSec());
        if (time_diff > timestamp_tolerance) {
            ROS_WARN("Timestamp difference (%f sec) exceeded tolerance (%f sec)", time_diff, timestamp_tolerance);
            return;
        }
        
        // 转换图像
        cv::Mat left_image, right_image;
        try {
            left_image = cv_bridge::toCvShare(left_msg, "mono8")->image;
            right_image = cv_bridge::toCvShare(right_msg, "mono8")->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        // 鱼眼去畸变，保留完整图像并填充黑色区域
        cv::Mat left_undistorted, right_undistorted;
        cv::fisheye::undistortImage(left_image, left_undistorted, K_left, D_left, K_left, left_image.size());
        cv::fisheye::undistortImage(right_image, right_undistorted, K_right, D_right, K_right, right_image.size());
        
        // 发布去畸变后的图像
        sensor_msgs::ImagePtr left_undistorted_msg = cv_bridge::CvImage(left_msg->header, "mono8", left_undistorted).toImageMsg();
        sensor_msgs::ImagePtr right_undistorted_msg = cv_bridge::CvImage(right_msg->header, "mono8", right_undistorted).toImageMsg();
        left_undistorted_pub.publish(left_undistorted_msg);
        right_undistorted_pub.publish(right_undistorted_msg);
        
        ROS_DEBUG("Processed stereo image pair with size %dx%d", left_image.cols, left_image.rows);
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    
    void loadCameraParameters() {
        // 获取左相机参数
        std::vector<double> left_k_vec, left_d_vec;
        if (!nh.getParam("left_k", left_k_vec) || !nh.getParam("left_d", left_d_vec)) {
            ROS_ERROR("Left camera parameters (left_k, left_d) not provided!");
            exit(-1);
        }
        
        K_left = cv::Mat(3, 3, CV_64F, left_k_vec.data()).clone();
        D_left = cv::Mat(1, 4, CV_64F, left_d_vec.data()).clone();
        
        // 获取右相机参数
        std::vector<double> right_k_vec, right_d_vec;
        if (!nh.getParam("right_k", right_k_vec) || !nh.getParam("right_d", right_d_vec)) {
            ROS_ERROR("Right camera parameters (right_k, right_d) not provided!");
            exit(-1);
        }
        
        K_right = cv::Mat(3, 3, CV_64F, right_k_vec.data()).clone();
        D_right = cv::Mat(1, 4, CV_64F, right_d_vec.data()).clone();
        
        ROS_INFO("Camera parameters loaded successfully");
    }

    ros::NodeHandle nh;
    ros::Publisher left_undistorted_pub;
    ros::Publisher right_undistorted_pub;
    
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> left_image_sub;
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> right_image_sub;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;
    
    std::string left_input_topic;
    std::string right_input_topic;
    std::string left_output_topic;
    std::string right_output_topic;
    double timestamp_tolerance;
    
    // 相机参数
    cv::Mat K_left, K_right;   // 内参矩阵
    cv::Mat D_left, D_right;   // 畸变系数
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_fisheye_undistort_node");
    
    // 设置日志级别
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info)) {
        ros::console::notifyLoggerLevelsChanged();
    }
    
    StereoFisheyeUndistortNode node;
    ros::spin();
    return 0;
}