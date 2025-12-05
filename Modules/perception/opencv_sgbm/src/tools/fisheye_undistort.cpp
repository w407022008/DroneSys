#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class FisheyeUndistortNode {
public:
    FisheyeUndistortNode(): nh("~") {
        // 获取参数
        nh.param<std::string>("input_topic", input_topic, "/camera/image_raw");
        nh.param<std::string>("output_topic", output_topic, "/camera/image_undistorted");
        nh.param<double>("timestamp_tolerance", timestamp_tolerance, 0.01);
        
        // 获取相机内参矩阵和畸变系数
        std::vector<double> k_vec, d_vec;
        if (!nh.getParam("k", k_vec) || !nh.getParam("d", d_vec)) {
            ROS_ERROR("Camera parameters (k, d) not provided!");
            exit(-1);
        }
        
        // 初始化内参矩阵和畸变系数
        K = cv::Mat(3, 3, CV_64F, k_vec.data()).clone();
        D = cv::Mat(1, 4, CV_64F, d_vec.data()).clone();
        
        // 设置图像订阅者
        image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, input_topic, 10));
        
        // 注册回调函数
        image_sub->registerCallback(boost::bind(&FisheyeUndistortNode::imageCallback, this, _1));
        
        // 创建发布者
        undistorted_pub = nh.advertise<sensor_msgs::Image>(output_topic, 10);
        
        ROS_INFO("FisheyeUndistortNode initialized:");
        ROS_INFO("  Input topic: %s", input_topic.c_str());
        ROS_INFO("  Output topic: %s", output_topic.c_str());
        ROS_INFO("  Camera matrix K: [%.3f, %.3f, %.3f; %.3f, %.3f, %.3f; %.3f, %.3f, %.3f]", 
                 K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
                 K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
                 K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2));
        ROS_INFO("  Distortion coefficients D: [%.3f, %.3f, %.3f, %.3f]", 
                 D.at<double>(0,0), D.at<double>(0,1), D.at<double>(0,2), D.at<double>(0,3));
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        // 转换图像
        cv::Mat image;
        try {
            image = cv_bridge::toCvShare(msg, "mono8")->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        // 鱼眼去畸变，保留完整图像并填充黑色区域
        cv::Mat undistorted_image;
        cv::fisheye::undistortImage(image, undistorted_image, K, D, K, image.size());
        
        // 发布去畸变后的图像
        sensor_msgs::ImagePtr undistorted_msg = cv_bridge::CvImage(msg->header, "mono8", undistorted_image).toImageMsg();
        undistorted_pub.publish(undistorted_msg);
        
        ROS_DEBUG("Processed image with size %dx%d", image.cols, image.rows);
    }

private:
    ros::NodeHandle nh;
    ros::Publisher undistorted_pub;
    
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub;
    
    std::string input_topic;
    std::string output_topic;
    double timestamp_tolerance;
    
    // 相机参数
    cv::Mat K;  // 内参矩阵
    cv::Mat D;  // 畸变系数
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "fisheye_undistort_node");
    
    // 设置日志级别
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info)) {
        ros::console::notifyLoggerLevelsChanged();
    }
    
    FisheyeUndistortNode node;
    ros::spin();
    return 0;
}