#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <deque>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <chrono>

class StereoSGBMNode {
public:
    StereoSGBMNode(): nh("~") {

        // 创建发布者
        disparity_pub = nh.advertise<sensor_msgs::Image>("/disparity", 10);

        // 获取参数
        nh.param<bool>("use_stereo_sgbm", use_stereo_sgbm, true);
        nh.param<bool>("is_fisheye", is_fisheye, false);
        nh.param<std::string>("left_image_topic", left_image_topic, "/camera/infra1/image_rect_raw");
        nh.param<std::string>("right_image_topic", right_image_topic, "/camera/infra2/image_rect_raw");
        nh.param<double>("timestamp_tolerance", timestamp_tolerance, 0.01); // 时间戳容忍度，默认10ms

        // 鱼眼相机参数
        if (is_fisheye) {
            // 获取相机内参矩阵
            std::vector<double> left_k_vec, right_k_vec;
            std::vector<double> left_d_vec, right_d_vec;
            
            if (nh.getParam("left_k", left_k_vec) && 
                nh.getParam("right_k", right_k_vec) &&
                nh.getParam("left_d", left_d_vec) && 
                nh.getParam("right_d", right_d_vec)) {
                
                // 初始化内参矩阵
                K_left = cv::Mat(3, 3, CV_64F, left_k_vec.data()).clone();
                K_right = cv::Mat(3, 3, CV_64F, right_k_vec.data()).clone();
                
                // 初始化畸变系数
                D_left = cv::Mat(1, 4, CV_64F, left_d_vec.data()).clone();
                D_right = cv::Mat(1, 4, CV_64F, right_d_vec.data()).clone();
                
                ROS_INFO("Loaded fisheye camera parameters");
            } else {
                ROS_WARN("Fisheye mode enabled but camera parameters not provided. Will use standard stereo.");
                is_fisheye = false;
            }
        }

        // 设置图像订阅者，使用message_filters实现更好的同步
        left_image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, left_image_topic, 10));
        right_image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, right_image_topic, 10));

        // 使用近似时间同步策略
        sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *left_image_sub, *right_image_sub));
        sync->registerCallback(boost::bind(&StereoSGBMNode::imageCallback, this, _1, _2));

        // 创建 StereoSGBM 对象
        int nDispFactor = 2;
        int window_size = 11;
        int min_disp = 0;
        int num_disp = 16 * nDispFactor - min_disp;

        if (use_stereo_sgbm){
            stereoSGBM = cv::StereoSGBM::create(
                min_disp,
                num_disp,
                window_size,
                8 * 1 * window_size * window_size,
                32 * 1 * window_size * window_size,
                0,
                63,
                10,
                200,
                16 * 2,
                cv::StereoSGBM::MODE_SGBM
            );
            ROS_INFO("Using StereoSGBM algorithm");
        }else{
            stereoBM = cv::StereoBM::create(16 * nDispFactor, 21);
            ROS_INFO("Using StereoBM algorithm");
        }

        ROS_INFO("StereoSGBMNode initialized:");
        ROS_INFO("  Left image topic: %s", left_image_topic.c_str());
        ROS_INFO("  Right image topic: %s", right_image_topic.c_str());
        ROS_INFO("  Timestamp tolerance: %f seconds", timestamp_tolerance);
        ROS_INFO("  Algorithm: %s", use_stereo_sgbm ? "SGBM" : "BM");
        ROS_INFO("  Camera type: %s", is_fisheye ? "Fisheye" : "Standard");
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& left_msg, 
                       const sensor_msgs::ImageConstPtr& right_msg) {
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ROS_DEBUG("Received images with timestamps: left=%f, right=%f, diff=%f", 
                  left_msg->header.stamp.toSec(), 
                  right_msg->header.stamp.toSec(), 
                  abs((left_msg->header.stamp - right_msg->header.stamp).toSec()));

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

        // 打印关键图像信息
        ROS_INFO_ONCE("Image info - Size: %dx%d, Type: %s", 
                      left_image.cols, left_image.rows, 
                      left_image.type() == CV_8UC1 ? "mono8" : "other");

        // 如果是鱼眼相机，先进行校正
        cv::Mat left_corrected, right_corrected;
        if (is_fisheye) {
            // 鱼眼相机校正
            cv::fisheye::undistortImage(left_image, left_corrected, K_left, D_left, K_left, left_image.size());
            cv::fisheye::undistortImage(right_image, right_corrected, K_right, D_right, K_right, right_image.size());
        } else {
            // 标准相机图像已经是校正的
            left_corrected = left_image;
            right_corrected = right_image;
        }

        // 计算视差图
        auto compute_start = std::chrono::high_resolution_clock::now();
        cv::Mat disparity;
        if (use_stereo_sgbm) {
            stereoSGBM->compute(left_corrected, right_corrected, disparity);
        } else {
            stereoBM->compute(left_corrected, right_corrected, disparity);
        }
        auto compute_end = std::chrono::high_resolution_clock::now();
        auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);

        // 归一化视差图以便显示
        auto normalize_start = std::chrono::high_resolution_clock::now();
        cv::Mat disparity_norm;
        cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        auto normalize_end = std::chrono::high_resolution_clock::now();
        auto normalize_duration = std::chrono::duration_cast<std::chrono::microseconds>(normalize_end - normalize_start);

        // 发布ROS消息
        auto publish_start = std::chrono::high_resolution_clock::now();
        sensor_msgs::ImagePtr disparity_msg = cv_bridge::CvImage(left_msg->header, "mono8", disparity_norm).toImageMsg();
        disparity_pub.publish(disparity_msg);
        auto publish_end = std::chrono::high_resolution_clock::now();
        auto publish_duration = std::chrono::duration_cast<std::chrono::microseconds>(publish_end - publish_start);

        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // 打印处理统计信息
        ROS_INFO("Processing time breakdown:");
        ROS_INFO("  Total:     %6.2f ms", total_duration.count() / 1000.0);
        ROS_INFO("  Compute:   %6.2f ms (%5.1f%%)", compute_duration.count() / 1000.0, 
                    100.0 * compute_duration.count() / total_duration.count());
        ROS_INFO("  Normalize: %6.2f ms (%5.1f%%)", normalize_duration.count() / 1000.0, 
                    100.0 * normalize_duration.count() / total_duration.count());
        ROS_INFO("  Publish:   %6.2f ms (%5.1f%%)", publish_duration.count() / 1000.0, 
                    100.0 * publish_duration.count() / total_duration.count());
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

    ros::NodeHandle nh;
    ros::Publisher disparity_pub;

    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> left_image_sub;
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> right_image_sub;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

    cv::Ptr<cv::StereoBM> stereoBM;
    cv::Ptr<cv::StereoSGBM> stereoSGBM;
    bool use_stereo_sgbm;
    bool is_fisheye;
    std::string left_image_topic;
    std::string right_image_topic;
    double timestamp_tolerance;
    
    // 鱼眼相机参数
    cv::Mat K_left, K_right;  // 内参矩阵
    cv::Mat D_left, D_right;  // 畸变系数
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_sgbm_node");
    
    // 设置日志级别
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info)) {
        ros::console::notifyLoggerLevelsChanged();
    }
    
    StereoSGBMNode node;
    ros::spin();
    return 0;
}