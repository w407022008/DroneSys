#include <iostream>
#include <sstream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
// #include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
// #include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include "SemiGlobalMatching.h"
#include <chrono>
using namespace std::chrono;

ros::Publisher k_depth_pub;
// SGM匹配参数设计
SemiGlobalMatching::SGMOption sgm_option;

void processStereo(const sensor_msgs::ImageConstPtr& msgLeft, const sensor_msgs::ImageConstPtr& msgRight)
{
    // 使用CvShare将ROS的图像转换为opencv的Mat
    cv::Mat leftImage, rightImage;
    double x,y,z,qx,qy,qz,qw;
    double timestamp;
    try
    {
        leftImage = cv_bridge::toCvShare(msgLeft)->image;
        rightImage = cv_bridge::toCvShare(msgRight)->image;

        if (leftImage.data == nullptr || rightImage.data == nullptr) {
            std::cout << "读取影像失败！" << std::endl;
            return;
        }
        if (leftImage.rows != rightImage.rows || leftImage.cols != rightImage.cols) {
            std::cout << "左右影像尺寸不一致！" << std::endl;
            return;
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if(leftImage.channels() == 3)
    {
        cv::cvtColor(leftImage, leftImage, CV_BGR2GRAY);
        cv::cvtColor(rightImage, rightImage, CV_BGR2GRAY);
    }


    //···············································································//
    const int32_t width = static_cast<uint32_t>(leftImage.cols);
    const int32_t height = static_cast<uint32_t>(rightImage.rows);

    // 左右影像的灰度数据
    auto bytes_left = new uint8_t[width * height];
    auto bytes_right = new uint8_t[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = leftImage.at<uint8_t>(i, j);
            bytes_right[i * width + j] = rightImage.at<uint8_t>(i, j);
        }
    }

    // 定义SGM匹配类实例
    SemiGlobalMatching sgm;

    //···············································································//
    // 初始化
	printf("SGM Initializing...\n");
    auto start = std::chrono::steady_clock::now();
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    auto tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("SGM Initializing Done! Timing : %lf s\n", tt.count() / 1000.0);

    //···············································································//
    // 匹配
	printf("SGM Matching...\n");
    start = std::chrono::steady_clock::now();
    // disparity数组保存子像素的视差结果
    cv::Mat disparity = cv::Mat::zeros(height, width, CV_32F);
    if (!sgm.Match(bytes_left, bytes_right, (float *) disparity.data)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return;
    }
    end = std::chrono::steady_clock::now();
    tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("SGM Matching...Done! Timing :   %lf s\n\n", tt.count() / 1000.0);


    sensor_msgs::ImagePtr depthMsg = cv_bridge::CvImage(std_msgs::Header(), "mono16", disparity).toImageMsg();
    depthMsg->header.stamp = msgLeft->header.stamp;
    k_depth_pub.publish(depthMsg);
    
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sgm_node");
    ros::NodeHandle nh;
    // 参数读取
    float fx, baseline;
    nh.param<float>("sgm_node/focal_length", fx, 711.9);
    nh.param<float>("sgm_node/base_line", baseline, 0.12);
    // census窗口类型
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    nh.param<int>("sgm_node/num_paths", sgm_option.num_paths, 4);// 聚合路径数
    nh.param<int32_t>("sgm_node/min_disparity", sgm_option.min_disparity, 0);// 候选视差范围
    nh.param<int32_t>("sgm_node/min_disparity", sgm_option.min_disparity, 128);
    nh.param<bool>("sgm_node/is_check_lr", sgm_option.is_check_lr, true);// 一致性检查
    nh.param<float>("sgm_node/lrcheck_thres", sgm_option.lrcheck_thres, 1.0);
    nh.param<bool>("sgm_node/is_check_unique", sgm_option.is_check_unique, true);// 唯一性约束
    nh.param<float>("sgm_node/uniqueness_ratio", sgm_option.uniqueness_ratio, 0.99);
    nh.param<bool>("sgm_node/is_remove_speckles", sgm_option.is_remove_speckles, true);// 剔除小连通区
    nh.param<int>("sgm_node/min_speckle_aera", sgm_option.min_speckle_aera, 50);
    nh.param<int32_t>("sgm_node/p1", sgm_option.p1, 10);// 惩罚项P1、P2
    nh.param<int32_t>("sgm_node/p2_init", sgm_option.p2_init, 150);
    nh.param<bool>("sgm_node/is_fill_holes", sgm_option.is_fill_holes, false);// 视差图填充,结果并不可靠，若工程，不建议填充，若科研，则可填充
    std::string leftTopic, rightTopic;
    nh.param<std::string>("sgm_node/left_topic", leftTopic, "/cam0/image_raw");
    nh.param<std::string>("sgm_node/right_topic", rightTopic, "/cam1/image_raw");
    //【发布】发布深度图和点云的话题
    k_depth_pub = nh.advertise<sensor_msgs::Image>("/sgm_depth", 1);

    printf("Loading Views...Done!\n");

    // 创建两个订阅者
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, leftTopic, 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, rightTopic, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), left_sub, right_sub);
    sync.registerCallback(boost::bind(&processStereo, _1, _2));

    std::cout << "Storing for point cloud" << std::endl;
    ros::spin();

    return 0;
}
