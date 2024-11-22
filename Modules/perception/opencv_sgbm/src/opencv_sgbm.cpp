#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <deque>

class StereoSGBMNode {
public:
    StereoSGBMNode(): nh("~") {

        // 创建发布者
        disparity_pub = nh.advertise<sensor_msgs::Image>("/disparity", 10);

        // 设置图像订阅者
        left_image_sub = nh.subscribe("/camera/infra1/image_rect_raw", 10, &StereoSGBMNode::leftImageCallback, this);
        right_image_sub = nh.subscribe("/camera/infra2/image_rect_raw", 10, &StereoSGBMNode::rightImageCallback, this);

        // 初始化队列
        left_queue = std::deque<std::pair<cv::Mat, ros::Time>>();
        right_queue = std::deque<std::pair<cv::Mat, ros::Time>>();

        // 获取参数选择算法
        nh.param<bool>("use_stereo_sgbm", use_stereo_sgbm, true);

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
            std::cout<<"use stereo_sgbm"<<std::endl;
        }else{
            stereoBM = cv::StereoBM::create(16 * nDispFactor, 21);
            std::cout<<"use stereo_bm"<<std::endl;
        }
    }

    void leftImageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat left_image = cv_bridge::toCvCopy(msg, "mono8")->image;
        left_queue.push_back(std::make_pair(left_image, msg->header.stamp));

        if (!right_queue.empty()) {
            processImages();
        }
    }

    void rightImageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat right_image = cv_bridge::toCvCopy(msg, "mono8")->image;
        right_queue.push_back(std::make_pair(right_image, msg->header.stamp));
    }

    void processImages() {
        auto left_img_pair = left_queue.front();
        auto right_img_pair = right_queue.front();

        // 比较时间戳
        if (std::abs((left_img_pair.second - right_img_pair.second).toNSec()) < 1e6) { // 1 ms
            // 计算视差图
            cv::Mat disparity;
            if (use_stereo_sgbm)
                stereoSGBM->compute(left_img_pair.first, right_img_pair.first, disparity);
            else 
                stereoBM->compute(left_img_pair.first, right_img_pair.first, disparity);

            // 归一化视差图以便显示
            cv::Mat disparity_norm;
            cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

            // 发布ROS消息
            sensor_msgs::ImagePtr disparity_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", disparity_norm).toImageMsg();
            disparity_msg->header.stamp = left_img_pair.second;
            disparity_pub.publish(disparity_msg);

            // 从队列中移除
            left_queue.pop_front();
            right_queue.pop_front();
        }
    }

private:
    ros::NodeHandle nh;
    ros::Publisher disparity_pub;
    ros::Subscriber left_image_sub;
    ros::Subscriber right_image_sub;

    std::deque<std::pair<cv::Mat, ros::Time>> left_queue;
    std::deque<std::pair<cv::Mat, ros::Time>> right_queue;
    cv::Ptr<cv::StereoBM> stereoBM;
    cv::Ptr<cv::StereoSGBM> stereoSGBM;
    bool use_stereo_sgbm;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_sgbm_node");
    StereoSGBMNode node;
    ros::spin();
    return 0;
}
