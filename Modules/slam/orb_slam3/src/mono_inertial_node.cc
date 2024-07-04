/**
* 
* Adapted from ORB-SLAM3: Examples/ROS/src/ros_mono_inertial.cc
*
*/

#include "common.h"

using namespace std;

bool whether_publish_tf_transform;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();
    void publish();

    queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBufMutex;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    std::deque<geometry_msgs::PoseStamped> pose_msgs;
    std::mutex mBufMutexPose;

    bool updated = false;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono_Inertial");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are neglected.");
    }

    ros::NodeHandle node_handler;
    std::string node_name = ros::this_node::getName();
    image_transport::ImageTransport image_transport(node_handler);

    std::string voc_file, settings_file;
    node_handler.param<std::string>(node_name + "/voc_file", voc_file, "file_not_set");
    node_handler.param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");

    if (voc_file == "file_not_set" || settings_file == "file_not_set")
    {
        ROS_ERROR("Please provide voc_file and settings_file in the launch file");       
        ros::shutdown();
        return 1;
    }

    node_handler.param<std::string>(node_name + "/map_frame_id", map_frame_id, "map");
    node_handler.param<std::string>(node_name + "/pose_frame_id", pose_frame_id, "pose");

    bool bEqual = false;
    node_handler.param<bool>(node_name + "/do_equalize", bEqual, false);
    bool bUseViewer = false;
    node_handler.param<bool>(node_name + "/use_viewer", bUseViewer, false);
    
    node_handler.param<bool>(node_name + "/publish_tf_transform", whether_publish_tf_transform, false);
    
    node_handler.param<float>(node_name + "/delay", delay, 0.0);
    
    // interpolation
    node_handler.param<bool>(node_name + "/interpolation", interpolation, false);
    node_handler.param<float>(node_name + "/interpolation_rate", interpolation_rate, 50.0);
    node_handler.param<int>(node_name + "/interpolation_order", interpolation_order, 2);
    node_handler.param<int>(node_name + "/interpolation_sample_num", interpolation_sample_num, 4);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(voc_file, settings_file, ORB_SLAM3::System::IMU_MONOCULAR, bUseViewer);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, bEqual);

    ros::Subscriber sub_imu = node_handler.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
    ros::Subscriber sub_img0 = node_handler.subscribe("/camera/image_raw", 100, &ImageGrabber::GrabImage, &igb);

    setup_ros_publishers(node_handler, image_transport);

    setup_tf_orb_to_ros(ORB_SLAM3::System::IMU_MONOCULAR);
    setup_interpolation();

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    std::thread sync_thread_pub(&ImageGrabber::publish, &igb); 

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutex.lock();
    if (!img0Buf.empty())
        img0Buf.pop();
    img0Buf.push(img_msg);
    mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    
    if(cv_ptr->image.type()==0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void ImageGrabber::SyncWithImu()
{
    while(1)
    {
        if (!img0Buf.empty()&&!mpImuGb->imuBuf.empty())
        {
            cv::Mat im;
            double tIm = 0;
            ros::Time current_frame_time;

            tIm = img0Buf.front()->header.stamp.toSec();
            if(tIm>mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;
            
            {
            this->mBufMutex.lock();
            im = GetImage(img0Buf.front());
            current_frame_time = img0Buf.front()->header.stamp;
            img0Buf.pop();
            this->mBufMutex.unlock();
            }

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tIm)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if(mbClahe)
                mClahe->apply(im,im);

            // Main algorithm runs here
            cv::Mat Tcw;
            Sophus::SE3f Tcw_SE3f = mpSLAM->TrackMonocular(im, tIm, vImuMeas);
            Eigen::Matrix4f Tcw_Matrix = Tcw_SE3f.matrix();
            cv::eigen2cv(Tcw_Matrix, Tcw);
            // cv::Mat Tcw = mpSLAM->TrackMonocular(im, tIm, vImuMeas);

            if (!Tcw.empty())
            {
                tf::Transform tf_transform = from_orb_to_ros_tf_transform (Tcw);

                if(whether_publish_tf_transform) 
                {
                    static tf::TransformBroadcaster tf_broadcaster;
                    tf_broadcaster.sendTransform(tf::StampedTransform(tf_transform, current_frame_time, map_frame_id, pose_frame_id));
                }

                tf::Stamped<tf::Pose> grasp_tf_pose(tf_transform, current_frame_time, map_frame_id);

                geometry_msgs::PoseStamped pose_msg;

                tf::poseStampedTFToMsg(grasp_tf_pose, pose_msg);
                
                this->mBufMutexPose.lock();
                pose_msgs.push_back(pose_msg);
                while(pose_msgs.size() > interpolation_sample_num)
                {
                    pose_msgs.pop_front();
                }
                updated = true;
                this->mBufMutexPose.unlock();
                // pose_pub.publish(pose_msg);
            }

            publish_ros_tracking_mappoints(mpSLAM->GetTrackedMapPoints(), current_frame_time);

            // publish_ros_tracking_img(mpSLAM->GetCurrentFrame(), current_frame_time);
        }

        std::chrono::milliseconds tSleep(1);
        std::this_thread::sleep_for(tSleep);
    }
}

void ImageGrabber::publish()
{
    while(1)
    {
        mBufMutexPose.lock();
        publish_ros_poseStamped(pose_msgs, updated);
        mBufMutexPose.unlock();
        std::chrono::milliseconds tSleep(20);
        std::this_thread::sleep_for(tSleep);
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}
