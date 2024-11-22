#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>

using namespace std;

main(int argc, char **argv)
{
    ros::init(argc, argv, "points_publisher");
    ros::NodeHandle nh("~");
    std::string pcl_topic_out;
    nh.param<string>("pcl_topic_out", pcl_topic_out,"/drone_msg/pcl_groundtruth");
    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(pcl_topic_out, 1);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    sensor_msgs::PointCloud2 output;


    std::string pcd_path;
    if (nh.getParam("pcd_path", pcd_path)) {
        ROS_INFO("Get the pcd_path : %s", pcd_path.c_str());
    } else {
        ROS_WARN("didn't find parameter pcd_path, use the default path");
        std::string ros_path = ros::package::getPath("simulation_gazebo");
        pcd_path = ros_path+"/maps/obstacle.pcd";
    }

    pcl::io::loadPCDFile(pcd_path, cloud);
    //Convert the cloud to ROS message
    pcl::toROSMsg(cloud, output);
    output.header.frame_id = "world"; //this has been done in order to be able to visualize our PointCloud2 message on the RViz visualizer
    ros::Rate loop_rate(2.0);
    while (ros::ok())
    {
        pcl_pub.publish(output);
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
