#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

using namespace std;

string octomap_topic_out, pcl_topic_in;
float kOctomapResolution;

ros::Publisher k_octomap_pub;

void cloud_to_octomap_callback(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg) {
  
  auto kOctoTree = std::make_shared<octomap::ColorOcTree>(kOctomapResolution);

  if (pointCloudMsg->data.size()) {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr globalCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::fromROSMsg(*pointCloudMsg, *globalCloud);

    kOctoTree->clear();
    for (auto p : globalCloud->points) {
      kOctoTree->updateNode(octomap::point3d(p.x, p.y, p.z), true);
      kOctoTree->integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
    }
    kOctoTree->updateInnerOccupancy();
  } else {
    ROS_WARN("no point in cloud to octomap");
  }

  octomap_msgs::Octomap octomapMsg;
  octomap_msgs::fullMapToMsg(*kOctoTree, octomapMsg);
  octomapMsg.header.frame_id = "world";
  octomapMsg.header.stamp = ros::Time::now();
  k_octomap_pub.publish(octomapMsg);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pointcloud_to_octomap");
  ros::NodeHandle nh;
    
  nh.param<float>("resolution", kOctomapResolution, 0.05f);
  nh.param<string>("octomap_topic_out", octomap_topic_out, "/octomap_topic_out");
  nh.param<string>("pcl_topic_in", pcl_topic_in, "/pcl_topic_in");
  ros::Subscriber sub = nh.subscribe(pcl_topic_in, 1, cloud_to_octomap_callback);
  k_octomap_pub = nh.advertise<octomap_msgs::Octomap>(octomap_topic_out, 1);
  ros::spin();

  return 0;
}
