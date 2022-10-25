#include "points_filter.h"

using namespace Points_Filter;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "points_filter_node");
  ros::NodeHandle nh("~");

  PointsFilter points_filter;
  points_filter.init(nh);

  ros::spin();

  return 0;
}

namespace Points_Filter
{

//--------------------------------------- Helper -----------------------------------------------
// R = R_z(psi) * R_y(theta) * R_x(phi)
Eigen::Matrix3f get_rotation_matrix(float phi, float theta, float psi)
{
  Eigen::Matrix3f Rota_Mat;
  float r11 = cos(theta)*cos(psi);
  float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
  float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
  float r21 = cos(theta)*sin(psi);
  float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
  float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
  float r31 = - sin(theta);
  float r32 = sin(phi)*cos(theta);
  float r33 = cos(phi)*cos(theta); 
  Rota_Mat << r11,r12,r13,r21,r22,r23,r31,r32,r33;
  return Rota_Mat;
}
 
void PointsFilter::init(ros::NodeHandle& nh)
{
  nh.param("points_filter/sensor_max_range", sensor_max_range, 3.0f);
  nh.param("points_filter/resolution", resolution, 0.2f);
  // Data type: 1: 2d laser <sensor_msgs::LaserScan>, 2: 3d points <sensor_msgs::PointCloud2>
  nh.param("points_filter/data_type", map_input, 2);

  // remove the ground
  nh.param("points_filter/ground_removal", flag_pcl_ground_removal, true);
  nh.param("points_filter/ground_height", max_ground_height, 0.3f);
  nh.param("points_filter/downsampling", downsampling, true);
  nh.param("points_filter/spatial", spatial, true);
  nh.param("points_filter/concatenate", concatenate, false);
  // frame name
  nh.param<string>("points_filter/frame_name", frame_name, "/world");
  if(map_input==1){
    nh.param<string>("points_filter/object_link_name", object_link_name, "/lidar_link");
  }else if(map_input==2){
    // rgbd ? 3d lidar
    nh.param("points_filter/is_rgbd", is_rgbd, true); 
    nh.param("points_filter/is_lidar", is_lidar, false); 
    if(is_rgbd){
      nh.param<string>("points_filter/object_link_name", object_link_name, "/realsense_camera_link");
    }else if(is_lidar){
      nh.param<string>("points_filter/object_link_name", object_link_name, "/3Dlidar_link");
    }
  }else if(map_input==3){
    nh.param<string>("points_filter/object_link_name", object_link_name, "/realsense_camera_link");
    // camera param
    nh.param("points_filter/dist_min", dist_min, 0.1);
    nh.param("points_filter/cut_edge", cut_edge, 0);
    nh.param("points_filter/interval", interval, 1);
  }
  
  // [sub]
  if (map_input == 1)
    local_point_clound_sub = nh.subscribe<sensor_msgs::LaserScan>("/points_filter/sensor", 1, &PointsFilter::Callback_2dlaserscan,this);
  else if (map_input == 2)
    local_point_clound_sub = nh.subscribe<sensor_msgs::PointCloud2>("/points_filter/sensor", 1, &PointsFilter::Callback_3dpointcloud,this);
  else if (map_input == 3){
    local_point_clound_sub = nh.subscribe<sensor_msgs::Image>("/camera/depth/image_rect_raw", 1, &PointsFilter::Callback_depthimage,this);
    camera_info_sub = nh.subscribe<sensor_msgs::CameraInfo>("/camera/depth/camera_info", 1, &PointsFilter::Callback_depthinfo,this);
  }
  // [pub]
  point_cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/local_obs_pcl", 10); 

  ros::spin();
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> callback function <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void PointsFilter::Callback_2dlaserscan(const sensor_msgs::LaserScanConstPtr &msg)
{    
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now(); 

  tf::StampedTransform transform;
  try{
    tfListener.waitForTransform(frame_name,object_link_name,msg->header.stamp,ros::Duration(4.0));
    tfListener.lookupTransform(frame_name, object_link_name, msg->header.stamp, transform);
  }
  catch (tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }
	
  tf::Quaternion q = transform.getRotation();
  tf::Vector3 Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
  tf::Matrix3x3 R_Body_to_ENU(q);
  //double roll,pitch,yaw;
  //tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
  //Eigen::Matrix3f R_Body_to_ENU = get_rotation_matrix(roll, pitch, yaw);
    
  sensor_msgs::LaserScan::ConstPtr _laser_scan;
  _laser_scan = msg;

  pcl::PointCloud<pcl::PointXYZ> _pointcloud;
  _pointcloud.clear();
    
  pcl::PointXYZ newPoint;
  tf::Vector3 _laser_point_body_body_frame,_laser_point_body_ENU_frame;
    
  double newPointAngle;
  int beamNum = _laser_scan->ranges.size();
  for (int i = 0; i < beamNum; i++)
  {
    double range = _laser_scan->ranges[i];
    if(range < 0.3) {continue;}
    if(range > sensor_max_range) {range = 1.2*sensor_max_range;}
    newPointAngle = _laser_scan->angle_min + _laser_scan->angle_increment * i;
    _laser_point_body_body_frame[0] = range * cos(newPointAngle);
    _laser_point_body_body_frame[1] = range * sin(newPointAngle);
    _laser_point_body_body_frame[2] = 0.0;
    _laser_point_body_ENU_frame = R_Body_to_ENU * _laser_point_body_body_frame;
    newPoint.x = Origin.getX() + _laser_point_body_ENU_frame[0];
    newPoint.y = Origin.getY() + _laser_point_body_ENU_frame[1];
    newPoint.z = Origin.getZ() + _laser_point_body_ENU_frame[2];
        
    _pointcloud.push_back(newPoint);
  }

  /* Filter out point cloud */
  if(_pointcloud.size()>0) 
  {
    if(flag_pcl_ground_removal){
      pcl::PassThrough<pcl::PointXYZ> ground_removal;
      ground_removal.setInputCloud (_pointcloud.makeShared());
      ground_removal.setFilterFieldName ("z");
      ground_removal.setFilterLimits (-10.0, max_ground_height);
      ground_removal.setFilterLimitsNegative (true);
      ground_removal.filter (_pointcloud);
    }

    if(concatenate)
      local_point_cloud += _pointcloud;
    else
      local_point_cloud = _pointcloud;

    /* Downsampling for all */
    if(downsampling){
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(local_point_cloud.makeShared());
      sor.setLeafSize(resolution, resolution, resolution);
      sor.filter(local_point_cloud);
    }
  }
	
  /* Nearby local point cloud */
  if(spatial){
    pcl::PassThrough<pcl::PointXYZ> sensor_range;
    sensor_range.setInputCloud (local_point_cloud.makeShared());
    sensor_range.setFilterFieldName ("x");
    sensor_range.setFilterLimits (Origin.getX()-sensor_max_range, Origin.getX()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
    sensor_range.setInputCloud (local_point_cloud.makeShared());
    sensor_range.setFilterFieldName ("y");
    sensor_range.setFilterLimits (Origin.getY()-sensor_max_range, Origin.getY()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
  }

  local_point_cloud.header.stamp = _laser_scan->header.stamp.toSec()*1e6;
  local_point_cloud.header.seq = _laser_scan->header.seq;	 	
  local_point_cloud.header.frame_id = "world";
  point_cloud_pub.publish(local_point_cloud);

    static int exec_num = 0;
    exec_num++;
    if(exec_num == 100)
    {
    	std::chrono::duration<double, std::milli> elapsed_seconds = std::chrono::system_clock::now() - start; 
        printf("[PCL_reader]: point_cloud processing takes %f [ms].\n", elapsed_seconds.count());
        exec_num=0;
    }
}

void PointsFilter::Callback_3dpointcloud(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now(); 

  tf::Quaternion q;
  tf::Vector3 Origin;
    
  tf::StampedTransform transform;
  if (is_rgbd)
    try{
      tfListener.waitForTransform(frame_name,object_link_name,msg->header.stamp,ros::Duration(4.0));
      tfListener.lookupTransform(frame_name, object_link_name, msg->header.stamp, transform);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      return;
    }

  if (is_lidar)
    try{
      tfListener.waitForTransform(frame_name,object_link_name,msg->header.stamp,ros::Duration(4.0));
      tfListener.lookupTransform(frame_name, object_link_name, msg->header.stamp, transform);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      return;
    }

  if(is_rgbd || is_lidar)
  {
    q = transform.getRotation();
    Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
  }
  else
  {
    q = tf::Quaternion(0.0,0.0,0.0,1.0);
    Origin = tf::Vector3(0.0,0.0,0.0);
  }
  tf::Matrix3x3 R_Body_to_ENU(q);
    
  pcl::fromROSMsg(*msg, latest_local_pcl_);
  int latest_local_pcl_size = (int)latest_local_pcl_.size();

  pcl::PointCloud<pcl::PointXYZ> _pointcloud;
  _pointcloud.clear();
    
  pcl::PointXYZ newPoint;
    
  tf::Vector3 point_body_body_frame, point_body_ENU_frame;
    
  /* Filter out point cloud */
  if(latest_local_pcl_size>0) 
  {
    for (int i = 0; i < latest_local_pcl_size; i++)
    {
      point_body_body_frame[0] = latest_local_pcl_.points[i].x;
      point_body_body_frame[1] = latest_local_pcl_.points[i].y;
      point_body_body_frame[2] = latest_local_pcl_.points[i].z;
      point_body_ENU_frame = R_Body_to_ENU * point_body_body_frame;
      newPoint.x = Origin.getX() + point_body_ENU_frame[0];
      newPoint.y = Origin.getY() + point_body_ENU_frame[1];
      newPoint.z = Origin.getZ() + point_body_ENU_frame[2];
        
      _pointcloud.push_back(newPoint);
    }
    /* Ground removal */
    if(flag_pcl_ground_removal){
      pcl::PassThrough<pcl::PointXYZ> ground_removal;
      ground_removal.setInputCloud (_pointcloud.makeShared());
      ground_removal.setFilterFieldName ("z");
      ground_removal.setFilterLimits (-10.0, max_ground_height);
      ground_removal.setFilterLimitsNegative (true);
      ground_removal.filter (_pointcloud);
    }
    
    if(concatenate)
      local_point_cloud += _pointcloud;
    else
      local_point_cloud = _pointcloud;

    /* Downsampling for all */
    if(downsampling){
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(local_point_cloud.makeShared());
      sor.setLeafSize(resolution, resolution, resolution);
      sor.filter(local_point_cloud);
    }
  }
	
  /* Nearby local point cloud */
  if(spatial){
    pcl::PassThrough<pcl::PointXYZ> sensor_range;
    sensor_range.setInputCloud (local_point_cloud.makeShared());
    sensor_range.setFilterFieldName ("x");
    sensor_range.setFilterLimits (Origin.getX()-sensor_max_range, Origin.getX()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
    sensor_range.setInputCloud (local_point_cloud.makeShared());
     sensor_range.setFilterFieldName ("y");
    sensor_range.setFilterLimits (Origin.getY()-sensor_max_range, Origin.getY()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
  }

  local_point_cloud.header = latest_local_pcl_.header;
  local_point_cloud.header.frame_id = "world";
//  local_point_cloud.height = 1;
//  local_point_cloud.width = local_point_cloud.points.size();
  point_cloud_pub.publish(local_point_cloud);

    static int exec_num = 0;
    exec_num++;
    if(exec_num == 100)
    {
    	std::chrono::duration<double, std::milli> elapsed_seconds = std::chrono::system_clock::now() - start; 
        printf("[PCL_reader]: point_cloud processing takes %f [ms].\n", elapsed_seconds.count());
        exec_num=0;
    }
}

void PointsFilter::Callback_depthinfo(const sensor_msgs::CameraInfoConstPtr &info)
{
  fx = info->K[0];
  cx = info->K[2];
  fy = info->K[4];
  cy = info->K[5];
  
  depth_width = info->width;
  depth_height = info->height;
}

void PointsFilter::Callback_depthimage(const sensor_msgs::ImageConstPtr &img)
{
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now(); 

  /* get camera pose */
  tf::Quaternion q;
  tf::Vector3 Origin;
    
  tf::StampedTransform transform;
  try{
    tfListener.waitForTransform(frame_name,object_link_name,img->header.stamp,ros::Duration(4.0));
    tfListener.lookupTransform(frame_name, object_link_name, img->header.stamp, transform);
  }
  catch (tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
    return;
  }

  q = transform.getRotation();
  Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
  tf::Matrix3x3 R_Body_to_ENU(q);

  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 1000.0);
  }
  cv_ptr->image.copyTo(depth_image_);

  /* get points */
  pcl::PointCloud<pcl::PointXYZ> _pointcloud;
  _pointcloud.clear();
  
  pcl::PointXYZ newPoint;
  
  tf::Vector3 point_body_body_frame, point_body_ENU_frame;

  uint16_t *row_ptr;
  double depth;
  
  for (int v = cut_edge; v < depth_height - cut_edge; v += interval)
  {
    row_ptr = depth_image_.ptr<uint16_t>(v) + cut_edge;

    for (int u = cut_edge; u < depth_width - cut_edge; u += interval)
    {
      depth = (*row_ptr) * 0.001; // mm -> m
      row_ptr += interval;

      if (*row_ptr == 0 || depth == 1 || depth > sensor_max_range)
      {
        //depth = sensor_max_range + 0.1;
        continue;
      }
      else if (depth < dist_min)
      {
        continue;
      }

      // project to ENU frame
      point_body_body_frame[0] = (u - cx) * depth / fx;
      point_body_body_frame[1] = (v - cy) * depth / fy;
      point_body_body_frame[2] = depth;

      point_body_ENU_frame = R_Body_to_ENU * point_body_body_frame;
      newPoint.x = Origin.getX() + point_body_ENU_frame[0];
      newPoint.y = Origin.getY() + point_body_ENU_frame[1];
      newPoint.z = Origin.getZ() + point_body_ENU_frame[2];

      _pointcloud.push_back(newPoint);
    }
  }
  
  /* Ground removal */
  if(flag_pcl_ground_removal){
    pcl::PassThrough<pcl::PointXYZ> ground_removal;
    ground_removal.setInputCloud (_pointcloud.makeShared());
    ground_removal.setFilterFieldName ("z");
    ground_removal.setFilterLimits (-10.0, max_ground_height);
    ground_removal.setFilterLimitsNegative (true);
    ground_removal.filter (_pointcloud);
  }
    
  if(concatenate)
    local_point_cloud += _pointcloud;
  else
    local_point_cloud = _pointcloud;

  /* Downsampling for all */
  if(downsampling){
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(local_point_cloud.makeShared());
    sor.setLeafSize(resolution, resolution, resolution);
    sor.filter(local_point_cloud);
  }
    
  /* Nearby local point cloud */
  if(spatial){
    pcl::PassThrough<pcl::PointXYZ> sensor_range;
    sensor_range.setInputCloud (local_point_cloud.makeShared());
    sensor_range.setFilterFieldName ("x");
    sensor_range.setFilterLimits (Origin.getX()-sensor_max_range, Origin.getX()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
    sensor_range.setInputCloud (local_point_cloud.makeShared());
    sensor_range.setFilterFieldName ("y");
    sensor_range.setFilterLimits (Origin.getY()-sensor_max_range, Origin.getY()+sensor_max_range);
    sensor_range.filter (local_point_cloud);
  }
  
  local_point_cloud.header.stamp = img->header.stamp.toSec()*1e6;
  local_point_cloud.header.seq = img->header.seq;
  local_point_cloud.header.frame_id = "world";
//  local_point_cloud.height = 1;
//  local_point_cloud.width = local_point_cloud.points.size();
  point_cloud_pub.publish(local_point_cloud);

    static int exec_num = 0;
    exec_num++;
    if(exec_num == 100)
    {
    	std::chrono::duration<double, std::milli> elapsed_seconds = std::chrono::system_clock::now() - start; 
        printf("[PCL_reader]: point_cloud processing takes %f [ms].\n", elapsed_seconds.count());
        exec_num=0;
    }
}
}
