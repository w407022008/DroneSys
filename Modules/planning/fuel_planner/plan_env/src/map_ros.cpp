#include <plan_env/sdf_map.h>
#include <plan_env/map_ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

#include <fstream>
#define DEBUG
namespace fast_planner {
MapROS::MapROS() {
}

MapROS::~MapROS() {
}

void MapROS::setMap(SDFMap* map) {
  this->map_ = map;
}

void MapROS::init() {
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);

  node_.param("map_ros/esdf_slice_height", esdf_slice_height_, -0.1);
  node_.param("map_ros/visualization_truncate_height", visualization_truncate_height_, -0.1);
  node_.param("map_ros/visualization_truncate_low", visualization_truncate_low_, -0.1);
  node_.param("map_ros/show_occ_time", show_occ_time_, false);
  node_.param("map_ros/show_esdf_time", show_esdf_time_, false);
  node_.param("map_ros/show_all_map", show_all_map_, false);
  node_.param("map_ros/show_local_map", show_local_map_, false);
  node_.param("map_ros/show_unknow_map", show_unknow_map_, false);
  node_.param("map_ros/show_esdf_slice", show_esdf_slice_, false);
  node_.param("map_ros/show_update_range", show_update_range_, false);
  node_.param("map_ros/show_depth_pcl", show_depth_pcl_, false);
  node_.param("map_ros/frame_id", frame_id_, string("world"));
  node_.param("map_ros/log/on", log_on_, false);
  node_.param("map_ros/log/dir", file_path_, string(""));

  node_.param("map_ros/use_d435i_vins", use_d435i_vins_, false);
  node_.param("map_ros/use_d435i_mavros", use_d435i_mavros_, false);
  node_.param("map_ros/use_fpga_vins", use_fpga_vins_, false);
  node_.param("map_ros/use_fpga_mavros", use_fpga_mavros_, false);
  node_.param("map_ros/use_sensors_inSim", use_sensors_inSim_, false);

  // proj_points_.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  cur_point_cloud_.points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  // proj_points_.reserve(640 * 480 / map_->mp_->skip_pixel_ / map_->mp_->skip_pixel_);
  cur_points_cnt = 0;

  grid_need_update_ = false;
  grid_update_count = 0;
  esdf_update_count = 0;
  fuse_time_ = 0.0;
  esdf_time_ = 0.0;
  max_fuse_time_ = 0.0;
  max_esdf_time_ = 0.0;
  fuse_num_ = 0;
  esdf_num_ = 0;
  depth_image_.reset(new cv::Mat);

  rand_noise_ = normal_distribution<double>(0, 0.1);
  random_device rd;
  eng_ = default_random_engine(rd());

  // Timer
  esdf_timer_ = node_.createTimer(ros::Duration(0.01), &MapROS::updateESDFCallback, this);
  // occ_timer_ = node_.createTimer(ros::Duration(0.01), &MapROS::updateGridCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.1), &MapROS::visCallback, this);

  // Puber
  map_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_all", 10);
  map_local_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local", 10);
  map_local_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  // Suber
  depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 50));
  cloud_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/map_ros/cloud", 50));
  pose_sub_.reset(
      new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/map_ros/pose", 25));

  sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
      MapROS::SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
  sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));

  sync_cloud_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyCloudPose>(
      MapROS::SyncPolicyCloudPose(100), *cloud_sub_, *pose_sub_));
  sync_cloud_pose_->registerCallback(boost::bind(&MapROS::cloudPoseCallback, this, _1, _2));

  map_start_time_ = ros::Time::now();
}

void MapROS::visCallback(const ros::TimerEvent& e) {
  if(show_local_map_) publishMapLocal();  // publish local(current pcl frame) range occupied map
  if (show_all_map_) {
    // Limit the frequency of all map
    static double tpass = 0.0;
    tpass += (e.current_real - e.last_real).toSec();
    if (tpass > 0.1) {
      publishMapAll(); // publish global range occupied map
      tpass = 0.0;
    }
  }

  if(show_unknow_map_) publishUnknown();  // publish local(current pcl frame) range unoccupied map
  if(show_esdf_slice_) publishESDFSlice();     // publish local(current pcl frame) range ESDF map slice with set height

  if(show_update_range_) publishUpdateRange();  // visualize ESDF update range
  if(show_depth_pcl_) publishDepth();   // publish depth to points
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (grid_need_update_ || esdf_update_count>=grid_update_count) return;

esdf_update_count++;
// if(esdf_update_count>100){
//   esdf_update_count-=100;
//   grid_update_count-=100;
// }
  auto t1 = ros::Time::now();
  map_->updateESDF3d();
  auto t2 = ros::Time::now();
  map_->copyDistance();
// #ifdef DEBUG
//   std::cout<<"distance buffer:"<<std::endl;
//   for(auto elem:map_->md_->distance_buffer_copy_)
//     std::cout<<elem<<", ";
//   std::cout<<std::endl;
// #endif
  if (show_esdf_time_){
    esdf_time_ += (t2 - t1).toSec();
    max_esdf_time_ = max(max_esdf_time_, (t2 - t1).toSec());
    esdf_num_++;
    ROS_WARN("ESDF t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), esdf_time_ / esdf_num_,
             max_esdf_time_);
  }
}

void MapROS::depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                               const nav_msgs::OdometryConstPtr& pose) {
  camera_pos_(0) = pose->pose.pose.position.x;
  camera_pos_(1) = pose->pose.pose.position.y;
  camera_pos_(2) = pose->pose.pose.position.z;
  if (!map_->isInMap(camera_pos_)){  // exceed mapped region
    ROS_WARN("Current pos exceed mapped region!");
    return;
  }
  
  Eigen::Quaterniond q_;
  q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                                 pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  cv_ptr->image.copyTo(*depth_image_);
  // generate point cloud, update map
  proessDepthImage(camera_pos_, q_);

  grid_need_update_ = true;
  grid_update_count++;
  auto t1 = ros::Time::now();
  map_->inputPointCloud(cur_point_cloud_, cur_points_cnt, camera_pos_);
  auto t2 = ros::Time::now();
  grid_need_update_ = false;

  map_->copyOccupancy();

  if (show_occ_time_){
    fuse_time_ += (t2 - t1).toSec();
    max_fuse_time_ = max(max_fuse_time_, (t2 - t1).toSec());
    fuse_num_ += 1;
    ROS_WARN("Fusion t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), fuse_time_ / fuse_num_,
             max_fuse_time_);
  }

#ifdef DEBUG
  static int cnt=1;
  std::cout<<"[DEBUG]:receive image: "<<cnt++<<", grid update: "<<grid_update_count<<", esdf update: "<<esdf_update_count<<std::endl;
#endif
}

void MapROS::cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                               const nav_msgs::OdometryConstPtr& pose) {
  camera_pos_(0) = pose->pose.pose.position.x;
  camera_pos_(1) = pose->pose.pose.position.y;
  camera_pos_(2) = pose->pose.pose.position.z;
  Eigen::Quaterniond q_;
  q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                                 pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  pcl::fromROSMsg(*msg, cur_point_cloud_); // original in ned
  grid_need_update_ = true;

  grid_update_count++;
  auto t1 = ros::Time::now();
  map_->inputPointCloud(cur_point_cloud_, cur_point_cloud_.points.size(), camera_pos_);
  auto t2 = ros::Time::now();
  grid_need_update_ = false;

  map_->copyOccupancy();

  if (show_occ_time_){
    fuse_time_ += (t2 - t1).toSec();
    max_fuse_time_ = max(max_fuse_time_, (t2 - t1).toSec());
    fuse_num_ += 1;
    ROS_WARN("Fusion t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), fuse_time_ / fuse_num_,
             max_fuse_time_);
  }

}

void MapROS::proessDepthImage(const Eigen::Vector3d& camera_pos, const Eigen::Quaterniond& camera_q) {
  cur_points_cnt = 0;

  uint16_t* row_ptr;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = camera_q.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / k_depth_scaling_factor_;

  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    row_ptr = depth_image_->ptr<uint16_t>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      depth = (*row_ptr) * inv_factor;
      row_ptr = row_ptr + skip_pixel_;

      // // filter depth
      // if (depth > 0.01)
      //   depth += rand_noise_(eng_);

      // TODO: simplify the logic here
      if (*row_ptr == 0 || depth > depth_filter_maxdist_)
        depth = depth_filter_maxdist_;
      else if (depth < depth_filter_mindist_)
        continue;

      if (use_d435i_vins_) {
        // setting for 2.5inch_d435 with running vins
        pt_cur(0) = (u - cx_) * depth / fx_;
        pt_cur(1) = -(v - cy_) * depth / fy_;
        pt_cur(2) = -depth;
        // ROS_WARN("begin transform for d435i vins");
      } else if (use_d435i_mavros_) {
        // setting for mavros
        pt_cur(0) = depth;
        pt_cur(1) = -(u - cx_) * depth / fx_;
        pt_cur(2) = -(v - cy_) * depth / fy_; 
        // ROS_WARN("begin transform for d435i mavros");
      } else if (use_fpga_vins_) {
        // setting for fpga_5inch_vins  
        pt_cur(0) = (v - cy_) * depth / fy_;
        pt_cur(1) = -(u - cx_) * depth / fx_; 
        pt_cur(2) = depth;
        // ROS_WARN("begin transform for fpga vins");
      } else if (use_fpga_mavros_ || use_sensors_inSim_) {
        // setting for fpga_5inch_mavros  
        // seeting for simulation, after changing pose from geometry to odometry in sim
        pt_cur(0) = depth;
        pt_cur(1) = -(u - cx_) * depth / fx_;
        pt_cur(2) = -(v - cy_) * depth / fy_; 
        // ROS_WARN("begin transform for fpga mavros or sim");
      }  

      pt_world = camera_r * pt_cur + camera_pos;
      auto& pt = cur_point_cloud_.points[cur_points_cnt++];
      pt.x = pt_world[0];
      pt.y = pt_world[1];
      pt.z = pt_world[2];
    }
  }

  // publishDepth();
}

void MapROS::publishMapAll() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;
  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        if (map_->getOccupancy(Eigen::Vector3i(x, y, z)) == map_->OCCUPIED) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud1.push_back(pt);
        }
      }
  cloud1.width = cloud1.points.size();
  cloud1.height = 1;
  cloud1.is_dense = true;
  cloud1.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud1, cloud_msg);
  map_all_pub_.publish(cloud_msg);

  // Output time and known volumn
  if(log_on_){
    double time_now = (ros::Time::now() - map_start_time_).toSec();
    double known_volumn = 0;

    for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
      for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
        for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
          if (map_->getOccupancy(Eigen::Vector3i(x, y, z)) > map_->UNKNOWN)
            known_volumn += 0.1 * 0.1 * 0.1;
        }

    ofstream file(file_path_, ios::app);
    file << time_now << ", " << known_volumn << std::endl;
  }
}

void MapROS::publishMapLocal() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointCloud<pcl::PointXYZ> cloud2;
  Eigen::Vector3i min_cut, max_cut;
  map_->getLocalBox(min_cut, max_cut);
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  // for (int z = min_cut(2); z <= max_cut(2); ++z)
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = map_->mp_->box_min_(2); z < map_->mp_->box_max_(2); ++z) {
        if (map_->getOccupancy(Eigen::Vector3i(x, y, z)) == map_->OCCUPIED) {
          // Occupied cells
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
        else if (map_->getInflateOccupancy(Eigen::Vector3i(x, y, z)) == 1)
        {
          // Inflated occupied cells
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_)
            continue;
          if (pos(2) < visualization_truncate_low_)
            continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud2.push_back(pt);
        }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  cloud2.width = cloud2.points.size();
  cloud2.height = 1;
  cloud2.is_dense = true;
  cloud2.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_local_pub_.publish(cloud_msg);
  pcl::toROSMsg(cloud2, cloud_msg);
  map_local_inflate_pub_.publish(cloud_msg);
}

void MapROS::publishUnknown() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  Eigen::Vector3i min_cut, max_cut;
  map_->getLocalBox(min_cut, max_cut);
  map_->boundIndex(max_cut);
  map_->boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (map_->getOccupancy(Eigen::Vector3i(x, y, z)) == map_->UNKNOWN) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void MapROS::publishDepth() {
  cur_point_cloud_.width = cur_points_cnt;
  cur_point_cloud_.height = 1;
  cur_point_cloud_.is_dense = true;
  cur_point_cloud_.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cur_point_cloud_, cloud_msg);
  depth_pub_.publish(cloud_msg);
}

void MapROS::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  Eigen::Vector3i min_cut, max_cut;
  map_->getLocalBox(min_cut, max_cut);
  map_->indexToPos(min_cut, esdf_min_pos);
  map_->indexToPos(max_cut, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;
  mk.header.frame_id = frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;
  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);
  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);
  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

void MapROS::publishESDFSlice() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;
  Eigen::Vector3i min_cut, max_cut;
  map_->getLocalBox(min_cut, max_cut);

  min_cut -= Eigen::Vector3i(map_->mp_->local_map_margin_,
                            map_->mp_->local_map_margin_,
                            map_->mp_->local_map_margin_);
  max_cut += Eigen::Vector3i(map_->mp_->local_map_margin_,
                            map_->mp_->local_map_margin_,
                            map_->mp_->local_map_margin_);
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      Eigen::Vector3d pos;
      map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = esdf_slice_height_;
      dist = map_->getDistance(pos);
      dist = min(dist, max_dist);
      dist = max(dist, min_dist);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}
}