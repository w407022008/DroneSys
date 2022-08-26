#ifndef _SDF_MAP_H
#define _SDF_MAP_H

#include <visualization_msgs/Marker.h>
#include <Eigen/Eigen>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Odometry.h>


//octomap
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>


using namespace std;

namespace Histo_Planning
{
class SDFMap
{
private:
  // data are saved in vector
  std::vector<int> occupancy_buffer_;  // 0 is free, 1 is occupied
  std::vector<double> distance_buffer_;
  std::vector<double> distance_buffer_neg_;
  std::vector<double> tmp_buffer1_, tmp_buffer2_;

  // map property
  Eigen::Vector3d min_range_, max_range_;  // map range in pos
  Eigen::Vector3i grid_size_;              // map range in index
  Eigen::Vector3i min_vec_, max_vec_;      // the min and max updated range, unit is 1

  bool isInMap(Eigen::Vector3d pos);
  void posToIndex(Eigen::Vector3d pos, Eigen::Vector3i& id);
  void indexToPos(Eigen::Vector3i id, Eigen::Vector3d& pos);

  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

  /* ---------- parameter ---------- */
  bool trigger;
  bool pub_inflate_cloud;
  double inflate_, update_range_, radius_ignore_;
  Eigen::Vector3d origin_, map_size_;
  double resolution_sdf_, resolution_inv_;
  double ceil_height_, max_ground_height_;
  double update_rate_, update_delay;

  /* ---------- callback ---------- */
  nav_msgs::Odometry odom_;
  bool have_odom_;

  pcl::PointCloud<pcl::PointXYZ> latest_cloud_, cloud_inflate_vis_;
  bool new_map_, map_valid_;

  ros::Publisher inflate_cloud_pub_;
  ros::Timer update_timer_;

  void updateCallback(const ros::TimerEvent& e);

  /* --------------------------------- */

public:
  SDFMap() {}
  SDFMap(Eigen::Vector3d origin, double resolution, Eigen::Vector3d map_size);
  ~SDFMap() {}
  void init(ros::NodeHandle& nh);

  /* set state */
  void set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr);
  void set_odom(nav_msgs::Odometry cur_odom);
  void resetUpdatedRange(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos, bool reset_all_ = false);
  
  /* get state */
  bool sdfState() {return trigger;}
  bool odomValid() { return have_odom_; }
  bool mapValid() { return map_valid_; }
  nav_msgs::Odometry getOdom() { return odom_; }
  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size) { ori = origin_, size = map_size_; }
  double getResolution() { return resolution_sdf_; }
  double getIgnoreRadius() { return radius_ignore_; }
  void getInterpolationData(const Eigen::Vector3d& pos, vector<Eigen::Vector3d>& pos_vec,
                            Eigen::Vector3d& diff);
  double getUpdateDelay() { if(map_valid_) return update_delay; else return -1.0;};

  // occupancy management
  void setOccupancy(Eigen::Vector3d pos, int is_occ = 1);
  int getOccupancy(Eigen::Vector3d pos);
  int getOccupancy(Eigen::Vector3i id);
  void getOccupancyMarker(visualization_msgs::Marker& m, int id, Eigen::Vector4d color);

  // distance field management
  void updateESDF3d(bool neg = false);
  double getMaxDistance();
  double getDistance(Eigen::Vector3d pos);
  double getDistance(Eigen::Vector3i id);
  double getDistWithGradTrilinear(Eigen::Vector3d pos, Eigen::Vector3d& grad);
  double getDistTrilinear(Eigen::Vector3d pos);
  void getESDFMarker(vector<visualization_msgs::Marker>& markers, int id, Eigen::Vector3d color);

  void publishESDF();

  typedef shared_ptr<SDFMap> Ptr;
};

}

#endif
