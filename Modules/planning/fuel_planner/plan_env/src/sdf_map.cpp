#include "plan_env/sdf_map.h"
#include "plan_env/map_ros.h"
#include <plan_env/raycast.h>

namespace fast_planner {
SDFMap::SDFMap() {
}

SDFMap::~SDFMap() {
}

void SDFMap::initMap(ros::NodeHandle& nh) {
  mp_.reset(new MapParam);
  md_.reset(new MapData);
  mr_.reset(new MapROS);

  // Params of map properties
  double x_size, y_size, z_size;
  nh.param("sdf_map/resolution", mp_->resolution_, -1.0);
  nh.param("sdf_map/map_size_x", x_size, -1.0);
  nh.param("sdf_map/map_size_y", y_size, -1.0);
  nh.param("sdf_map/map_size_z", z_size, -1.0);
  nh.param("sdf_map/obstacles_inflation", mp_->obstacles_inflation_, -1.0);
  nh.param("sdf_map/local_bound_inflate", mp_->local_bound_inflate_, 1.0);
  nh.param("sdf_map/local_map_margin", mp_->local_map_margin_, 1);
  nh.param("sdf_map/ground_height", mp_->ground_height_, 1.0);
  nh.param("sdf_map/default_dist", mp_->default_dist_, 5.0);
  nh.param("sdf_map/optimistic", mp_->optimistic_, true);
  nh.param("sdf_map/signed_dist", mp_->signed_dist_, false);

  mp_->local_bound_inflate_ = max(mp_->resolution_, mp_->local_bound_inflate_);
  mp_->resolution_inv_ = 1 / mp_->resolution_;
  mp_->map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_->ground_height_);
  mp_->map_size_ = Eigen::Vector3d(x_size, y_size, z_size);
  for (int i = 0; i < 3; ++i)
    mp_->map_voxel_num_(i) = ceil(mp_->map_size_(i) / mp_->resolution_);
  mp_->map_min_boundary_ = mp_->map_origin_;
  mp_->map_max_boundary_ = mp_->map_origin_ + mp_->map_size_;

  // Params of raycasting-based fusion
  nh.param("sdf_map/p_hit", mp_->p_hit_, 0.70);
  nh.param("sdf_map/p_miss", mp_->p_miss_, 0.35);
  nh.param("sdf_map/p_min", mp_->p_min_, 0.12);
  nh.param("sdf_map/p_max", mp_->p_max_, 0.97);
  nh.param("sdf_map/p_occ", mp_->p_occ_, 0.80);
  nh.param("sdf_map/max_ray_length", mp_->max_ray_length_, -0.1);
  nh.param("sdf_map/virtual_ceil_height", mp_->virtual_ceil_height_, -0.1);

  auto logit = [](const double& x) { return log(x / (1 - x)); };
  mp_->prob_hit_log_ = logit(mp_->p_hit_);
  mp_->prob_miss_log_ = logit(mp_->p_miss_);
  mp_->clamp_min_log_ = logit(mp_->p_min_);
  mp_->clamp_max_log_ = logit(mp_->p_max_);
  mp_->min_occupancy_log_ = logit(mp_->p_occ_);
  mp_->unknown_flag_ = 0.01;
  cout << "hit: " << mp_->prob_hit_log_ << ", miss: " << mp_->prob_miss_log_
       << ", min: " << mp_->clamp_min_log_ << ", max: " << mp_->clamp_max_log_
       << ", thresh: " << mp_->min_occupancy_log_ << endl;

  // Initialize data buffer of map
  int buffer_size = mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2);
  md_->occupancy_buffer_ = vector<double>(buffer_size, mp_->clamp_min_log_ - mp_->unknown_flag_);
  md_->occupancy_buffer_copy_ = vector<double>(buffer_size, mp_->clamp_min_log_ - mp_->unknown_flag_);
  md_->occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);
  md_->occupancy_buffer_inflate_copy_ = vector<char>(buffer_size, 0);
  if (mp_->signed_dist_)  md_->distance_buffer_neg_ = vector<double>(buffer_size, mp_->default_dist_);
  md_->distance_buffer_ = vector<double>(buffer_size, mp_->default_dist_);
  md_->distance_buffer_copy_ = vector<double>(buffer_size, mp_->default_dist_);
  md_->count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_->count_hit_ = vector<short>(buffer_size, 0);
  md_->count_miss_ = vector<short>(buffer_size, 0);
  md_->tmp_buffer1_ = vector<double>(buffer_size, 0);
  md_->tmp_buffer2_ = vector<double>(buffer_size, 0);
  md_->update_min_ = md_->update_max_ = Eigen::Vector3d(0, 0, 0);

  // Try retriving bounding box of map, set box to map size if not specified
  vector<string> axis = { "x", "y", "z" };
  for (int i = 0; i < 3; ++i) {
    nh.param("sdf_map/box_min_" + axis[i], mp_->box_mind_[i], mp_->map_min_boundary_[i]);
    nh.param("sdf_map/box_max_" + axis[i], mp_->box_maxd_[i], mp_->map_max_boundary_[i]);
  }
  posToIndex(mp_->box_mind_, mp_->box_min_);
  posToIndex(mp_->box_maxd_, mp_->box_max_);

  // Initialize ROS wrapper
  mr_->setMap(this);
  mr_->node_ = nh;
  mr_->init();

  raycaster_.reset(new RayCaster);
  raycaster_->setParams(mp_->resolution_, mp_->map_origin_);
}

void SDFMap::resetBuffer() {
  resetBuffer(mp_->map_min_boundary_, mp_->map_max_boundary_);
  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_->map_voxel_num_ - Eigen::Vector3i::Ones();
}

void SDFMap::resetBuffer(const Eigen::Vector3d& min_pos, const Eigen::Vector3d& max_pos) {
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  for (int x = min_id[0]; x <= max_id[0]; ++x)
    for (int y = min_id[1]; y <= max_id[1]; ++y)
      for (int z = min_id[2]; z <= max_id[2]; ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_->distance_buffer_[toAddress(x, y, z)] = mp_->default_dist_;
      }
}

template <typename F_get_val, typename F_set_val>
void SDFMap::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[mp_->map_voxel_num_(dim)];
  double z[mp_->map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q)
      k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

void SDFMap::updateESDF3d() {
  Eigen::Vector3i local_bound_min, local_bound_max;
  getLocalBox(local_bound_min, local_bound_max);

  if (mp_->optimistic_) {
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; x++)
      for (int y = local_bound_min[1]; y <= local_bound_max[1]; y++) {
        fillESDF(
            [&](int z) {
              return getInflateOccupancy(Eigen::Vector3i(x, y, z)) == 1 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, local_bound_min[2],
            local_bound_max[2], 2);
      }
  } else {
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; x++)
      for (int y = local_bound_min[1]; y <= local_bound_max[1]; y++) {
        fillESDF(
            [&](int z) {
              return (getInflateOccupancy(Eigen::Vector3i(x, y, z)) == 1 ||
                      getOccupancy(Eigen::Vector3i(x, y, z)) == UNKNOWN) ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, local_bound_min[2],
            local_bound_max[2], 2);
      }
  }

  for (int x = local_bound_min[0]; x <= local_bound_max[0]; x++)
    for (int z = local_bound_min[2]; z <= local_bound_max[2]; z++) {
      fillESDF(
          [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
          [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, local_bound_min[1],
          local_bound_max[1], 1);
    }
  for (int y = local_bound_min[1]; y <= local_bound_max[1]; y++)
    for (int z = local_bound_min[2]; z <= local_bound_max[2]; z++) {
      fillESDF(
          [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
          [&](int x, double val) {
            md_->distance_buffer_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
          },
          local_bound_min[0], local_bound_max[0], 0);
    }

  if (mp_->signed_dist_) {
    // Compute negative distance
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; x++)
      for (int y = local_bound_min[1]; y <= local_bound_max[1]; y++) {
        fillESDF(
            [&](int z) {
              return getInflateOccupancy(Eigen::Vector3i(x, y, z)) == 0 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, local_bound_min[2],
            local_bound_max[2], 2);
      }
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; x++)
      for (int z = local_bound_min[2]; z <= local_bound_max[2]; z++) {
        fillESDF(
            [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
            [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, local_bound_min[1],
            local_bound_max[1], 1);
      }
    for (int y = local_bound_min[1]; y <= local_bound_max[1]; y++)
      for (int z = local_bound_min[2]; z <= local_bound_max[2]; z++) {
        fillESDF(
            [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
            [&](int x, double val) {
              md_->distance_buffer_neg_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
            },
            local_bound_min[0], local_bound_max[0], 0);
      }
    // Merge negative distance with positive
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; ++x)
      for (int y = local_bound_min[1]; y <= local_bound_max[1]; ++y)
        for (int z = local_bound_min[2]; z <= local_bound_max[2]; ++z) {
          int idx = toAddress(x, y, z);
          if (md_->distance_buffer_neg_[idx] > 0.0)
            md_->distance_buffer_[idx] += (-md_->distance_buffer_neg_[idx] + mp_->resolution_);
        }
  }
}

void SDFMap::setCacheOccupancy(const int& adr, const int& occ) {
  // Add to update list if first visited
  if (md_->count_hit_[adr] == 0 && md_->count_miss_[adr] == 0) md_->cache_voxel_.push(adr);

  if (occ == 0)
    md_->count_miss_[adr] = 1;
  else if (occ == 1)
    md_->count_hit_[adr] += 1;

  // md_->count_hit_and_miss_[adr] += 1;
  // if (occ == 1)
  //   md_->count_hit_[adr] += 1;
  // if (md_->count_hit_and_miss_[adr] == 1)
  //   md_->cache_voxel_.push(adr);
}

// input point cloud from map_ros wrapper
void SDFMap::inputPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num,
    const Eigen::Vector3d& camera_pos) {
  if (point_num == 0) return;

  Eigen::Vector3d update_min = camera_pos;
  Eigen::Vector3d update_max = camera_pos;

  Eigen::Vector3d pt_w, tmp;
  Eigen::Vector3i idx;
  int vox_adr;
  double length;
  for (int i = 0; i < point_num; ++i) {
    auto& pt = points.points[i];
    pt_w << pt.x, pt.y, pt.z;
    int tmp_flag;
    // Set flag for projected point
    if (!isInMap(pt_w)) {
      // Find closest point in map and set free
      pt_w = closetPointOnMapBoundary(pt_w, camera_pos);
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_)
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
      if (pt_w[2] < 0.2) continue;
      tmp_flag = 0;
    } else {
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_) {
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
        if (pt_w[2] < 0.2) continue;
        tmp_flag = 0;
      } else
        tmp_flag = 1;
    }
    posToIndex(pt_w, idx);
    vox_adr = toAddress(idx);
    setCacheOccupancy(vox_adr, tmp_flag);

    for (int k = 0; k < 3; ++k) {
      update_min[k] = min(update_min[k], pt_w[k]);
      update_max[k] = max(update_max[k], pt_w[k]);
    }

    raycaster_->setInput(pt_w, camera_pos);
    raycaster_->nextId(idx);
    while (raycaster_->nextId(idx))
      setCacheOccupancy(toAddress(idx), 0);
  }

  Eigen::Vector3d bound_inf(mp_->local_bound_inflate_, mp_->local_bound_inflate_, 0);
  updateLocalBox((update_min - bound_inf), (update_max + bound_inf));
  Eigen::Vector3i local_bound_min, local_bound_max;
  getLocalBox(local_bound_min, local_bound_max);

  // Bounding box for subsequent updating
{
std::unique_lock<std::shared_mutex> lock(box_mtx);
  md_->update_min_ = update_min;//md_->update_min_.array().min(update_min.array()).matrix();
  md_->update_max_ = update_max;//md_->update_max_.array().min(update_max.array()).matrix();
}
  // update occupied prob
  while (!md_->cache_voxel_.empty()) {
    int adr = md_->cache_voxel_.front();
    md_->cache_voxel_.pop();
    double log_odds_update =
        md_->count_hit_[adr] >= md_->count_miss_[adr] ? mp_->prob_hit_log_ : mp_->prob_miss_log_;
    md_->count_hit_[adr] = md_->count_miss_[adr] = 0;
    if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)
      md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;

    md_->occupancy_buffer_[adr] = std::min(
        std::max(md_->occupancy_buffer_[adr] + log_odds_update, mp_->clamp_min_log_),
        mp_->clamp_max_log_);
  }

  // clean local space
  for (int x = local_bound_min[0]; x <= local_bound_max[0]; ++x)
    for (int y = local_bound_min[1]; y <= local_bound_max[1]; ++y)
      for (int z = local_bound_min[2]; z <= local_bound_max[2]; ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  // inflate local occupied cells
  for (int x = local_bound_min[0]; x <= local_bound_max[0]; ++x)
    for (int y = local_bound_min[1]; y <= local_bound_max[1]; ++y)
      for (int z = local_bound_min[2]; z <= local_bound_max[2]; ++z) {
        int id1 = toAddress(x, y, z);
        if (md_->occupancy_buffer_[id1] > mp_->min_occupancy_log_) {
          vector<Eigen::Vector3i> inf_pts;
          int inf_step = ceil(mp_->obstacles_inflation_ / mp_->resolution_);
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);
          for (auto inf_pt : inf_pts) {
            int idx_inf = toAddress(inf_pt);
            if (idx_inf >= 0 &&
                idx_inf <
                    mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2)) {
              md_->occupancy_buffer_inflate_[idx_inf] = 1;
            }
          }
        }
      }

  // add virtual ceiling to limit flight height
  if (mp_->virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_->virtual_ceil_height_ - mp_->map_origin_[2]) * mp_->resolution_inv_);
    for (int x = local_bound_min[0]; x <= local_bound_max[0]; ++x)
      for (int y = local_bound_min[1]; y <= local_bound_max[1]; ++y) {
        // md_->occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
        md_->occupancy_buffer_[toAddress(x, y, ceil_id)] = mp_->clamp_max_log_;
      }
  }
}

Eigen::Vector3d
SDFMap::closetPointOnMapBoundary(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt) {
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_->map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_->map_min_boundary_ - camera_pt;
  double min_t = 1000000;
  for (int i = 0; i < 3; ++i) {
    if (fabs(diff[i]) > 0) {
      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t) min_t = t1;
      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t) min_t = t2;
    }
  }
  return camera_pt + (min_t - 1e-3) * diff;
}

void SDFMap::clearAndInflateLocalMap() {
  // int inf_step = ceil(mp_->obstacles_inflation_ / mp_->resolution_);
  // vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  // inf_pts.resize(4 * inf_step + 3);

}

// Helper function
double SDFMap::getResolution() {
  return mp_->resolution_;
}

int SDFMap::getVoxelNum() {
  return mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1] * mp_->map_voxel_num_[2];
}

void SDFMap::getFullMap(Eigen::Vector3d& ori, Eigen::Vector3d& size) {
  ori = mp_->map_origin_, size = mp_->map_size_;
}

void SDFMap::getFullBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) {
  bmin = mp_->box_mind_;
  bmax = mp_->box_maxd_;
}

void SDFMap::getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) {
std::shared_lock<std::shared_mutex> lock(box_mtx);
  bmin = md_->update_min_;
  bmax = md_->update_max_;
}

void SDFMap::updateLocalBox(const Eigen::Vector3d& bmin, const Eigen::Vector3d& bmax){
std::unique_lock<std::shared_mutex> lock(local_box_mtx);
  posToIndex(bmax, md_->local_bound_max_);
  posToIndex(bmin, md_->local_bound_min_);
  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);
}

void SDFMap::getLocalBox(Eigen::Vector3i& bmin, Eigen::Vector3i& bmax){
std::shared_lock<std::shared_mutex> lock(local_box_mtx);
  bmin = md_->local_bound_min_;
  bmax = md_->local_bound_max_;
}

void SDFMap::copyOccupancy() {
std::unique_lock<std::shared_mutex> lock(occ_buffer_mtx);
  // std::copy(md_->occupancy_buffer_.begin(),md_->occupancy_buffer_.end(),md_->occupancy_buffer_copy_.begin());
  // std::copy(md_->occupancy_buffer_inflate_.begin(),md_->occupancy_buffer_inflate_.end(),md_->occupancy_buffer_inflate_copy_.begin());
  md_->occupancy_buffer_copy_ = md_->occupancy_buffer_;
  md_->occupancy_buffer_inflate_copy_ = md_->occupancy_buffer_inflate_;
}

int SDFMap::getOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
std::shared_lock<std::shared_mutex> lock(occ_buffer_mtx);
  double occ = md_->occupancy_buffer_copy_[toAddress(id)];
  if (occ < mp_->clamp_min_log_ - 1e-3) return UNKNOWN;
  if (occ > mp_->min_occupancy_log_) return OCCUPIED;
  return FREE;
}

int SDFMap::getOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getOccupancy(id);
}

void SDFMap::setInflatedMapOccupancy(const Eigen::Vector3d& pos, const int& occ) {
  assert(isInMap(pos));
  Eigen::Vector3i id;
  posToIndex(pos, id);
  md_->occupancy_buffer_inflate_[toAddress(id)] = occ;
}

int SDFMap::getInflateOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
std::shared_lock<std::shared_mutex> lock(occ_buffer_mtx);
  int occ = md_->occupancy_buffer_inflate_copy_[toAddress(id)];
  return occ;
}

int SDFMap::getInflateOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getInflateOccupancy(id);
}


void SDFMap::copyDistance() {
std::unique_lock<std::shared_mutex> lock(occ_buffer_mtx);
  // std::copy(md_->distance_buffer_.begin(),md_->distance_buffer_.end(),md_->distance_buffer_copy_.begin());
  md_->distance_buffer_copy_ = md_->distance_buffer_;
}

double SDFMap::getDistance(const Eigen::Vector3i& id) {
  assert(isInMap(id));
std::shared_lock<std::shared_mutex> lock(dist_buffer_mtx);
  double dist = md_->distance_buffer_copy_[toAddress(id)];
  return dist;
}

double SDFMap::getDistance(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getDistance(id);
}

double SDFMap::getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) {
  if (!isInMap(pos)) {
    grad.setZero();
    return 0;
  }

  /* trilinear interpolation */
  Eigen::Vector3d pos_m = pos - 0.5 * mp_->resolution_ * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  posToIndex(pos_m, idx);
  Eigen::Vector3d idx_pos, diff;
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_->resolution_inv_;

  double values[2][2][2];
  double last = 0.0;
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        if (!isInMap(current_idx)){
          values[x][y][z] = last;
          continue;
        }
        values[x][y][z] = getDistance(current_idx);
        last = values[x][y][z];
      }

  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  grad[2] = (v1 - v0) * mp_->resolution_inv_;
  grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * mp_->resolution_inv_;
  grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
  grad[0] *= mp_->resolution_inv_;

  return dist;
}
}  // namespace fast_planner
// SDFMap
