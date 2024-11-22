#include "histogram.h"
#define show_time false
using namespace std::chrono; 
namespace Histo_Planning
{

int width_ = 3; // width_extended

inline auto sign=[](double v)->double
{
  return v<0.0? -1.0:1.0;
};

inline double average(double a[],int n)
{
  double sum=0.0;
  for (int i=0;i<n;i++)
    sum+=a[i];
  return sum/n;
}

inline double minimum(double a[],int n)
{
  double min_val = a[0];
  for (int i=1;i<n;i++)
    if (a[i]<min_val)
      min_val=a[i];
  return min_val;
}

inline double maximum(double a[],int n)
{
  double max_val = a[0];
  for (int i=1;i<n;i++)
    if (a[i]>max_val)
      max_val=a[i];
  return max_val;
}

inline double fun_norm(double input, double std)
{
  double u = 0; // mean
  return exp(-0.5*pow((input-u)/std,2));// input==0 -> 1; input==1 -> 0.04
}

// input: 0 ~ 1; min_val: 0 ~ 1; return: min_val ~ 1
inline double fun_cos(double input, double pow_, double min_val)
{
  return (1-min_val)*pow(input,pow_) + min_val;
}

void HIST::init(ros::NodeHandle& nh)
{
  latest_local_pcl_.clear();
  has_local_map_ = true;
  has_odom_ = false;
  has_best_dir = false;

  nh.param("histo_planner/gen_guide_point", gen_guide_point, true);   // Whether to generate guide point
  nh.param("histo_planner/ground_height", ground_height, 0.1);        // Ground height
  nh.param("histo_planner/ceil_height", ceil_height, 5.0);            // Ceiling height
  nh.param("histo_planner/sensor_max_range", sensor_max_range, 3.0);  // Maximum Sensing Distance
  nh.param("histo_planner/forbidden_range", forbidden_range, 0.50); // Obstacle inflation distance
  nh.param("histo_planner/max_tracking_error", safe_distance, 0.2);   // Safe stopping distance
  forbidden_plus_safe_distance = safe_distance + forbidden_range;
  nh.param("histogram/is_2D", is_2D, false);                                // Whether 2D planning
  nh.param("histogram/isCylindrical", isCylindrical, false);                // If 3D planning, whether build cylindrical histogram
  nh.param("histogram/std_normal", _std_, 0.6);                             // Standard deviation of normal distribution function, for Cylindrical Histogram
  nh.param("histogram/isSpherical", isSpherical, true);                     // If 3D planning, whether build spherical histogram
  nh.param("histogram/min_fun_cos_value", min_value, 0.2);                  // Minimum of cos distribution, for Spherical Histogram; The greater the minimum value the greater the likelihood that the track will turn around
  nh.param("histogram/fun_cos_pow", _pow_, 1.0);                            // exponent of cos distribution, for Spherical Histogram，指数越高越锁定于当前速度方向 0.9～1.1
  if(!is_2D && !isCylindrical && !isSpherical) isSpherical = true;    // Spherical Histogram by default
  if(is_2D) {Vcnt = 1;isCylindrical=false; isSpherical=false;}
  if(isCylindrical || isSpherical) is_2D=false;
  nh.param("histogram/h_cnt", Hcnt, 180);                                   // Number of histogram columns (even)
  nh.param("histogram/v_cnt", Vcnt, 90);                                    // Number of histogram rows (even)
  nh.param("histogram/min_vel_default", min_vel_default, 0.1);              // Default minimum speed
  nh.param("histogram/max_planning_vel", limit_v_norm, 0.4);                // Maximum flight speed
    
  nh.param("histogram/piecewise_interpolation_num", piecewise_interpolation_num, 20); // Polygonal approximation of initial trajectory
  point_cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/histo_planner/local_hist_pcl", 10); // point could filtered repub

  Hres = 2*M_PI/Hcnt;
  if(isSpherical)
    Vres = M_PI/Vcnt; // +/- pi/2
  else if(isCylindrical)
    Vres = 2.0/Vcnt; // default: +/- 1m
    
  Obstacle_3d = new double*[Vcnt]();
  for( int i=0; i<Vcnt; i++ )
    Obstacle_3d[i] = new double[Hcnt]();
    
  Histogram_3d = new double*[Vcnt]();
  for( int i=0; i<Vcnt; i++ )
    Histogram_3d[i] = new double[Hcnt]();
    
  Obs_buff_3d = new double**[4]();
  for( int i=0; i<4; i++ ){
    Obs_buff_3d[i] = new double*[Vcnt]();
    for(int j=0; j<Vcnt; j++)
      Obs_buff_3d[i][j] = new double[Hcnt](); // xyzt
  }

  Env_3d = new double**[4]();
  for( int i=0; i<4; i++ ){
    Env_3d[i] = new double*[Vcnt]();
    for(int j=0; j<Vcnt; j++)
      Env_3d[i][j] = new double[Hcnt](); // xyzt
  }

  if(isSpherical){
    Weights_3d = new double*[Vcnt/2]();
    for(int i=0;i<Vcnt/2;i++){
      Weights_3d[i] = new double[Hcnt/2]();
      for(int j=0;j<Hcnt/2;j++){
        Weights_3d[i][j] = fun_cos((1+cos((i+0.5)*Vres))/2, _pow_, min_value) * fun_cos((1+cos((j+0.5)*Hres))/2, _pow_, min_value);
      }
    }
  } else if(isCylindrical || is_2D){
    Weights_3d = new double*[Vcnt]();
    for(int i=0;i<Vcnt;i++){
      Weights_3d[i] = new double[Hcnt/2]();
      for(int j=0;j<Hcnt/2;j++){
        Weights_3d[i][j] = fun_norm((i+0.5)/Vcnt,_std_) * fun_cos((cos((j+0.5)*Hres)+1)/2, _pow_, min_value);
      }
    }
  }
}

/* ************** Obstacle Histogram generation *********************
* latest_pcl_xyzt_buff: Point clouds with time stamp within short-term memory
* pcl_ptr: New sensor measurements
* Obs_buff_3d: Point clouds updated buffer. Obscured, time-out, and out-of-range point clouds will be forgotten
* Obstacle_3d: Obs_histogram
* latest_local_pcl_: The currently activated point cloud which will be used to generate the artificial potential field.
*/
void HIST::set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr)
{
  // Unreliable points are filtered out based on the maximum sensing distance,
  // while a pure environmental histogram is generated for viewing and training NN.
  if (!has_odom_)
    return;
    
  float pcl_new_time = (float)ros::Time::now().toSec();

  if(show_time) start_pcl = std::chrono::system_clock::now(); 
  
  mutex.lock();
  
  /* ***** reset the Histogram as max sensor range ***** */
  for(int i=0; i<Vcnt; i++)
    std::fill(Obstacle_3d[i],Obstacle_3d[i]+Hcnt,sensor_max_range);

  if(pcl_ptr->points.size() + latest_pcl_xyzt_buff.points.size() == 0)
  {
    has_local_map_= false;
    latest_local_pcl_.clear();
    has_local_map_= true;
    if(show_time){
      std::chrono::duration<double, std::micro> elapsed_ = std::chrono::system_clock::now() - start_pcl; 
      printf(" reduced Obs takes %f [us].\n", elapsed_.count());
    }
    mutex.unlock();
    return;
  }

  /* ***** reset the Obs_buff ***** */
  for(int i=0; i<4; i++)
    for(int j=0; j<Vcnt; j++)
      std::fill(Obs_buff_3d[i][j],Obs_buff_3d[i][j]+Hcnt,0.0);

  /* ***************** Generate Obs HIST ***************** */
  Eigen::Vector3d p3d;
  /* ****** update old pcl ****** */
  for (size_t id = 0; id < latest_pcl_xyzt_buff.points.size(); ++id) 
  {
    // Time filter
    float pcl_old_time = latest_pcl_xyzt_buff.points[id].t;
    if(pcl_new_time - pcl_old_time > 5) continue;
  
    p3d(0) = latest_pcl_xyzt_buff.points[id].x;
    p3d(1) = latest_pcl_xyzt_buff.points[id].y;
    p3d(2) = latest_pcl_xyzt_buff.points[id].z; // World_ENU frame

    Eigen::Vector3d uav2obs = p3d - capture_pos; 
    double obs_dist = uav2obs.norm();
    // Spacial filter
    if(obs_dist > sensor_max_range) continue;

    // update Obs_buff
    if (is_2D)
    {
      if (fabs(uav2obs(2))>1.0) continue; // default: +/- 1.0m
		    
      uav2obs(2) = 0.0;
      obs_dist = uav2obs.norm();
      double angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / uav2obs.norm());// angle_cen: -pi ~ pi
      angle_cen = angle_cen<0 ? angle_cen+2*M_PI : angle_cen; // 0 ~ 2pi
      int cnt = floor(angle_cen/Hres);
      
      if(Obs_buff_3d[3][0][cnt] == 0.0){
        Obs_buff_3d[0][0][cnt] = p3d[0];
        Obs_buff_3d[1][0][cnt] = p3d[1];
        Obs_buff_3d[2][0][cnt] = capture_pos[2];
        Obs_buff_3d[3][0][cnt] = pcl_old_time;
        Obstacle_3d[0][cnt] = obs_dist;
      }else{
        double temp_x = (Obs_buff_3d[0][0][cnt] - capture_pos[0]) * (Obs_buff_3d[0][0][cnt] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][0][cnt] - capture_pos[1]) * (Obs_buff_3d[1][0][cnt] - capture_pos[1]);
        double _dist_last = sqrt(temp_x+temp_y);
        if(_dist_last > obs_dist){
          Obs_buff_3d[0][0][cnt] = p3d[0];
          Obs_buff_3d[1][0][cnt] = p3d[1];
          Obs_buff_3d[2][0][cnt] = capture_pos[2];
          Obs_buff_3d[3][0][cnt] = pcl_old_time;
          Obstacle_3d[0][cnt] = obs_dist;
        }
      }
    }
    else if (isCylindrical)
    {
      if (fabs(uav2obs(2))>1) continue; // default: +/- 1m

      double dist = Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm();
      double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / dist);// obs_hor_angle_cen: -pi ~ pi	
      obs_hor_angle_cen = obs_hor_angle_cen<0 ? obs_hor_angle_cen+2*M_PI : obs_hor_angle_cen; // 0 ~ 2pi	    
      int obs_ver_idx_cen = floor(min(double(Vcnt),max(0.0,Vcnt/2 + uav2obs(2)/Vres)));
      int cnt = floor(obs_hor_angle_cen/Hres);
      
      if(Obs_buff_3d[3][obs_ver_idx_cen][cnt] == 0.0){
        Obs_buff_3d[0][obs_ver_idx_cen][cnt] = p3d[0];
        Obs_buff_3d[1][obs_ver_idx_cen][cnt] = p3d[1];
        Obs_buff_3d[2][obs_ver_idx_cen][cnt] = p3d[2];
        Obs_buff_3d[3][obs_ver_idx_cen][cnt] = pcl_old_time;
        Obstacle_3d[obs_ver_idx_cen][cnt] = obs_dist;
      }else{
        double temp_x = (Obs_buff_3d[0][obs_ver_idx_cen][cnt] - capture_pos[0]) * (Obs_buff_3d[0][obs_ver_idx_cen][cnt] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][obs_ver_idx_cen][cnt] - capture_pos[1]) * (Obs_buff_3d[1][obs_ver_idx_cen][cnt] - capture_pos[1]);
        double temp_z = (Obs_buff_3d[2][obs_ver_idx_cen][cnt] - capture_pos[2]) * (Obs_buff_3d[2][obs_ver_idx_cen][cnt] - capture_pos[2]);
        double _dist_last = sqrt(temp_x+temp_y+temp_z);
        if (_dist_last > obs_dist){
          Obs_buff_3d[0][obs_ver_idx_cen][cnt] = p3d[0];
          Obs_buff_3d[1][obs_ver_idx_cen][cnt] = p3d[1];
          Obs_buff_3d[2][obs_ver_idx_cen][cnt] = p3d[2];
          Obs_buff_3d[3][obs_ver_idx_cen][cnt] = pcl_old_time;
          Obstacle_3d[obs_ver_idx_cen][cnt] = obs_dist;
        }
      }
    }
    else if (isSpherical)
    {
      double dist = Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm();
      double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / dist);// obs_hor_angle_cen: -pi ~ pi
      obs_hor_angle_cen = obs_hor_angle_cen<0 ? obs_hor_angle_cen+2*M_PI : obs_hor_angle_cen; // 0(axis +x) ~ 2pi
      double obs_ver_angle_cen = sign(uav2obs(2)) * acos(dist / uav2obs.norm());// obs_ver_angle_cen: -pi/2 ~ pi/2
      obs_ver_angle_cen = obs_ver_angle_cen + M_PI/2; // 0(axis -z) ~ pi

      int i = floor(obs_ver_angle_cen/Vres);
      int j = floor(obs_hor_angle_cen/Hres);
      
      if(Obs_buff_3d[3][i][j] == 0.0){
        Obs_buff_3d[0][i][j] = p3d[0];
        Obs_buff_3d[1][i][j] = p3d[1];
        Obs_buff_3d[2][i][j] = p3d[2];
        Obs_buff_3d[3][i][j] = pcl_old_time;
        Obstacle_3d[i][j] = obs_dist;
      }else{
        double temp_x = (Obs_buff_3d[0][i][j] - capture_pos[0]) * (Obs_buff_3d[0][i][j] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][i][j] - capture_pos[1]) * (Obs_buff_3d[1][i][j] - capture_pos[1]);
        double temp_z = (Obs_buff_3d[2][i][j] - capture_pos[2]) * (Obs_buff_3d[2][i][j] - capture_pos[2]);
        double _dist_last = sqrt(temp_x+temp_y+temp_z);
        if (_dist_last > obs_dist){
          Obs_buff_3d[0][i][j] = p3d[0];
          Obs_buff_3d[1][i][j] = p3d[1];
          Obs_buff_3d[2][i][j] = p3d[2];
          Obs_buff_3d[3][i][j] = pcl_old_time;
          Obstacle_3d[i][j] = obs_dist;
        }
      }
    }
  }

  /* ****** update new pcl ****** */
  for (size_t i = 0; i < pcl_ptr->points.size(); ++i) 
  {
    p3d(0) = pcl_ptr->points[i].x;
    p3d(1) = pcl_ptr->points[i].y;
    p3d(2) = pcl_ptr->points[i].z; // World_ENU frame

    Eigen::Vector3d uav2obs = p3d - capture_pos; 
    double obs_dist = uav2obs.norm();
    // Spacial filter
    //if(obs_dist > sensor_max_range) continue;
    if(p3d(2) > ceil_height || p3d(2) < ground_height) continue;

    // update Obs_buff
    if (is_2D)
    {
      if (fabs(uav2obs(2))>1.0) continue; // default: +/- 1.0m
		    
      uav2obs(2) = 0.0;
      obs_dist = uav2obs.norm();
      double angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / uav2obs.norm());// angle_cen: -pi ~ pi
      angle_cen = angle_cen<0 ? angle_cen+2*M_PI : angle_cen; // 0 ~ 2pi
      int cnt = floor(angle_cen/Hres);

      if(Obs_buff_3d[3][0][cnt] < pcl_new_time){
        if(obs_dist > sensor_max_range){// Spacial filter
          Obs_buff_3d[3][0][cnt] = 0.0;
          Obstacle_3d[0][cnt] = sensor_max_range;
        }else{
          Obs_buff_3d[0][0][cnt] = p3d[0];
          Obs_buff_3d[1][0][cnt] = p3d[1];
          Obs_buff_3d[2][0][cnt] = capture_pos[2];
          Obs_buff_3d[3][0][cnt] = pcl_new_time;
          Obstacle_3d[0][cnt] = obs_dist;
        }
      }else{
        double temp_x = (Obs_buff_3d[0][0][cnt] - capture_pos[0]) * (Obs_buff_3d[0][0][cnt] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][0][cnt] - capture_pos[1]) * (Obs_buff_3d[1][0][cnt] - capture_pos[1]);
        double _dist_last = sqrt(temp_x+temp_y);
        if(_dist_last > obs_dist){
          Obs_buff_3d[0][0][cnt] = p3d[0];
          Obs_buff_3d[1][0][cnt] = p3d[1];
          Obs_buff_3d[2][0][cnt] = capture_pos[2];
          // Obs_buff_3d[3][0][cnt] = pcl_new_time;
          Obstacle_3d[0][cnt] = obs_dist;
        }
      }
    }
    else if (isCylindrical)
    {
      if (fabs(uav2obs(2))>1) continue; // default: +/- 1m

      double dist = Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm();
      double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / dist);// obs_hor_angle_cen: -pi ~ pi	
      obs_hor_angle_cen = obs_hor_angle_cen<0 ? obs_hor_angle_cen+2*M_PI : obs_hor_angle_cen; // 0 ~ 2pi	    
      int obs_ver_idx_cen = floor(min(double(Vcnt),max(0.0,Vcnt/2 + uav2obs(2)/Vres)));
      int cnt = floor(obs_hor_angle_cen/Hres);
      
      if(Obs_buff_3d[3][obs_ver_idx_cen][cnt] < pcl_new_time){
        if(obs_dist > sensor_max_range){// Spacial filter
          Obs_buff_3d[3][obs_ver_idx_cen][cnt] = 0.0;
          Obstacle_3d[obs_ver_idx_cen][cnt] = sensor_max_range;
        }else{
          Obs_buff_3d[0][obs_ver_idx_cen][cnt] = p3d[0];
          Obs_buff_3d[1][obs_ver_idx_cen][cnt] = p3d[1];
          Obs_buff_3d[2][obs_ver_idx_cen][cnt] = p3d[2];
          Obs_buff_3d[3][obs_ver_idx_cen][cnt] = pcl_new_time;
          Obstacle_3d[obs_ver_idx_cen][cnt] = obs_dist;
        }
      }else{
        double temp_x = (Obs_buff_3d[0][obs_ver_idx_cen][cnt] - capture_pos[0]) * (Obs_buff_3d[0][obs_ver_idx_cen][cnt] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][obs_ver_idx_cen][cnt] - capture_pos[1]) * (Obs_buff_3d[1][obs_ver_idx_cen][cnt] - capture_pos[1]);
        double temp_z = (Obs_buff_3d[2][obs_ver_idx_cen][cnt] - capture_pos[2]) * (Obs_buff_3d[2][obs_ver_idx_cen][cnt] - capture_pos[2]);
        double _dist_last = sqrt(temp_x+temp_y+temp_z);
        if (_dist_last > obs_dist){
          Obs_buff_3d[0][obs_ver_idx_cen][cnt] = p3d[0];
          Obs_buff_3d[1][obs_ver_idx_cen][cnt] = p3d[1];
          Obs_buff_3d[2][obs_ver_idx_cen][cnt] = p3d[2];
          // Obs_buff_3d[3][obs_ver_idx_cen][cnt] = pcl_new_time;
          Obstacle_3d[obs_ver_idx_cen][cnt] = obs_dist;
        }
      }
    }
    else if (isSpherical)
    {
      double dist = Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm();
      double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / dist);// obs_hor_angle_cen (ccw, 0 at x): -pi ~ pi
      obs_hor_angle_cen = obs_hor_angle_cen<0 ? obs_hor_angle_cen+2*M_PI : obs_hor_angle_cen; // 0 ~ 2pi
      double obs_ver_angle_cen = sign(uav2obs(2)) * acos(dist / uav2obs.norm());// obs_ver_angle_cen (up, 0 at xy): -pi/2 ~ pi/2
      obs_ver_angle_cen = obs_ver_angle_cen + M_PI/2; // 0 ~ pi

      int i = floor(obs_ver_angle_cen/Vres);
      int j = floor(obs_hor_angle_cen/Hres);
      
      if(Obs_buff_3d[3][i][j] < pcl_new_time){
        if(obs_dist > sensor_max_range){// Spacial filter
          Obs_buff_3d[3][i][j] = 0.0;
          Obstacle_3d[i][j] = sensor_max_range;
        }else{
          Obs_buff_3d[0][i][j] = p3d[0];
          Obs_buff_3d[1][i][j] = p3d[1];
          Obs_buff_3d[2][i][j] = p3d[2];
          Obs_buff_3d[3][i][j] = pcl_new_time;
          Obstacle_3d[i][j] = obs_dist;
        }
      }else{
        double temp_x = (Obs_buff_3d[0][i][j] - capture_pos[0]) * (Obs_buff_3d[0][i][j] - capture_pos[0]);
        double temp_y = (Obs_buff_3d[1][i][j] - capture_pos[1]) * (Obs_buff_3d[1][i][j] - capture_pos[1]);
        double temp_z = (Obs_buff_3d[2][i][j] - capture_pos[2]) * (Obs_buff_3d[2][i][j] - capture_pos[2]);
        double _dist_last = sqrt(temp_x+temp_y+temp_z);
        if (_dist_last > obs_dist){
          Obs_buff_3d[0][i][j] = p3d[0];
          Obs_buff_3d[1][i][j] = p3d[1];
          Obs_buff_3d[2][i][j] = p3d[2];
          // Obs_buff_3d[3][i][j] = pcl_new_time;
          Obstacle_3d[i][j] = obs_dist;
        }
      }
    }
  }

  /* ************* Generate reduced PCL ************* */
  has_local_map_= false;
  latest_local_pcl_.clear();
  latest_pcl_xyzt_buff.clear();
  pcl::PointXYZ newPoint;
  PointXYZT XYZT;
  if(is_2D)
  {
    for(int i=0; i<Hcnt; i++)
      if(Obs_buff_3d[3][0][i]>0.0){
        newPoint.x = Obs_buff_3d[0][0][i];
        newPoint.y = Obs_buff_3d[1][0][i];
        newPoint.z = Obs_buff_3d[2][0][i];
        latest_local_pcl_.push_back(newPoint);
        XYZT.x = Obs_buff_3d[0][0][i];
        XYZT.y = Obs_buff_3d[1][0][i];
        XYZT.z = Obs_buff_3d[2][0][i];
        XYZT.t = Obs_buff_3d[3][0][i];
        latest_pcl_xyzt_buff.push_back(XYZT);
      }
  }
  else if (isCylindrical)
  {
    for(int i=0; i<Vcnt; i++)
      for(int j=0; j<Hcnt; j++)
        if(Obs_buff_3d[3][i][j]>0.0){
          newPoint.x = Obs_buff_3d[0][i][j];
          newPoint.y = Obs_buff_3d[1][i][j];
          newPoint.z = Obs_buff_3d[2][i][j];
          latest_local_pcl_.push_back(newPoint);
          XYZT.x = Obs_buff_3d[0][i][j];
          XYZT.y = Obs_buff_3d[1][i][j];
          XYZT.z = Obs_buff_3d[2][i][j];
          XYZT.t = Obs_buff_3d[3][i][j];
          latest_pcl_xyzt_buff.push_back(XYZT);
        }
  }
  else if (isSpherical)
  {
    for(int i=0; i<Vcnt; i++)
      for(int j=0; j<Hcnt; j++)
        if(Obs_buff_3d[3][i][j]>0.0){
          //if(Obs_buff_3d[3][i][j]<pcl_new_time){
          //  cout<<"u,v: "<<i*Vres<<", "<<j*Hres<<endl;
          //}
          newPoint.x = Obs_buff_3d[0][i][j];
          newPoint.y = Obs_buff_3d[1][i][j];
          newPoint.z = Obs_buff_3d[2][i][j];
          latest_local_pcl_.push_back(newPoint);
          XYZT.x = Obs_buff_3d[0][i][j];
          XYZT.y = Obs_buff_3d[1][i][j];
          XYZT.z = Obs_buff_3d[2][i][j];
          XYZT.t = Obs_buff_3d[3][i][j];
          latest_pcl_xyzt_buff.push_back(XYZT);
        }
  }
  for(int i=0;i<4;i++) 
    for(int j=0;j<Vcnt;j++) // Vcnt
      for(int k=0;k<Hcnt;k++) // Hcnt
        Env_3d[i][j][k] = Obs_buff_3d[i][j][k];
  has_local_map_= true;

  //cout << latest_local_pcl_.points.size() << " ";
  latest_local_pcl_.header.seq++;
  latest_local_pcl_.header.stamp = pcl_ptr->header.stamp;
  latest_local_pcl_.header.frame_id = "world";
  point_cloud_pub.publish(latest_local_pcl_);

  mutex.unlock();

  if(show_time){
    std::chrono::duration<double, std::micro> elapsed_ = std::chrono::system_clock::now() - start_pcl; 
    printf(" reduced Obs takes %f [us].\n", elapsed_.count());
  }
}

/* ************** Odometry of Sensor ***********************
* capture_pos: The current position of the sensor. 
* High frequency messages therefore do not take into account time differences.
*/
void HIST::set_odom(nav_msgs::Odometry cur_odom)
{
  mutex.lock();

  /* ***************** Initialize state ***************** */ 
  // cur_odom_ = cur_odom;
  capture_pos[0] = cur_odom.pose.pose.position.x;
  capture_pos[1] = cur_odom.pose.pose.position.y;
  capture_pos[2] = cur_odom.pose.pose.position.z;

  mutex.unlock();

  has_odom_ = true;
}

/* ***************** Guidance point generation *********************
* PolarCoordinateHist: Point clouds inflated in Polar coord, saved in Histogram_3d. 2D case
* CylindricalCoordinateHist: Point clouds inflated in Cylindrical coord, saved in Histogram_3d. Cyl case
* SphericalCoordinateHist: Point clouds inflated in Spherical coord, saved in Histogram_3d. Sph case
* Histogram_3d: Weighted histogram used for finding guidance point
* Weights_3d: Quarter histogram weights table. Query to get the weights.
* filter_w: kernal filter
*/
int HIST::generate(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &goal, Eigen::Vector3d &desired)
{
  if(!has_local_map_ || !has_odom_){
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if(!has_local_map_ || !has_odom_){
      cout << "[HIST Err] Check map input and odom input!" << endl;
      return 0;
    }
  }
  
  mutex.lock();
  if((int)latest_local_pcl_.points.size() == 0) 
  {
    cout << "[HIST Wrn] no point cloud" << endl;
    desired = start_pos;
    mutex.unlock();
    return 1;
  }
  	    
  if (isnan(goal(0)) || isnan(goal(1)) || isnan(goal(2)))
  {
    cout << "[HIST Err] Goal Unkown!" << endl;
    return 0;
  }

  /* ***************** Initialize state ***************** */
  Eigen::Vector3d current_pos;
  current_pos[0] = start_pos[0];
  current_pos[1] = start_pos[1];
  current_pos[2] = start_pos[2];
  if (current_pos[2]>ceil_height || current_pos[2]<ground_height)
  {
    cout << "[HIST Err] Height is not in range!" << endl;
    return 0;
  }
  Eigen::Vector3d current_vel;
  current_vel[0] = start_vel[0];
  current_vel[1] = start_vel[1];
  current_vel[2] = start_vel[2];
  Eigen::Vector3d uav2goal = goal - current_pos;

  int safe_cnt=0;
  if(show_time) start_gen = std::chrono::system_clock::now(); 

  /* ***** reset the Histogram as max sensor range ***** */
  for(int i=0; i<Vcnt; i++)
    std::fill(Histogram_3d[i],Histogram_3d[i]+Hcnt,1.2*sensor_max_range);

  /* ********* set the virtual ground and ciel ********* */
  for(int j=0; j<Hcnt; j++)
  {
    if(isSpherical)
    {
      // virtual ground
      double delta = current_pos[2]-ground_height;
      if(delta > sensor_max_range) continue;
      double i_max = acos(delta/sensor_max_range)/Vres;
      for(int i=0; i<i_max; i++){
        double dist = delta/cos((i+0.5)*Vres);
        if (Obstacle_3d[i][j] > dist){
          Histogram_3d[i][j] = dist;
          Obstacle_3d[i][j] = dist;
        }
      }
      // virtual ciel
      delta = ceil_height-current_pos[2];
      if(delta > sensor_max_range) continue;
      double i_min = Vcnt-1-acos(delta/sensor_max_range)/Vres;
      for(int i=Vcnt-1; i>i_min; i--){
        double dist = delta/cos((Vcnt-1-i+0.5)*Vres);
        if (Obstacle_3d[i][j] > dist){
          Histogram_3d[i][j] = dist;
          Obstacle_3d[i][j] = dist;
        }
      }
    }
    else if(isCylindrical)
    {
      // virtual ground
      double delta = current_pos[2]-ground_height;
      double i_max = Vcnt/2.0-delta/Vres;
      for(int i=0; i<i_max; i++)
        if (Obstacle_3d[i][j] > 0.0){
          Histogram_3d[i][j] = 0.0;
          Obstacle_3d[i][j] = 0.0;
        }
      // virtual ciel
      delta = ceil_height-current_pos[2];
      double i_min = Vcnt/2.0-1+delta/Vres;
      for(int i=Vcnt-1; i>i_min; i--)
        if (Obstacle_3d[i][j] > 0.0){
          Histogram_3d[i][j] = 0.0;
          Obstacle_3d[i][j] = 0.0;
        }
    }
  }

// cout<<latest_local_pcl_.points.size()<<endl;
  /* ************ Generate Inflated Obs HIST ************ */
  // Traverse all points in the point cloud
  Eigen::Vector3d p3d;
  for (size_t i = 0; i < latest_local_pcl_.points.size(); ++i) 
  {
    p3d(0) = latest_local_pcl_.points[i].x;
    p3d(1) = latest_local_pcl_.points[i].y;
    p3d(2) = latest_local_pcl_.points[i].z; // World_ENU frame

    Eigen::Vector3d uav2obs = p3d - current_pos; 
    double obs_dist = uav2obs.norm();
    
    double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / Eigen::Vector2d(uav2obs(0),uav2obs(1)).norm());// obs_hor_angle_cen: -pi ~ pi	
    double angle_range = M_PI/2;// angle_range: 0~pi/2
    if(obs_dist>forbidden_range)
      angle_range = asin(forbidden_range/obs_dist);
    else
      safe_cnt++;  // Dangerous??
    
    if (is_2D)
    {
      PolarCoordinateHist(obs_hor_angle_cen, angle_range, max(Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm()-forbidden_range,1e-6));
    }
    else if (isCylindrical)
    {
      double obs_ver_idx_cen = min(double(Vcnt),max(0.0,Vcnt/2 + uav2obs(2)/Vres));
      double hor_obs_dist = Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm();
      CylindricalCoordinateHist(obs_hor_angle_cen, obs_ver_idx_cen, hor_obs_dist);
    }
    else if (isSpherical)
    { 
      double obs_ver_angle_cen = sign(uav2obs(2)) * acos(Eigen::Vector2d(uav2obs(0),uav2obs(1)).norm() / obs_dist);// obs_ver_angle_cen: -pi/2 ~ pi/2
      SphericalCoordinateHist(obs_hor_angle_cen, obs_ver_angle_cen, angle_range, obs_dist);
    }
  }

  if(show_time){
   std::chrono::duration<double, std::micro> elapsed_ = std::chrono::system_clock::now() - start_gen; 
   printf("Obs Histogram generation takes %f [us].\n", elapsed_.count());
 }
  // Obs histogram
  /*
    for (int i=0; i<Vcnt; i++) {
      for (int j=0; j<Hcnt; j++) 
        cout << Histogram_3d[i][j] << ", ";
      cout << endl;
    }
    cout << endl;
  */

  /* ***************** Update and Search ***************** */
  double uav2goal_norm = uav2goal.norm();
  double uav2goal_xy_norm = Eigen::Vector2d(uav2goal(0),uav2goal(1)).norm();
  double hor_current_vel_norm = Eigen::Vector2d(current_vel(0),current_vel(1)).norm();
  // Taget & Movement
  if (is_2D)
  {
    int best_idx = -1;
    double best_value = 0;
    // Calculation of target orientation
    double goal_heading = sign(uav2goal(1)) * acos(uav2goal(0)/uav2goal_xy_norm); // -pi ~ pi
    goal_heading = goal_heading<0 ? goal_heading+2*M_PI : goal_heading; // 0 ~ 2*pi
    // Determine if need to find a guide point (not necessary if target point is visible)
    if( uav2goal_norm > Histogram_3d[0][int(goal_heading/Hres)]){
      // Calculation of moving orientation
      double current_heading = sign(current_vel(1)) * acos(current_vel(0)/hor_current_vel_norm); // -pi ~ pi
      current_heading = current_heading<0 ? current_heading+2*M_PI : current_heading; // 0 ~ 2*pi
      // Modify the weights on histogram
      for(int i=0; i<Hcnt; i++)
      {
        double angle_i = (i + 0.5)* Hres; // 0 ~ 2pi

        double angle_err_goal = fabs(angle_i - goal_heading);
        // if (angle_err_goal>M_PI) angle_err_goal = 2*M_PI - angle_err_goal; // It'd be better if used, but not necessary

        // The smaller the angle difference, the higher weight
        int X0=Hcnt/4;
        double X=fmod(angle_err_goal/M_PI,1.0)*2*X0;
        int v_j=X0+(X-X0)*(1-2*int(angle_err_goal/M_PI));
        Histogram_3d[0][i] *= Weights_3d[0][v_j];
        // Histogram_3d[0][i] *= fun_cos((cos(angle_err_goal)+1)/2, pow_goal, min_value);
        if (hor_current_vel_norm>min_vel_default){
          double angle_err_vel = fabs(angle_i - current_heading);
          // if (angle_err_vel>M_PI) angle_err_vel = 2*M_PI - angle_err_vel; // It'd be better if used, but not necessary

          double X=fmod(angle_err_vel/M_PI,1.0)*2*X0;
          int v_j=X0+(X-X0)*(1-2*int(angle_err_vel/M_PI));
          Histogram_3d[0][i] *= Weights_3d[0][v_j];
          // Histogram_3d[0][i] *= fun_cos((cos(angle_err_vel)+1)/2, _pow_, min_value);
        }
      }

      // Finding out the optimal direction
      for(int i=0; i<Hcnt; i++){
        int half_width_ = (width_-1)/2;
        double filter[width_];
        for (int u=0; u<width_; u++)
        {
          int _i_ = i+u-half_width_;
          filter[u] = Histogram_3d[0][_i_<0 ? Hcnt+_i_ : _i_>=Hcnt ? _i_-Hcnt : _i_];
        }
        double value_ = average(filter,width_) + minimum(filter,width_);
        if(value_>best_value){
          best_value = value_;
          best_idx = i;
        }
      }
    }
		
    // Return to Optimal Guidance
    if (best_idx == -1){
      best_dir = Eigen::Vector3d(0.0,0.0,0.0);
    }else {
      has_best_dir = true;
      double best_heading  = (best_idx + 0.5)* Hres;
      best_dir(0) = cos(best_heading);
      best_dir(1) = sin(best_heading);
      best_dir(2) = 0.0;

      if(gen_guide_point){
        int i = best_idx;
        int half_width_ = (width_-1)/2;
        double filter[width_];
        for (int u=0; u<width_; u++){
          int _i_ = i+u-half_width_;
          filter[u] = Histogram_3d[0][_i_<0 ? Hcnt+_i_ : _i_>=Hcnt ? _i_-Hcnt : _i_];
        }
        max_distance = min(max(1.0,average(filter,width_)),min(uav2goal_norm,Obstacle_3d[0][best_idx]));
      }
    }
  }
  else if (isCylindrical)
  {
    int best_idx[2];
    best_idx[0] = -1;
    best_idx[1] = -1;
    double best_value = 0;
    // Calculation of moving orientation
    double goal_height_v = min(double(Vcnt),max(0.0,Vcnt/2 + uav2goal(2)/Vres)); // 0 ~ Vcnt
    double goal_heading_h = sign(uav2goal(1)) * acos(uav2goal(0) / uav2goal_xy_norm); // -pi ~ pi
    goal_heading_h = goal_heading_h<0 ? goal_heading_h+2*M_PI : goal_heading_h; // 0 ~ 2*pi
    // Determine if need to find a guide point (not necessary if target point is visible)
    double current_height_v = Vcnt/2;
    if( uav2goal_norm > Histogram_3d[int(goal_height_v)][int(goal_heading_h/Hres)] ){
      double current_heading_h = sign(current_vel(1)) *acos(current_vel(0) / hor_current_vel_norm); // -pi ~ pi
      // Calculation of moving orientation
      current_heading_h = current_heading_h<0 ? current_heading_h+2*M_PI : current_heading_h; // 0 ~ 2*pi
      // Modify the weights on histogram
      for(int v=0; v<Vcnt; v++)
        for(int h=0; h< Hcnt; h++){
          double height_v = (v + 0.5); // 0 ~ Vcnt
          double angle_h = (h + 0.5)* Hres; // 0 ~ 2pi
          double height_err_goal_v = fabs(height_v - goal_height_v)/Vcnt; // 0 ~ 1
          double angle_err_goal_h = fabs(angle_h - goal_heading_h); // 0 ~ 2pi
          // if (angle_err_goal_h>M_PI) angle_err_goal_h = 2*M_PI - angle_err_goal_h; // 0 ~ pi

          // The smaller the angle difference, the higher weight
          int X0=Hcnt/4;
          int Y0=Vcnt/2;
          double X=fmod(angle_err_goal_h/M_PI,1.0)*2*X0;
          double Y=height_err_goal_v*2*Y0;
          int v_j=X0+(X-X0)*(1-2*int(angle_err_goal_h/M_PI));
          int u_i=Y;
          Histogram_3d[v][h] *= (Weights_3d[u_i][v_j]);

          // Histogram_3d[v][h] *= fun_norm(height_err_goal_v,_std_)*fun_cos((cos(angle_err_goal_h)+1)/2, pow_goal, min_value);
          if ( current_vel.norm()>min_vel_default){
            double height_err_vel_v = fabs(height_v - current_height_v)/current_height_v; // 0 ~ 1
            double angle_err_vel_h = fabs(angle_h - current_heading_h); // 0 ~ 2pi

            double X=fmod(angle_err_vel_h/M_PI,1.0)*2*X0;
            double Y=height_err_vel_v*2*Y0;
            int v_j=X0+(X-X0)*(1-2*int(angle_err_vel_h/M_PI));
            int u_i=Y;
            Histogram_3d[v][h] *= (Weights_3d[u_i][v_j]);
            // Histogram_3d[v][h] *= fun_norm(height_err_vel_v,std_normal_vel)*fun_cos((cos(angle_err_vel_h)+1)/2, _pow_, min_value);
          }
        }
      // Finding out the optimal direction
      for(int i=0; i<Vcnt; i++)
        for(int j=0; j<Hcnt; j++){
          int half_width_ = (width_-1)/2;
          double filter[width_*width_];
          for (int u=0; u<width_; u++)
            for (int v=0; v<width_; v++)
            {
              int _i_ = i+u-half_width_;
              int _j_ = j+v-half_width_;
              filter[u*width_+v] = Histogram_3d[_i_<0 ? -_i_-1 : (_i_>=Vcnt ? 2*Vcnt-1-_i_ : _i_)][((_i_<0) || (_i_>=Vcnt)) ? ((_j_+Hcnt/2)%Hcnt) : (_j_<0 ? Hcnt+_j_ : (_j_>=Hcnt ? _j_-Hcnt : _j_))];
            }
          double value_ = average(filter,width_*width_) + minimum(filter,width_*width_);
          if(value_>best_value){
            best_value = value_;
            best_idx[0] = i;
            best_idx[1] = j;
            if(gen_guide_point) max_distance = average(filter,width_*width_);
          }
        }
    }

    // Return to Optimal Guidance
    if (best_idx[0] == -1 || best_idx[1] == -1){
      best_dir = Eigen::Vector3d(0.0,0.0,0.0);
    }else {
      has_best_dir = true;
      max_distance = min(max(1.0,max_distance),min(uav2goal_norm,Obstacle_3d[best_idx[0]][best_idx[1]]));
      double best_heading_h  = (best_idx[1] + 0.5)* Hres; // 0 ~ 2*pi
      best_dir(0) = max_distance*cos(best_heading_h); // adjustable: 0.9
      best_dir(1) = max_distance*sin(best_heading_h);
      best_dir(2) = (best_idx[0] + 0.5 - Vcnt/2)*Vres;

      if(gen_guide_point) max_distance=1.0;
      else best_dir.normalize();
    }

  }
  else if (isSpherical)
  {
    int best_idx[2];
    best_idx[0] = -1;
    best_idx[1] = -1;
    double best_value = 0;
    // Calculation of moving orientation
    double goal_heading_v = sign(uav2goal(2)) * acos(uav2goal_xy_norm / uav2goal_norm); // -pi/2 ~ pi/2
    double goal_heading_h = sign(uav2goal(1)) * acos(uav2goal(0) / uav2goal_xy_norm); // -pi ~ pi
    goal_heading_h = goal_heading_h<0 ? goal_heading_h+2*M_PI : goal_heading_h; // 0 ~ 2*pi
    //Determine if need to find a guide point (not necessary if target point is visible)
    if( uav2goal_norm > Histogram_3d[int(min(double(Vcnt),max(0.0,Vcnt/2 + goal_heading_v/Vres)))][int(goal_heading_h/Hres)] )
    {
      // Modify the weights on histogram
      for(int v=0; v<Vcnt; v++)
        for(int h=0; h<Hcnt; h++)
        {
          double angle_v = (v + 0.5)* Vres - M_PI/2; // -pi/2 ~ pi/2
          double angle_h = (h + 0.5)* Hres; // 0 ~ 2pi
          double angle_err_goal_v = fabs(angle_v - goal_heading_v); // 0 ~ pi
          double angle_err_goal_h = fabs(angle_h - goal_heading_h); // 0 ~ 2*pi
          // if (angle_err_goal_h>M_PI) angle_err_goal_h = 2*M_PI - angle_err_goal_h; // 0 ~ pi

          // The smaller the angle difference, the higher weight
          int X0=Hcnt/4;
          int Y0=Vcnt/4;
          double X=fmod(angle_err_goal_h/M_PI,1.0)*2*X0;
          double Y=fmod(angle_err_goal_v/(M_PI/2),1.0)*2*Y0;
          int v_j=X0+(X-X0)*(1-2*((int(angle_err_goal_h/M_PI)+int(angle_err_goal_v/(M_PI/2)))%2));
          int u_i=Y0+(Y-Y0)*(1-2*(int(angle_err_goal_v/(M_PI/2))%2));
          Histogram_3d[v][h] *= (Weights_3d[u_i][v_j]);

          // Histogram_3d[v][h] *= (fun_norm(angle_err_goal_v,0.9)+min_value)/(1+min_value) * (fun_norm(angle_err_goal_h,1.2)+min_value)/(1+min_value);
          // Histogram_3d[v][h] *= (fun_cos((cos(angle_err_goal_v)+1)/2, _pow_, min_value) * fun_cos((cos(angle_err_goal_h)+1)/2, _pow_, min_value));
          if ( current_vel.norm()>min_vel_default)
          {
            // Calculation of moving orientation
            double current_heading_h = sign(current_vel(1)) *acos(current_vel(0) / hor_current_vel_norm); // -pi ~ pi
            current_heading_h = current_heading_h<0 ? current_heading_h+2*M_PI : current_heading_h; // 0 ~ 2*pi
            double angle_err_vel_v = fabs(angle_v - sign(current_vel(2)) *acos(hor_current_vel_norm / current_vel.norm())); // 0 ~ pi
            double angle_err_vel_h = fabs(angle_h - current_heading_h); // 0 ~ 2*pi
            // if (angle_err_vel_h>M_PI) angle_err_vel_h = 2*M_PI - angle_err_vel_h; // 0 ~ pi

            double X=fmod(angle_err_vel_h/M_PI,1.0)*2*X0;
            double Y=fmod(angle_err_vel_v/(M_PI/2),1.0)*2*Y0;
            int v_j=X0+(X-X0)*(1-2*((int(angle_err_vel_h/M_PI)+int(angle_err_vel_v/(M_PI/2)))%2));
            int u_i=Y0+(Y-Y0)*(1-2*(int(angle_err_vel_v/(M_PI/2))%2));
            Histogram_3d[v][h] *= (Weights_3d[u_i][v_j]);

            // Histogram_3d[v][h] *= (fun_norm(angle_err_vel_v,0.9)+min_value)/(1+min_value) * (fun_norm(angle_err_vel_h,1.2)+min_value)/(1+min_value);
            // Histogram_3d[v][h] *= (fun_cos((cos(angle_err_vel_v)+1)/2, _pow_, min_value) * fun_cos((cos(angle_err_vel_h)+1)/2, _pow_, min_value));
          }
        }
      // Return to Optimal Guidance
      for(int i=0; i<Vcnt; i++)
        for(int j=0; j<Hcnt; j++)
        {
          int half_width_ = (width_-1)/2;
          double filter_w[width_*width_];
          // double filter_o[width_*width_];
          for (int u=0; u<width_; u++)
            for (int v=0; v<width_; v++)
            {
              int _i_ = i+u-half_width_;
              int _j_ = j+v-half_width_;
              int u_i, v_j;
              if(_i_ < 0){
                u_i = -_i_-1;
                v_j = (_j_+Hcnt/2)%Hcnt;
              } else if (_i_>=Vcnt) {
                u_i = 2*Vcnt-1-_i_;
                v_j = (_j_+Hcnt/2)%Hcnt;
              } else {
                u_i = _i_;
                if(_j_<0) v_j = Hcnt+_j_;
                else if(_j_>=Hcnt) v_j = _j_-Hcnt;
                else v_j = _j_;
              }
              filter_w[u*width_+v] = Histogram_3d[u_i][v_j];
              // filter_o[u*width_+v] = Obstacle_3d[u_i][v_j];
            }
          double value_ = average(filter_w,width_*width_) + minimum(filter_w,width_*width_);
          if(value_>best_value){
            best_value = value_;
            best_idx[0] = i;
            best_idx[1] = j;
            if(gen_guide_point) max_distance = average(filter_w,width_*width_);
          }
        }
    }
    //  cout << "the best direction: row=" << best_idx[0] << "[" << Hcnt << "], col=" << best_idx[1] << "[" << Vcnt << "]" << endl;
    if (best_idx[0] == -1 || best_idx[1] == -1){
      best_dir = Eigen::Vector3d(0.0,0.0,0.0);
    }else {
      has_best_dir = true;
      double best_heading_v  = (best_idx[0] + 0.5)* Vres-M_PI/2; // -pi/2 ~ pi/2
      double best_heading_h  = (best_idx[1] + 0.5)* Hres; // 0 ~ 2*pi
      best_dir(0) = cos(best_heading_v)*cos(best_heading_h);
      best_dir(1) = cos(best_heading_v)*sin(best_heading_h);
      best_dir(2) = sin(best_heading_v);
			
      if(gen_guide_point) max_distance = min(max(0.5,max_distance),min(uav2goal_norm,Obstacle_3d[best_idx[0]][best_idx[1]]));
    }
  }
  
  mutex.unlock();

  if(show_time){
   std::chrono::duration<double, std::micro> elapsed__ = std::chrono::system_clock::now() - start_gen; 
   printf("Weighted Histogram generate takes %f [us].\n", elapsed__.count());
  }

  // weights histogram
  /*
    for (int i=0; i<Vcnt; i++) {
      for (int j=0; j<Hcnt; j++) 
        cout << Histogram_3d[i][j] << ", ";
      cout << endl;
    }
    cout << endl;
  */

  /* ***************** Return ***************** */
  // 1 if ok but dangerous; 0 if safe
  int planner_state = 1;  
  if(safe_cnt<3)
    planner_state = 0;  // Successful planning, and safe
  
  if(gen_guide_point)
    desired = start_pos + best_dir * max_distance;
  else
    desired = start_pos + best_dir * limit_v_norm;

  if(show_time){
   std::chrono::duration<double, std::micro> elapsed_seconds = std::chrono::system_clock::now() - start_gen; 
   printf("Histogram total calculation takes %f [us].\n", elapsed_seconds.count());
  }

  has_best_dir = false;
    
  return planner_state;
}

// angle_cen: -pi ~ pi; angle_range: 0~pi/2
void HIST::PolarCoordinateHist(double angle_cen, double angle_range, double val)
{
  angle_cen = angle_cen<0 ? angle_cen+2*M_PI : angle_cen; // 0 ~ 2pi
  double angle_max = angle_cen + angle_range;
  double angle_min = angle_cen - angle_range; // -pi/2 ~ 5 pi/2
  int cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
  int cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);
  if(cnt_min>cnt_max)
  {
    for(int i=cnt_min; i<Hcnt; i++)
    {
      if (Histogram_3d[0][i] > val) Histogram_3d[0][i] = val;
    }
    for(int i=0;i<=cnt_max; i++)
    {
      if (Histogram_3d[0][i] > val) Histogram_3d[0][i] =val;
    }
  }else
  {
    for(int i=cnt_min; i<=cnt_max; i++)
    {
      if (Histogram_3d[0][i] > val) Histogram_3d[0][i] = val;
    }
  }
}

// hor_angle_cen: -pi ~ pi; ver_angle_cen: -pi/2 ~ pi/2; angle_range: 0~pi/2
void HIST::SphericalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double angle_range, double obs_dist)
{
  hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
  double val = obs_dist - forbidden_range;

  ver_angle_cen = ver_angle_cen + M_PI/2; // 0 ~ pi
  double ver_angle_max = ver_angle_cen + angle_range;
  double ver_angle_min = ver_angle_cen - angle_range; // -pi/2 ~ 3pi/2

  if(ver_angle_min < 0){
    for(int i =0; i<-ver_angle_min/Vres; i++)
      for(int j=0; j<Hcnt; j++)
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
    ver_angle_min = -ver_angle_min + Vres; // 0 ~ pi
  }

  if(ver_angle_max > M_PI){
    for(int i =(2*M_PI-ver_angle_max)/Vres; i<Vcnt; i++)
      for(int j=0; j<Hcnt; j++)
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
    ver_angle_max = (2*M_PI-ver_angle_max) - Vres; // 0 ~ pi
  }
  int ver_cnt_min = int(ver_angle_min/Vres);
  int ver_cnt_max = int(ver_angle_max/Vres);
  for(int i=ver_cnt_min; i<=ver_cnt_max; i++){
    double r = obs_dist*obs_dist - forbidden_range*forbidden_range;
    r = sqrt(max(0.0,r));

    if (i==int(ver_angle_cen/Vres)){
      angle_range = atan(forbidden_range	/(r*cos(ver_angle_cen-M_PI/2)));// alpha/2 < pi/2
    }else if (i<ver_angle_cen/Vres){
      double delta = ver_angle_cen-(i+1)*Vres;
      angle_range = atan(sqrt(pow(forbidden_range*cos(delta),2)-pow(r*sin(delta),2))/(r*cos((i+1)*Vres-M_PI/2)));// alpha/2 < pi/2
    }else{
      double delta = (i)*Vres-ver_angle_cen;
      angle_range = atan(sqrt(pow(forbidden_range*cos(delta),2)-pow(r*sin(delta),2))/(r*cos((i)*Vres-M_PI/2)));// alpha/2 < pi/2
    }

    double hor_angle_max = hor_angle_cen + angle_range;
    double hor_angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5pi/2
    int hor_cnt_min = int((hor_angle_min<0 ? hor_angle_min+2*M_PI : hor_angle_min>2*M_PI ? hor_angle_min-2*M_PI : hor_angle_min)/Hres);
    int hor_cnt_max = int((hor_angle_max<0 ? hor_angle_max+2*M_PI : hor_angle_max>2*M_PI ? hor_angle_max-2*M_PI : hor_angle_max)/Hres);

    if(hor_cnt_min>hor_cnt_max){
      for(int j=hor_cnt_min; j<Hcnt; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
      for(int j=0;j<=hor_cnt_max; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
    }else{
      for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
    }
  }
}

// hor_angle_cen: -pi ~ pi; ver_idx_cen: 0 ~ Vcnt; idx_range: 0~Vcnt/2; hor_obs_dist; val
void HIST::CylindricalCoordinateHist(double hor_angle_cen, double ver_idx_cen, double hor_obs_dist)
{
  hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
  double idx_range = forbidden_range/Vres;
  double ver_idx_min = max(ver_idx_cen - idx_range,0.0);
  double ver_idx_max = min(ver_idx_cen + idx_range,double(Vcnt));

  double angle_range;
  for(int i=int(ver_idx_min); i<ver_idx_max; i++){
    if (i==int(ver_idx_cen)){
      angle_range = asin(min(forbidden_range/hor_obs_dist,1.0));// alpha/2 < pi/2
    }else if (i<int(ver_idx_cen)){
      angle_range = asin(min(1.0,sqrt(forbidden_range*forbidden_range - pow((ver_idx_cen-(i+1))*Vres,2)) / hor_obs_dist));// alpha/2 < pi/2
    }else{
      angle_range = asin(min(1.0,sqrt(forbidden_range*forbidden_range - pow((ver_idx_cen-(i))*Vres,2)) / hor_obs_dist));// alpha/2 < pi/2
    }			
    double hor_angle_max = hor_angle_cen + angle_range;
    double hor_angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5pi/2
    int hor_cnt_min = int((hor_angle_min<0 ? hor_angle_min+2*M_PI : hor_angle_min>2*M_PI ? hor_angle_min-2*M_PI : hor_angle_min)/Hres);
    int hor_cnt_max = int((hor_angle_max<0 ? hor_angle_max+2*M_PI : hor_angle_max>2*M_PI ? hor_angle_max-2*M_PI : hor_angle_max)/Hres);

    double val = hor_obs_dist - forbidden_range;
    if(hor_cnt_min>hor_cnt_max)
    {
      for(int j=hor_cnt_min; j<Hcnt; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
      for(int j=0;j<=hor_cnt_max; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
    }else{
      for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
      }
    }
  }
}

// Get the location and distance of the nearest obstacle point
double HIST::getDist(Eigen::Vector3d pos, Eigen::Vector3d &closest)
{
  double dist = min(abs(pos[2]-ceil_height),abs(pos[2]-ground_height));
  closest << pos[0], pos[1], (abs(pos[2]-ceil_height)>abs(pos[2]-ground_height))? ground_height : ceil_height;
  
  if(!has_local_map_){
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if(!has_local_map_){
      cout << "[getDist] Check map input!" << endl;
      return dist;
    }
  }
	
  mutex.lock();
  
  if(latest_local_pcl_.points.size() == 0){
    mutex.unlock();
    return dist;
  }

  Eigen::Vector3d diff = pos - capture_pos;
  double eff_dist = 2*forbidden_range+safe_distance;

  if(diff.norm()<eff_dist){
    for (size_t i = 0; i < latest_local_pcl_.points.size(); ++i) 
    {
      diff[0] = pos[0] - latest_local_pcl_.points[i].x;
      diff[1] = pos[1] - latest_local_pcl_.points[i].y;
      diff[2] = pos[2] - latest_local_pcl_.points[i].z;
      
      double d = diff.norm();
      if(d<dist){
        dist = d;
        closest[0] = latest_local_pcl_.points[i].x;
        closest[1] = latest_local_pcl_.points[i].y;
        closest[2] = latest_local_pcl_.points[i].z;
      }
    }
  }else{
    double pos_dist = diff.norm();
    double hor_angle_cen = sign(diff(1)) * acos(diff(0) / Eigen::Vector2d(diff(0),diff(1)).norm());// hor_angle_cen: -pi ~ pi	
    double angle_range = asin(eff_dist/pos_dist);// angle_range: 0~pi/2
    if (is_2D)
    {
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);
      if(cnt_min>cnt_max)
      {
        for(int i=cnt_min; i<Hcnt; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];

            double d = diff.norm();
            if(d<dist){
              dist = d;
              closest[0] = Env_3d[0][0][i];
              closest[1] = Env_3d[1][0][i];
              closest[2] = Env_3d[2][0][i];
            }
          }
        }
        for(int i=0;i<=cnt_max; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];
            
            double d = diff.norm();
            if(d<dist){
              dist = d;
              closest[0] = Env_3d[0][0][i];
              closest[1] = Env_3d[1][0][i];
              closest[2] = Env_3d[2][0][i];
            }
          }
        }
      }else
      {
        for(int i=cnt_min; i<=cnt_max; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];
            
            double d = diff.norm();
            if(d<dist){
              dist = d;
              closest[0] = Env_3d[0][0][i];
              closest[1] = Env_3d[1][0][i];
              closest[2] = Env_3d[2][0][i];
            }
          }
        }
      }
    }
    else if (isCylindrical)
    {
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int hor_cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int hor_cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);

      double ver_idx_cen = min(double(Vcnt),max(0.0,Vcnt/2 + diff(2)/Vres));
      double idx_range = eff_dist/Vres;
      double ver_idx_min = max(ver_idx_cen - idx_range,0.0);
      double ver_idx_max = min(ver_idx_cen + idx_range,double(Vcnt));

      for(int i=int(ver_idx_min); i<ver_idx_max; i++){
        if(hor_cnt_min>hor_cnt_max)
        {
          for(int j=hor_cnt_min; j<Hcnt; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];

              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
          for(int j=0;j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
        }else{
          for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
        }
      }
    }
    else if (isSpherical)
    { 
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int hor_cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int hor_cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);

      double ver_angle_cen = sign(diff(2)) * acos(Eigen::Vector2d(diff(0),diff(1)).norm() / pos_dist);// ver_angle_cen: -pi/2 ~ pi/2
      ver_angle_cen = ver_angle_cen + M_PI/2; // 0 ~ pi
      double ver_angle_max = ver_angle_cen + angle_range;
      double ver_angle_min = ver_angle_cen - angle_range; // -pi/2 ~ 3pi/2

      if(ver_angle_min < 0){
        for(int i =0; i<-ver_angle_min/Vres; i++)
          for(int j=0; j<Hcnt; j++)
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
        ver_angle_min = -ver_angle_min + Vres; // 0 ~ pi
      }

      if(ver_angle_max > M_PI){
        for(int i =(2*M_PI-ver_angle_max)/Vres; i<Vcnt; i++)
          for(int j=0; j<Hcnt; j++)
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
        ver_angle_max = (2*M_PI-ver_angle_max) - Vres; // 0 ~ pi
      }
      int ver_cnt_min = int(ver_angle_min/Vres);
      int ver_cnt_max = int(ver_angle_max/Vres);
      for(int i=ver_cnt_min; i<=ver_cnt_max; i++){
        if(hor_cnt_min>hor_cnt_max){
          for(int j=hor_cnt_min; j<Hcnt; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
          for(int j=0;j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
        }else{
          for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
                closest[0] = Env_3d[0][i][j];
                closest[1] = Env_3d[1][i][j];
                closest[2] = Env_3d[2][i][j];
              }
            }
          }
        }
      }
    }
  }
  
  mutex.unlock();

  return dist;
}

// pos: interest position; grad: negative gradient, Direction away from obstacles
double HIST::getDistWithGrad(Eigen::Vector3d pos, Eigen::Vector3d &grad)
{
  double grad_eff_dist = 2*forbidden_range+safe_distance;
  double dist_ground = pos[2]-ground_height;
  double dist_ceil = ceil_height-pos[2];
  grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0;
  double dist;
  if(abs(dist_ceil) > abs(dist_ground)){  // ground
    dist = abs(dist_ground);
    if(dist_ground < grad_eff_dist)
      grad[2] = max(1/(max(dist_ground-forbidden_range,0.01))-1/(forbidden_plus_safe_distance),0.0);
  }else{ // ceil
    dist = abs(dist_ceil);
    if(dist_ceil < grad_eff_dist)
      grad[2] = -max(1/(max(dist_ceil-forbidden_range,0.01))-1/(forbidden_plus_safe_distance),0.0);
  }
  if(!has_local_map_){
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if(!has_local_map_){
      return dist;
    }
  }

  mutex.lock();

  if(latest_local_pcl_.points.size() == 0){
    mutex.unlock();
    return dist;
  }

  Eigen::Vector3d diff = pos - capture_pos;
  if(diff.norm()<grad_eff_dist){
    for (size_t i = 0; i < latest_local_pcl_.points.size(); ++i) 
    {
      diff[0] = pos[0] - latest_local_pcl_.points[i].x;
      diff[1] = pos[1] - latest_local_pcl_.points[i].y;
      diff[2] = pos[2] - latest_local_pcl_.points[i].z;
      
      double d = diff.norm();
      if(d<dist){
        dist = d;
      }

      if(d<grad_eff_dist){
        double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
        grad[0] += diff[0]*grad_val;
        grad[1] += diff[1]*grad_val;
        grad[2] += diff[2]*grad_val;
      }
    }
  }else{
    double pos_dist = diff.norm();
    double hor_angle_cen = sign(diff(1)) * acos(diff(0) / Eigen::Vector2d(diff(0),diff(1)).norm());// hor_angle_cen: -pi ~ pi	
    double angle_range = asin(grad_eff_dist/pos_dist);// angle_range: 0~pi/2
    if (is_2D)
    {
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);
      if(cnt_min>cnt_max)
      {
        for(int i=cnt_min; i<Hcnt; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];

            double d = diff.norm();
            if(d<dist){
              dist = d;
            }

            if(d<grad_eff_dist){
              double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
              grad[0] += diff[0]*grad_val;
              grad[1] += diff[1]*grad_val;
              grad[2] += diff[2]*grad_val;
            }
          }
        }
        for(int i=0;i<=cnt_max; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];
            
            double d = diff.norm();
            if(d<dist){
              dist = d;
            }

            if(d<grad_eff_dist){
              double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
              grad[0] += diff[0]*grad_val;
              grad[1] += diff[1]*grad_val;
              grad[2] += diff[2]*grad_val;
            }
          }
        }
      }else
      {
        for(int i=cnt_min; i<=cnt_max; i++)
        {
          if (Env_3d[3][0][i] > 0.0){
            diff[0] = pos[0] - Env_3d[0][0][i];
            diff[1] = pos[1] - Env_3d[1][0][i];
            diff[2] = pos[2] - Env_3d[2][0][i];
            
            double d = diff.norm();
            if(d<dist){
              dist = d;
            }

            if(d<grad_eff_dist){
              double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
              grad[0] += diff[0]*grad_val;
              grad[1] += diff[1]*grad_val;
              grad[2] += diff[2]*grad_val;
            }
          }
        }
      }
    }
    else if (isCylindrical)
    {
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int hor_cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int hor_cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);

      double ver_idx_cen = min(double(Vcnt),max(0.0,Vcnt/2 + diff(2)/Vres));
      double idx_range = grad_eff_dist/Vres;
      double ver_idx_min = max(ver_idx_cen - idx_range,0.0);
      double ver_idx_max = min(ver_idx_cen + idx_range,double(Vcnt));

      for(int i=int(ver_idx_min); i<ver_idx_max; i++){
        if(hor_cnt_min>hor_cnt_max)
        {
          for(int j=hor_cnt_min; j<Hcnt; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];

              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
          for(int j=0;j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
        }else{
          for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
        }
      }
    }
    else if (isSpherical)
    { 
      hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
      double angle_max = hor_angle_cen + angle_range;
      double angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5 pi/2
      int hor_cnt_min = floor((angle_min<0 ? angle_min+2*M_PI : angle_min>2*M_PI ? angle_min-2*M_PI : angle_min)/Hres);
      int hor_cnt_max = floor((angle_max<0 ? angle_max+2*M_PI : angle_max>2*M_PI ? angle_max-2*M_PI : angle_max)/Hres);

      double ver_angle_cen = sign(diff(2)) * acos(Eigen::Vector2d(diff(0),diff(1)).norm() / pos_dist);// ver_angle_cen: -pi/2 ~ pi/2
      ver_angle_cen = ver_angle_cen + M_PI/2; // 0 ~ pi
      double ver_angle_max = ver_angle_cen + angle_range;
      double ver_angle_min = ver_angle_cen - angle_range; // -pi/2 ~ 3pi/2

      if(ver_angle_min < 0){
        for(int i =0; i<-ver_angle_min/Vres; i++)
          for(int j=0; j<Hcnt; j++)
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
        ver_angle_min = -ver_angle_min + Vres; // 0 ~ pi
      }

      if(ver_angle_max > M_PI){
        for(int i =(2*M_PI-ver_angle_max)/Vres; i<Vcnt; i++)
          for(int j=0; j<Hcnt; j++)
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
        ver_angle_max = (2*M_PI-ver_angle_max) - Vres; // 0 ~ pi
      }
      int ver_cnt_min = int(ver_angle_min/Vres);
      int ver_cnt_max = int(ver_angle_max/Vres);
      for(int i=ver_cnt_min; i<=ver_cnt_max; i++){
        if(hor_cnt_min>hor_cnt_max){
          for(int j=hor_cnt_min; j<Hcnt; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
          for(int j=0;j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
        }else{
          for(int j=hor_cnt_min; j<=hor_cnt_max; j++){
            if (Env_3d[3][i][j] > 0.0){
              diff[0] = pos[0] - Env_3d[0][i][j];
              diff[1] = pos[1] - Env_3d[1][i][j];
              diff[2] = pos[2] - Env_3d[2][i][j];
              
              double d = diff.norm();
              if(d<dist){
                dist = d;
              }

              if(d<grad_eff_dist){
                double grad_val = 1/(max(d-forbidden_plus_safe_distance,0.01))-1/(forbidden_range);
                grad[0] += diff[0]*grad_val;
                grad[1] += diff[1]*grad_val;
                grad[2] += diff[2]*grad_val;
              }
            }
          }
        }
      }
    }
  }
  mutex.unlock();

  return dist;
}

// generate hermite spline by interpolating guidance point
vector<Eigen::Vector3d> HIST::getSamples(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &start_acc, Eigen::Vector3d &goal, Eigen::Vector3d &goal_vel, Eigen::Vector3d &goal_acc, Eigen::Vector3d &guide_point, double& ts)
{
  /* ************ Target Visible ************ */ 
  if((start_pos-guide_point).norm() < forbidden_range){
    Eigen::Vector3d start_dir, goal_dir;
		
    if(goal_vel.norm()>min_vel_default)
      goal_dir = goal_vel;
    else
      goal_dir = min_vel_default*(goal-start_pos).normalized();

    if(start_vel.norm()>min_vel_default)
      start_dir = start_vel;
    else
      start_dir = min_vel_default*(goal-start_pos).normalized();

    int K = min(30.0,max(3.0,ceil((start_pos-goal).norm()/limit_v_norm/ts)));// the number of minddle path points except for start&end
    double tau = ts*(K+1);
    double tau2 = tau*tau;
    double tau3 = tau2*tau;
    Eigen::Matrix4d M;
    M << 1,0,0,0,
         0,1,0,0,
         -3/tau2,-2/tau,3/tau2,-1/tau,
         2/tau3,1/tau2,-2/tau3,1/tau2;
    //Eigen::MatrixXd samples(3, K + 6);
    vector<Eigen::Vector3d> samples;
		
    Eigen::MatrixXd T_(1,4);
    Eigen::MatrixXd P_(4,3);
    // start_point to goal
    P_.row(0) = start_pos.transpose();
    P_.row(1) = start_dir.transpose();
    P_.row(2) = goal.transpose();
    P_.row(3) = goal_dir.transpose();
    for(double i=0.0; i<K+2; i+=1.0){
      double t = tau*i/(K+1); // [0~1]
      T_ << 1, t, t*t, t*t*t;
      Eigen::MatrixXd point(1,3);
      point = T_*M*P_;
      //samples.block(0, i, 3, 1) = point.transpose();
      samples.push_back(point.transpose());
      //cout << "point: " << point << endl;
    }
		
    //samples.col(K + 2) = 10*start_vel;
    //samples.col(K + 3) = 10*goal_vel;
    //samples.col(K + 4) = 10*start_acc;
    //samples.col(K + 5) = 10*goal_acc;
    samples.push_back(start_vel);
    samples.push_back(goal_vel);
    samples.push_back(start_acc);
    samples.push_back(goal_acc);
    return samples;
  }
	
  /* ************ Target Unvisible ************ */ 
  /* Option 1: 3p1v Cubic Hermite Interpolating Spline */ 
  // double angle_factor = start_vel.norm()*acos(start_vel.dot(guide_point-start_pos)/start_vel.norm()/(guide_point-start_pos).norm())/M_PI+1; //1~2
  // double s = angle_factor*(guide_point-start_pos).norm() / (angle_factor*(guide_point-start_pos).norm()+(guide_point-goal).norm());// time at guide_point
  // Eigen::Matrix4d M;
  // Eigen::MatrixXd T_(1,4);
  // Eigen::MatrixXd P_(4,3);
  // P_.row(0) = start_pos.transpose();
  // P_.row(1) = start_vel.transpose();
  // P_.row(2) = guide_point.transpose();
  // P_.row(3) = goal.transpose();
  
  // M <<     (1+s)/s/s,       1/s,  1/(s*s*(s-1)),  1/(1-s),
  //       -1-(1+s)/s/s,  -(s+1)/s,  1/(s*s*(1-s)),  s/(s-1),
  //                   0,         1,              0,        0,
  //                   1,         0,              0,        0;

  // double curve_length = 0.0;
  // for(int i=0; i<100; i++){
  //   T_ << 1e-6*(3*i*i + 3*i + 1), 1e-4*(2*i + 1), 0.01, 0;
  //   curve_length += (T_*M*P_).norm();
  // }
  
  // int K = min(20.0,max(10.0,ceil(curve_length/limit_v_norm/ts)-1));
  // cout << "[HIST]: curve_length: " << curve_length << ", limit_v_norm: " << limit_v_norm << ", ts: " << ts << ", K: " << K << endl;

  // //cout << "samples: " << endl;
  // //Eigen::MatrixXd samples(3, K + 6);
  // vector<Eigen::Vector3d> samples;
  
  // for(double i=0.0; i<K+2; i+=1.0){
  //   double t = i/(K+1); // 0~1
  //   //cout << "t: " << t << "\tpos: ";
  //   T_ << t*t*t, t*t, t, 1;
  //   Eigen::MatrixXd point(1,3);
  //   point = T_*M*P_;
  //   //cout << point << endl;
  //   //samples.block(0, i, 3, 1) = point.transpose();
  //   samples.push_back(point.transpose());
  // }

  // //samples.col(K + 2) = start_vel;
  // //samples.col(K + 3) = goal_vel;
  // //samples.col(K + 4) = start_acc;
  // //samples.col(K + 5) = goal_acc;
  // samples.push_back(start_vel);
  // samples.push_back(goal_vel);
  // samples.push_back(start_acc);
  // samples.push_back(goal_acc);
  // return samples;


  /* Option 2: 2p2v Cubic Hermite Interpolating Spline */
  int K_1 = min(15.0,max(2.0,ceil((start_pos-guide_point).norm()/limit_v_norm/ts)+2));
  int K_2 = min(15.0,max(2.0,ceil((guide_point-goal).norm()/limit_v_norm/ts)));
  //int K = K_1+K_2+1; // the number of minddle path points except for start&end
  
  double tau_1, tau_12, tau_13, tau_2, tau_22, tau_23;
  tau_1 = ts*(K_1+1);
  tau_12 = tau_1*tau_1;
  tau_13 = tau_12*tau_1;
  tau_2 = ts*(K_2+1);
  tau_22 = tau_2*tau_2;
  tau_23 = tau_22*tau_2;
  
  Eigen::Matrix4d M_1, M_2;
  M_1 <<1,              0,        0,       0,
        0,              1,        0,       0,
        -3/tau_12,-2/tau_1, 3/tau_12,-1/tau_1,
        2/tau_13,1/tau_12,-2/tau_13,1/tau_12;
  M_2 <<1,              0,        0,       0,
        0,              1,        0,       0,
        -3/tau_22,-2/tau_2, 3/tau_22,-1/tau_2,
        2/tau_23,1/tau_22,-2/tau_23,1/tau_22;
        
  Eigen::Vector3d start_dir, goal_dir, guide_point_dir;

  if(goal_vel.norm()>min_vel_default)
    goal_dir = goal_vel;
  else
    goal_dir = min_vel_default*(goal-guide_point).normalized();

  if(start_vel.norm()>min_vel_default)
    start_dir = start_vel;
  else
    start_dir = min_vel_default*(guide_point-start_pos).normalized();

  guide_point_dir = (3/tau_22*(goal-guide_point) - 3/tau_12*(start_pos-guide_point) - goal_dir/tau_2 - start_dir/tau_1)
                    /  (2/tau_1+2/tau_2);
  
  //Eigen::MatrixXd samples(3, K + 6);
  vector<Eigen::Vector3d> samples;
  
  Eigen::MatrixXd T_(1,4);
  Eigen::MatrixXd P_(4,3);
  // start_point to guide_point
  P_.row(0) = start_pos.transpose();
  P_.row(1) = start_dir.transpose();
  P_.row(2) = guide_point.transpose();
  P_.row(3) = guide_point_dir.transpose();
  for(double i=0.0; i<K_1+1; i+=1.0){
    double t = tau_1*i/(K_1+1); // [0~1)
    T_ << 1, t, t*t, t*t*t;
    Eigen::MatrixXd point(1,3);
    point = T_*M_1*P_;
    //samples.block(0, i, 3, 1) = point.transpose();
    samples.push_back(point.transpose());
  }
  // guide_point to goal
  P_.row(0) = guide_point.transpose();
  P_.row(1) = guide_point_dir.transpose();
  P_.row(2) = goal.transpose();
  P_.row(3) = goal_dir.transpose();
  for(double i=0.0; i<K_2+2; i+=1.0){
    double t = tau_2*i/(K_2+1); // [0~1]
    T_ << 1, t, t*t, t*t*t;
    Eigen::MatrixXd point(1,3);
    point = T_*M_2*P_;
    //samples.block(0, K_1+1+i, 3, 1) = point.transpose();
    samples.push_back(point.transpose());
  }

  //samples.col(K + 2) = start_vel;
  //samples.col(K + 3) = goal_vel;
  //samples.col(K + 4) = start_acc;
  //samples.col(K + 5) = goal_acc;
  samples.push_back(start_vel);
  samples.push_back(goal_vel);
  samples.push_back(start_acc);
  samples.push_back(goal_acc);
  return samples;


  /* Option 3: 2p2v Polygonal Chain */
  // Eigen::Matrix4d M;
  // M <<  1,0,0,0,
  //       0,1,0,0,
  //       -3,-2,3,-1,
  //       2,1,-2,1;
  // Eigen::Vector3d start_dir, goal_dir, guide_point_dir;
  
  // if(goal_vel.norm()>min_vel_default)
  //   goal_dir = goal_vel;
  // else
  //   goal_dir = min_vel_default*(goal-guide_point).normalized();

  // if(start_vel.norm()>min_vel_default)
  //   start_dir = start_vel;
  // else
  //   start_dir = min_vel_default*(guide_point-start_pos).normalized();
  
  // guide_point_dir = 3/4*(goal-start_pos) - 1/4*(goal_dir+ start_dir);

  // int K_1 = min(10.0,max(4.0,ceil((start_pos-guide_point).norm()/limit_v_norm/ts+2)));
  // int K_2 = min(10.0,max(2.0,ceil((guide_point-goal).norm()/limit_v_norm/ts)));
  // int K = K_1+K_2+1; // the number of minddle path points except for start&end
  // //Eigen::MatrixXd samples(3, K + 6);
  // vector<Eigen::Vector3d> samples;
  // //samples.col(0) = start_pos;
  // samples.push_back(start_pos);

  // // polygonal chain
  // int NUM = piecewise_interpolation_num;
  // Eigen::MatrixXd points(3, NUM*(K + 1)+1);
  // Eigen::MatrixXd T_(1,4);
  // Eigen::MatrixXd P_(4,3);
  // // start_point to guide_point
  // P_.row(0) = start_pos.transpose();
  // P_.row(1) = start_dir.transpose();
  // P_.row(2) = guide_point.transpose();
  // P_.row(3) = guide_point_dir.transpose();
  // for(double i=0.0; i<NUM*(K_1+1); i+=1.0){
  //   double t = i/(NUM*(K_1+1)); // [0~1)
  //   T_ << 1, t, t*t, t*t*t;
  //   Eigen::MatrixXd point(1,3);
  //   point = T_*M*P_;
  //   points.block(0, i, 3, 1) = point.transpose();
  //   //samples.push_back(point.transpose());
  // }
  // // guide_point to goal
  // P_.row(0) = guide_point.transpose();
  // P_.row(1) = guide_point_dir.transpose();
  // P_.row(2) = goal.transpose();
  // P_.row(3) = goal_dir.transpose();
  // for(double i=0.0; i<NUM*(K_2+1)+1; i+=1.0){
  //   double t = i/(NUM*(K_2+1)); // [0~1]
  //   T_ << 1, t, t*t, t*t*t;
  //   Eigen::MatrixXd point(1,3);
  //   point = T_*M*P_;
  //   points.block(0, NUM*(K_1+1)+i, 3, 1) = point.transpose();
  //   //samples.push_back(point.transpose());
  // }

  // double length_total = 0.0;
  // for(int i=0;i<NUM*(K+1);i++)
  //   length_total += (points.block(0, i, 3, 1) - points.block(0, i+1, 3, 1)).norm();
  // //	cout << "length_total: " << length_total << endl;
  // //	cout << "segments: " << K+1 << endl;
  // double metric = length_total/(K+1);
  // //	cout << "metric: " << metric << endl;
  // double sum = 0.01*metric; int idx = 0;
  // for(int i=0;i<NUM*(K+1);i++){
  //   if(sum>metric){
  //     sum -= metric;
  //     idx++;
  //     //samples.block(0, idx, 3, 1) = points.block(0, i, 3, 1);
  //     samples.push_back(points.block(0, i, 3, 1));
  //   }
  //   sum += (points.block(0, i, 3, 1) - points.block(0, i+1, 3, 1)).norm();
  // }

  // //samples.col(K + 1) = goal;
  // samples.push_back(goal);
  // //samples.col(K + 2) = start_vel;
  // //samples.col(K + 3) = goal_vel;
  // //samples.col(K + 4) = start_acc;
  // //samples.col(K + 5) = goal_acc;
  // samples.push_back(start_vel);
  // samples.push_back(goal_vel);
  // samples.push_back(start_acc);
  // samples.push_back(goal_acc);
  // return samples;
}

}
