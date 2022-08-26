#include "histogram.h"
#include "string.h"
#include "math.h"
#include "chrono" 
#include "ctime"
#include "iostream"

using namespace std::chrono; 
namespace Local_Planning
{
// Inner Parameter
int width_ = 3; // width_extended
double min_value = 0.3;

auto sign=[](double v)->double
{
    return v<0.0? -1.0:1.0;
};

double average(double a[],int n)
{
	int sum=0;
	for (int i=0;i<n;i++)
		sum+=a[i];
	return sum/n;
}

double minimum(double a[],int n)
{
	double min_val = a[0];
	for (int i=1;i<n;i++)
		if (a[i]<min_val)
			min_val=a[i];
	return min_val;
}

double normal(double input, double std)
{
	double u = 0; // mean
//	double std = 0.4; // std
    return exp(-0.5*((input-u)/std)*((input-u)/std));// input==0 -> 1; input==1 -> 0.04
}

// input: 0 ~ 1; min_val: 0 ~ 1; return: min_val ~ 1
double fun_cos(double input, double min_val)
{
	return (1-min_val)*input + min_val;
}

void HIST::init(ros::NodeHandle& nh)
{
    has_local_map_ = false;
    has_odom_ = false;
    has_best_dir = false;

	nh.param("local_planner/gen_guide_point", gen_guide_point, false);	// 是否生成histogram指导点
	nh.param("local_planner/max_ground_height", ground_height, 0.1);	// 地面高度
    nh.param("local_planner/ceil_height", ceil_height, 5.0);			// 天花板高度
    nh.param("local_planner/max_planning_vel", limit_v_norm, 0.4);		// 最大飞行速度
    nh.param("local_planner/sensor_max_range", sensor_max_range, 3.0);	// 最大探测距离
    nh.param("local_planner/forbidden_range", forbidden_range, 0.50);	// 障碍物影响距离
    nh.param("local_planner/safe_distance", safe_distance, 0.2);		// 安全停止距离
    forbidden_plus_safe_distance = safe_distance + forbidden_range;
    nh.param("histogram/is_2D", is_2D, false); 								// 是否2D平面规划
    nh.param("histogram/isCylindrical", isCylindrical, false); 				// 3D规划，是否柱面建图
    nh.param("histogram/isSpherical", isSpherical, false); 					// 3D规划，是否球面建图
    if(!is_2D && !isCylindrical && !isSpherical) isSpherical = true;	// 默认球面建图
    if(is_2D) {isCylindrical=false; isSpherical=false;}
    if(isCylindrical || isSpherical) is_2D=false;
    nh.param("histogram/h_cnt", Hcnt, 180); 									// 直方图横向个数(偶数)
    nh.param("histogram/v_cnt", Vcnt, 90); 									// 直方图纵向个数(偶数)
    
    Hres = 2*M_PI/Hcnt;
    if(isSpherical)
	    Vres = M_PI/Vcnt; // +/- pi/2
    else if(isCylindrical)
    	Vres = 2/Vcnt; // default: +/- 1m
    
    if(is_2D){
	    Histogram_2d = new double[Hcnt]();
    }else{
    	Histogram_3d = new double*[Vcnt]();
		for( int i=0; i<Vcnt; i++ )
			Histogram_3d[i] = new double[Hcnt]();
    }
}

void HIST::set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr)
{
    latest_local_pcl_ = *pcl_ptr;
    has_local_map_=true;
}

void HIST::set_odom(nav_msgs::Odometry cur_odom)
{
    cur_odom_ = cur_odom;
    has_odom_=true;
}

int HIST::generate(Eigen::Vector3d  &goal, Eigen::Vector3d &desired)
{
    if(!has_local_map_|| !has_odom_){
    	cout << "[Err] Check map input and odom input!" << endl;
        return 0;
	}
    if ((int)latest_local_pcl_.points.size() == 0) 
	    cout << "[Wrn] no point cloud" << endl;
	    
    if (isnan(goal(0)) || isnan(goal(1)) || isnan(goal(2))){
    	cout << "[Err] Goal Unkown!" << endl;
        return 0;
	}

    // 状态量
    std::chrono::time_point<std::chrono::system_clock> start, end; 
    start = std::chrono::system_clock::now(); 
    static int exec_num=0;
    
    Eigen::Vector3d current_pos;
    current_pos[0] = cur_odom_.pose.pose.position.x;
    current_pos[1] = cur_odom_.pose.pose.position.y;
    current_pos[2] = cur_odom_.pose.pose.position.z;
    if (current_pos[2]>ceil_height || current_pos[2]<ground_height){
    	cout << "[Err] Height is not in range!" << endl;
        return 0;
	}
    Eigen::Vector3d current_vel;
    current_vel[0] = cur_odom_.twist.twist.linear.x;
    current_vel[1] = cur_odom_.twist.twist.linear.y;
    current_vel[2] = cur_odom_.twist.twist.linear.z;
    Eigen::Vector3d uav2goal = goal - current_pos;

    // reset the Histogram
    // max sensor range
    if(is_2D)
		for(int i=0; i<Hcnt; i++)
		    Histogram_2d[i] = sensor_max_range;
	else
		for(int i=0; i<Vcnt; i++)
			for(int j=0; j<Hcnt; j++)
				Histogram_3d[i][j] = sensor_max_range;
	// virtual ground
	if(isSpherical)
		for(int i=0; i<acos((current_pos[2]-ground_height)/sensor_max_range)/Vres; i++)
			for(int j=0; j<Hcnt; j++)
				Histogram_3d[i][j] = min((current_pos[2]-ground_height)/cos((i+0.5)*Vres),sensor_max_range);
	else if(isCylindrical)
		for(int i=0; i<Vcnt/2-(current_pos[2]-ground_height)/Vres; i++)
			for(int j=0; j<Hcnt; j++)
				Histogram_3d[i][j] = 0.0;
	// virtual ciel
	if(isSpherical)
		for(int i=Vcnt-1; i>Vcnt-1-acos((ceil_height-current_pos[2])/sensor_max_range)/Vres; i--)
			for(int j=0; j<Hcnt; j++)
				Histogram_3d[i][j] = min((ceil_height-current_pos[2])/cos((Vcnt-1-i+0.5)*Vres),sensor_max_range);
	else if(isCylindrical)
		for(int i=Vcnt-1; i<Vcnt-1-(ceil_height-current_pos[2])/Vres; i--)
			for(int j=0; j<Hcnt; j++)
				Histogram_3d[i][j] = 0.0;
//    for(int i=0; i<Vcnt; i++){
//    	for(int j=0; j<Hcnt; j++){
//    		cout << Histogram_3d[i][j] << " , ";
//    	}
//    	cout << endl;
//    }
//	cout << endl;

    // 0 for not init; 1 for safe; 2 for ok but dangerous
    int local_planner_state=0;  
    int safe_cnt=0;

//    Eigen::Quaterniond cur_rotation_local_to_global(cur_odom_.pose.pose.orientation.w, cur_odom_.pose.pose.orientation.x, cur_odom_.pose.pose.orientation.y, cur_odom_.pose.pose.orientation.z); 
//    Eigen::Matrix<double,3,3> rotation_mat_local_to_global = cur_rotation_local_to_global.toRotationMatrix();
//    Eigen::Vector3d eulerAngle_yrp = rotation_mat_local_to_global.eulerAngles(2, 1, 0);// R_z*R_y*R_x
//    rotation_mat_local_to_global = Eigen::AngleAxisd(eulerAngle_yrp(0), Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // 遍历点云中的所有点
    Eigen::Vector3d p3d;
    vector<Eigen::Vector3d> obstacles;
    for (size_t i = 0; i < latest_local_pcl_.points.size(); ++i) 
    {
        p3d(0) = latest_local_pcl_.points[i].x;
        p3d(1) = latest_local_pcl_.points[i].y;
        p3d(2) = latest_local_pcl_.points[i].z; // World_ENU frame

        //　超出最大考虑距离，则忽略该点
        Eigen::Vector3d uav2obs = p3d - current_pos; 
        double dist_push = (uav2obs).norm();
        if(dist_push > sensor_max_range || isnan(dist_push))
            continue;

        double obs_dist = uav2obs.norm();
        if (is_2D){
		    //　不考虑地面上的点的排斥力
		    if(fabs(p3d(2))<ground_height)
		        continue;

		    if (fabs(uav2obs(2))>0.5) continue;
		    
		    uav2obs(2) = 0.0;
		    double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / uav2obs.norm());// obs_hor_angle_cen: -pi ~ pi
		    double angle_range;// angle_range: 0~pi/2
		    if(obs_dist>forbidden_plus_safe_distance)
		        angle_range = asin(forbidden_plus_safe_distance/obs_dist);
		    else if (obs_dist<=forbidden_plus_safe_distance)
		    {
		        angle_range = M_PI/2;
		        safe_cnt++;  // 非常危险
		    }
		    
		    PolarCoordinateHist(obs_hor_angle_cen, angle_range, max(obs_dist-forbidden_plus_safe_distance,0.0));
		}else if (isCylindrical){
		    if (fabs(uav2obs(2))>1) continue;
		    
		    double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm());// obs_hor_angle_cen: -pi ~ pi		    
		    double obs_ver_idx_cen = min(double(Vcnt),max(0.0,Vcnt/2 + uav2obs(2)/Vres));
		    
		    CylindricalCoordinateHist(obs_hor_angle_cen, obs_ver_idx_cen, forbidden_plus_safe_distance/Vres, Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm(), obs_dist-forbidden_plus_safe_distance);
		}else if (isSpherical){
		    double angle_range;// angle_range: 0~pi/2
		    if(obs_dist>forbidden_plus_safe_distance)
		        angle_range = asin(forbidden_plus_safe_distance/obs_dist);
		    else if (obs_dist<=forbidden_plus_safe_distance)
		    {
		        angle_range = M_PI/2;
		        safe_cnt++;  // 非常危险
		    }  
		    
		    double obs_hor_angle_cen = sign(uav2obs(1)) * acos(uav2obs(0) / Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm());// obs_hor_angle_cen: -pi ~ pi
		    double obs_ver_angle_cen = sign(uav2obs(2)) * acos(Eigen::Vector3d(uav2obs(0),uav2obs(1),0.0).norm() / uav2obs.norm());// obs_ver_angle_cen: -pi/2 ~ pi/2
		    
		    SphericalCoordinateHist(obs_hor_angle_cen, obs_ver_angle_cen, angle_range, max(obs_dist-forbidden_plus_safe_distance,1e-6));
//		    for(int i=0; i<Vcnt; i++){
//		    	for(int j=0; j<Hcnt; j++){
//		    		cout << Histogram_3d[i][j] << " , ";
//		    	}
//		    	cout << endl;
//		    }
//	    	cout << endl;
		}

        obstacles.push_back(p3d);
    }
//    cout << "size of obs considered: " << obstacles.size() << " w.s.t: " << latest_local_pcl_.points.size() << endl;

// distance histogram
//if (is_2D){
//	for (int i=0; i<Hcnt; i++)
//		cout << Histogram_2d[i] < ", ";
//	cout << "" << endl;
//	cout << "" << endl;
//}
//else{
//	for (int i=0; i<Vcnt; i++) {
//		for (int j=0; j<Hcnt; j++) 
//			cout << Histogram_3d[i][j] << ", ";
//		cout << "" << endl;
//	}
//	cout << "" << endl;
//}

    // 目标点&当前速度 相关
    if (is_2D){
		uav2goal(2) = 0.0;
		double goal_heading = sign(uav2goal(1)) * acos(uav2goal(0)/uav2goal.norm()); // -pi ~ pi
		double hor_current_vel_norm = Eigen::Vector3d(current_vel(0),current_vel(1),0.0).norm();
		double current_heading = sign(current_vel(1)) *acos(current_vel(0)/hor_current_vel_norm); // -pi ~ pi

		for(int i=0; i<Hcnt; i++)
		{
		    double angle_i = (i + 0.5)* Hres; // 0 ~ 2pi
		    double angle_err_goal = fabs(angle_i - (goal_heading<0 ? goal_heading+2*M_PI : goal_heading)); // 0 ~ 2*pi
//		    if (angle_err_goal>M_PI) angle_err_goal = 2*M_PI - angle_err_goal; // It'd better if used, but not necessary
			double angle_err_vel = fabs(angle_i - (current_heading<0 ? current_heading+2*M_PI : current_heading)); // 0 ~ 2*pi
//		    if (angle_err_vel>M_PI) angle_err_vel = 2*M_PI - angle_err_vel; // It'd better if used, but not necessary
		    
		    // 角度差的越小越好
			if (hor_current_vel_norm>0.1)
		    	Histogram_2d[i] *= sqrt(fun_cos((cos(angle_err_goal)+1)/2,min_value) + fun_cos((cos(angle_err_vel)+1)/2,min_value) * pow(hor_current_vel_norm/limit_v_norm,2));
			else
				Histogram_2d[i] *= sqrt(fun_cos((cos(angle_err_goal)+1)/2,min_value));
		}

		// 寻找最优方向
		int best_idx = -1;
		double best_value = 0;
		for(int i=0; i<Hcnt; i++){
			int half_width_ = (width_-1)/2;
		    double filter[width_];
			for (int u=0; u<width_; u++)
		    	filter[u] = Histogram_2d[i+u-half_width_<0 ? Hcnt+i+u-half_width_ : i+u-half_width_>=Hcnt ? i+u-half_width_-Hcnt : i+u-half_width_];
		    double value_ = average(filter,width_) + minimum(filter,width_);
		    if(value_>best_value){
		        best_value = value_;
		        best_idx = i;
		    }
		}
		
		if (best_idx == -1){
			best_dir = Eigen::Vector3d(0.0,0.0,0.0);
		    max_distance = 0.0;
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
				for (int u=0; u<width_; u++)
					filter[u] = Histogram_2d[i+u-half_width_<0 ? Hcnt+i+u-half_width_ : i+u-half_width_>=Hcnt ? i+u-half_width_-Hcnt : i+u-half_width_];
				max_distance = min(0.8*average(filter,width_),0.667*uav2goal.norm());
			}
		}
	} else if (isCylindrical){
		double goal_height_v = min(double(Vcnt),max(0.0,Vcnt/2 + uav2goal(2)/Vres)); // 0 ~ Vcnt
		double goal_heading_h = sign(uav2goal(1)) * acos(uav2goal(0) / Eigen::Vector3d(uav2goal(0),uav2goal(1),0.0).norm()); // -pi ~ pi
		double current_height_v = Vcnt/2;
		double current_heading_h = sign(current_vel(1)) *acos(current_vel(0) / Eigen::Vector3d(current_vel(0),current_vel(1),0.0).norm()); // -pi ~ pi

		for(int v=0; v<Vcnt; v++)
			for(int h=0; h< Hcnt; h++){
				double height_v = (v + 0.5); // 0 ~ Vcnt
				double angle_h = (h + 0.5)* Hres; // 0 ~ 2pi
				
				double height_err_goal_v = fabs(height_v - goal_height_v)/Vcnt; // 0 ~ 1
				double angle_err_goal_h = fabs(angle_h - (goal_heading_h<0 ? goal_heading_h+2*M_PI : goal_heading_h)); // 0 ~ 2*pi
//				if (angle_err_goal_h>M_PI) angle_err_goal_h = 2*M_PI - angle_err_goal_h; // 0 ~ pi
				
				double height_err_vel_v = fabs(height_v - current_height_v)/current_height_v; // 0 ~ 1
				double angle_err_vel_h = fabs(angle_h - (current_heading_h<0 ? current_heading_h+2*M_PI : current_heading_h)); // 0 ~ 2*pi
//				if (angle_err_goal_h>M_PI) angle_err_goal_h = 2*M_PI - angle_err_goal_h; // 0 ~ pi

				// 差的越小越好
				if ( current_vel.norm()>0.1)
					Histogram_3d[v][h] *= sqrt( normal(height_err_goal_v,0.6)*fun_cos((cos(angle_err_goal_h)+1)/2,min_value) + normal(height_err_vel_v,0.3)*fun_cos((cos(angle_err_vel_h)+1)/2,min_value) * pow(current_vel.norm()/limit_v_norm,2));
				else
					Histogram_3d[v][h] *= sqrt( normal(height_err_goal_v,0.6)*fun_cos((cos(angle_err_goal_h)+1)/2,min_value)); // adjustable: 0.6, 0.3
			}

		// 寻找最优方向
		int best_idx[2];
		best_idx[0] = -1;
		best_idx[1] = -1;
		double best_value = 0;
		for(int i=0; i<Vcnt; i++)
			for(int j=0; j<Hcnt; j++){
				int half_width_ = (width_-1)/2;
				double filter[width_*width_];
				for (int u=0; u<width_; u++)
					for (int v=0; v<width_; v++)
						filter[u*width_+v] = Histogram_3d[i+u-half_width_<0 ? -(i+u-half_width_)-1 : (i+u-half_width_>=Vcnt ? 2*Vcnt-1-(i+u-half_width_) : i+u-half_width_)][i+u-half_width_<0 ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (i+u-half_width_>=Vcnt ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (j+v-half_width_<0 ? Hcnt+j+v-half_width_ : (j+v-half_width_>=Hcnt ? j+v-half_width_-Hcnt : j+v-half_width_)))];
				double value_ = average(filter,width_*width_) + minimum(filter,width_*width_);
				if(value_>best_value){
				    best_value = value_;
				    best_idx[0] = i;
				    best_idx[1] = j;
				}
			}
			
		if (best_idx[0] == -1 || best_idx[1] == -1){
			best_dir = Eigen::Vector3d(0.0,0.0,0.0);
			max_distance = 0.0;
		}else {
			has_best_dir = true;
			double best_height_v  = fabs((best_idx[0] + 0.5) - current_height_v)/current_height_v; // 0 ~ 1
//			cout << "best_height_v: " << best_height_v << endl;
			double best_heading_h  = (best_idx[1] + 0.5)* Hres; // 0 ~ 2*pi
			best_dir(0) = normal(best_height_v,0.9)*cos(best_heading_h); // adjustable: 0.9
			best_dir(1) = normal(best_height_v,0.9)*sin(best_heading_h);
			best_dir(2) = sign((best_idx[0] + 0.5) - current_height_v) * sqrt(1-pow(normal(best_height_v,0.9),2));
			
			if(gen_guide_point){
				int i = best_idx[0];
				int j = best_idx[1];
				int half_width_ = (width_-1)/2;
				double filter[width_*width_];
				for (int u=0; u<width_; u++)
					for (int v=0; v<width_; v++)
						filter[u*width_+v] = Histogram_3d[i+u-half_width_<0 ? -(i+u-half_width_)-1 : (i+u-half_width_>=Vcnt ? 2*Vcnt-1-(i+u-half_width_) : i+u-half_width_)][i+u-half_width_<0 ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (i+u-half_width_>=Vcnt ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (j+v-half_width_<0 ? Hcnt+j+v-half_width_ : (j+v-half_width_>=Hcnt ? j+v-half_width_-Hcnt : j+v-half_width_)))];
				max_distance = min(0.8*average(filter,width_*width_),0.667*uav2goal.norm());
			}
		}

	} else if (isSpherical){
		double goal_heading_v = sign(uav2goal(2)) * acos(Eigen::Vector3d(uav2goal(0),uav2goal(1),0.0).norm() / uav2goal.norm()); // -pi/2 ~ pi/2
		double goal_heading_h = sign(uav2goal(1)) * acos(uav2goal(0) / Eigen::Vector3d(uav2goal(0),uav2goal(1),0.0).norm()); // -pi ~ pi
		double current_heading_v = sign(current_vel(2)) *acos(Eigen::Vector3d(current_vel(0),current_vel(1),0.0).norm() / current_vel.norm()); // -pi/2 ~ pi/2
		double current_heading_h = sign(current_vel(1)) *acos(current_vel(0) / Eigen::Vector3d(current_vel(0),current_vel(1),0.0).norm()); // -pi ~ pi

		for(int v=0; v<Vcnt; v++)
			for(int h=0; h<Hcnt; h++)
			{
				double angle_v = (v + 0.5)* Vres - M_PI/2; // -pi/2 ~ pi/2
				double angle_h = (h + 0.5)* Hres; // 0 ~ 2pi
				
				double angle_err_goal_v = fabs(angle_v - goal_heading_v); // 0 ~ pi
				double angle_err_goal_h = fabs(angle_h - (goal_heading_h<0 ? goal_heading_h+2*M_PI : goal_heading_h)); // 0 ~ 2*pi
//				if (angle_err_goal_h>M_PI) angle_err_goal_h = 2*M_PI - angle_err_goal_h; // 0 ~ pi
				
				double angle_err_vel_v = fabs(angle_v - current_heading_v); // 0 ~ pi
				double angle_err_vel_h = fabs(angle_h - (current_heading_h<0 ? current_heading_h+2*M_PI : current_heading_h)); // 0 ~ 2*pi
//				if (angle_err_vel_h>M_PI) angle_err_vel_h = 2*M_PI - angle_err_vel_h; // 0 ~ pi
				
				// 角度差的越小越好
				if ( current_vel.norm()>0.1)
					Histogram_3d[v][h] *= sqrt(fun_cos((cos(angle_err_goal_v)+1)/2,min_value) * fun_cos((cos(angle_err_goal_h)+1)/2,min_value) + fun_cos((cos(angle_err_vel_v)+1)/2,min_value) * fun_cos((cos(angle_err_vel_h)+1)/2,min_value) * pow(current_vel.norm()/limit_v_norm,2));
				else
					Histogram_3d[v][h] *= sqrt(fun_cos((cos(angle_err_goal_v)+1)/2,min_value) * fun_cos((cos(angle_err_goal_h)+1)/2,min_value));
			}

		// 寻找最优方向
		int best_idx[2];
		best_idx[0] = -1;
		best_idx[1] = -1;
		double best_value = 0;
		for(int i=0; i<Vcnt; i++)
			for(int j=0; j<Hcnt; j++){
				int half_width_ = (width_-1)/2;
				double filter[width_*width_];
				for (int u=0; u<width_; u++)
					for (int v=0; v<width_; v++)
						filter[u*width_+v] = Histogram_3d[i+u-half_width_<0 ? -(i+u-half_width_)-1 : (i+u-half_width_>=Vcnt ? 2*Vcnt-1-(i+u-half_width_) : i+u-half_width_)][i+u-half_width_<0 ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (i+u-half_width_>=Vcnt ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (j+v-half_width_<0 ? Hcnt+j+v-half_width_ : (j+v-half_width_>=Hcnt ? j+v-half_width_-Hcnt : j+v-half_width_)))];
				double value_ = average(filter,width_*width_) + minimum(filter,width_*width_);
				if(value_>best_value){
				    best_value = value_;
				    best_idx[0] = i;
				    best_idx[1] = j;
				}
			}
			
		if (best_idx[0] == -1 || best_idx[1] == -1){
			best_dir = Eigen::Vector3d(0.0,0.0,0.0);
			max_distance = 0.0;
		}else {
			has_best_dir = true;
			double best_heading_v  = (best_idx[0] + 0.5)* Vres-M_PI/2; // -pi/2 ~ pi/2
			double best_heading_h  = (best_idx[1] + 0.5)* Hres; // 0 ~ 2*pi
			best_dir(0) = cos(best_heading_v)*cos(best_heading_h);
			best_dir(1) = cos(best_heading_v)*sin(best_heading_h);
			best_dir(2) = sin(best_heading_v);
			
			if(gen_guide_point){
				int i = best_idx[0];
				int j = best_idx[1];
				int half_width_ = (width_-1)/2;
				double filter[width_*width_];
				for (int u=0; u<width_; u++)
					for (int v=0; v<width_; v++)
						filter[u*width_+v] = Histogram_3d[i+u-half_width_<0 ? -(i+u-half_width_)-1 : (i+u-half_width_>=Vcnt ? 2*Vcnt-1-(i+u-half_width_) : i+u-half_width_)][i+u-half_width_<0 ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (i+u-half_width_>=Vcnt ? (j+v-half_width_<0 ? (j+v-half_width_+Hcnt/2)%Hcnt : (j+v-half_width_>=Hcnt ? (j+v-half_width_-Hcnt/2)%Hcnt : (j+v-half_width_+Hcnt/2)%Hcnt )) : (j+v-half_width_<0 ? Hcnt+j+v-half_width_ : (j+v-half_width_>=Hcnt ? j+v-half_width_-Hcnt : j+v-half_width_)))];
				max_distance = min(0.8*average(filter,width_*width_),0.667*uav2goal.norm());
			}
		}
	}


// weights histogram
//if (is_2D){
//	for (int i=0; i<Hcnt; i++)
//		cout << Histogram_2d[i] << ", ";
//	cout << "" << endl;
//	cout << "" << endl;
//}
//else{
//	for (int i=0; i<Vcnt; i++) {
//		for (int j=0; j<Hcnt; j++) 
//			cout << Histogram_3d[i][j] << ", ";
//		cout << "" << endl;
//	}
//	cout << "" << endl;
//}


    // 如果不安全的点超出指定数量
    if(safe_cnt>5)
        local_planner_state = 2;  //成功规划，但是飞机不安全
    else
        local_planner_state =1;  //成功规划， 安全
        
    if(gen_guide_point){
    	desired = best_dir * max(max_distance,0.001);
    }else{
    	desired = best_dir * limit_v_norm;
	}
	
    exec_num++;
	end = std::chrono::system_clock::now();
    if(exec_num == 10)
    {
    	std::chrono::duration<double> elapsed_seconds = end - start; 
        printf("Histogram calculation takes %f [us].\n", elapsed_seconds.count()/10*1e6);
        exec_num=0;
    }
    
    has_local_map_ = false;
    has_odom_ = false;
    has_best_dir = false;
    
    return local_planner_state;
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
            if (Histogram_2d[i] > val) Histogram_2d[i] = val;
        }
        for(int i=0;i<=cnt_max; i++)
        {
            if (Histogram_2d[i] > val) Histogram_2d[i] =val;
        }
    }else
    {
        for(int i=cnt_min; i<=cnt_max; i++)
        {
            if (Histogram_2d[i] > val) Histogram_2d[i] = val;
        }
    }
     
}

// hor_angle_cen: -pi ~ pi; ver_angle_cen: -pi/2 ~ pi/2; angle_range: 0~pi/2
void HIST::SphericalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double angle_range, double val)
{
	hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
    double obs_dist = val + forbidden_plus_safe_distance;
    
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
		double r = sqrt(obs_dist*obs_dist - forbidden_plus_safe_distance*forbidden_plus_safe_distance);
		
		if (i==int(ver_angle_cen/Vres)){
			angle_range = atan(forbidden_plus_safe_distance/(r*cos(ver_angle_cen-M_PI/2)));// alpha/2 < pi/2
		}else if (i<ver_angle_cen/Vres){
            angle_range = atan(sqrt(max(pow(obs_dist*cos(ver_angle_cen-(i+1)*Vres),2)-r*r,1e-6))/(r*cos((i+1)*Vres-M_PI/2)));// alpha/2 < pi/2
        }else{
            angle_range = atan(sqrt(max(pow(obs_dist*cos(ver_angle_cen-(i)*Vres),2)-r*r,1e-6))/(r*cos((i)*Vres-M_PI/2)));// alpha/2 < pi/2
        }
        
		double hor_angle_max = hor_angle_cen + angle_range;
		double hor_angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5pi/2
		int hor_cnt_min = int((hor_angle_min<0 ? hor_angle_min+2*M_PI : hor_angle_min>2*M_PI ? hor_angle_min-2*M_PI : hor_angle_min)/Hres);
		int hor_cnt_max = int((hor_angle_max<0 ? hor_angle_max+2*M_PI : hor_angle_max>2*M_PI ? hor_angle_max-2*M_PI : hor_angle_max)/Hres);
    	
	    if(hor_cnt_min>hor_cnt_max)
		{
		    for(int j=hor_cnt_min; j<Hcnt; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		    for(int j=0;j<=hor_cnt_max; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		}else
		{
		    for(int j=hor_cnt_min; j<=hor_cnt_max; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		}
    }
    
}

// hor_angle_cen: -pi ~ pi; ver_angle_cen: 0 ~ Vcnt; idx_range: 0~Vcnt/2; hor_obs_dist; val
void HIST::CylindricalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double idx_range, double hor_obs_dist, double val)
{
	hor_angle_cen = hor_angle_cen<0 ? hor_angle_cen+2*M_PI : hor_angle_cen; // 0 ~ 2pi
	double ver_idx_min = max(ver_angle_cen - idx_range,0.0);
	double ver_idx_max = min(ver_angle_cen + idx_range,double(Vcnt));

	double angle_range;
	for(int i=int(ver_idx_min); i<ver_idx_max; i++){
		if (i==int(ver_angle_cen)){
			angle_range = asin(min(forbidden_plus_safe_distance/hor_obs_dist,1.0));// alpha/2 < pi/2
		}else if (i<ver_angle_cen){
			angle_range = asin(min(sqrt(pow(forbidden_plus_safe_distance,2) - pow((ver_angle_cen-(i+1))*Vres,2)) / hor_obs_dist,1.0));// alpha/2 < pi/2
		}else{
			angle_range = asin(min(sqrt(pow(forbidden_plus_safe_distance,2) - pow((ver_angle_cen-(i))*Vres,2)) / hor_obs_dist,1.0));// alpha/2 < pi/2
		}			

		double hor_angle_max = hor_angle_cen + angle_range;
		double hor_angle_min = hor_angle_cen - angle_range; // -pi/2 ~ 5pi/2
		int hor_cnt_min = int((hor_angle_min<0 ? hor_angle_min+2*M_PI : hor_angle_min>2*M_PI ? hor_angle_min-2*M_PI : hor_angle_min)/Hres);
		int hor_cnt_max = int((hor_angle_max<0 ? hor_angle_max+2*M_PI : hor_angle_max>2*M_PI ? hor_angle_max-2*M_PI : hor_angle_max)/Hres);
		    	
	    if(hor_cnt_min>hor_cnt_max)
		{
		    for(int j=hor_cnt_min; j<Hcnt; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		    for(int j=0;j<=hor_cnt_max; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		}else
		{
		    for(int j=hor_cnt_min; j<=hor_cnt_max; j++)
		    {
		        if (Histogram_3d[i][j] > val) Histogram_3d[i][j] = val;
		    }
		}
	}
}

}
