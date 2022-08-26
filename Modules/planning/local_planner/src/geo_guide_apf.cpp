#include "geo_guide_apf.h"
#include "math.h"

namespace Local_Planning
{
Eigen::Vector3d last_desired;
    	        
auto min=[](double v1, double v2)->double
{
    return v1<v2 ? v1 : v2;
};    	        
auto max=[](double v1, double v2)->double
{
    return v1<v2 ? v1 : v2;
};
auto sign=[](double v)->double
{
	return v<0.0? -1.0:1.0;
};

void GeoGuideAPF::init(ros::NodeHandle& nh)
{
    has_local_map_ = false;

    nh.param("local_planner/forbidden_range", forbidden_range, 0.20);  // 最小障碍物距离
    nh.param("local_planner/sensor_max_range", sensor_max_range, 2.5);  // 感知障碍物距离
    nh.param("local_planner/max_planning_vel", max_planning_vel, 0.5); // 最大飞行速度
    nh.param("apf/k_push", k_push, 0.8);                         // 推力增益
    nh.param("apf/k_att", k_att, 0.4);                                  // 引力增益
    nh.param("apf/max_att_dist", max_att_dist, 5.0);             // 最大吸引距离
    nh.param("local_planner/ground_height", ground_height, 0.1);  // 地面高度
    nh.param("apf/ground_safe_height", ground_safe_height, 0.2);  // 地面安全距离
    nh.param("local_planner/safe_distance", safe_distance, 0.15); // 安全停止距离

    // TRUE代表2D平面规划及搜索,FALSE代表3D 
    nh.param("local_planner/is_2D", is_2D, true); 
    
    sensor_max_range = max(sensor_max_range,3*forbidden_range);
}

void GeoGuideAPF::set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr)
{
    latest_local_pcl_ = *pcl_ptr;
    has_local_map_=true;
}

void GeoGuideAPF::set_odom(nav_msgs::Odometry cur_odom)
{
    cur_odom_ = cur_odom;
    has_odom_=true;
}

int GeoGuideAPF::generate(Eigen::Vector3d &goal, Eigen::Vector3d &desired)
{
    // 0 for not init; 1for safe; 2 for dangerous
    int local_planner_state=0;  
    int safe_cnt=0;

    if(!has_local_map_|| !has_odom_)
        return 0;

    if ((int)latest_local_pcl_.points.size() == 0) 
        return 0;

    if (isnan(goal(0)) || isnan(goal(1)) || isnan(goal(2)))
        return 0;

    //　当前状态
    Eigen::Vector3d current_pos;
    current_pos[0] = cur_odom_.pose.pose.position.x;
    current_pos[1] = cur_odom_.pose.pose.position.y;
    current_pos[2] = cur_odom_.pose.pose.position.z;
    Eigen::Vector3d current_vel;
    current_vel[0] = cur_odom_.twist.twist.linear.x;
    current_vel[1] = cur_odom_.twist.twist.linear.y;
    current_vel[2] = cur_odom_.twist.twist.linear.z;
    float  current_vel_norm = current_vel.norm();
    ros::Time begin_collision = ros::Time::now();

    // 引力
    Eigen::Vector3d uav2goal = goal - current_pos;
    double dist_att = uav2goal.norm();
    if(dist_att > max_att_dist)
    {
        uav2goal = max_att_dist * uav2goal/dist_att ;
    }
    //　计算吸引力
    attractive_force = k_att * uav2goal;

    // 排斥力
    double uav_height = cur_odom_.pose.pose.position.z;
    repulsive_force = Eigen::Vector3d(0.0, 0.0, 0.0);
    guide_force = Eigen::Vector3d(0.0, 0.0, 0.0);
    int count;
    double max_guide_force = 0.0;

    Eigen::Vector3d p3d;
    vector<Eigen::Vector3d> obstacles;
    
    //　根据局部点云计算排斥力（是否可以考虑对点云进行降采样？）
    for (size_t i = 0; i < latest_local_pcl_.points.size(); ++i) 
    {
        p3d(0) = latest_local_pcl_.points[i].x;
        p3d(1) = latest_local_pcl_.points[i].y;
        p3d(2) = latest_local_pcl_.points[i].z; // World-ENU frame

        Eigen::Vector3d uav2obs = p3d - current_pos;

        //　低速悬浮时不考虑
		if(current_vel_norm<0.3)
			continue;
        //　不考虑地面上的点的排斥力
        if(fabs(p3d(2))<ground_height)
            continue;

        //　超出感知范围，则不考虑该点的排斥力
        double dist_push = (uav2obs).norm();

		
    	obs_angle = acos(uav2obs.dot(current_vel) / uav2obs.norm() / current_vel_norm);
        if(isnan(dist_push) || (dist_push > min(3*forbidden_range,sensor_max_range) && obs_angle > M_PI/6)){
        	continue;            
		}
		
        // 如果当前的观测点中，包含小于安全停止距离的点，进行计数
        if(dist_push < safe_distance+forbidden_range)
        {
            safe_cnt++;
            desired += 1000 * (-uav2obs)/dist_push;
            if(is_2D)
            {
                desired[2] = 0.0;
            }
            if(safe_cnt>3)
            {
                desired /= safe_cnt;
                return 2;  //成功规划，但是飞机不安全
            }
        }else if(dist_push < 3*forbidden_range){
		    double push_gain = k_push * (1/(dist_push - forbidden_range) - 1/(2*forbidden_range));
            repulsive_force += push_gain * (-uav2obs)/dist_push;
			count ++;
		}
		
		if (obs_angle < M_PI/6){
			double force = cos(obs_angle) * min(3,1/(max(forbidden_range,dist_push) - forbidden_range + 1e-6) - 1/(sensor_max_range-forbidden_range));
			guide_force += current_vel.cross((-uav2obs).cross(current_vel)) / pow(current_vel_norm,2) / dist_push * force; // or / pow(dist_push,2)
			
			if (max_guide_force<force) max_guide_force = force;
		}
        obstacles.push_back(p3d);
    }

    //　平均排斥力
    if(count != 0)
    {
        repulsive_force=repulsive_force/count; //obstacles.size();
    }
    
    // guide force
	if(count==int(obstacles.size()))
	{
		guide_force=Eigen::Vector3d(0.0,0.0,0.0);
	}else{
		double guide_force_norm = guide_force.norm();
		if(guide_force_norm<1e-6) 
			guide_force += uav2goal;
		guide_force = max_guide_force*guide_force/guide_force.norm();
	}

    // 地面排斥力
    if (current_pos[2] <2*ground_safe_height)
		repulsive_force += Eigen::Vector3d(0.0, 0.0, 1.0) * k_push * (1/max(max(ground_safe_height,current_pos[2]) - ground_safe_height, 1e-6) - 1/(ground_safe_height));

    if(dist_att<1.0)
    {
        repulsive_force *= pow(dist_att,2);  // to gaurantee to reach the goal.
        guide_force *= pow(dist_att,2);  // to gaurantee to reach the goal.
    }
    // 合力
    desired = 0.3*repulsive_force + 0.4*attractive_force + 0.3*guide_force; // ENU frame

    if(is_2D)
        desired[2] = 0.0;

	if(max_planning_vel<desired.norm())
		desired = desired/desired.norm()*max_planning_vel;
		
	desired = 0.5*last_desired + 0.5*desired;
    last_desired = desired;
    
	cout << "guide_force: " << max_guide_force << endl;
	cout << "repulsive_force: " << repulsive_force.norm() << endl;
	cout << "attractive_force: " << attractive_force.norm() << endl;
	cout << "desired_vel: " << desired.norm() << endl;
	cout << " " << endl;
	
    local_planner_state =1;  //成功规划， 安全

    static int exec_num=0;
    exec_num++;

    // 此处改为根据循环时间计算的数值
    if(exec_num == 100)
    {
        printf("APF calculate take %f [s].\n",   (ros::Time::now()-begin_collision).toSec());
        exec_num=0;
    }  

    return local_planner_state;
}



}
