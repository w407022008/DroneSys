#include "local_planning.h"
#include <string> 	
#include <time.h>
#include "chrono" 

namespace Local_Planning
{

// 局部规划算法 初始化函数
void Local_Planner::init(ros::NodeHandle& nh)
{
    // 是否为仿真模式
    nh.param("local_planner/sim_mode", sim_mode, false); 
    // 局部避障算法: [0]: GeoGuideAPF,[1]: Histogram
    nh.param("local_planner/algorithm_mode", algorithm_mode, 0);
    // 轨迹追踪使能
	nh.param("local_planner/path_tracking_enable", planner_enable_default, false);
    	
	// 环境输入类型：0: octomap点云数据类型<sensor_msgs::PointCloud2>, 1: 2d传感器数据类型<sensor_msgs::LaserScan>, 2: 3d传感器数据类型<sensor_msgs::PointCloud2>
    nh.param("local_planner/map_input", map_input, 0);
    nh.param("local_planner/ground_removal", flag_pcl_ground_removal, false);
    nh.param("local_planner/max_ground_height", max_ground_height, 0.1);
    nh.param("local_planner/ceil_height", ceil_height, 5.0);
    nh.param("local_planner/downsampling", flag_pcl_downsampling, false);
    nh.param("local_planner/resolution", size_of_voxel_grid, 0.1);
    nh.param("local_planner/timeSteps_fusingSamples", timeSteps_fusingSamples, 5);
    
    // TRUE代表2D平面规划及搜索,FALSE代表3D 
    nh.param("local_planner/is_2D", is_2D, false); 
    // 如果采用2维Lidar，需要一定的yawRate来探测地图
    nh.param("local_planner/yaw_tracking_mode", yaw_tracking_mode, 1); 
    // 2D规划时,定高高度
    nh.param("local_planner/fly_height_2D", fly_height_2D, 1.0);  
    
    nh.param("local_planner/is_rgbd", is_rgbd, false); 
    nh.param("local_planner/is_lidar", is_lidar, false); 
    
    // 最小障碍物距离
    nh.param("local_planner/forbidden_range", forbidden_range, 0.20);
    // 探测最大距离
    nh.param("local_planner/sensor_max_range", sensor_max_range, 3.0);
    // 最大速度
    nh.param("local_planner/max_planning_vel", max_planning_vel, 0.4);
    // 最小竖直目标高度
    nh.param("local_planner/min_goal_height", min_goal_height, 1.0);
    // 是否使用joy control作为第二输入; 0：disable, 1：control in Body Frame，2：control in Custom Frame
    nh.param("local_planner/control_from_joy", control_from_joy, 0);
    // 最大水平目标距离（when joy control）
    nh.param("local_planner/_max_goal_range_xy", _max_goal_range_xy, 3.0);
    // 最大竖直目标距离（when joy control）
    nh.param("local_planner/_max_goal_range_z", _max_goal_range_z, 3.0);
    // 最大转向速度（when joy control）
    nh.param("local_planner/_max_manual_yaw_rate", _max_manual_yaw_rate, 1.0);
    

    // 订阅开关
    planner_switch_sub = nh.subscribe<std_msgs::Bool>("/drone_msg/switch/local_planner", 10, &Local_Planner::planner_switch_cb, this);

	// 订阅目标点，
	// 从 Manual Control Setpoint Input
	manual_control_sub = nh.subscribe<drone_msgs::RCInput>("/joy/RCInput", 10, &Local_Planner::manual_control_cb, this);
	// 从终端输入点坐标
	goal_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 1, &Local_Planner::goal_cb, this);

	
    // 订阅 无人机状态
    drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state", 10, &Local_Planner::drone_state_cb, this);

    // 订阅传感器点云信息,该话题名字可在launch文件中任意指定
    if (map_input == 0)
    {
        local_point_clound_sub = nh.subscribe<sensor_msgs::PointCloud2>("/planning/local_pcl", 1, &Local_Planner::localcloudCallback, this);
    }else if (map_input == 1)
    {
        local_point_clound_sub = nh.subscribe<sensor_msgs::LaserScan>("/planning/local_pcl", 1, &Local_Planner::Callback_2dlaserscan, this);
    }else if (map_input == 2)
    {
        local_point_clound_sub = nh.subscribe<sensor_msgs::PointCloud2>("/planning/local_pcl", 1, &Local_Planner::Callback_3dpointcloud, this);
    }

    // 发布 期望速度
    command_pub = nh.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 10);
    
    // 发布提示消息
    message_pub = nh.advertise<drone_msgs::Message>("/drone_msg/message", 10);

    // 发布指导位置或者方向用于显示
    rviz_guide_pub = nh.advertise<geometry_msgs::PointStamped >("/local_planner/guide", 10); 
    
    // 发布局部点云用于显示
    point_cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/local_planner/local_point_cloud", 10); 

    // 任务执行环，10Hz
    mainloop_timer = nh.createTimer(ros::Duration(0.1), &Local_Planner::mainloop_cb, this);

    // 轨迹追踪环，20Hz
    control_timer = nh.createTimer(ros::Duration(0.05), &Local_Planner::control_cb, this);

    // 选择避障算法
    if(algorithm_mode==0){
        local_alg_ptr.reset(new GeoGuideAPF);
        local_alg_ptr->init(nh);
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "GeoGuideAPF init.");
    }
    else if(algorithm_mode==1)
    {
        local_alg_ptr.reset(new HIST);
        local_alg_ptr->init(nh);
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Histogram init.");
    }

    // init state machine
    exec_state = EXEC_STATE::WAIT_GOAL;
    odom_ready = false;
    drone_ready = false;
    goal_ready = false;
    sensor_ready = false;
    path_ok = false;

    // init command 
    Command_Now.header.stamp = ros::Time::now();
    Command_Now.Mode  = drone_msgs::ControlCommand::Idle;
    Command_Now.Command_ID = 0;
    Command_Now.source = NODE_NAME;
    desired_yaw = 0.0;

    //　仿真模式下直接发送切换模式与起飞指令
    int start_flag = 0;
    if(sim_mode == true)
    {
        // Waiting for type in 
        while(start_flag == 0)
        {
            cout << "Please type in 1 for taking off:"<<endl;
            cin >> start_flag;
        }
        // Takeoff
        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode  = drone_msgs::ControlCommand::Idle;
        Command_Now.Command_ID = 1;
        Command_Now.source = NODE_NAME;
        Command_Now.Reference_State.yaw_ref = 999;
        command_pub.publish(Command_Now);   
        do{
            ros::spinOnce();
        }while(!drone_ready);
        cout << "Switched to OFFBOARD and armed, drone will take off after 1.0s"<<endl;
        ros::Duration(1.0).sleep();

        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
        Command_Now.Command_ID = Command_Now.Command_ID + 1;
        Command_Now.source = NODE_NAME;
        command_pub.publish(Command_Now);
        cout << "Takeoff"<<endl;
        ros::Duration(1.0).sleep();
        do{
            ros::spinOnce();
        }while(fabs(_DroneState.velocity[2])>0.1);
    }else
    {
        //　真实飞行情况：等待飞机状态变为offboard模式，然后发送起飞指令
    }

    goal_pos[0] = _DroneState.position[0];
    goal_pos[1] = _DroneState.position[1];
    goal_pos[2] = _DroneState.position[2];
    planner_enable = planner_enable_default;
    ros::spin();
}

void Local_Planner::planner_switch_cb(const std_msgs::Bool::ConstPtr& msg)
{
    if (!planner_enable && msg->data){
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Planner is enable.");
    }else if (planner_enable && !msg->data){
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Planner is disable.");
        exec_state = EXEC_STATE::WAIT_GOAL;
    }
    planner_enable = msg->data;
}

void Local_Planner::goal_cb(const geometry_msgs::PoseStampedConstPtr& msg)
{
    if(_DroneState.connected == true){
		if (_DroneState.armed == false){
			// 起飞
		    Command_Now.header.stamp = ros::Time::now();
		    Command_Now.Mode  = drone_msgs::ControlCommand::Idle;
		    Command_Now.Command_ID = Command_Now.Command_ID + 1;
		    Command_Now.source = NODE_NAME;
		    Command_Now.Reference_State.yaw_ref = 999;
		    command_pub.publish(Command_Now);   
		    cout << "Switch to OFFBOARD and arm ..."<<endl;
		    ros::Duration(3.0).sleep();
		    
		    // Command_Now.header.stamp = ros::Time::now();
		    // Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
		    // Command_Now.Command_ID = Command_Now.Command_ID + 1;
		    // Command_Now.source = NODE_NAME;
		    // command_pub.publish(Command_Now);
		    // cout << "Takeoff ..."<<endl;
		    // ros::Duration(3.0).sleep();
		}
	} else{
		cout << "Disconnected!" << endl;
		goal_ready = false;
		return;
	}
		
    if (is_2D == true)
    {
        goal_pos << msg->pose.position.x, msg->pose.position.y, fly_height_2D;
    }else
    {
    	if(msg->pose.position.z < min_goal_height)
        	goal_pos << msg->pose.position.x, msg->pose.position.y, min_goal_height;
        else
        	goal_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    }

    goal_ready = true;
    

    // 获得新目标点
    if(planner_enable){
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");

        cout << "Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;

        if(goal_pos(0) == 99 && goal_pos(1) == 99 )
        {
            path_ok = false;
            goal_ready = false;
        	yaw_tracking_mode = 0;
            exec_state = EXEC_STATE::LANDING;
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Land");
        }else if(exec_state == EXEC_STATE::LANDING && _DroneState.position[2]<ceil_height && _DroneState.position[2]>max_ground_height)
        	exec_state = EXEC_STATE::WAIT_GOAL;
    }
}

void Local_Planner::manual_control_cb(const drone_msgs::RCInputConstPtr& msg)
{
	if(!drone_ready){
//    	cout << "is ready to remote control?" << endl;
		return;
	}
	
	if(!odom_ready){
//    	cout << "has odom?" << endl;
		return;
	}
	
	if(control_from_joy == 0){
//    	cout << "should remote control from joystick?" << endl;
		return;
	}
		
	// 计算所在坐标系的方向
    // tf::Quaternion q;
    Eigen::Quaternionf q;
    if(control_from_joy == 1){
        // in Body Frame
        q = Eigen::Quaternionf {_DroneState.attitude_q.w, _DroneState.attitude_q.x, _DroneState.attitude_q.y, _DroneState.attitude_q.z};
        // tf::quaternionMsgToTF(_DroneState.attitude_q, q);
    }else if (control_from_joy == 2){
        // tf::StampedTransform transform_local;
        // tf::TransformListener tfListener;
        
        // // in User Heading Consistent Frame
        // try{
        //   tfListener.waitForTransform("/world","/joystick_link", msg->header.stamp, ros::Duration(4.0));
        //   tfListener.lookupTransform("/world", "/joystick_link", msg->header.stamp, transform_local);
        // }
        // catch (tf::TransformException ex){
        //   ROS_ERROR("%s",ex.what());
        //   ros::Duration(1.0).sleep();
        // }
        // q = transform_local.getRotation();
        // q = tf::Quaternion(0.0, 0.0, 0.0, 1.0);

        q = Eigen::Quaternionf{1.0, 0.0, 0.0, 0.0};
    }else if (control_from_joy == 3){
        if(drone_yaw_init != 0.0 && user_yaw_init != 0.0)
        {
        double yaw_diff = user_yaw - user_yaw_init + drone_yaw_init;
        yaw_diff = yaw_diff>M_PI ? yaw_diff-2*M_PI : (yaw_diff<-M_PI ? yaw_diff+2*M_PI : yaw_diff);
        q = quaternion_from_rpy(Eigen::Vector3d{0.0,0.0,yaw_diff});
        // cout<<yaw_diff<<" "<<user_yaw_init<<" "<<drone_yaw_init<<endl;
        }else{
        q = Eigen::Quaternionf{1.0, 0.0, 0.0, 0.0};
        }
    }

    // double roll,pitch,yaw;
    // tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
    // Eigen::Matrix3f R_Local_to_Joy = euler2matrix(roll, pitch, yaw);

    Eigen::Matrix3f R_Local_to_Joy(q);

	rc_x = msg->rc_x;
	rc_y = msg->rc_y;
	rc_z = msg->rc_z;
	rc_r = msg->rc_r;
	
    Eigen::Vector3f goal_in_local_frame, goal_in_map_frame;
    goal_in_local_frame[0] = _max_goal_range_xy * rc_x;
    goal_in_local_frame[1] = _max_goal_range_xy * rc_y;
    goal_in_local_frame[2] = _max_goal_range_z * rc_z;

    goal_in_map_frame = R_Local_to_Joy * goal_in_local_frame;
    
    if (is_2D)
    	goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], fly_height_2D;
    else
    	if(_DroneState.position[2] + goal_in_map_frame[2] < min_goal_height)
        	goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], min_goal_height;
        else if(_DroneState.position[2] + goal_in_map_frame[2] > ceil_height)
        	goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], ceil_height-0.1;
        else
    		goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], _DroneState.position[2] + goal_in_map_frame[2];
    	
	
    goal_ready = true;
    // 获得新目标点
    if(planner_enable){
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");

        cout << "Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;

        if(int(rc_x) == -1 && int(rc_y) == -1 && int(rc_z) == -1 && int(rc_r) == -1)
        {
            path_ok = false;
            goal_ready = false;
        	yaw_tracking_mode = 0;
            exec_state = EXEC_STATE::LANDING;
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Land");
        }else if(exec_state == EXEC_STATE::LANDING && _DroneState.position[2]<ceil_height && _DroneState.position[2]>max_ground_height)
        	exec_state = EXEC_STATE::WAIT_GOAL;
    }
}

void Local_Planner::drone_state_cb(const drone_msgs::DroneStateConstPtr& msg)
{
    _DroneState = *msg; // ENU系

    if (is_2D == true)
    {
        start_pos << msg->position[0], msg->position[1], fly_height_2D;
        start_vel << msg->velocity[0], msg->velocity[1], 0.0;

        if(abs(fly_height_2D - msg->position[2]) > 0.2)
        {
            pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME,"Drone is not in the desired height.");
        }
    }else
    {
        start_pos << msg->position[0], msg->position[1], msg->position[2];
        start_vel << msg->velocity[0], msg->velocity[1], msg->velocity[2];
    }

    odom_ready = true;

    if (_DroneState.connected == true && _DroneState.armed == true )
    {
        drone_ready = true;
    }else
    {
        drone_ready = false;
    }

    Drone_odom.header = _DroneState.header;
    Drone_odom.child_frame_id = "base_link";

    Drone_odom.pose.pose.position.x = _DroneState.position[0];
    Drone_odom.pose.pose.position.y = _DroneState.position[1];
    Drone_odom.pose.pose.position.z = _DroneState.position[2];

    Drone_odom.pose.pose.orientation = _DroneState.attitude_q;
    Drone_odom.twist.twist.linear.x = _DroneState.velocity[0];
    Drone_odom.twist.twist.linear.y = _DroneState.velocity[1];
    Drone_odom.twist.twist.linear.z = _DroneState.velocity[2];

    local_alg_ptr->set_odom(Drone_odom);

}


void Local_Planner::Callback_2dlaserscan(const sensor_msgs::LaserScanConstPtr &msg)
{
    /* need odom_ for center radius sensing */
    if (!odom_ready) 
    {
//    	cout << "has odom?" << endl;
        return;
    }
    
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now(); 
    
	tf::StampedTransform transform;
	try{
		tfListener.waitForTransform("/map","/lidar_link",msg->header.stamp,ros::Duration(4.0));
		tfListener.lookupTransform("/map", "/lidar_link", msg->header.stamp, transform);
	}
		catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
		ros::Duration(1.0).sleep();
	}
	
	tf::Quaternion q = transform.getRotation();
	tf::Vector3 Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());
	
    double roll,pitch,yaw;
    tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
    Eigen::Matrix3f R_Body_to_ENU = get_rotation_matrix(roll, pitch, yaw);
    
    sensor_msgs::LaserScan::ConstPtr _laser_scan;
    _laser_scan = msg;

    pcl::PointCloud<pcl::PointXYZ> _pointcloud;
    _pointcloud.clear();
    
    pcl::PointXYZ newPoint;
    Eigen::Vector3f _laser_point_body_body_frame,_laser_point_body_ENU_frame;
    
    double newPointAngle;
    int beamNum = _laser_scan->ranges.size();
    for (int i = 0; i < beamNum; i++)
    {
    	if(_laser_scan->ranges[i] < forbidden_range) continue;
        newPointAngle = _laser_scan->angle_min + _laser_scan->angle_increment * i;
        _laser_point_body_body_frame[0] = _laser_scan->ranges[i] * cos(newPointAngle);
        _laser_point_body_body_frame[1] = _laser_scan->ranges[i] * sin(newPointAngle);
        _laser_point_body_body_frame[2] = 0.0;
        _laser_point_body_ENU_frame = R_Body_to_ENU * _laser_point_body_body_frame;
        newPoint.x = Origin.getX() + _laser_point_body_ENU_frame[0];
        newPoint.y = Origin.getY() + _laser_point_body_ENU_frame[1];
        newPoint.z = Origin.getZ() + _laser_point_body_ENU_frame[2];
        
        _pointcloud.push_back(newPoint);
    }
    
//	concatenate_PointCloud += _pointcloud;
//	static int frame_id = 0;
//	if (frame_id == timeSteps_fusingSamples){
////		cout << "point cloud size: " << ", " << (int)concatenate_PointCloud.points.size();
//		if(flag_pcl_ground_removal){
//			pcl::PassThrough<pcl::PointXYZ> ground_removal;
//			ground_removal.setInputCloud (concatenate_PointCloud.makeShared());
//			ground_removal.setFilterFieldName ("z");
//			ground_removal.setFilterLimits (-1.0, max_ground_height);
//			ground_removal.setFilterLimitsNegative (true);
//			ground_removal.filter (concatenate_PointCloud);
//		}
//		
//		if (flag_pcl_downsampling){
//			pcl::VoxelGrid<pcl::PointXYZ> sor;
//			sor.setInputCloud(concatenate_PointCloud.makeShared());
//			sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
//			sor.filter(concatenate_PointCloud);
//		}
////		cout << " to " << (int)concatenate_PointCloud.points.size() << endl;
//		
//		local_point_cloud = concatenate_PointCloud;
//		frame_id = 0;
//		concatenate_PointCloud.clear();
//	} else {
//		local_point_cloud = local_point_cloud;
//		frame_id++;
//	}

	if(flag_pcl_ground_removal){
		pcl::PassThrough<pcl::PointXYZ> ground_removal;
		ground_removal.setInputCloud (_pointcloud.makeShared());
		ground_removal.setFilterFieldName ("z");
		ground_removal.setFilterLimits (-1.0, max_ground_height);
		ground_removal.setFilterLimitsNegative (true);
		ground_removal.filter (_pointcloud);
	}
	if (flag_pcl_downsampling){
		static int frame_id = 0;
		frame_id++;
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(_pointcloud.makeShared());
		sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
		sor.filter(_pointcloud);
		
		concatenate_PointCloud += _pointcloud;
		
		if(frame_id % timeSteps_fusingSamples == 0){
			static int max_point_num = 1000;
			if(concatenate_PointCloud.points.size()>max_point_num){
//				cout << "point cloud size: " << (int)concatenate_PointCloud.points.size();
				sor.setInputCloud(concatenate_PointCloud.makeShared());
				sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
				sor.filter(concatenate_PointCloud);
				max_point_num = (int(concatenate_PointCloud.points.size()/1000) + 1)*1000;
//				cout << " to " << (int)concatenate_PointCloud.points.size() << " max_point_num " << max_point_num << endl;
			}
				
			if(frame_id % (5*timeSteps_fusingSamples) == 0){
//				cout << "outlier removal: " << (int)concatenate_PointCloud.points.size();
				pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;
				radiusoutlier.setInputCloud(concatenate_PointCloud.makeShared());
				radiusoutlier.setRadiusSearch(2*size_of_voxel_grid);
				radiusoutlier.setMinNeighborsInRadius(15);
				radiusoutlier.filter(concatenate_PointCloud); 
//				cout << " to " << (int)concatenate_PointCloud.points.size() << endl;
			}
		}
	}else
		concatenate_PointCloud += _pointcloud;
	
	pcl::PassThrough<pcl::PointXYZ> sensor_range;
	sensor_range.setInputCloud (concatenate_PointCloud.makeShared());
	sensor_range.setFilterFieldName ("x");
	sensor_range.setFilterLimits (Origin.getX()-sensor_max_range, Origin.getX()+sensor_max_range);
	sensor_range.filter (local_point_cloud);
	sensor_range.setInputCloud (local_point_cloud.makeShared());
	sensor_range.setFilterFieldName ("y");
	sensor_range.setFilterLimits (Origin.getY()-sensor_max_range, Origin.getY()+sensor_max_range);
	sensor_range.filter (local_point_cloud);
	
	
	local_point_cloud.header.seq++;
	local_point_cloud.header.stamp = (msg->header.stamp).toNSec()/1e3;
	local_point_cloud.header.frame_id = "world";
	point_cloud_pub.publish(local_point_cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr = local_point_cloud.makeShared();
    local_alg_ptr->set_local_map_pcl(pcl_ptr);
    
    sensor_ready = true;
    static int exec_num = 0;
    exec_num++;
    if(exec_num == 10)
    {
    	std::chrono::duration<double, std::milli> elapsed_seconds = std::chrono::system_clock::now() - start; 
        printf("point_cloud processing takes %f [ms].\n", elapsed_seconds.count());
        exec_num=0;
    }
}

void Local_Planner::Callback_3dpointcloud(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    /* need odom_ for center radius sensing */
    if (!odom_ready || (!is_rgbd && !is_lidar)) 
    {
//    	cout << "has odom? is_rgbd or is_lidar?" << endl;
        return;
    }
    
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now(); 
    
	tf::StampedTransform transform;
	if (is_rgbd)
		try{
			tfListener.waitForTransform("/map","/realsense_camera_link",msg->header.stamp,ros::Duration(4.0));
			tfListener.lookupTransform("/map", "/realsense_camera_link", msg->header.stamp, transform);
		}
			catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}

	if (is_lidar)
		try{
			tfListener.waitForTransform("/map","/3Dlidar_link",msg->header.stamp,ros::Duration(4.0));
			tfListener.lookupTransform("/map", "/3Dlidar_link", msg->header.stamp, transform);
		}
			catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}
	

	tf::Quaternion q = transform.getRotation();
	tf::Matrix3x3 Rotation(q);
	tf::Vector3 Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());

	pcl::fromROSMsg(*msg, latest_local_pcl_);
	
    pcl::PointCloud<pcl::PointXYZ> _pointcloud;

    _pointcloud.clear();
    pcl::PointXYZ newPoint;
    tf::Vector3 _laser_point_body_body_frame,_laser_point_body_ENU_frame;
    
    for (int i = 0; i < (int)latest_local_pcl_.points.size(); i++)
    {
		if(is_rgbd && latest_local_pcl_.points[i].z == 1) continue;
        _laser_point_body_body_frame[0] = latest_local_pcl_.points[i].x;
        _laser_point_body_body_frame[1] = latest_local_pcl_.points[i].y;
        _laser_point_body_body_frame[2] = latest_local_pcl_.points[i].z;
        _laser_point_body_ENU_frame = Rotation * _laser_point_body_body_frame;
        newPoint.x = Origin.getX() + _laser_point_body_ENU_frame[0];
        newPoint.y = Origin.getY() + _laser_point_body_ENU_frame[1];
        newPoint.z = Origin.getZ() + _laser_point_body_ENU_frame[2];
        
        _pointcloud.push_back(newPoint);
    }
    
//	concatenate_PointCloud += _pointcloud;
//	static int frame_id = 0;
//	if (frame_id == timeSteps_fusingSamples){
//		if(flag_pcl_ground_removal){
//			pcl::PassThrough<pcl::PointXYZ> ground_removal;
//			ground_removal.setInputCloud (concatenate_PointCloud.makeShared());
//			ground_removal.setFilterFieldName ("z");
//			ground_removal.setFilterLimits (-1.0, max_ground_height);
//			ground_removal.setFilterLimitsNegative (true);
//			ground_removal.filter (concatenate_PointCloud);
//		}
//		
//		if (flag_pcl_downsampling){
//			pcl::VoxelGrid<pcl::PointXYZ> sor;
//			sor.setInputCloud(concatenate_PointCloud.makeShared());
//			sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
//			sor.filter(concatenate_PointCloud);
//		}
//		
//		local_pcl_tm1 = concatenate_PointCloud;
//		frame_id = 0;
//		concatenate_PointCloud.clear();
//		
//		local_point_cloud = local_pcl_tm1;
//		local_point_cloud += local_pcl_tm2;
//		local_point_cloud += local_pcl_tm3;
//		local_pcl_tm3 = local_pcl_tm2;
//		local_pcl_tm2 = local_pcl_tm1;
//		local_pcl_tm1.clear();
//	} else {
//		local_point_cloud = local_point_cloud;
//		frame_id++;
//	}

	if(flag_pcl_ground_removal){
		pcl::PassThrough<pcl::PointXYZ> ground_removal;
		ground_removal.setInputCloud (_pointcloud.makeShared());
		ground_removal.setFilterFieldName ("z");
		ground_removal.setFilterLimits (-1.0, max_ground_height);
		ground_removal.setFilterLimitsNegative (true);
		ground_removal.filter (_pointcloud);
	}
	if (flag_pcl_downsampling){
		static int frame_id = 0;
		frame_id++;
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(_pointcloud.makeShared());
		sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
		sor.filter(_pointcloud);
		
		concatenate_PointCloud += _pointcloud;
		
		if(frame_id % timeSteps_fusingSamples == 0){
			static int max_point_num = 1000;
			if(concatenate_PointCloud.points.size()>max_point_num){
//				cout << "downsampling: " << (int)concatenate_PointCloud.points.size();
				sor.setInputCloud(concatenate_PointCloud.makeShared());
				sor.setLeafSize(size_of_voxel_grid, size_of_voxel_grid, size_of_voxel_grid);
				sor.filter(concatenate_PointCloud);
				max_point_num = (int(concatenate_PointCloud.points.size()/1000) + 1)*1000;
//				cout << " to " << (int)concatenate_PointCloud.points.size() << " max_point_num " << max_point_num << endl;
			}
				
			if(frame_id % (5*timeSteps_fusingSamples) == 0){
//				cout << "outlier removal: " << (int)concatenate_PointCloud.points.size();
				pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;
				radiusoutlier.setInputCloud(concatenate_PointCloud.makeShared());
				radiusoutlier.setRadiusSearch(2*size_of_voxel_grid);
				radiusoutlier.setMinNeighborsInRadius(15);
				radiusoutlier.filter(concatenate_PointCloud); 
//				cout << " to " << (int)concatenate_PointCloud.points.size() << endl;
			}
		}
	}else
		concatenate_PointCloud += _pointcloud;
	
	pcl::PassThrough<pcl::PointXYZ> sensor_range;
	sensor_range.setInputCloud (concatenate_PointCloud.makeShared());
	sensor_range.setFilterFieldName ("x");
	sensor_range.setFilterLimits (Origin.getX()-sensor_max_range, Origin.getX()+sensor_max_range);
	sensor_range.filter (local_point_cloud);
	sensor_range.setInputCloud (local_point_cloud.makeShared());
	sensor_range.setFilterFieldName ("y");
	sensor_range.setFilterLimits (Origin.getY()-sensor_max_range, Origin.getY()+sensor_max_range);
	sensor_range.filter (local_point_cloud);
	
	
	local_point_cloud.header = latest_local_pcl_.header;
	local_point_cloud.header.frame_id = "world";
//	local_point_cloud.height = 1;
//	local_point_cloud.width = local_point_cloud.points.size();
	point_cloud_pub.publish(local_point_cloud);

	
//	cout << "Rotation: " << endl;
//	cout << "[ [" << Rotation[0][0] << ", " << Rotation[0][1] << ", " << Rotation[0][2] << "]," << endl;
//	cout << "  [" << Rotation[1][0] << ", " << Rotation[1][1] << ", " << Rotation[1][2] << "]," << endl;
//	cout << "  [" << Rotation[2][0] << ", " << Rotation[2][1] << ", " << Rotation[2][2] << "] ]" << endl;
//	cout << "local obstacle points size: " << (int)local_point_cloud.points.size() << endl;
//	cout << "header.seq: " << (int)local_point_cloud.header.seq << endl;
//	cout << "header.stamp: " << (int)local_point_cloud.header.stamp << endl;
//	cout << "header.frame_id: " << local_point_cloud.header.frame_id << endl;
//	cout << "height: " << (int)local_point_cloud.height << endl;
//	cout << "width: " << (int)local_point_cloud.width << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr = local_point_cloud.makeShared();
    local_alg_ptr->set_local_map_pcl(pcl_ptr);
    
    sensor_ready = true;
    static int exec_num = 0;
    exec_num++;
    if(exec_num == 10)
    {
    	std::chrono::duration<double, std::milli> elapsed_seconds = std::chrono::system_clock::now() - start; 
        printf("point_cloud processing takes %f [ms].\n", elapsed_seconds.count());
        exec_num=0;
    }
}

void Local_Planner::localcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    /* need odom_ for center radius sensing */
    if (!odom_ready) 
    {
        return;
    }

    sensor_ready = true;
    
    pcl::fromROSMsg(*msg, local_point_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr = local_point_cloud.makeShared();
    local_alg_ptr->set_local_map_pcl(pcl_ptr);
}

void Local_Planner::control_cb(const ros::TimerEvent& e)
{
    if (!odom_ready || !drone_ready || !planner_enable)
        return;
		
	if (!path_ok){
		if (yaw_tracking_mode){
            Command_Now.header.stamp                        = ros::Time::now();
            Command_Now.source                              = NODE_NAME;
            Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
			Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
			Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XY_VEL_Z_POS;
			Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
		    Command_Now.Reference_State.position_ref[2]     = goal_pos[2];
			Command_Now.Reference_State.velocity_ref[0]     = 0.0;
			Command_Now.Reference_State.velocity_ref[1]     = 0.0;
			
			desired_yaw = desired_yaw + 0.05;
			Command_Now.Reference_State.yaw_ref             = desired_yaw;
		    command_pub.publish(Command_Now);
		}
        return;
	}

    distance_to_goal = Eigen::Vector3d((start_pos - goal_pos)[0],(start_pos - goal_pos)[1],0.0).norm();

    // arrived
    if(distance_to_goal < MIN_DIS)
    {
        Command_Now.header.stamp                        = ros::Time::now();
        Command_Now.source                              = NODE_NAME;
        Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
        Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
        Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
        Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
        Command_Now.Reference_State.position_ref[0]     = goal_pos[0];
        Command_Now.Reference_State.position_ref[1]     = goal_pos[1];
        Command_Now.Reference_State.position_ref[2]     = goal_pos[2];

        Command_Now.Reference_State.yaw_ref             = desired_yaw;
        command_pub.publish(Command_Now);
        
        if (!planner_enable_default)
            pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "Reach the goal! The planner will be disable automatically.");
        else
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Reach the goal! The planner is still enable.");
        
        // stop
        path_ok = false;
        planner_enable = planner_enable_default;
        // wait for a new
        exec_state = EXEC_STATE::WAIT_GOAL;
        return;
    }

    if (is_2D)
    {
		Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
		Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XY_VEL_Z_POS;
		Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
		Command_Now.Reference_State.velocity_ref[0]     = desired_vel[0];
		Command_Now.Reference_State.velocity_ref[1]     = desired_vel[1];
		Command_Now.Reference_State.velocity_ref[2]     = 0.0;
		Command_Now.Reference_State.position_ref[2]     = fly_height_2D;
    }else{
		Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
		Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_VEL;
		Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
		Command_Now.Reference_State.velocity_ref[0]     = desired_vel[0];
		Command_Now.Reference_State.velocity_ref[1]     = desired_vel[1];
		Command_Now.Reference_State.velocity_ref[2]     = desired_vel[2];
    }
		

    // desired yaw
    if (yaw_tracking_mode == 1)
    {
        auto sign=[](double v)->double
        {
            return v<0.0? -1.0:1.0;
        };
        Eigen::Vector3d ref_vel;
        ref_vel[0] = desired_vel[0];
        ref_vel[1] = desired_vel[1];
        //ref_vel[2] = desired_vel[2];

        if( sqrt( ref_vel[1]* ref_vel[1] + ref_vel[0]* ref_vel[0])  >  0.05 || exec_state == EXEC_STATE::PLANNING)
        {
	        /* vel direction is the desired yaw, because of Tait-Bryan z-y-x(3-2-1) angle ratation in px4.
	         * if desired yaw is just the vel direction, the pitch rotation along body-y axis is zero. 
	         */
        	float next_desired_yaw_vel = sign(ref_vel(1)) * acos(ref_vel(0) / ref_vel.norm());
//			cout << "desired_yaw " << desired_yaw << ", next_desired_yaw_vel " << next_desired_yaw_vel << endl;
	
            if (fabs(desired_yaw-next_desired_yaw_vel)<M_PI)
            	desired_yaw = (0.3*desired_yaw + 0.7*next_desired_yaw_vel);
            else
            	desired_yaw = next_desired_yaw_vel + sign(next_desired_yaw_vel) * 0.3/(0.3+0.7)*(2*M_PI-fabs(desired_yaw-next_desired_yaw_vel));
        } else {
            desired_yaw = desired_yaw + 0.05;
        }
    		
    }else if (yaw_tracking_mode == 2)
		desired_yaw = desired_yaw + _max_manual_yaw_rate * 0.05 * rc_r; // 0.05 -> tracking_loop，20Hz
		
    else
        desired_yaw = 0.0;
    
    
	if(desired_yaw>M_PI)
		desired_yaw -= 2*M_PI;
	else if (desired_yaw<-M_PI)
		desired_yaw += 2*M_PI;

    Command_Now.Reference_State.yaw_ref             = desired_yaw;

    Command_Now.header.stamp                        = ros::Time::now();
    Command_Now.source                              = NODE_NAME;
    Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
    command_pub.publish(Command_Now);

}

void Local_Planner::mainloop_cb(const ros::TimerEvent& e)
{
    static int exec_num=0;
    exec_num++;

    if(!odom_ready || !drone_ready || !sensor_ready || !planner_enable)
    {
        if(exec_num == 10)
        {
            if(!planner_enable)
            {
                message = "Planner is disable by default! If you want to enable it, pls set the param [local_planner/enable] as true!";
            }else if(!odom_ready)
            {
                message = "Need Odom.";
            }else if(!drone_ready)
            {
                message = "Drone is not ready.";
            }else if(!sensor_ready)
            {
                message = "Need sensor info.";
            } 

            pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, message);
            exec_num=0;
        }  

        return;
    }else
    {
        odom_ready = false;
        drone_ready = false;
        sensor_ready = false;
    }

    switch (exec_state)
    {
        case WAIT_GOAL:
        {
            path_ok = false;
            if(!goal_ready)
            {
                if(exec_num == 20)
                {
                    message = "Waiting for a new goal.";
                    pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME,message);
                    exec_num=0;
                }
            }else
            {
                exec_state = EXEC_STATE::PLANNING;
                goal_ready = false;
            }
            
            break;
        }
        case PLANNING:
        {
            Eigen::Vector3d vel_;
            static Eigen::Vector3d vel_last_;
            planner_state = local_alg_ptr->generate(goal_pos, vel_);
			
			if(!planner_state){
				exec_state = EXEC_STATE::LANDING;
				break;
			}
			
            path_ok = true;
			
			guide_rviz.header.seq++;
	        guide_rviz.header.stamp = ros::Time::now();
	        guide_rviz.header.frame_id = "world";
	        //　rviz
	        guide_rviz.point.x = _DroneState.position[0]+vel_[0];
	        guide_rviz.point.y = _DroneState.position[1]+vel_[1];
	        guide_rviz.point.z = _DroneState.position[2]+vel_[2];
	        double vel_norm = vel_.norm();
	        if(vel_norm > max_planning_vel){
	        	vel_ *= max_planning_vel / vel_norm;
	        }
            desired_vel = 0.5*vel_last_ + 0.5*vel_;
            vel_last_ = vel_;
	        cout << "planned_vel: " << vel_norm << " max_planning_vel: " << max_planning_vel << 
                    " vel: " << desired_vel(0) << " " << desired_vel(1) << " " << desired_vel(2) << endl;
	        rviz_guide_pub.publish(guide_rviz);

            if(exec_num==20)
            {
                if(planner_state == 0)
                {
                    message = "path generation stoped!";
                }else if(planner_state == 1)
                {
                    message = "local planning desired vel: [" + std::to_string(desired_vel(0)) + "," + std::to_string(desired_vel(1)) + "," + std::to_string(desired_vel(2)) + "]";
                }else if(planner_state == 2)
                {
                    message = "Dangerous!";
                }
                
                pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, message);
                exec_num=0;
            }

            break;
        }
        case  LANDING:
        {
        	pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "start to land.\n");
        	
            Command_Now.header.stamp = ros::Time::now();
            Command_Now.Mode         = drone_msgs::ControlCommand::Land;
            Command_Now.Command_ID   = Command_Now.Command_ID + 1;
            Command_Now.source = NODE_NAME;

            command_pub.publish(Command_Now);
            break;
        }
    }

}

}


