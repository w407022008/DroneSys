#include "global_planner.h"

namespace Global_Planning
{

void Global_Planner::init(ros::NodeHandle& nh)
{
    nh.param("global_planner/is_2D", is_2D, false); 
    nh.param("global_planner/fly_height_2D", fly_height_2D, 1.0);  
    nh.param("global_planner/yaw_tracking_mode", yaw_tracking_mode, false); 
    nh.param("global_planner/safe_distance", safe_distance, 0.05); 
    nh.param("global_planner/time_per_path", time_per_path, 1.0); 
    nh.param("global_planner/replan_time", replan_time, 2.0); 
    
    nh.param("global_planner/sim_mode", sim_mode, false); 

    nh.param("global_planner/map_groundtruth", map_groundtruth, false); 

    // Sub
    goal_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 1, &Global_Planner::goal_cb, this);

    nh.param("global_planner/planner_enable", planner_enable_default, false); 
    planner_enable = planner_enable_default;
    planner_switch_sub = nh.subscribe<std_msgs::Bool>("/global_planner/switch", 10, &Global_Planner::planner_switch_cb, this);

    drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state", 10, &Global_Planner::drone_state_cb, this);

    nh.param("global_planner/map_input", map_input, 0); 
    if(map_input == 0){
        Gpointcloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/global_planner/global_pcl", 1, &Global_Planner::Gpointcloud_cb, this); //groundtruth点云、SLAM全局点云
    }else if(map_input == 1){
        Lpointcloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/global_planner/local_pcl", 1, &Global_Planner::Lpointcloud_cb, this); //RGBD相机、三维激光雷达
    }else if(map_input == 2){
        laserscan_sub = nh.subscribe<sensor_msgs::LaserScan>("/global_planner/laser_scan", 1, &Global_Planner::laser_cb, this); //2维激光雷达
    }

    // Pub
    command_pub = nh.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 10);

    path_cmd_pub   = nh.advertise<nav_msgs::Path>("/global_planner/path_cmd",  10); 


    // Timer
    safety_timer = nh.createTimer(ros::Duration(time_per_path), &Global_Planner::safety_cb, this); 

    mainloop_timer = nh.createTimer(ros::Duration(0.2), &Global_Planner::mainloop_cb, this);  
      
    track_path_timer = nh.createTimer(ros::Duration(time_per_path), &Global_Planner::track_path_cb, this);        

    nh.param("global_planner/algorithm_mode", algorithm_mode, 0); 
    if(algorithm_mode==0){
        global_alg_ptr.reset(new Astar);
        global_alg_ptr->init(nh);
        cout << "A_star init." << endl;
    }
    else if(algorithm_mode==1){
        global_alg_ptr.reset(new KinodynamicAstar);
        global_alg_ptr->init(nh);
        cout << "Kinodynamic A_star init." << endl;
    }

    // init state
    exec_state = EXEC_STATE::WAIT_GOAL;
    odom_ready = false;
    drone_ready = false;
    goal_ready = false;
    sensor_ready = false;
    is_safety = true;
    is_new_path = false;

    // init cmd
    Command_Now.header.stamp = ros::Time::now();
    Command_Now.Mode  = drone_msgs::ControlCommand::Idle;
    Command_Now.Command_ID = 0;
    Command_Now.source = NODE_NAME;
    desired_yaw = 0.0;

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
    while(!drone_ready){
      ros::spinOnce();
    }
    cout << "Switched to OFFBOARD and armed, drone will take off after 1.0s"<<endl;
    ros::Duration(1.0).sleep();

    Command_Now.header.stamp = ros::Time::now();
    Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
    Command_Now.Command_ID = Command_Now.Command_ID + 1;
    Command_Now.source = NODE_NAME;
    command_pub.publish(Command_Now);
    cout << "Takeoff"<<endl;
  }else
  {
    // Real flight situation: Manually switch to offboard mode and send a take-off command
    while(start_flag == 0)
    {
        cout << "Please manually switch to offboard mode and arm it and then type in 1 for takeoff:"<<endl;
        cin >> start_flag;
    }
    Command_Now.header.stamp = ros::Time::now();
    Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
    Command_Now.Command_ID = Command_Now.Command_ID + 1;
    Command_Now.source = NODE_NAME;
    command_pub.publish(Command_Now);
    //ros::Duration(5.0).sleep();
    cout << "Takeoff"<<endl;
  }
}

void Global_Planner::planner_switch_cb(const std_msgs::Bool::ConstPtr& msg){
    if (!planner_enable && msg->data){
        ROS_WARN("Enable planner!");
    }else if (planner_enable && !msg->data){
        ROS_WARN("Disable planner!");
        exec_state = EXEC_STATE::WAIT_GOAL;
    }
    planner_enable = msg->data;
}

void Global_Planner::goal_cb(const geometry_msgs::PoseStampedConstPtr& msg){
    if (is_2D == true){
        goal_pos << msg->pose.position.x, msg->pose.position.y, fly_height_2D;
    }else{
        if(msg->pose.position.z < 1.0)
            goal_pos << msg->pose.position.x, msg->pose.position.y, 1.0;
        else
            goal_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    }
        
    goal_vel.setZero();

    goal_ready = true;

    if(planner_enable){
        cout << "Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;

        if(goal_pos(0) == 99 && goal_pos(1) == 99 ){
            path_ok = false;
            goal_ready = false;
            exec_state = EXEC_STATE::LANDING;
            ROS_WARN("Landing");
        }
    }
}

void Global_Planner::drone_state_cb(const drone_msgs::DroneStateConstPtr& msg){
    _DroneState = *msg;

    if (is_2D == true){
        start_pos << msg->position[0], msg->position[1], fly_height_2D;
        start_vel << msg->velocity[0], msg->velocity[1], 0.0;
    }else{
        start_pos << msg->position[0], msg->position[1], msg->position[2];
        start_vel << msg->velocity[0], msg->velocity[1], msg->velocity[2];
    }

    start_acc << 0.0, 0.0, 0.0;

    odom_ready = true;

    if (_DroneState.connected == true && _DroneState.armed == true ){
        drone_ready = true;
    }else{
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
}

void Global_Planner::Gpointcloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg){
    /* need odom_ for center radius sensing */
    if (!odom_ready) 
        return;
    sensor_ready = true;
    global_alg_ptr->Occupy_map_ptr->map_update_gpcl(msg);
    global_alg_ptr->Occupy_map_ptr->inflate_point_cloud(); 
}

void Global_Planner::Lpointcloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg){
    /* need odom_ for center radius sensing */
    if (!odom_ready) 
        return;    
    sensor_ready = true;
    global_alg_ptr->Occupy_map_ptr->map_update_lpcl(msg);
    global_alg_ptr->Occupy_map_ptr->inflate_point_cloud(); 
}

void Global_Planner::laser_cb(const sensor_msgs::LaserScanConstPtr &msg){
    /* need odom_ for center radius sensing */
    if (!odom_ready) 
        return;
    sensor_ready = true;
    global_alg_ptr->Occupy_map_ptr->map_update_laser(msg);
    global_alg_ptr->Occupy_map_ptr->inflate_point_cloud(); 
}

void Global_Planner::track_path_cb(const ros::TimerEvent& e){
	if (!planner_enable_default)
		return;
		
	static ros::Time last_cmd_pub_time = ros::Time::now();
    if(!path_ok){
		if (yaw_tracking_mode && exec_state == EXEC_STATE::WAIT_GOAL ){
			Command_Now.header.stamp                        = ros::Time::now();
			Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
			Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
			Command_Now.source                              = NODE_NAME;
			Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XY_VEL_Z_POS;
			Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
		    Command_Now.Reference_State.position_ref[2]     = _DroneState.position[2];
			Command_Now.Reference_State.velocity_ref[0]     = 0.0;
			Command_Now.Reference_State.velocity_ref[1]     = 0.0;
			
			desired_yaw = desired_yaw + 0.5*(ros::Time::now()-last_cmd_pub_time).toSec();
			if(desired_yaw>M_PI)
				desired_yaw -= 2*M_PI;
			Command_Now.Reference_State.yaw_ref             = desired_yaw;
		    command_pub.publish(Command_Now);
		    last_cmd_pub_time = ros::Time::now();
		}
        return;
    }

    is_new_path = false;

    if(cur_id == Num_total_wp - 1)
    {
        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
        Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
        Command_Now.source = NODE_NAME;
        Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
        Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
        Command_Now.Reference_State.position_ref[0]     = goal_pos[0];
        Command_Now.Reference_State.position_ref[1]     = goal_pos[1];
        Command_Now.Reference_State.position_ref[2]     = goal_pos[2];

        Command_Now.Reference_State.yaw_ref             = desired_yaw;
        command_pub.publish(Command_Now);

        planner_enable = planner_enable_default;
        if (planner_enable_default)
            ROS_WARN("Reach the goal! The planner will be disable automatically.");
        else
            ROS_WARN("Reach the goal! The planner is still enable.");
        
        path_ok = false;
        exec_state = EXEC_STATE::WAIT_GOAL;
        return;
    }
 
    int i = cur_id;
    cout << "Moving to waypoint: [ " << cur_id << " / "<< Num_total_wp<< " ] "<<endl;
    cout << "Moving to waypoint:"   << path_cmd.poses[i].pose.position.x  << " [m] "
                                    << path_cmd.poses[i].pose.position.y  << " [m] "
                                    << path_cmd.poses[i].pose.position.z  << " [m] "<<endl; 

    Command_Now.header.stamp                        = ros::Time::now();
    Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
    Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
    Command_Now.source = NODE_NAME;
    Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
    Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
    Command_Now.Reference_State.position_ref[0]     = path_cmd.poses[i].pose.position.x;
    Command_Now.Reference_State.position_ref[1]     = path_cmd.poses[i].pose.position.y;
    Command_Now.Reference_State.position_ref[2]     = path_cmd.poses[i].pose.position.z;
    //Command_Now.Reference_State.velocity_ref[0]     = (path_cmd.poses[i].pose.position.x - _DroneState.position[0])/time_per_path;
    //Command_Now.Reference_State.velocity_ref[1]     = (path_cmd.poses[i].pose.position.y - _DroneState.position[1])/time_per_path;
    //Command_Now.Reference_State.velocity_ref[2]     = (path_cmd.poses[i].pose.position.z - _DroneState.position[2])/time_per_path;

    if (yaw_tracking_mode){
        auto sign=[](double v)->double {return v<0.0? -1.0:1.0;};
        
        // Eigen::Vector3d ref_vel;
        // ref_vel[0] = _DroneState.velocity[0];
        // ref_vel[1] = _DroneState.velocity[1];
        // ref_vel[2] = 0.0;
            
        Eigen::Vector3d ref_pos;
        ref_pos[0] = path_cmd.poses[i].pose.position.x;
        ref_pos[1] = path_cmd.poses[i].pose.position.y;
        ref_pos[2] = 0.0;

        Eigen::Vector3d curr_pos;
        curr_pos[0] = _DroneState.position[0];
        curr_pos[1] = _DroneState.position[1];
        curr_pos[2] = 0.0;

        Eigen::Vector3d diff_pos = ref_pos - curr_pos;
        if (diff_pos.norm()>1e-3) {
            // float next_desired_yaw_vel      = sign(ref_vel(1)) * acos(ref_vel(0) / ref_vel.norm());
            float next_desired_yaw_pos      = sign(diff_pos(1)) * acos(diff_pos(0) / diff_pos.norm());

            if (fabs(desired_yaw-next_desired_yaw_pos)<M_PI)
                desired_yaw = (0.4*desired_yaw + 0.6*next_desired_yaw_pos);
            else
                desired_yaw = next_desired_yaw_pos + sign(next_desired_yaw_pos) * 0.4/(0.4+0.6)*(2*M_PI-fabs(desired_yaw-next_desired_yaw_pos));
        
            if(desired_yaw>M_PI)
                desired_yaw -= 2*M_PI;
            else if (desired_yaw<-M_PI)
                desired_yaw += 2*M_PI;
        }
    }else
        desired_yaw = 0.0;

    Command_Now.Reference_State.yaw_ref             = desired_yaw;
    command_pub.publish(Command_Now);
	last_cmd_pub_time = ros::Time::now();
	
    cur_id = cur_id + 1;
}
 
void Global_Planner::mainloop_cb(const ros::TimerEvent& e)
{
    static int exec_num=0;
    exec_num++;

    if(!odom_ready || !drone_ready || !sensor_ready ||!planner_enable){
        if(exec_num == 10)
        {
            if(!planner_enable)
            {
                ROS_WARN("Planner is disable by default! If you want to enable it, pls set the param [global_planner/enable] as true!");
            }else if(!odom_ready)
            {
                ROS_WARN("Need Odom.");
            }else if(!drone_ready)
            {
                ROS_WARN("Drone is not ready.");
            }else if(!sensor_ready)
            {
                ROS_WARN("Need sensor info.");
            } 
            exec_num=0;
        }  
        return;
    }else{
        odom_ready = false;
        drone_ready = false;
        sensor_ready = false;
    }
    
    switch (exec_state){
        case WAIT_GOAL:{
            path_ok = false;
            if(!goal_ready){
                if(exec_num == 10){
                    std::cout<<"Waiting for a new goal."<<std::endl;
                    exec_num=0;
                }
            }else{
                exec_state = EXEC_STATE::PLANNING;
                goal_ready = false;
            }
            
            break;
        }
        case PLANNING:{
            global_alg_ptr->reset();
            // Astar algorithm
            int astar_state;
            if(algorithm_mode==0)
                astar_state = global_alg_ptr->search(start_pos, Eigen::Vector3d(0,0,0), Eigen::Vector3d(0,0,0), goal_pos, Eigen::Vector3d(0,0,0), false, false, 0);
            else if(algorithm_mode==1)
                astar_state = global_alg_ptr->search(start_pos, start_vel, start_acc, goal_pos, goal_vel, false, true, 0);

            if(astar_state==Astar::NO_PATH){
                path_ok = false;
                exec_state = EXEC_STATE::WAIT_GOAL;
                ROS_ERROR("Planner can't find path, please reset the goal!");
            }else{
                path_ok = true;
                is_new_path = true;
                path_cmd = global_alg_ptr->get_ros_path();
                Num_total_wp = path_cmd.poses.size();
                start_point_index = get_start_point_id();
                cur_id = start_point_index;
                tra_start_time = ros::Time::now();
                exec_state = EXEC_STATE::TRACKING;
                path_cmd_pub.publish(path_cmd);
                std::cout<<"Get a new path!"<<std::endl;
            }

            break;
        }
        case TRACKING:{
            if((ros::Time::now() - tra_start_time).toSec() >= replan_time){
                exec_state = EXEC_STATE::PLANNING;
                std::cout<<"time to replan"<<std::endl;
                exec_num = 0;
            }

            break;
        }
        case  LANDING:{
            Command_Now.header.stamp = ros::Time::now();
            Command_Now.Mode         = drone_msgs::ControlCommand::Land;
            Command_Now.Command_ID   = Command_Now.Command_ID + 1;
            Command_Now.source = NODE_NAME;

            command_pub.publish(Command_Now);
            break;
        }
    }
}

float Global_Planner::get_time_in_sec(const ros::Time& begin_time)
{
    ros::Time time_now = ros::Time::now();
    float currTimeSec = time_now.sec - begin_time.sec;
    float currTimenSec = time_now.nsec / 1e9 - begin_time.nsec / 1e9;
    return (currTimeSec + currTimenSec);
}

void Global_Planner::safety_cb(const ros::TimerEvent& e){
    // Eigen::Vector3d cur_pos(_DroneState.position[0], _DroneState.position[1], _DroneState.position[2]);
    // is_safety = global_alg_ptr->check_safety(cur_pos, safe_distance);
    for(auto cur_pos : global_alg_ptr->path_pos){
        if(global_alg_ptr->Occupy_map_ptr->getOccupancy(cur_pos)){
            exec_state = EXEC_STATE::PLANNING;
            std::cout<<"========== safety replan =========="<<std::endl;
            std::cout<<cur_pos.transpose()<<std::endl;
            break;
        }
    }
}

int Global_Planner::get_start_point_id(void){
    int id = 0;
    float distance_to_wp_min = abs(path_cmd.poses[0].pose.position.x - _DroneState.position[0])
                                + abs(path_cmd.poses[0].pose.position.y - _DroneState.position[1])
                                + abs(path_cmd.poses[0].pose.position.z - _DroneState.position[2]);
    
    float distance_to_wp;

    for (int j=1; j<Num_total_wp;j++){
        distance_to_wp = abs(path_cmd.poses[j].pose.position.x - _DroneState.position[0])
                                + abs(path_cmd.poses[j].pose.position.y - _DroneState.position[1])
                                + abs(path_cmd.poses[j].pose.position.z - _DroneState.position[2]);
        
        if(distance_to_wp < distance_to_wp_min){
            distance_to_wp_min = distance_to_wp;
            id = j;
        }
    }

    if(id + 2 < Num_total_wp)
        id = id + 2;
    return id;
}
}
