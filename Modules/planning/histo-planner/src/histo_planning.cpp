#include "histo_planning.h"
#include <string> 	
#include <time.h>
#include <chrono>
#include <fstream> 
#include <math.h>
#define debug false

namespace Histo_Planning
{

inline auto sign=[](double v)->double
{
  return v<0.0? -1.0:1.0;
};

void Histo_Planner::init(ros::NodeHandle& nh)
{
  /* ---------- Parameter initialisation ---------- */
  nh.param("geo_fence/x_min", geo_fence_x_min, -100.0);
  nh.param("geo_fence/x_max", geo_fence_x_max, 100.0);
  nh.param("geo_fence/y_min", geo_fence_y_min, -100.0);
  nh.param("geo_fence/y_max", geo_fence_y_max, 100.0);
  nh.param("geo_fence/z_min", geo_fence_z_min, -100.0);
  nh.param("geo_fence/z_max", geo_fence_z_max, 100.0);
  /* -------------- Histogram -------------*/
  nh.param("histogram/min_vel_default", min_vel_default, 1.0);                                   // At low speeds, trajectories are generated without considering initial velocity
  /* -------------- B-Spline -------------*/
  nh.param("bspline/limit_vel", Bspline::limit_vel_, -1.0);                         // velocity limit
  nh.param("bspline/limit_acc", Bspline::limit_acc_, -1.0);                         // accel limit
  /* -------------- Planner -------------*/
  nh.param("histo_planner/sim_mode", sim_mode, true);                                      // simulation ?
  nh.param("histo_planner/path_tracking_enable", tracking_controller_enable, true);        // Whether to enable track tracking and pub reference
  nh.param("histo_planner/CNNLogEnable", CNNLogEnable, false);                             // log data used to train NN
  
  nh.param("histo_planner/is_2D", is_2D, false);                                           // 2D planning? fixed height
  nh.param("histo_planner/fly_height_2D", fly_height_2D, 1.0);  
  /* -- Whether to use joy control as a second input? 
  *  -- 0：disable
  *  -- 1：control in Body Frame
  *  -- 2：control in User Heading Frame
  */
  nh.param("histo_planner/control_from_joy", control_from_joy, 0);
  nh.param("histo_planner/joy_goal_xy_max", _max_goal_range_xy, 3.0);                      // Maximum relative distance in the horizontal direction of the target（when joy control）
  nh.param("histo_planner/joy_goal_z_max", _max_goal_range_z, 3.0);                        // Maximum relative height of the target in the vertical direction（when joy control）
  /* -- Yaw auto-hold？
  *  -- 0：Disable 
  *  -- 1：Auto-tracking 
  *  -- 2：Manually-tracking
  */
  nh.param("histo_planner/yaw_tracking_mode", yaw_tracking_mode, 1); 
  nh.param("histo_planner/spinning_once_first", spinning_once_first, false);               // Spinning once turn first before starting tracking a new traj
  nh.param("histo_planner/time_forward_facing_toward", time_forward_facing_toward, 2.0);   // Towards the waypoint n second later
  nh.param("histo_planner/yaw_rate", yaw_rate, 3.0);                                       // Maximum yaw speed (whenever yaw auto-hold or joy input)
  nh.param("histo_planner/yaw_tracking_err_max", yaw_tracking_err_max, 0.5);               // maximum yaw tracking error
  Bspline::limit_omega_ = yaw_rate;
  Bspline::yaw_track = yaw_tracking_mode>0;
  /* -- Regenerate goal when it is unreachable ?
  *  -- 1：Move the original goal in the direction of the falling gradient of the obstacle map! [Not recommended]
  *  -- 2：Searching within the cylindrical space centred on the original goal, first searching for positions closer to cur_pos and the goal.  [Not recommended]
  *  -- 3：Back to the feasible position along the trajectory
  */
  nh.param("histo_planner/goal_regenerate_mode", goal_regenerate_mode, 3);
  nh.param("histo_planner/min_goal_height", min_goal_height, 1.0);                       // Minimum vertical height of end-point regenerated
  
  nh.param("histo_planner/forbidden_range", forbidden_range, 0.20);                      // Obstacle inflation distance
  nh.param("histo_planner/max_tracking_error", safe_distance, 0.2);                             // safe stopping distance
  forbidden_plus_safe_distance = safe_distance + forbidden_range;
  nh.param("histo_planner/sensor_max_range", sensor_max_range, 3.0);                       // maximum range sensed
  nh.param("histo_planner/range_near_start", range_near_start, 0.1);
  nh.param("histo_planner/range_near_end", range_near_end, 0.1);                           // start-end zone
  nh.param("histo_planner/time_traj_interval", time_interval, 0.1);                        // time interval
  nh.param("histo_planner/time_to_replan", time_to_replan, 5.0);                             // regular replanning interval

  nh.param("histo_planner/ground_height", max_ground_height, 0.1);
  nh.param("histo_planner/ceil_height", ceil_height, 5.0);

  /* ---------- [SUB] ---------- */
  // [SUB] planner enbale
  swith_sub = nh.subscribe<std_msgs::Bool>("/histo_planner/switch", 10, &Histo_Planner::switchCallback, this);  

  // [SUB] goal
  user_yaw_sub = nh.subscribe<sensor_msgs::Imu>("/wit/imu", 10, &Histo_Planner::user_yaw_cb, this);
  manual_control_sub = nh.subscribe<drone_msgs::RCInput>("/joy/RCInput", 10, &Histo_Planner::manual_control_cb, this);// Radio Control input
  goal_sub = nh.subscribe<geometry_msgs::PoseStamped>("/histo_planner/goal", 1, &Histo_Planner::goal_cb, this);// terminal input

  // [SUB] drone state
  drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/histo_planner/drone_state", 10, &Histo_Planner::drone_state_cb, this);
  
  // [SUB] point cloud
  local_point_clound_sub = nh.subscribe<sensor_msgs::PointCloud2>("/histo_planner/local_pcl", 1, &Histo_Planner::localcloudCallback, this);
  
  /* ---------- [PUB] ---------- */
  // [PUB] Flight commands
  command_pub = nh.advertise<drone_msgs::ControlCommand>("/histo_planner/control_command", 10);
  
  // [PUB] message
  message_pub = nh.advertise<drone_msgs::Message>("/drone_msg/message", 10);
  
  // [PUB] visualisation
  rviz_guide_pub = nh.advertise<geometry_msgs::PointStamped >("/histo_planner/guide", 10); // guide point
  rviz_closest_pub = nh.advertise<geometry_msgs::PointStamped >("/histo_planner/closest", 10); // closest obs
  rviz_joy_goal_pub = nh.advertise<geometry_msgs::PointStamped >("/histo_planner/goal", 10); // joy goal set
  image_transport::ImageTransport it(nh);
  obs_img_pub = it.advertise("/histo_planner/obs_image",1);
  his_img_pub = it.advertise("/histo_planner/his_image",1);

  // [PUB] Post new goal
  // goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 10);

  /* ---------- [Timer] ---------- */
  // Joystick Goal Point Pub Loop，10Hz
  if(control_from_joy > 0) joy_loop = nh.createTimer(ros::Duration(0.1), &Histo_Planner::joy_cb, this);
  // Task execution loop，50Hz
  mission_loop = nh.createTimer(ros::Duration(0.02), &Histo_Planner::mission_cb, this);
  // Tracking control loop, 100Hz
  control_loop = nh.createTimer(ros::Duration(0.01), &Histo_Planner::control_cb, this);
  delta_yaw = yaw_rate * 0.01;

  /* ---------- Fuction initialisation ---------- */
  // Initialising the local target search algorithm
  histo_planning_.reset(new HIST);
  histo_planning_->init(nh);
  pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Environment init.");
  ROS_INFO("--- environment init finished! ---");
  // Initialisation of the trajectory optimisation algorithm
  bspline_optimizer_.reset(new BsplineOptimizer);
  bspline_optimizer_->setParam(nh);
  bspline_optimizer_->setEnvironment(histo_planning_);
  pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "BsplineOptimizer init.");
  ROS_INFO("--- bspline opt init finished! ---");
  // Initial visualisation
  visualization_.reset(new PlanningVisualization(nh));
  pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "PlanningVisualization init.");
  ROS_INFO("--- visualization init finished! ---");

  // Initialisation of planner state parameters
  exec_state = EXEC_STATE::WAIT_GOAL;
  planner_enable = true;
  odom_ready = false;
  drone_ready = false;
  goal_ready = false;
  raw_goal_ready = false;
  map_ready = false;
  path_ok = false;
  is_generating = false;
  escape_mode = false;
  planner_state = 0;
  flag_tracking = 2;
  yaw_ref_comd = 0.0;

  cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>VFHB Planner<<<<<<<<<<<<<<<<<<<<<<<<<<< "<< endl;
  if(control_from_joy > 0) cout << "[Init]: joystick remote enable" << endl;
  
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

  start_flag = 0;
  while(start_flag == 0)
  {
    cout << "Please type in 1 if you ready for navigation:"<<endl;
    cin >> start_flag;
  }
  ros::spinOnce();
  while(!odom_ready)
  {
    ros::spinOnce();
    cout << "[Init]: wait for odom" << endl;
  }
  stop_pos = cur_pos;
  cur_pos_ref = cur_pos;
  cur_vel_ref.setZero();
  cur_acc_ref.setZero();
  yaw_ref_comd = cur_yaw;
  cout << "[Init]: Current Position: " << stop_pos.transpose() << " current yaw: "<< cur_yaw << endl;

  ros::spin();
}

// Activate to track traj
void Histo_Planner::switchCallback(const std_msgs::Bool::ConstPtr &msg)
{
  if (!planner_enable && msg->data){
    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Planner is enable.");
  }else if (planner_enable && !msg->data){
    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Planner is disable.");
    exec_state = EXEC_STATE::WAIT_GOAL;
  }
  planner_enable = msg->data;
}

// Detects the operator orientation so that the published goal is in the operator's frame.
void Histo_Planner::user_yaw_cb(const sensor_msgs::ImuConstPtr& msg)
{
  Eigen::Quaterniond q_fcu = Eigen::Quaterniond(msg->orientation.w, msg->orientation.y, msg->orientation.x, -msg->orientation.z); // NED -> ENU
  Eigen::Vector3d euler_fcu = quaternion_to_euler(q_fcu);
  user_yaw = euler_fcu[2];

  static int flag = 0;
  if(!flag && _DroneState.armed) flag = 1;
  if(flag == 1)
  {
    flag = 2;
    user_yaw_init = euler_fcu[2]; // by default: user stands directly behind the drone and face to it at the beginning
  }
}

// Directly acquire the goal point.
void Histo_Planner::goal_cb(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if(!_DroneState.connected){
    cout << "[goal_cb]: Drone disconnected!" << endl;
    goal_ready = false;
    return;
  }
  // Got a new goal
  if (is_2D == true){
    goal_pos << msg->pose.position.x, msg->pose.position.y, fly_height_2D;
  }else{
    if(msg->pose.position.z<min_goal_height)
      goal_pos << msg->pose.position.x, msg->pose.position.y, min_goal_height;
    else
      goal_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  }

  goal_vel.setZero();
  goal_acc.setZero();
  goal_ready = true;

  if(planner_enable){
    if(exec_state == EXEC_STATE::EXEC_TRAJ){
      traj_duration_ = min((ros::Time::now() - time_traj_start).toSec() + 1.0, traj_duration_);
      changeExecState(REPLAN_TRAJ, "Goal Operator");
      
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");
      cout << "[Planner]: Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;
      }else if(goal_pos(0) == 99 && goal_pos(1) == 99 ){
      path_ok = false;
      goal_ready = false;
      changeExecState(LANDING, "Goal Operator");
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Land");
    }
  }else if(exec_state == EXEC_STATE::LANDING && _DroneState.position[2]<ceil_height && _DroneState.position[2]>max_ground_height){
    planner_enable = true;
    changeExecState(GEN_NEW_TRAJ, "Goal Operator");
    stop_pos = cur_pos;
    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");
    cout << "[Planner]: Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;
  }
}

// Get goal point via joystick or px4 transferred RC stick measurements.
void Histo_Planner::manual_control_cb(const drone_msgs::RCInputConstPtr& msg)
{
  if(control_from_joy == 0 || !drone_ready || !odom_ready){
    return;
  }
		
  // select command frame, Yaw adjust only
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

  if(goal_in_local_frame.norm() < 0.1)
    return;

  goal_in_map_frame = R_Local_to_Joy * goal_in_local_frame;
  if (is_2D)
    raw_goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], fly_height_2D;
  else
    if(_DroneState.position[2] + goal_in_map_frame[2] < min_goal_height)
      raw_goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], min_goal_height;
    else if(_DroneState.position[2] + goal_in_map_frame[2] > ceil_height)
      raw_goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], ceil_height-0.1;
    else
      raw_goal_pos << _DroneState.position[0] + goal_in_map_frame[0], _DroneState.position[1] + goal_in_map_frame[1], _DroneState.position[2] + goal_in_map_frame[2];
    
  // geofence
  raw_goal_pos[0] = max(geo_fence_x_min+0.2,min(geo_fence_x_max-0.2,raw_goal_pos[0]));
  raw_goal_pos[1] = max(geo_fence_y_min+0.2,min(geo_fence_y_max-0.2,raw_goal_pos[1]));
  raw_goal_pos[2] = max(geo_fence_z_min+0.2,min(geo_fence_z_max-0.2,raw_goal_pos[2]));
  
  // Post to rviz to show the location of goal point
  joy_goal_rviz.header.seq++;
  joy_goal_rviz.header.stamp = ros::Time::now();
  joy_goal_rviz.header.frame_id = "world";
  joy_goal_rviz.point.x = raw_goal_pos[0];
  joy_goal_rviz.point.y = raw_goal_pos[1];
  joy_goal_rviz.point.z = raw_goal_pos[2];
  rviz_joy_goal_pub.publish(joy_goal_rviz);
    
  if(msg->goal_enable==1)
  {
    raw_goal_ready = true;
  }
}

// Drone state in real time
void Histo_Planner::drone_state_cb(const drone_msgs::DroneStateConstPtr& msg)
{
    _DroneState = *msg; // in ENU frame

    static int flag = 0;
    if(!flag && _DroneState.armed) flag = 1;
    if(flag == 1)
    {
      flag = 2;
      drone_yaw_init = _DroneState.attitude[2]; // by default: user stands directly behind the drone and face to it at the beginning
    }

    if (_DroneState.connected == true && _DroneState.armed == true )
        drone_ready = true;
    else
        drone_ready = false;

    cur_pos << msg->position[0], msg->position[1], msg->position[2];
    cur_vel << msg->velocity[0], msg->velocity[1], msg->velocity[2];
    cur_acc.setZero();
    tf::Quaternion q(
        _DroneState.attitude_q.x,
        _DroneState.attitude_q.y,
        _DroneState.attitude_q.z,
        _DroneState.attitude_q.w);
    tf::Matrix3x3(q).getRPY(cur_roll, cur_pitch, cur_yaw);

    cur_yaw_rate = msg->attitude_rate[2];

    Drone_odom.header = _DroneState.header;
    Drone_odom.child_frame_id = "base_link";

    Drone_odom.pose.pose.position.x = _DroneState.position[0];
    Drone_odom.pose.pose.position.y = _DroneState.position[1];
    Drone_odom.pose.pose.position.z = _DroneState.position[2];

    Drone_odom.pose.pose.orientation = _DroneState.attitude_q;
    Drone_odom.twist.twist.linear.x = _DroneState.velocity[0];
    Drone_odom.twist.twist.linear.y = _DroneState.velocity[1];
    Drone_odom.twist.twist.linear.z = _DroneState.velocity[2];

    odom_ready = true;
    histo_planning_->set_odom(Drone_odom);
}

// Obstacle detection in real time
void Histo_Planner::localcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  pcl::PointCloud<pcl::PointXYZ> local_point_cloud; // point cloud
  pcl::fromROSMsg(*msg, local_point_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr = local_point_cloud.makeShared();
  histo_planning_->set_local_map_pcl(pcl_ptr);
  map_ready = true;

  // Publish Obstacle histogram
  // axis +z    ---
  //    |       ---
  //    |     axis -x   axis +y   axis +x   axis -y   axis -x
  //    |       ---
  // axis -z    ---
  cv_bridge::CvImage out_msg;
  out_msg.header.frame_id = "base_link";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1; 

  double** Obstacle_3d = histo_planning_->get_Obstacle_3d();
  int Hcnt = histo_planning_->get_Hcnt();
  int Vcnt = histo_planning_->get_Vcnt();
  if(Vcnt == 1){
    Vcnt = 10;
    cv::Mat image(Vcnt,Hcnt,CV_32F);
    for(int j=0; j<Hcnt; j++){
      float val = (float)Obstacle_3d[0][(Hcnt-j+Hcnt/2-1)%Hcnt];
      for(int i=0; i<Vcnt; i++){
        ((float*)image.data)[i*Hcnt+j] = val;
      }
    }
    out_msg.image    = image;
  }else{
    cv::Mat image(Vcnt,Hcnt,CV_32F);
    for(int i=0; i<Vcnt; i++)
      for(int j=0; j<Hcnt; j++){
        ((float*)image.data)[i*Hcnt+j] = (float)Obstacle_3d[Vcnt-1-i][(Hcnt-j+Hcnt/2-1)%Hcnt];
      }
    out_msg.image    = image;
  }

  obs_img_pub.publish(out_msg.toImageMsg());
}

// [Core] Generate Traj
bool Histo_Planner::generateTrajectory(Eigen::Vector3d start_pos_, Eigen::Vector3d start_vel_, Eigen::Vector3d start_acc_, Eigen::Vector3d end_pos_, Eigen::Vector3d end_vel_, Eigen::Vector3d end_acc_)
{
  is_generating = true;
  /*
  cout << "==============================================" << endl;
  cout << "======= [planner]: Planning Starts Now! ======" << endl;
  cout << "==============================================" << endl;
  cout << "start: \t pos: " << start_pos_.transpose() << "\n\t vel: " << start_vel_.transpose() << "\n\t acc: " << start_acc_.transpose()
        << "\ngoal: \t pos: " << end_pos_.transpose() << "\n\t vel: " << end_vel_.transpose() << "\n\t acc: " << end_acc_.transpose() << endl;
  */
  std::chrono::time_point<std::chrono::system_clock> tic, toc; 

  /* ---------- local guide point finding ---------- */
  //cout << "======= [planner]: Initialization =======" << endl;
  tic = std::chrono::system_clock::now(); 
  /*
  input:
    start_pos_, start_vel_, end_pos_
  output: 
    guide_point
    state:
      0: search successfully; 
      1: aircraft not safe (too close to some obstacles)
  */
  int state = histo_planning_->generate(start_pos_, start_vel_, end_pos_, guide_point);
	// planner_state = planner_state*state+state;
  // if(planner_state==5)
  // {
  //   planner_state = 0;
  //   escape_mode = true;
  // }

  // cout << "guide point pos: " << guide_point.transpose() << endl;

  /* ---------- B-Spline parameterization ---------- */
  //cout << "======= [planner]: Initialize B Spline =======" << endl;
  // Get sampling points on init curve with obs avoidance unawareness
  vector<Eigen::Vector3d> samples;
  if(escape_mode){
    end_pos_ = guide_point;
    end_vel_.setZero();
    end_acc_.setZero();
    samples = histo_planning_->getSamples(start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_, start_pos_, time_interval);
  }else
    samples = histo_planning_->getSamples(start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_, guide_point, time_interval);
  
  // fitting sampling points with B-Spline
  Eigen::MatrixXd control_pts;
  Bspline::cubicSamplePts_to_BsplineCtlPts(samples, time_interval, control_pts);
  traj_init_ = Bspline(control_pts, 3, time_interval);// B-Spline Unoptimised

  toc = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_planning = toc - tic; // time using for finding init traj with a little obs avoidance awareness

  //cout << "sample: " << endl;
  //for(int i =0;i<samples.size();i++)
  //{
  //  cout << samples[i].transpose() << endl;
  //}
  //cout << "init_ctl:" << endl;
  //for(int i =0;i<control_pts.rows();i++)
  //{
  //  cout << control_pts.row(i) << endl;
  //}


  /* ---------- optimize trajectory ---------- */
  //cout << "=========== [planner]: Optimization ==========" << endl;
  tic = std::chrono::system_clock::now(); 
  // if((start_pos_-guide_point).norm() > range_near_start){
    bspline_optimizer_->setControlPoints(control_pts);// set control pts
    bspline_optimizer_->setBSplineInterval(time_interval);// set interval
    bspline_optimizer_->setBoundaryCondition(start_pos_, end_pos_, start_vel_, end_vel_, start_acc_, end_acc_);
    bspline_optimizer_->optimize(BsplineOptimizer::ALL, escape_mode ? BsplineOptimizer::POS : BsplineOptimizer::ALL);
    control_pts = bspline_optimizer_->getControlPoints();// got control pts optimized
  // }

  //cout << "opt_ctl:" << endl;
  //for(int i =0;i<control_pts.rows();i++)
  //{
  //  cout << control_pts.row(i) << endl;
  //}

  /* ---------- time adjustment ---------- */
  //cout << "========= [planner]: Time Adjustment =========" << endl;
  Bspline pos = Bspline(control_pts, 3, time_interval);// get B-Spline 
  double tm, tmp, to, tn;
  pos.getTimeSpan(tm, tmp);
  if(tmp < tm) return false;
  to = tmp - tm;
  int iter_num = 0;
  while (ros::ok()){
    ++iter_num;
    /* actually it not needed, converges within 10 iterations */
    if (pos.reallocateTime(false, time_interval) || iter_num >= 50)
      break;
  }
  toc = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_optimize = toc - tic; // time using for control pts optimized
  pos.getTimeSpan(tm, tmp);
  tn = tmp - tm;
  cout << "[Adjustment]: iter num: " << iter_num << ", Reallocate ratio: " << tn / to << endl;

  //cout << "================================== " << endl;

  std::chrono::duration<double, std::milli> t_total = t_planning + t_optimize;// time using total
  cout << "[planner]: total: \033[31m" << t_total.count() << "\033[0m[ms], search: \033[32m" << t_planning.count() << "\033[0m[ms], optimize: \033[32m" << t_optimize.count() << "\033[0m[ms]\n" << endl;
  
  /* ---------- save result ---------- */
  traj_pos_ = pos;
  traj_vel_ = traj_pos_.getDerivative();
  traj_acc_ = traj_vel_.getDerivative();
  
  // Publish the location of guide point
  guide_rviz.header.seq++;
  guide_rviz.header.stamp = ros::Time::now();
  guide_rviz.header.frame_id = "world";
  guide_rviz.point.x = guide_point[0];
  guide_rviz.point.y = guide_point[1];
  guide_rviz.point.z = guide_point[2];
  rviz_guide_pub.publish(guide_rviz);

  time_planning_ = t_planning.count();
  time_optimize_ = t_optimize.count();

  // Publish Weighted histogram
  // axis +z    ---
  //    |       ---
  //    |     axis -x   axis +y   axis +x   axis -y   axis -x
  //    |       ---
  // axis -z    ---
  cv_bridge::CvImage out_msg;
  out_msg.header.frame_id = "base_link";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  double** Histogram_3d = histo_planning_->get_Histogram_3d();
  int Hcnt = histo_planning_->get_Hcnt();
  int Vcnt = histo_planning_->get_Vcnt();
  if(Vcnt == 1){
    Vcnt = 10;
    cv::Mat image(Vcnt,Hcnt,CV_32F);
    for(int j=0; j<Hcnt; j++){
      float val = (float)Histogram_3d[0][(Hcnt-j+Hcnt/2-1)%Hcnt];
      for(int i=0; i<Vcnt; i++){
        ((float*)image.data)[i*Hcnt+j] = val;
      }
    }
    out_msg.image    = image;
  }else{
    cv::Mat image(Vcnt,Hcnt,CV_32F);
    for(int i=0; i<Vcnt; i++)
      for(int j=0; j<Hcnt; j++){
        ((float*)image.data)[i*Hcnt+j] = (float)Histogram_3d[Vcnt-1-i][(Hcnt-j+Hcnt/2-1)%Hcnt];
      }
    out_msg.image    = image;
  } 

  his_img_pub.publish(out_msg.toImageMsg());

  static int exec_num=0;
  if(CNNLogEnable)
  { 
    try
    {
      csv_writer(start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_, control_pts, exec_num);
    }
    catch (const std::exception& ex)
    {
      cout<<"exception was thrown: " << ex.what() << endl;
    }
	  exec_num++;
  }

  return true;
}

// save log
void Histo_Planner::csv_writer(Eigen::Vector3d start_pos_, Eigen::Vector3d start_vel_, Eigen::Vector3d start_acc_, Eigen::Vector3d end_pos_, Eigen::Vector3d end_vel_, Eigen::Vector3d end_acc_, Eigen::MatrixXd control_pts, int index)
{
  string path_ = "/home/sique/src/Prometheus/Modules/planning/vfhb-planner/dataLog/forCNNTraining/example_"+std::to_string(index)+".csv";
  std::ofstream csv(path_);
  cout << "output data path: "+path_ << endl;
    
  int Hcnt = histo_planning_->get_Hcnt();
  int Vcnt = histo_planning_->get_Vcnt();
  csv << Hcnt << ";";
  csv << Vcnt << ";";
  csv << start_pos_ .transpose() << ";";
  csv << start_vel_ .transpose() << ";";
  csv << start_acc_ .transpose() << ";";
  csv << end_pos_ .transpose() << ";";
  csv << end_vel_ .transpose() << ";";
  csv << end_acc_ .transpose() << ";";
  double** Histogram;
  Histogram = histo_planning_->get_Obstacle_3d();
  for(int i = 0; i < Vcnt; ++i){
    for(int j = 0; j < Hcnt; ++j){
      csv << Histogram[i][j] << " ";
    }
  }
  csv  << ";";
  csv << control_pts.rows() << ";";
  for(int i = 0; i < control_pts.rows(); ++i){
    csv << control_pts.row(i) << " ";
  }
  cout << "finish output" << endl;
  csv.close();
}

// [Core] Safety checking
int Histo_Planner::safetyCheck()
{
  /* ---------- check current position safety ---------- */
  Eigen::Vector3d pos = traj_pos_.evaluateDeBoor(traj_time_after_(0.001));
  double cur_dis = histo_planning_->getDist(pos,closest_obs);
  if(closest_obs[0] != pos[0] && closest_obs[1] != pos[1]){
    closest_rviz.header.seq++;
    closest_rviz.header.stamp = ros::Time::now();
    closest_rviz.header.frame_id = "world";
    closest_rviz.point.x = closest_obs[0];
    closest_rviz.point.y = closest_obs[1];
    closest_rviz.point.z = closest_obs[2];
    rviz_closest_pub.publish(closest_rviz);
  }
  if (cur_dis < forbidden_range+min(0.1,0.2*safe_distance))
  {
    goal_ready = false;
    cout << "[safetyCheck]: current position in collision. cur_dis= "<<cur_dis<<" forbidden_range= "<<forbidden_range << endl;
    pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME,  "[safetyCheck]: current position in collision. ");
    return 3;
  }
  // }
  /* ---------- check front section of trajectory safety ---------- */
  if (checkTrajCollision(0.02, 0.5,forbidden_plus_safe_distance-min(0.2,0.4*safe_distance),0.02,0.05))
  {
    cout << "[safetyCheck]: front section of trajectory in collision. " << endl;
    pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME,  "[safetyCheck]: front section of trajectory in collision. ");
    return 2;
  }

  /* ---------- check goal safety ---------- */
  bool is_found_feasible_goal = false;
  bool check_goal_safty = (cur_pos-goal_pos).norm() < sensor_max_range;
  // Option 1: Move the original goal in the direction of the descent gradient of the obstacle map!
  // [Not recommended]: Due to the use of local maps, the gradient direction is not exactly the true principle obstacle direction.
  if(check_goal_safty && goal_regenerate_mode==1){
    Eigen::Vector3d grad;
    double dist;
		
    dist = histo_planning_->getDistWithGrad(goal_pos,grad);
			
    if (dist < forbidden_plus_safe_distance)
    {
      pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "[safetyCheck]: goal dangerous\n");
      const double dr = (forbidden_plus_safe_distance)/2;
      Eigen::Vector3d new_pos;
      for (int r = 1; r <= 6; r++)
      {
        new_pos = goal_pos + r*dr*grad;
        cout << "new_pos: [" << new_pos(0) << "," << new_pos(1) << "," << new_pos(2) << "]" << endl;

        dist = histo_planning_->getDist(new_pos,closest_obs);

        if (dist > forbidden_plus_safe_distance + 0.1)
        {
          // geofence
          new_pos[0] = max(geo_fence_x_min+0.2,min(geo_fence_x_max-0.2,new_pos[0]));
          new_pos[1] = max(geo_fence_y_min+0.2,min(geo_fence_y_max-0.2,new_pos[1]));
          new_pos[2] = max(geo_fence_z_min+0.2,min(geo_fence_z_max-0.2,new_pos[2]));
  
          /* reset goal_pos */
          goal_pos = new_pos;
          goal_vel.setZero();
          goal_acc.setZero();
          is_found_feasible_goal = true;

          // cout << "change goal, replan." << endl;
          pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "change goal. ");
          cout << "[safetyCheck]: change goal at : [" << goal_pos(0) << ", " << goal_pos(1) << ", " << goal_pos(2) << "], dist= " << dist << endl;

          break;
        }
      }

      if(is_found_feasible_goal){
        goal_ready = true;
        /* pub new goal */
        // geometry_msgs::PoseStamped new_goal_;
        // new_goal_.header.seq = goalPoints_seq_id++;
        // new_goal_.header.stamp = ros::Time::now();
        // new_goal_.header.frame_id = "world";
        // new_goal_.pose.position.x = goal_pos(0);
        // new_goal_.pose.position.y = goal_pos(1);
        // new_goal_.pose.position.z = goal_pos(2);
        // goal_pub.publish(new_goal_);
        return 4;
      }else{
        goal_ready = false;
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[safetyCheck]: cannot find a new goal. ");
        return 0;
      }
    }
  }
  // Option 2: Searching within the cylindrical space centred on the original goal, first searching for positions closer to cur_pos_ref and the goal. 
  // [Not recommended]: For voluminous obstacles, it is possible that the point found falls inside obs's body
  else if(check_goal_safty && goal_regenerate_mode==2){
    double dist;

    dist = histo_planning_->getDist(goal_pos,closest_obs);
    
    if (dist < forbidden_plus_safe_distance)
    {
      pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "[safetyCheck]: goal dangerous\n");

      /* try to find a max distance goal around */
      const double dr = (forbidden_plus_safe_distance)/2, dtheta = M_PI/12, dz = (forbidden_plus_safe_distance)/2;
      double new_x, new_y, new_z;
      Eigen::Vector3d goal;
      Eigen::Vector3d goal_cur = cur_pos_ref - goal_pos;
      double theta_init = sign(goal_cur(1)) * acos(goal_cur(0) / Eigen::Vector3d(goal_cur[0], goal_cur[1], 0.0).norm());
      for (int theta = 0; theta < 13; theta++)
      {
        for (int nz = 0; nz <= 4; nz++)
        {
          for (int r = 1; r <= 4; r++)
          {
            new_x = goal_pos(0) + r*dr * cos(((pow(-1,theta)*int((theta+1)/2))*dtheta+theta_init));
            new_y = goal_pos(1) + r*dr * sin(((pow(-1,theta)*int((theta+1)/2))*dtheta+theta_init));
            new_z = goal_pos(2) + (pow(-1,nz)*int((nz+1)/2))*dz;
						
            dist = histo_planning_->getDist(Eigen::Vector3d(new_x, new_y, new_z),closest_obs);
						
            if (dist > forbidden_plus_safe_distance + 0.1)
            {
              // geofence
              new_x = max(geo_fence_x_min+0.2,min(geo_fence_x_max-0.2,new_x));
              new_y = max(geo_fence_y_min+0.2,min(geo_fence_y_max-0.2,new_y));
              new_z = max(geo_fence_z_min+0.2,min(geo_fence_z_max-0.2,new_z));
              /* reset goal_pos */
              goal(0) = new_x;
              goal(1) = new_y;
              goal(2) = new_z;
              is_found_feasible_goal = true;
              goal_pos = goal;
              goal_vel.setZero();
              goal_acc.setZero();

              cout << "[safetyCheck]: change goal at : [" << goal_pos(0) << ", " << goal_pos(1) << ", " << goal_pos(2) << "], dist= " << dist << endl;

              break;
            }
          }
          if(is_found_feasible_goal) break;
        }
        if(is_found_feasible_goal) break;
      }

      if (is_found_feasible_goal)
      {
        goal_ready = true;

        /* pub new goal */
        // geometry_msgs::PoseStamped new_goal_;
        // new_goal_.header.seq = goalPoints_seq_id++;
        // new_goal_.header.stamp = ros::Time::now();
        // new_goal_.header.frame_id = "world";
        // new_goal_.pose.position.x = new_x;
        // new_goal_.pose.position.y = new_y;
        // new_goal_.pose.position.z = new_z;
        // goal_pub.publish(new_goal_);
        return 4;
      }else{
        goal_ready = false;
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[safetyCheck]: cannot find a new goal. ");
        return 0;
      }
    }
  }
  // Option 3: Back to the feasible position along the trajectory
  else if(check_goal_safty && goal_regenerate_mode==3){	
    double dist;

    dist = histo_planning_->getDist(goal_pos,closest_obs);
      
    if (dist < forbidden_plus_safe_distance)
    {
      for(double t=0.1; t<traj_duration_; t+=0.1)
      {
        Eigen::Vector3d new_pos = traj_pos_.evaluateDeBoor(t_end_-t);

        dist = histo_planning_->getDist(new_pos,closest_obs);

        if (dist > forbidden_plus_safe_distance + 0.1)
        {
          // geofence
          new_pos[0] = max(geo_fence_x_min+0.2,min(geo_fence_x_max-0.2,new_pos[0]));
          new_pos[1] = max(geo_fence_y_min+0.2,min(geo_fence_y_max-0.2,new_pos[1]));
          new_pos[2] = max(geo_fence_z_min+0.2,min(geo_fence_z_max-0.2,new_pos[2]));
  
          /* reset goal_pos */
          goal_pos = new_pos;
          goal_vel.setZero();
          goal_acc.setZero();
          is_found_feasible_goal = true;

          cout << "[safetyCheck]: change goal at : [" << goal_pos(0) << ", " << goal_pos(1) << ", " << goal_pos(2) << "], dist= " << dist << endl;

          break;
        }
      }
			
      if(is_found_feasible_goal){
        goal_ready = true;
        /* pub new goal */
        // geometry_msgs::PoseStamped new_goal_;
        // new_goal_.header.seq = goalPoints_seq_id++;
        // new_goal_.header.stamp = ros::Time::now();
        // new_goal_.header.frame_id = "world";
        // new_goal_.pose.position.x = goal_pos(0);
        // new_goal_.pose.position.y = goal_pos(1);
        // new_goal_.pose.position.z = goal_pos(2);
        // goal_pub.publish(new_goal_);
        return 4;
      }else{
        goal_ready = false;
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[safetyCheck]: cannot find a new goal. ");
        return 0;
      }
    }
  }
	
  /* ---------- check trajectory safety ---------- */
  if (checkTrajCollision(0.5, traj_duration_*0.5,forbidden_plus_safe_distance-min(0.1,0.2*safe_distance),0.05,0.1))
  {
    cout << "[safetyCheck]: middle section of traj in collision. " << endl;
    pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME,  "[safetyCheck]: middle section of traj in collision. ");
    return 1;
  }
  return -1;
}

// Detects points on the trajectory from near to far for collision
bool Histo_Planner::checkTrajCollision(double t_start, double t_end, double safe_range, double dt_min, double dt_max, bool force)
{
  double start_check_time = force ? t_start : traj_time_after_(t_start);
  double end_check_time = force ? t_end : traj_time_after_(t_end);
  double scale = (dt_max-dt_min)/(end_check_time-start_check_time);
  double dt = dt_min;
  /* check collision for a segment of traj within specified time period */
  for (double t = start_check_time; t <= end_check_time; t += dt)
  {
    dt *= 1+scale;
    Eigen::Vector3d pos = traj_pos_.evaluateDeBoor(t);
    if((pos-cur_pos).norm() > sensor_max_range){
      return false;
    }
    double dist;

    dist = histo_planning_->getDist(pos,closest_obs);
	
    if(closest_obs[0] == pos[0] && closest_obs[1] == pos[1] && dist < safe_range){
      cout << "[Traj_Colis_Check]: pos in " << t-start_check_time+t_start << "s later could be too close to the ceil or ground, the shortest distance is " << dist << "[" << safe_range << "]" << endl;
      return true;
    }else if ((dist < safe_range)){
      cout << "[Traj_Colis_Check]: pos in " << t-start_check_time+t_start << "s later in collision, the shortest distance is " << dist << "[" << safe_range << "]" << endl;
      return true;
    }
  }
  return false;
}

// Obtain the new goal point, if manual_control_cb is working
void Histo_Planner::joy_cb(const ros::TimerEvent& e)
{
  if(!raw_goal_ready)
    return;
  raw_goal_ready = false;
  static int exec_num=0;
  float delta = (raw_goal_pos-goal_pos).norm();
  // Ignore it if moving displacement too short or time interval too short
  if(delta<0.1)
    return;
  else if(delta< 1.0 && exec_num < 10){
    exec_num++;
    return;
  }
  exec_num = 0;
  goal_ready = true;

  // Got the new goal
  if(planner_enable)
  {
    if(int(rc_x) == -1 && int(rc_y) == -1 && int(rc_z) == -1 && int(rc_r) == -1)
    {
      path_ok = false;
      goal_ready = false;
      changeExecState(LANDING, "Goal Operator");
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Land");
    }
    else if(exec_state == EXEC_STATE::WAIT_GOAL)
    {
      goal_pos = raw_goal_pos;
      goal_vel.setZero();
      goal_acc.setZero();
      
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");
      cout << "[Joystick]: Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;
    }
    else if(exec_state == EXEC_STATE::EXEC_TRAJ)
    {
      goal_pos = raw_goal_pos;
      goal_vel.setZero();
      goal_acc.setZero();
	
      // replan, otherwise stop in 1s.
      traj_duration_ = min((ros::Time::now() - time_traj_start).toSec() + 1.0, traj_duration_);
      changeExecState(REPLAN_TRAJ, "Goal Operator");
      
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");
      cout << "[Joystick]: Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;
    }
  }
  else if(exec_state == EXEC_STATE::LANDING && _DroneState.position[2]<ceil_height && _DroneState.position[2]>max_ground_height)
  {
    goal_pos = raw_goal_pos;
    goal_vel.setZero();
    goal_acc.setZero();
	
    planner_enable = true;
    changeExecState(GEN_NEW_TRAJ, "Goal Operator");
    stop_pos = cur_pos;
    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME,"Get a new goal point");
    cout << "[Joystick]: Get a new goal point:"<< goal_pos(0) << " [m] "  << goal_pos(1) << " [m] "  << goal_pos(2)<< " [m] "   <<endl;
  }
}

// [Core] Planner Manager
void Histo_Planner::mission_cb(const ros::TimerEvent& e)
{
  static long int exec_num=0;
  static int num_gen_traj = 0;
  exec_num++;
  if(exec_num == 500)
  {
    pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "ExecState: " + state_str[int(exec_state)]);
    exec_num = 0;
  }

  // security check
  if(!odom_ready || !drone_ready || !planner_enable)
  {
    // 5Hz
    if(exec_num % 100 == 0)
    {
      string message;
      if(!planner_enable){
        message = "Planner is disable!";
      }else if(!odom_ready){
        message = "Odom is not ready!";
      }else if(!drone_ready){
        message = "Drone is not ready!";
      }
      pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, message);
    }
    return;
  }
 
  // Task execution
  switch (exec_state)
  {
    case WAIT_GOAL:
    {
      num_gen_traj = 0;
      if (goal_ready)
        changeExecState(GEN_NEW_TRAJ, "Planner");
      else
        if(exec_num % 500 == 0)
          pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "Waiting for a new goal.");
    break;
    }
    case GEN_NEW_TRAJ:
    {
      if(is_generating) {break;}
      if ( generateTrajectory(cur_pos_ref, cur_vel_ref, cur_acc_ref, goal_pos, goal_vel, goal_acc) ){
        is_generating = false;
        // start time tracking
        flag_tracking = 1; // start after having oriented in line with traj
        if(spinning_once_first){
          flag_tracking = 2; // start after spining one turn
        }

        // knot instance start and end , total duration of track
        traj_pos_.getTimeSpan(t_start_, t_end_);
        traj_duration_ = t_end_ - t_start_;

        static uint cnt = 0;
        if(!escape_mode && checkTrajCollision(0.02, traj_duration_*0.5,forbidden_plus_safe_distance-min(0.1,0.2*safe_distance),0.05,0.1))
        {
          cnt++;
          if(cnt==5){
            escape_mode = true;cout<<"======escape_mode======="<<endl;
            cnt = 0;
          }
          break;
        }

        /* visulization */
        visualization_->drawBspline(traj_init_, 0.1, Eigen::Vector4d(1, 0, 0, 1), false, 0.12, Eigen::Vector4d(0, 0, 0, 1));  // red bspline; without black ctr_pts
        visualization_->drawBspline(traj_pos_, 0.1, Eigen::Vector4d(1, 1, 0, 1), true, 0.12, Eigen::Vector4d(0, 1, 0, 1), 0, 0, true);   // yellow bspline; with green ctr_pts

        last_mission_exec_time = ros::Time::now();
        changeExecState(EXEC_TRAJ, "Planner");
        num_gen_traj = 0;
        path_ok = true;
      }else{
        path_ok = false;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "generate new traj fail.\n");
        num_gen_traj++;
        // If planning fails three times consecutively, switch to WAIT_GOAL Task, otherwise continue planning
        if (num_gen_traj >= 3){
          stop_pos = cur_pos_ref;
          yaw_ref_comd = cur_yaw;
          pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "fail to replan, wait a new goal.\n");
          changeExecState(WAIT_GOAL, "Planner");
        }
      }
      goal_ready = false;
    break;
    }
    case EXEC_TRAJ:
    {
      // Waiting until the preparation is completed, e.g. spin once around and camera heading alignment.
      // if(flag_tracking>0) {
      //   if(checkTrajCollision(t_start_+0.02, t_start_+min(traj_duration_,max(0.3,traj_duration_*0.5)),forbidden_plus_safe_distance-min(0.1,0.2*safe_distance),0.02,0.1,true)){
      //     cout << "[EXEC]: Obs detected during the first collision check" << endl;
      //     num_gen_traj++;
      //     path_ok = false;
      //     if(num_gen_traj++ >= 3){
          
      //     }
      //     // While waiting, trajectories are found to have the potential for collision and then replanning
      //     pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "Obs detected during the first collision check, replan\n");
      //     changeExecState(REPLAN_TRAJ, "SAFETY");
      //   }
      //   break;
      // }

      num_gen_traj = 0;
      /* determine if need to replan */
      //  std::chrono::time_point<std::chrono::system_clock> tic, toc; 
      //  tic = std::chrono::system_clock::now(); 
      //  cout << "[safetyCheck]: time: ";
      int trigger = safetyCheck();
      //  toc = std::chrono::system_clock::now();
      //  std::chrono::duration<double, std::milli> delta = toc - tic;
      //  cout << delta.count() << "[ms]" << endl;
      double time_out = 100.0;
      float _traj_tmp_cur_ = traj_time_after_(0.001);
      if (trigger == 0){ // end position is unreachable, so stop in 0.5s and wait for a new goal
        time_out = 0.1;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "goal unreachable, wait for a new goal");
        changeExecState(WAIT_GOAL, "SAFETY");
        yaw_ref_comd = cur_yaw;
        stop_pos = cur_pos_ref;
        path_ok = false;
      }
      else if(trigger == 1){ // While tracking, trajectories are found to have the potential for collision and then replanning
        time_out = 1.0;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "obs detected on road, replan");
        changeExecState(REPLAN_TRAJ, "SAFETY");
      }
      else if (trigger == 2){ // There is a collision on the track near the current position, so stop immediately and regenerate trajectory
        time_out = 0.1;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "DANGEROUS! stop immediately and regenerate a new traj");
        changeExecState(GEN_NEW_TRAJ, "SAFETY");
        yaw_ref_comd = cur_yaw;
        stop_pos = cur_pos_ref;
        path_ok = false;
      }
      else if (trigger == 3){ // The current position is dangerous, so stop immediately and wait for a new goal
        time_out = 0.0;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "DAMN IT! Collided!!! wait a new goal");
        goal_ready = false;
        changeExecState(WAIT_GOAL, "SAFETY");
        yaw_ref_comd = cur_yaw;
        stop_pos = cur_pos_ref;
        path_ok = false;
      }
      else if(trigger == 4){ // While tracking, trajectories are found to have the potential for collision and then replanning
        time_out = 1.0;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "change goal, replan");
        changeExecState(REPLAN_TRAJ, "SAFETY");
      }
      else if((ros::Time::now()-last_mission_exec_time).toSec() > time_to_replan && (goal_pos-cur_pos).norm() > sensor_max_range){ // It's time to replan, otherwise stop in 1s
        time_out = 1.0;
        cout << "[Planner]: time to REPLAN." << endl;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "It's time to replan");
        changeExecState(REPLAN_TRAJ, "Planner");
      }
      else if(false && (cur_pos-traj_pos_.evaluateDeBoor(_traj_tmp_cur_)).norm()>0.5) // tracking reference lost, re-track a closer reference on traj
      {
        /* ---------- check tracking error ---------- */
        float _traj_tmp_ = 0.0;
        float _smp_tmp_init_ = max(t_start_+0.001,_traj_tmp_cur_-2.0);
        float sam_tmp = 0.0;
        float _dis_min_ = numeric_limits<float>::max();
        int idx = 0;
        while(true) {
          sam_tmp = _smp_tmp_init_ + idx*0.4;
          if(sam_tmp-_traj_tmp_cur_>2.0 || sam_tmp>t_end_) break;
          float _dis_ = (cur_pos-traj_pos_.evaluateDeBoor(sam_tmp)).norm();
          if(_dis_ < _dis_min_) {
            _dis_min_ = _dis_;
            _traj_tmp_ = sam_tmp;
          }
          idx++;
        }
        time_traj_start = ros::Time::now() - ros::Duration(_traj_tmp_-t_start_+0.3);
      }
			
      traj_duration_ = min((ros::Time::now() - time_traj_start).toSec() + time_out, traj_duration_); // when stop
    break;
    }
    case REPLAN_TRAJ:
    {
      if(is_generating) {break;}
			
      Eigen::Vector3d start_pos_, start_vel_, start_acc_;
      // if(flag_tracking>0){// || cur_vel.norm() < min_vel_default){
        start_pos_ = cur_pos_ref;
        start_vel_ = cur_vel_ref;
        start_acc_ = cur_acc_ref;
      // }else{
      //   double time_replan_ = traj_time_after_(0.0);
      //   start_pos_ = traj_pos_.evaluateDeBoor(time_replan_);
      //   start_vel_ = traj_vel_.evaluateDeBoor(time_replan_);
      //   start_acc_ = traj_acc_.evaluateDeBoor(time_replan_);
      //   // if((cur_pos-start_pos_).norm()>0.5){
      //   //   start_pos_ = cur_pos;
      //   //   start_vel_ = cur_vel;
      //   //   start_acc_ = cur_acc;
      //   // }
      // }

      if ( generateTrajectory(start_pos_, start_vel_, start_acc_, goal_pos, goal_vel, goal_acc) ){
        is_generating = false;
        // knot instance start and end , total duration of track
        traj_pos_.getTimeSpan(t_start_, t_end_);
        traj_duration_ = t_end_ - t_start_;

        // start time tracking
        if(flag_tracking == 0){
          Eigen::Vector3d pos_ref;
          if(start_vel_.norm()>min_vel_default){
            pos_ref = traj_pos_.evaluateDeBoor(t_start_ + min(0.1,traj_duration_));
          }else{
            pos_ref = traj_pos_.evaluateDeBoor(t_start_ + min(time_forward_facing_toward,traj_duration_));
          }
          pos_ref[0] -= start_pos_[0];
          pos_ref[1] -= start_pos_[1];
          pos_ref[2] = 0.0;
          stop_pos = start_pos_;
          double dyaw = fabs(sign(pos_ref[1]) * acos(pos_ref[0] / pos_ref.norm())-cur_yaw);
          double yaw_err_th = max(0.05,10*delta_yaw);
          if(dyaw<yaw_err_th || 2*M_PI-dyaw<yaw_err_th)
            time_traj_start = ros::Time::now();
          else
            flag_tracking = 1;
        }

        /* visulization */
        visualization_->drawBspline(traj_init_, 0.1, Eigen::Vector4d(1, 0, 0, 1), false, 0.2, Eigen::Vector4d(0, 0, 0, 1));  // red bspline; with black ctr_pts
        visualization_->drawBspline(traj_pos_, 0.1, Eigen::Vector4d(1, 1, 0, 1), true, 0.12, Eigen::Vector4d(0, 1, 0, 1), 0, 0, true);   // yellow bspline; with green ctr_pts

        last_mission_exec_time = ros::Time::now();
        changeExecState(EXEC_TRAJ, "Planner");
        num_gen_traj = 0;
        path_ok = true;
      }else{
        path_ok = false;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "generate new traj fail.\n");
        num_gen_traj++;
        // If planning fails three times consecutively, switch to WAIT_GOAL Task, otherwise continue planning
        if (num_gen_traj >= 3){
          stop_pos = cur_pos_ref;
          yaw_ref_comd = cur_yaw;
          pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "fail to replan, wait a new goal.\n");
          changeExecState(WAIT_GOAL, "Planner");
        }
      }
      goal_ready = false;
    break;
    }
    case LANDING:
    {
      pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "start to land.\n");
      planner_enable = false;
      Command_Now.header.stamp	= ros::Time::now();
      Command_Now.Mode			= drone_msgs::ControlCommand::Land;
      Command_Now.Command_ID		= Command_Now.Command_ID + 1;
      Command_Now.source			= NODE_NAME;
      command_pub.publish(Command_Now);
    break;
    }
  }
}

// Trajectory tracking, publishing reference position and orientation.
void Histo_Planner::control_cb(const ros::TimerEvent& e)
{
  if (!odom_ready || !drone_ready || !tracking_controller_enable)
    return;
		
  if (!path_ok)
  {
    Command_Now.header.stamp                        = ros::Time::now();
    Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
    Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
    Command_Now.source                              = NODE_NAME;
    Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::TRAJECTORY;
    Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
    Command_Now.Reference_State.position_ref[0]     = stop_pos[0];
    Command_Now.Reference_State.position_ref[1]     = stop_pos[1];
    Command_Now.Reference_State.position_ref[2]     = stop_pos[2];
    Command_Now.Reference_State.velocity_ref[0]     = 0.0;
    Command_Now.Reference_State.velocity_ref[1]     = 0.0;
    Command_Now.Reference_State.velocity_ref[2]     = 0.0;
    if(yaw_tracking_mode>0){
      yaw_ref_comd += delta_yaw * rc_r;
    }
    Command_Now.Reference_State.yaw_ref             = yaw_ref_comd;
    command_pub.publish(Command_Now);
    cur_pos_ref = stop_pos;
    cur_vel_ref.setZero();
    cur_acc_ref.setZero();
    return;
  }

  // goal reached?
  if(Eigen::Vector3d(cur_pos - goal_pos).norm() < range_near_end)
  {
    yaw_ref_comd = cur_yaw;
    Command_Now.header.stamp                        = ros::Time::now();
    Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
    Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
    Command_Now.source                              = NODE_NAME;
    Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::TRAJECTORY;
    Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
    Command_Now.Reference_State.position_ref[0]     = goal_pos[0];
    Command_Now.Reference_State.position_ref[1]     = goal_pos[1];
    Command_Now.Reference_State.position_ref[2]     = goal_pos[2];
    Command_Now.Reference_State.velocity_ref[0]     = 0.0;
    Command_Now.Reference_State.velocity_ref[1]     = 0.0;
    Command_Now.Reference_State.velocity_ref[2]     = 0.0;
    Command_Now.Reference_State.yaw_ref             = yaw_ref_comd;
    command_pub.publish(Command_Now);
        
    if (!tracking_controller_enable)
      pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "Reach the goal! The planner will be disable automatically.");
    else
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Reach the goal! The planner is still enable.");
    cout << "Reach the goal" << endl;
        
    // yes, stop there
    path_ok = false;
    stop_pos = goal_pos;
    cur_pos_ref = stop_pos;
    cur_vel_ref.setZero();
    cur_acc_ref.setZero();
    planner_enable = tracking_controller_enable;
    // wait for a new goal
    goal_ready = false;
    changeExecState(WAIT_GOAL, "Planner");
    return;
  }

  float desired_yaw;
  static double yaw_diff;
  Eigen::Vector3d pos, vel, acc;
  switch(flag_tracking)
  {
    case 2: //spinning once turn at least
    {
      static float accu_delta_yaw = 0.0;
      if(accu_delta_yaw<2*M_PI){
        accu_delta_yaw += delta_yaw;
        yaw_ref_comd += delta_yaw;
      }else{
        accu_delta_yaw = 0.0;
        flag_tracking = 1;
      }
      pos = stop_pos;
      vel.setZero();
      acc.setZero();
    break;
    }
    
    case 1: // oriented in line with the traj
    {
      Eigen::Vector3d pos_rel, pos_rel_1, pos_rel_2;
      pos_rel_1 = stop_pos;
      pos_rel_2 = traj_pos_.evaluateDeBoor(t_start_ + min(time_forward_facing_toward,traj_duration_));
      pos_rel << pos_rel_2[0]-pos_rel_1[0] , pos_rel_2[1]-pos_rel_1[1] , 0.0;
      desired_yaw = sign(pos_rel[1]) * acos(pos_rel[0] / pos_rel.norm());
      double dyaw_n = fabs(cur_yaw-desired_yaw);
      double yaw_err_th = max(0.05,10*delta_yaw);
      if (yaw_tracking_mode == 0){
          flag_tracking = 0;
          time_traj_start = ros::Time::now();
          return;
      }else if (yaw_tracking_mode == 1){
        double dyaw_n = fabs(cur_yaw-desired_yaw);
        double dyaw_s = sign(desired_yaw-cur_yaw);
        if (dyaw_n<M_PI){
          yaw_ref_comd += (pow(dyaw_n,0.3))*dyaw_s*delta_yaw*sign(yaw_rate-fabs(cur_yaw_rate));
        }else{
          dyaw_n = 2*M_PI-dyaw_n;
          yaw_ref_comd -= (pow(dyaw_n,0.3))*dyaw_s*delta_yaw*sign(yaw_rate-fabs(cur_yaw_rate));
        }
        if(dyaw_n < yaw_err_th){
          yaw_ref_comd = desired_yaw;
          flag_tracking = 0;
          time_traj_start = ros::Time::now();
          return;
        }
      }else if (yaw_tracking_mode == 2){
        yaw_ref_comd += delta_yaw * rc_r * sign(yaw_rate-fabs(cur_yaw_rate));
        if (dyaw_n < yaw_err_th || dyaw_n > 2*M_PI-yaw_err_th){
          yaw_ref_comd = desired_yaw;
          flag_tracking = 0;
          time_traj_start = ros::Time::now();
          return;
        }
      }
      yaw_diff = dyaw_n;
      pos = pos_rel_1;
      vel.setZero();
      acc.setZero();
    break;
    }
    
    case 0: // tracking
    {
      //if(exec_state != EXEC_STATE::EXEC_TRAJ) break;
      double t_cur = max(0.0,(ros::Time::now() - time_traj_start).toSec());

      // Yaw auto-hold
      if (yaw_tracking_mode > 0)  // == 1)
      {
        Eigen::Vector3d pos_rel, pos_rel_1, pos_rel_2;
        pos_rel_1 = traj_pos_.evaluateDeBoor(t_start_ + min(t_cur,max(0.0,traj_duration_-time_forward_facing_toward)));
        pos_rel_2 = traj_pos_.evaluateDeBoor(t_start_ + min(t_cur+time_forward_facing_toward,traj_duration_));

        if(Eigen::Vector3d(pos_rel_2 - goal_pos).norm() > forbidden_range){
          pos_rel << pos_rel_2[0]-pos_rel_1[0] , pos_rel_2[1]-pos_rel_1[1] , 0.0;
        }else{
          pos_rel << goal_pos[0]-pos_rel_1[0] , goal_pos[1]-pos_rel_1[1] , 0.0;
        }
        desired_yaw = sign(pos_rel[1]) * acos(pos_rel[0] / pos_rel.norm());
        double dyaw_n = fabs(cur_yaw-desired_yaw);
        double dyaw_s = sign(desired_yaw-cur_yaw);
        if (dyaw_n<M_PI){
          yaw_ref_comd += (pow(dyaw_n,0.3))*dyaw_s*delta_yaw*sign(yaw_rate-fabs(cur_yaw_rate));
        }else{
          dyaw_n = 2*M_PI-dyaw_n;
          yaw_ref_comd -= (pow(dyaw_n,0.3))*dyaw_s*delta_yaw*sign(yaw_rate-fabs(cur_yaw_rate));
        }
        yaw_diff = 0.2*yaw_diff + 0.8*dyaw_n;
        if(yaw_diff < 0.5*delta_yaw){
          yaw_ref_comd = desired_yaw;
        }

        static int count=0;
        if(yaw_diff > yaw_tracking_err_max && cur_vel.norm() > min_vel_default){
          count++;
          if(count > 10){
            cout << "[Tracker]: Heading tracking lost. delta_yaw = "<<dyaw_n*57.29<<" deg" << endl;
            traj_duration_ = min((ros::Time::now() - time_traj_start).toSec() + 0.1, traj_duration_);
            changeExecState(GEN_NEW_TRAJ, "Planner");
            stop_pos = cur_pos;
            yaw_ref_comd = cur_yaw;
            count = 0;
          }else if(count == 5){
            cout << "[Tracker]: Heading tracking lost. delta_yaw = "<<dyaw_n*57.29<<" deg" << endl;
            traj_duration_ = min((ros::Time::now() - time_traj_start).toSec() + 1.0, traj_duration_);
            changeExecState(REPLAN_TRAJ, "Planner");
          }
        }else{
          count = 0;
        }
      }
      // Yaw manual-hold
      else if (yaw_tracking_mode == 2)
        yaw_ref_comd += delta_yaw * rc_r;
      // No yaw hold
      else
        yaw_ref_comd = 0.0;

      if (t_cur < traj_duration_) {
        pos = traj_pos_.evaluateDeBoor(t_start_ + t_cur);
        if(escape_mode && t_cur > 0.8*traj_duration_){
          escape_mode = false;
          changeExecState(REPLAN_TRAJ, "Planner");
        }
        vel = traj_vel_.evaluateDeBoor(t_start_ + t_cur);
        acc = traj_acc_.evaluateDeBoor(t_start_ + t_cur);
      } else if (t_cur >= traj_duration_) {
        /* hover when timeout */
        pos = traj_pos_.evaluateDeBoor(t_start_ + traj_duration_);
        stop_pos = pos;
        yaw_ref_comd = cur_yaw;
        vel.setZero();
        acc.setZero();
        path_ok = false;
        goal_ready = false;
        if(exec_state == EXEC_STATE::EXEC_TRAJ) changeExecState(WAIT_GOAL, "SAFETY");
      }
      //  double vel_norm = vel.norm();
      //  Eigen::Vector3d vel_u = vel/vel_norm;
      //  cout << "t_cur: " << t_cur << "(" <<  traj_duration_ << "),pos=" << pos.transpose() << "(" << cur_pos.transpose() <<"),vel= " << vel_norm << "(" << cur_vel.norm() << "),acc_t= " << acc.dot(vel_u) << ",acc_n= " << vel_u.cross(acc.cross(vel_u)).norm() << endl;
    break;
    }
  }

  if(fabs(yaw_ref_comd)>6.3)
    yaw_ref_comd = 0.0;
  else if(yaw_ref_comd>M_PI)
    yaw_ref_comd -= 2*M_PI;
  else if (yaw_ref_comd<-M_PI)
    yaw_ref_comd += 2*M_PI;

	
  // Out of range?
  if(pos[0] < geo_fence_x_min || pos[0] > geo_fence_x_max ||
    pos[1] < geo_fence_y_min || pos[1] > geo_fence_y_max || 
    pos[2] < geo_fence_z_min || pos[2] > geo_fence_z_max)
  {
    stop_pos[0] = max(geo_fence_x_min+0.05,min(geo_fence_x_max-0.05,cur_pos[0]));
    stop_pos[1] = max(geo_fence_y_min+0.05,min(geo_fence_y_max-0.05,cur_pos[1]));
    stop_pos[2] = max(geo_fence_z_min+0.05,min(geo_fence_z_max-0.05,cur_pos[2]));
    yaw_ref_comd = cur_yaw;
    Command_Now.header.stamp                        = ros::Time::now();
    Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
    Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
    Command_Now.source                              = NODE_NAME;
    Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::TRAJECTORY;
    Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
    Command_Now.Reference_State.position_ref[0]     = stop_pos[0];
    Command_Now.Reference_State.position_ref[1]     = stop_pos[1];
    Command_Now.Reference_State.position_ref[2]     = stop_pos[2];
    Command_Now.Reference_State.velocity_ref[0]     = 0.0;
    Command_Now.Reference_State.velocity_ref[1]     = 0.0;
    Command_Now.Reference_State.velocity_ref[2]     = 0.0;
    Command_Now.Reference_State.yaw_ref             = yaw_ref_comd;
    command_pub.publish(Command_Now);
        
    if (!tracking_controller_enable)
      pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "Out of geofence! The planner will be disable automatically.");
    else
      pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Out of geofence! The planner is still enable.");
        
    // yes, stop there
    cur_pos_ref = stop_pos;
    cur_vel_ref.setZero();
    cur_acc_ref.setZero();
    path_ok = false;
    planner_enable = tracking_controller_enable;
    // wait for a new goal
    goal_ready = false;
    changeExecState(WAIT_GOAL, "Planner");
    return;
  }

  Command_Now.header.stamp                        = ros::Time::now();
  Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
  Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
  Command_Now.source                              = NODE_NAME;
  Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::TRAJECTORY;
  Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
  Command_Now.Reference_State.position_ref[0]     = pos[0];
  Command_Now.Reference_State.position_ref[1]     = pos[1];
  Command_Now.Reference_State.position_ref[2]     = pos[2];
  Command_Now.Reference_State.velocity_ref[0]     = vel[0];
  Command_Now.Reference_State.velocity_ref[1]     = vel[1];
  Command_Now.Reference_State.velocity_ref[2]     = vel[2];

  Command_Now.Reference_State.yaw_ref             = yaw_ref_comd;
  command_pub.publish(Command_Now);
  cur_pos_ref = pos;
  cur_vel_ref = vel;
  cur_acc_ref = acc;
}

}
