#include <string> 
#include <planning_fsm.h>
// #define DEBUG
#define NODE_NAME "fast_planner"

namespace dyn_planner
{
void PlanningFSM::init(ros::NodeHandle& nh)
{
  /* ---------- init global param---------- */
  nh.param("bspline/limit_vel", NonUniformBspline::limit_vel_, -1.0);
  nh.param("bspline/limit_acc", NonUniformBspline::limit_acc_, -1.0);
  nh.param("bspline/limit_ratio", NonUniformBspline::limit_ratio_, -1.0);

  /* ---------- fsm param ---------- */
  nh.param("fsm/algorithm_mode", algorithm_mode, 1); // 0: A*; 1:kino_A*
  nh.param("fsm/flight_type", flight_type_, -1);
  nh.param("fsm/thresh_replan", thresh_replan_, -1.0);
  nh.param("fsm/thresh_no_replan", thresh_no_replan_, -1.0);
  nh.param("fsm/safety_distance", safety_distance, 0.01);
  nh.param("fsm/wp_num", wp_num_, -1);

  for (int i = 0; i < wp_num_; i++)
  {
    nh.param("fsm/wp" + to_string(i) + "_x", waypoints_[i][0], -1.0);
    nh.param("fsm/wp" + to_string(i) + "_y", waypoints_[i][1], -1.0);
    nh.param("fsm/wp" + to_string(i) + "_z", waypoints_[i][2], -1.0);
  }

  current_wp_ = 0;
  exec_state_ = EXEC_STATE::INIT;
  have_goal_ = false;

  /* ---------- init edt environment ---------- */
  // now we use global map
  nh.param("sdf_map/SDF_MODE", sdf_mode, 0);
  // ROS_INFO("sdf_mode: %s", sdf_mode==0? "local sdf" : "global sdf");

  if(sdf_mode==0){
      sdf_map_.reset(new SDFMap);
      sdf_map_->init(nh);

      edt_env_.reset(new EDTEnvironment);
      edt_env_->setMap(sdf_map_);
  } else{
      sdf_map_global.reset(new SDFMap_Global);
      sdf_map_global->init(nh);

      edt_env_.reset(new EDTEnvironment);
      edt_env_->setMap(sdf_map_global);
  }


  /* ---------- init path finder and optimizer ---------- */
    if(algorithm_mode==0){ // Not support now, TBD
        path_finder0_.reset(new Astar);
        path_finder0_->setParam(nh);
        path_finder0_->setEnvironment(edt_env_);
        path_finder0_->init();
        cout << "A_star init." << endl;
    }
    else if(algorithm_mode==1)
    {
        path_finder_.reset(new KinodynamicAstar);
        path_finder_->setParam(nh);
        path_finder_->setEnvironment(edt_env_);
        path_finder_->init();
        cout << "Kinodynamic A_star init." << endl;
    }

  // init local optimization - bspline optimizer 
  // ROS_INFO("---init bspline optimizer!---");
  bspline_optimizer_.reset(new BsplineOptimizer);
  bspline_optimizer_->setParam(nh);
  bspline_optimizer_->setEnvironment(edt_env_);

  // init planner manage 
  // ROS_INFO("---init planner manage!---");
  planner_manager_.reset(new DynPlannerManager);
  planner_manager_->setParam(nh);
  if(algorithm_mode==0){
    planner_manager_->setPathFinder0(path_finder0_);
  }
  else if(algorithm_mode==1)
  {
    planner_manager_->setPathFinder(path_finder_);
  }
  planner_manager_->setOptimizer(bspline_optimizer_);
  planner_manager_->setEnvironment(edt_env_);

  // init visualization
  // ROS_INFO("---init visualization!---");
  visualization_.reset(new PlanningVisualization(nh));

  /* ---------- callback ---------- */
  exec_timer_ = node_.createTimer(ros::Duration(0.02), &PlanningFSM::execFSMCallback, this);
  safety_timer_ = node_.createTimer(ros::Duration(0.25), &PlanningFSM::safetyCallback, this);

  /* ---------- sub -----------*/
  waypoint_sub_ = node_.subscribe("/fast_planner/goal", 1, &PlanningFSM::waypointCallback, this);

  swith_sub = node_.subscribe<std_msgs::Bool>("/fast_planner/switch", 10, &PlanningFSM::switchCallback, this);  

  /* ---------- pub -----------*/
  // new goal
  goalpoint_pub_ = node_.advertise<geometry_msgs::PoseStamped>("/fast_planner/goal", 10);
  
  // 
  replan_pub_ = node_.advertise<std_msgs::Empty>("/fast_planner/replan", 10);
  
  // emergency stop
  safety_pub_ = node_.advertise<std_msgs::Int8>("/fast_planner/stop_cmd", 10);
  
  bspline_pub_ = node_.advertise<drone_msgs::Bspline>("/fast_planner/bspline", 10);

  message_pub = node_.advertise<drone_msgs::Message>("/drone_msg/message", 10);
  
  // ROS_INFO("---planning_fsm: init finished!---");
  trigger_ = true;

}


void PlanningFSM::waypointCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if(flag_new_goal){
  	flag_new_goal = false;
  	return;
  }

 #ifdef DEBUG 
  cout << "[waypointCallback]: Triggered!" << endl;
#endif

  if (msg->pose.position.z < 0.1)  // the minimal goal height 
  {
#ifdef    DEBUG
    cout << "the goal's height is limited within [1.0, 2.0], but it is too low (hiehgt= " << msg->pose.position.z << "<0.1m) to reset" << endl;
#endif
  }
    
  double goal_z;
  waypoint_seq = msg->header.seq;

  // two mode: 1. manual setting goal from rviz; 2. preset goal in launch file.
  auto conf=[](double v, double min_v, double max_v)->double
  {
    return v<min_v? min_v:(v>max_v?max_v:v);
  };

  if (flight_type_ == FLIGHT_TYPE::MANUAL_GOAL)
  {
    goal_z = msg->pose.position.z;
    // end_pt_ << msg->pose.position.x, msg->pose.position.y, 1.0;
    // if (msg->pose.position.z < 0.3) goal_z = 0.3;
    // if (msg->pose.position.z > 3.5) goal_z = 3.5;
    goal_z = conf(msg->pose.position.z, 1.0, 2.0);
    end_pt_ << msg->pose.position.x, msg->pose.position.y, goal_z;
  }
  else if (flight_type_ == FLIGHT_TYPE::PRESET_GOAL)
  {
    end_pt_(0) = waypoints_[current_wp_][0];
    end_pt_(1) = waypoints_[current_wp_][1];
    end_pt_(2) = waypoints_[current_wp_][2];
    current_wp_ = (current_wp_ + 1) % wp_num_;
  }else if(flight_type_ == FLIGHT_TYPE::INPUT_MANUAL)
  {
    // cout << "please input waypoints_" << endl;
  }
#ifdef DEBUG
  cout << "---planning_fsm: get waypoint: [" << end_pt_(0) << end_pt_(1) << end_pt_(2) << "]!---" << endl;
#endif

  pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "Get a new goal: [ " + to_string(end_pt_(0)) + "," + to_string(end_pt_(1)) + "," + to_string(end_pt_(2)) + "]!---");

  end_vel_.setZero();
  have_goal_ = true;

  if (exec_state_ == EXEC_TRAJ)
    changeExecState(REPLAN_TRAJ, "TRIG");
}

void PlanningFSM::changeExecState(EXEC_STATE new_state, string pos_call)
{
  string state_str[5] = { "INIT", "WAIT_GOAL", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ" };
  int pre_s = int(exec_state_);
  exec_state_ = new_state;
  pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)]);
  // cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
}

void PlanningFSM::printExecState()
{
  string state_str[5] = { "INIT", "WAIT_GOAL", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ" };
  pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "ExecState: " + state_str[int(exec_state_)]);
  // cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
}

void PlanningFSM::execFSMCallback(const ros::TimerEvent& e)
{
  static int fsm_num = 0;
  static int num_gen_traj = 0;
  fsm_num++;
  if (fsm_num == 100)
  {
    printExecState();
    if (!trigger_)
      pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "haven't trigger!.\n");

    if (!edt_env_->odomValid())
      pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "no odom.\n");
   
    if (!edt_env_->mapValid())
      pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "no map\n.");

    fsm_num = 0;
  }

  switch (exec_state_)
  {
    case INIT:
    {
      if (!trigger_)
      {
        return;
      }
      if (!edt_env_->odomValid())
      {
        return;
      }
      if (!edt_env_->mapValid())
      {
        return;
      }
      changeExecState(WAIT_GOAL, "FSM");
      break;
    }

    case WAIT_GOAL:
    {
      if (!have_goal_)
        return;
      else
      {
        changeExecState(GEN_NEW_TRAJ, "FSM");
        num_gen_traj = 0;
      }
      break;
    }

    case GEN_NEW_TRAJ:
    {
      nav_msgs::Odometry odom = edt_env_->getOdom();
      start_pt_(0) = odom.pose.pose.position.x;
      start_pt_(1) = odom.pose.pose.position.y;
      start_pt_(2) = odom.pose.pose.position.z;

      start_vel_(0) = odom.twist.twist.linear.x;
      start_vel_(1) = odom.twist.twist.linear.y;
      start_vel_(2) = odom.twist.twist.linear.z;

      start_acc_.setZero();

      bool success = planSearchOpt();

      if (success)
      {
#ifdef DEBUG
      ROS_INFO("---planing_fsm: planning successful!---");
#endif

        changeExecState(EXEC_TRAJ, "FSM");
        num_gen_traj = 0;
      }
      else
      {
        num_gen_traj++;
        if (num_gen_traj >= 3){
          have_goal_ = false;
          changeExecState(WAIT_GOAL, "FSM");
//        }else{
//          changeExecState(GEN_NEW_TRAJ, "FSM");
        }
      }
      break;
    }

    case EXEC_TRAJ:
    {
      /* determine if need to replan */
      ros::Time time_now = ros::Time::now();
      double t_cur = (time_now - planner_manager_->time_traj_start_).toSec();
      //t_cur = min(planner_manager_->traj_duration_, t_cur);
      Eigen::Vector3d pos = planner_manager_->traj_pos_.evaluateDeBoor(planner_manager_->t_start_ + min(planner_manager_->traj_duration_, t_cur));

      /* && (end_pt_ - pos).norm() < 0.5 */
      if (t_cur > planner_manager_->traj_duration_)
      {
        if  ((end_pt_ - pos).norm() < 0.5)
        {
          have_goal_ = false;
          // for static debug, to commment this line
          changeExecState(WAIT_GOAL, "FSM");
          return;
        }else{
          changeExecState(REPLAN_TRAJ, "FSM");
        }
      }
      else if ((end_pt_ - pos).norm() < thresh_no_replan_)
      {
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "near end.");
        return;
      }
      else if ((planner_manager_->pos_traj_start_ - pos).norm() < thresh_replan_)
      {
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "near start");
        return;
      }
      else
      {
        // changeExecState(REPLAN_TRAJ, "FSM");
      }
      break;
    }

    case REPLAN_TRAJ:
    {
      // ros::Time time_now = ros::Time::now();
      // double t_cur = (time_now - planner_manager_->time_traj_start_).toSec();
      // start_pt_ = planner_manager_->traj_pos_.evaluateDeBoor(planner_manager_->t_start_ + t_cur);
      // start_vel_ = planner_manager_->traj_vel_.evaluateDeBoor(planner_manager_->t_start_ + t_cur);
      // start_acc_ = planner_manager_->traj_acc_.evaluateDeBoor(planner_manager_->t_start_ + t_cur);
      // cout << "t_cur: " << t_cur << endl;
      // cout << "start pt: " << start_pt_.transpose() << endl;

      nav_msgs::Odometry odom = edt_env_->getOdom();
      start_pt_(0) = odom.pose.pose.position.x;
      start_pt_(1) = odom.pose.pose.position.y;
      start_pt_(2) = odom.pose.pose.position.z;

      start_vel_(0) = odom.twist.twist.linear.x;
      start_vel_(1) = odom.twist.twist.linear.y;
      start_vel_(2) = odom.twist.twist.linear.z;
      start_acc_.setZero();

      /* inform server */
      std_msgs::Empty replan_msg;
      replan_pub_.publish(replan_msg);

      bool success = planSearchOpt();
      if (success)
      {
        changeExecState(EXEC_TRAJ, "FSM");
      }
      else
      {
        // have_goal_ = false;
        // changeExecState(WAIT_GOAL, "FSM");
        changeExecState(GEN_NEW_TRAJ, "FSM");
        num_gen_traj++;
      }
      break;
    }
  }
}

void PlanningFSM::safetyCallback(const ros::TimerEvent& e)
{
  if (!edt_env_->mapValid()){
    pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[safety callback]: no map.\n");
    return;
  }

  /* ---------- check current position safety ---------- */
  nav_msgs::Odometry odom = edt_env_->getOdom();
  Eigen::Matrix<double, 3, 1> cur_pt(
    odom.pose.pose.position.x,
    odom.pose.pose.position.y,
    odom.pose.pose.position.z
  );
  double cur_safe_dist = edt_env_->evaluateCoarseEDT(cur_pt, -1.0);
  if(cur_safe_dist< safety_distance){
    replan.data = 1; // stop!
    pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "[safetyCallback]: current position dangerous, stop and wait a new goal\n");
    changeExecState(WAIT_GOAL, "FSM");
    std_msgs::Empty emt;
    replan_pub_.publish(emt);
  }else{
    replan.data = 0;
  }
  safety_pub_.publish(replan); // useless
  
    
  /* ---------- check goal safety ---------- */
  if (have_goal_ && replan.data == 0)
  {
    double dist =
        planner_manager_->dynamic_ ?
            edt_env_->evaluateCoarseEDT(end_pt_, planner_manager_->time_start_ + planner_manager_->traj_duration_) :
            edt_env_->evaluateCoarseEDT(end_pt_, -1.0);
    // ROS_INFO("goal sdf: %f", dist);
    double margin_ = planner_manager_->margin_;
    if (dist <= margin_)
    {
      pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "[safetyCallback]: goal dangerous\n");
      
      // ROS_INFO("goal pos: [%f, %f, %f], goal sdf: %f", end_pt_(0), end_pt_(1), end_pt_(2), dist);
      /* try to find a max distance goal around */
      const double dr = margin_/2, dtheta = 30, dz = 0.3;
      double new_x, new_y, new_z, max_dist = -1.0;
      Eigen::Vector3d goal;
      bool is_found_feasible_goal = false;
      for (double r = dr; r <= 5 * dr + 1e-3; r += dr)
      {
        for (double theta = -90; theta <= 270; theta += dtheta)
        {
          for (double nz = 1 * dz; nz >= -1 * dz; nz -= dz)
          {
            new_x = end_pt_(0) + r * cos(theta / 57.3);
            new_y = end_pt_(1) + r * sin(theta / 57.3);
            new_z = end_pt_(2) + dz;
            Eigen::Vector3d new_pt(new_x, new_y, new_z);
            dist = planner_manager_->dynamic_ ?
                       edt_env_->evaluateCoarseEDT(new_pt,
                                                   planner_manager_->time_start_ + planner_manager_->traj_duration_) :
                       edt_env_->evaluateCoarseEDT(new_pt, -1.0);
            if (dist > 2*margin_)
            {
              /* reset end_pt_ */
              goal(0) = new_x;
              goal(1) = new_y;
              goal(2) = new_z;
              max_dist = dist;
              is_found_feasible_goal = true;
             
#ifdef DEBUG
              ROS_INFO("[safetyCallback]: change goal");
              ROS_INFO("goal pos: [%f, %f, %f], goal sdf: %f", goal(0), goal(1), goal(2), dist);
#endif

              // char* sp;
              // sprintf(sp, "[safetyCallback]: change goal; **goal pos: [%f, %f, %f], goal sdf: %f\n", goal(0), goal(1), goal(2), dist);
              // pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, sp);
              break;
            }
          }
          if(is_found_feasible_goal) break;
        }
        if(is_found_feasible_goal) break;
      }

      if (max_dist > 2*margin_)
      {
        // cout << "change goal, replan." << endl;
        pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "change goal, replan.\n");
        end_pt_ = goal;
        //have_goal_ = true;
        end_vel_.setZero();

        if (exec_state_ == EXEC_TRAJ)
        {
          changeExecState(REPLAN_TRAJ, "SAFETY");
        }

      	/* pub new goal */
      	flag_new_goal = true;
      	geometry_msgs::PoseStamped new_goal_;
      	new_goal_.header.seq = waypoint_seq+1;
      	new_goal_.header.stamp = ros::Time::now();
      	new_goal_.header.frame_id = "world";
      	new_goal_.pose.position.x = new_x;
      	new_goal_.pose.position.y = new_y;
      	new_goal_.pose.position.z = new_z;
      	
      	goalpoint_pub_.publish(new_goal_);
      	
      }
      else
      {
         have_goal_ = false;
         changeExecState(WAIT_GOAL, "SAFETY");
#ifdef DEBUG
              cout << "goal near collision, stop and wait a new goal" << endl;
#endif
        std_msgs::Empty emt;
        replan_pub_.publish(emt);
      }
    }
  }

  /* ---------- check trajectory ---------- */
  if (exec_state_ == EXEC_STATE::EXEC_TRAJ)
  {
    bool safe = planner_manager_->checkTrajCollision();

    if (!safe)
    {
      changeExecState(REPLAN_TRAJ, "SAFETY");
    }
  }
}

bool PlanningFSM::planSearchOpt()
{
  bool plan_success = planner_manager_->generateTrajectory(start_pt_, start_vel_, start_acc_, end_pt_, end_vel_);

  if (plan_success)
  {
    planner_manager_->retrieveTrajectory();

    /* publish traj */
    drone_msgs::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = planner_manager_->time_traj_start_;
    bspline.traj_id = planner_manager_->traj_id_;
    Eigen::MatrixXd ctrl_pts = planner_manager_->traj_pos_.getControlPoint();
    for (int i = 0; i < ctrl_pts.rows(); ++i)
    {
      Eigen::Vector3d pvt = ctrl_pts.row(i);
      geometry_msgs::Point pt;
      pt.x = pvt(0);
      pt.y = pvt(1);
      pt.z = pvt(2);
      bspline.pts.push_back(pt);
    }

    Eigen::VectorXd knots = planner_manager_->traj_pos_.getKnot();
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }
    
    bspline_pub_.publish(bspline);

    /* visulization */
    vector<Eigen::Vector3d> _path;
    if(algorithm_mode==0){
      _path = path_finder0_->getPath(); // A* path
    }
    else if(algorithm_mode==1)
    {
      _path = path_finder_->getKinoTraj(0.02);// kinodynamic A* path
    }

    visualization_->drawPath(_path, 0.1, Eigen::Vector4d(0, 0, 1, 1)); // blue
    visualization_->drawBspline(planner_manager_->traj_pos_, 0.1, Eigen::Vector4d(0.0, 0.5, 0.5, 1), true, 0.12, Eigen::Vector4d(0, 1, 0, 1), 0, 0);   // bspline; ctr_pts
    return true;
  }
  else
  {

    pub_message(message_pub, drone_msgs::Message::WARN,  NODE_NAME, "generate new traj fail.\n");
    return false;
  }
}

void PlanningFSM::switchCallback(const std_msgs::Bool::ConstPtr &msg){
    trigger_= msg->data;
}

}
