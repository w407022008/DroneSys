//头文件
#include <ros/ros.h>
#include <Eigen/Dense>
#include <iostream>

#include <math_utils.h>
#include <drone_msgs/RCInput.h>
#include "drone_msgs/DroneState.h"
#include <geometry_msgs/PoseStamped.h>
#include "sensor_msgs/Imu.h"
#include <sensor_msgs/Joy.h>
#include <tf/transform_listener.h>

using namespace std;

double drone_yaw_init, user_yaw_init = 0.0;
double user_yaw;
double geo_fence_x_min,geo_fence_x_max,geo_fence_y_min,geo_fence_y_max,geo_fence_z_min,geo_fence_z_max;
bool is_2D;
bool joy_raw;
double fly_height_2D;
double min_goal_height, ceil_height;
int control_from_joy;
double _max_goal_range_xy, _max_goal_range_z;
Eigen::Vector3d raw_goal_pos, goal_pos;
drone_msgs::DroneState _DroneState;

ros::Subscriber joy_sub, manual_control_sub, drone_state_sub, user_yaw_sub;
ros::Publisher rviz_joy_goal_pub, joy_goal_pub;

Eigen::Matrix3f euler2matrix(float phi, float theta, float psi)
{
    Eigen::Matrix3f Rota_Mat;

    float r11 = cos(theta)*cos(psi);
    float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
    float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
    float r21 = cos(theta)*sin(psi);
    float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
    float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
    float r31 = - sin(theta);
    float r32 = sin(phi)*cos(theta);
    float r33 = cos(phi)*cos(theta); 
    Rota_Mat << r11,r12,r13,r21,r22,r23,r31,r32,r33;

    return Rota_Mat;
}

void drone_state_cb(const drone_msgs::DroneStateConstPtr& msg)
{
  _DroneState = *msg; // in ENU frame
  static int flag = 0;
  if(!flag && _DroneState.armed) flag = 1;
  if(flag == 1)
  {
    flag = 2;
    drone_yaw_init = _DroneState.attitude[2]; // by default: user stands directly behind the drone and face to it at the beginning
  }
}

void user_yaw_cb(const sensor_msgs::ImuConstPtr& msg)
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

void RCinput_cb(const drone_msgs::RCInputConstPtr& msg)
{
  if(joy_raw) joy_raw = false;

  Eigen::Vector3f goal_in_local_frame, goal_in_map_frame;
  goal_in_local_frame[0] = _max_goal_range_xy * msg->rc_x;
  goal_in_local_frame[1] = _max_goal_range_xy * msg->rc_y;
  goal_in_local_frame[2] = _max_goal_range_z * msg->rc_z;

  if(goal_in_local_frame.norm() < 0.1)
    return;

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
      Eigen::Quaterniond q_ = quaternion_from_rpy(Eigen::Vector3d{0.0,0.0,yaw_diff});
      q = Eigen::Quaternionf{q_.w(), q_.x(), q_.y(), q_.z()};
      cout<<yaw_diff<<" "<<user_yaw_init<<" "<<drone_yaw_init<<endl;
    }else{
      q = Eigen::Quaternionf{1.0, 0.0, 0.0, 0.0};
    }
  }

  // double roll,pitch,yaw;
  // tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
  // Eigen::Matrix3f R_Local_to_Joy = euler2matrix(roll, pitch, yaw);

  Eigen::Matrix3f R_Local_to_Joy(q);
	
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
  geometry_msgs::PointStamped  joy_goal_rviz; 
  joy_goal_rviz.header.seq++;
  joy_goal_rviz.header.stamp = ros::Time::now();
  joy_goal_rviz.header.frame_id = "world";
  joy_goal_rviz.point.x = raw_goal_pos[0];
  joy_goal_rviz.point.y = raw_goal_pos[1];
  joy_goal_rviz.point.z = raw_goal_pos[2];
  rviz_joy_goal_pub.publish(joy_goal_rviz);

  if(int(msg->goal_enable)==1)
  {
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
    
    geometry_msgs::PoseStamped  joy_goal; 
    joy_goal.header.seq++;
    joy_goal.header.stamp = ros::Time::now();
    joy_goal.header.frame_id = "world";
    joy_goal.pose.position.x = raw_goal_pos[0];
    joy_goal.pose.position.y = raw_goal_pos[1];
    joy_goal.pose.position.z = raw_goal_pos[2];
    joy_goal_pub.publish(joy_goal);
    goal_pos = raw_goal_pos;
  }
}

void joy_raw_cb(const sensor_msgs::Joy::ConstPtr &msg)
{
  if (joy_raw && msg->header.frame_id == "/dev/input/js0")
  {
    float axes[8];
    float dead_band=0.1;
    for(int i=0;i<8;i++){
      axes[i] = msg->axes[i];
      if(fabs(axes[i]) <= dead_band) axes[i] = 0.0f;
      else if(axes[i] < -dead_band) axes[i] = (axes[i]+dead_band)/(1.0f-dead_band);
      else if(axes[i] > dead_band) axes[i] = (axes[i]-dead_band)/(1.0f-dead_band);
    }

    // in ROS-ENU frame or Body-FLU
    float rc_x = axes[4]; // F
    float rc_y = axes[3]; // L
    float rc_z = axes[1]; // U
    float rc_r = axes[0]; // left: turn left, right turn right
    bool goal_enable = axes[2]<0.0f;  // L2


    Eigen::Vector3f goal_in_local_frame, goal_in_map_frame;
    goal_in_local_frame[0] = _max_goal_range_xy * rc_x;
    goal_in_local_frame[1] = _max_goal_range_xy * rc_y;
    goal_in_local_frame[2] = _max_goal_range_z * rc_z;

    if(goal_in_local_frame.norm() < 0.1)
      return;

    Eigen::Quaternionf q;
    if(control_from_joy == 1){
      // in Body Frame
      q = Eigen::Quaternionf {_DroneState.attitude_q.w, _DroneState.attitude_q.x, _DroneState.attitude_q.y, _DroneState.attitude_q.z};
    }else if (control_from_joy == 2){
      q = Eigen::Quaternionf{1.0, 0.0, 0.0, 0.0};
    }else if (control_from_joy == 3){
      if(drone_yaw_init != 0.0 && user_yaw_init != 0.0)
      {
        double yaw_diff = user_yaw - user_yaw_init + drone_yaw_init;
        yaw_diff = yaw_diff>M_PI ? yaw_diff-2*M_PI : (yaw_diff<-M_PI ? yaw_diff+2*M_PI : yaw_diff);
        Eigen::Quaterniond q_ = quaternion_from_rpy(Eigen::Vector3d{0.0,0.0,yaw_diff});
        q = Eigen::Quaternionf{q_.w(), q_.x(), q_.y(), q_.z()};
        cout<<yaw_diff<<" "<<user_yaw_init<<" "<<drone_yaw_init<<endl;
      }else{
        q = Eigen::Quaternionf{1.0, 0.0, 0.0, 0.0};
      }
    }

    Eigen::Matrix3f R_Local_to_Joy(q);
    
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
    geometry_msgs::PointStamped  joy_goal_rviz; 
    joy_goal_rviz.header.seq++;
    joy_goal_rviz.header.stamp = ros::Time::now();
    joy_goal_rviz.header.frame_id = "world";
    joy_goal_rviz.point.x = raw_goal_pos[0];
    joy_goal_rviz.point.y = raw_goal_pos[1];
    joy_goal_rviz.point.z = raw_goal_pos[2];
    rviz_joy_goal_pub.publish(joy_goal_rviz);

    if(int(goal_enable)==1)
    {
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
      
      geometry_msgs::PoseStamped  joy_goal; 
      joy_goal.header.seq++;
      joy_goal.header.stamp = ros::Time::now();
      joy_goal.header.frame_id = "world";
      joy_goal.pose.position.x = raw_goal_pos[0];
      joy_goal.pose.position.y = raw_goal_pos[1];
      joy_goal.pose.position.z = raw_goal_pos[2];
      joy_goal_pub.publish(joy_goal);
      goal_pos = raw_goal_pos;
    }
  }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "joy_remote");
    ros::NodeHandle nh("~");

    nh.param("geo_fence/x_min", geo_fence_x_min, -100.0);
    nh.param("geo_fence/x_max", geo_fence_x_max, 100.0);
    nh.param("geo_fence/y_min", geo_fence_y_min, -100.0);
    nh.param("geo_fence/y_max", geo_fence_y_max, 100.0);
    nh.param("geo_fence/z_min", geo_fence_z_min, -100.0);
    nh.param("geo_fence/z_max", geo_fence_z_max, 100.0);

    nh.param("is_2D", is_2D, false);                                           // 2D planning? fixed height
    nh.param("fly_height_2D", fly_height_2D, 1.0);  
    nh.param("min_goal_height", min_goal_height, 1.0);     
    nh.param("ceil_height", ceil_height, 5.0);
    /* -- Whether to use joy control as a second input? 
    *  -- 0：disable
    *  -- 1：control in Body Frame
    *  -- 2：control in User Heading Frame
    */
    nh.param("control_from_joy", control_from_joy, 3);
    nh.param("joy_goal_xy_max", _max_goal_range_xy, 3.0);                      // Maximum relative distance in the horizontal direction of the target（when joy control）
    nh.param("joy_goal_z_max", _max_goal_range_z, 3.0);    

    // [SUB]
    joy_raw = true;
    joy_sub = nh.subscribe<sensor_msgs::Joy>("/joy", 100, joy_raw_cb);
    manual_control_sub = nh.subscribe<drone_msgs::RCInput>("/joy/RCInput", 10, RCinput_cb);// Radio Control input
    drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state", 10, drone_state_cb);
    user_yaw_sub = nh.subscribe<sensor_msgs::Imu>("/wit/imu", 10, user_yaw_cb);

    // [PUB]
    rviz_joy_goal_pub = nh.advertise<geometry_msgs::PointStamped >("/joy/goal", 10); // joy goal set
    joy_goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 10);// terminal input

    goal_pos.setZero();
    
    ros::spin();

    return 0;
}
