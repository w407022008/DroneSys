#include <ros/ros.h>

#include <iostream>
#include <deque>
#include <Eigen/Dense>
#include "state_from_mavros.h"
#include <sensor_msgs/Joy.h>
#include "math_utils.h"
#include "message_utils.h"

using namespace std;
#define TRA_WINDOW 1000
#define DEBUG false
#define NODE_NAME "pos_estimator"

// general param
int input_source;
int optitrack_frame;
float rate_hz;
Eigen::Vector3f device_pos_, cur_pos;
float yaw_offset;
bool armed_last = false, armed_cur=false;
ros::Time cur_pos_time;
// ros::Time last_stamp_lidar, last_stamp_t265, last_stamp_mocap, last_stamp_gazebo, last_stamp_slam;
string object_name;
bool d435i_with_imu;

// interpolation
bool interpolation;
float interpolation_rate;
int interpolation_sample_num, interpolation_order;
bool updated=false;
std::deque<geometry_msgs::PoseStamped> poses;
ros::Time time_stamp_header;

// RC input
drone_msgs::RCInput _RCInput, _RCInput_last;
float axes[8];
float dead_band;
int8_t buttons[13];

// Drone Trajectory
std::vector<geometry_msgs::PoseStamped> posehistory_vector_;

// [PUB]
ros::Publisher vision_pose_pub;
ros::Publisher vision_odom_pub;
ros::Publisher drone_state_pub;
ros::Publisher RC_input_pub;
ros::Publisher message_pub;
ros::Publisher odom_pub;
ros::Publisher trajectory_pub;

//[SUB]
ros::Subscriber optitrack_sub;
ros::Subscriber lidar_sub;
ros::Subscriber gazebo_sub;
ros::Subscriber t265_sub;
ros::Subscriber vins_fusion_sub;
ros::Subscriber orb_slam3_sub;
ros::Subscriber joy_sub;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Helper Funcion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void vision_pos_interpolation();
void pub_to_nodes(drone_msgs::DroneState State_from_fcu);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Callback Function <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void lidar_cb(const tf2_msgs::TFMessage::ConstPtr &msg)
{
    if (msg->transforms[0].header.frame_id == "map" && msg->transforms[0].child_frame_id == "base_link" && input_source == 1)  
    {
        geometry_msgs::PoseStamped pose_msg;

        pose_msg.pose.position.x = msg->transforms[0].transform.translation.x - device_pos_[0];
        pose_msg.pose.position.y = msg->transforms[0].transform.translation.y - device_pos_[1];
        pose_msg.pose.position.z = msg->transforms[0].transform.translation.z - device_pos_[2]; 

        // Read the Quaternion from the Carto Package [Frame: lidar[ENU]]
        Eigen::Quaterniond q_(msg->transforms[0].transform.rotation.w, msg->transforms[0].transform.rotation.x, msg->transforms[0].transform.rotation.y, msg->transforms[0].transform.rotation.z);

         // Transform the Quaternion to Euler Angles
        if(fabs(yaw_offset)>0.001){
        	Eigen::Vector3d euler_;
            euler_ = quaternion_to_euler(q_);
            euler_[2] += yaw_offset;
            q_ = quaternion_from_rpy(euler_);
        }
        
        pose_msg.pose.orientation.w = q_.w();
        pose_msg.pose.orientation.x = q_.x();
        pose_msg.pose.orientation.y = q_.y();
        pose_msg.pose.orientation.z = q_.z();

        pose_msg.header.stamp = msg->transforms[0].header.stamp;
        
        // ros::Time time_now = ros::Time::now();
	    // if( (time_now - last_stamp_lidar).toSec()> 0.1)
	    // {
	    //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "Cartographer Timeout.");
	    // }
        // last_stamp_lidar = time_now;
        
        if(interpolation){
            poses.push_back(pose_msg);
			while(poses.size()>interpolation_sample_num)
				poses.pop_front();
		    updated = true;
        }
        else
        {
            pose_msg.header.frame_id = "/world";
            vision_pose_pub.publish(pose_msg);
        }
    }
}

void t265_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (msg->header.frame_id == "t265_odom_frame")
    {
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.pose.position.x = msg->pose.pose.position.x - device_pos_[0];
        pose_msg.pose.position.y = msg->pose.pose.position.y - device_pos_[1];
        pose_msg.pose.position.z = msg->pose.pose.position.z - device_pos_[2];

        Eigen::Quaterniond q_ = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        if(fabs(yaw_offset)>0.001){
        	Eigen::Vector3d euler_;
            euler_ = quaternion_to_euler(q_);
            euler_[2] += yaw_offset;
            q_ = quaternion_from_rpy(euler_);
        }
        
        pose_msg.pose.orientation.w = q_.w();
        pose_msg.pose.orientation.x = q_.x();
        pose_msg.pose.orientation.y = q_.y();
        pose_msg.pose.orientation.z = q_.z();

        pose_msg.header.stamp = msg->header.stamp;
        
        // ros::Time time_now = ros::Time::now();
	    // if( (time_now - last_stamp_t265).toSec()> 0.1)
	    // {
	    //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "T256 Timeout.");
	    // }
        // last_stamp_t265 = time_now;
        
        if(interpolation){
            poses.push_back(pose_msg);
			while(poses.size()>interpolation_sample_num)
				poses.pop_front();
		    updated = true;
        }
        else
        {
            pose_msg.header.frame_id = "/world";
            vision_pose_pub.publish(pose_msg);
        }
    }
    else
    {
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "wrong t265 frame id.");
    }
}

void mocap_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    geometry_msgs::PoseStamped pose_msg;
    // optitrack frame to ENU frame
    Eigen::Quaterniond q_;
    if (optitrack_frame == 0)
    {
        // Read the Drone Position from the Vrpn Package [Frame: Vicon-Z-up]  (Vicon(XYZ-ENU) to ENU frame)
        pose_msg.pose.position.x = msg->pose.position.x;
        pose_msg.pose.position.y = msg->pose.position.y;
        pose_msg.pose.position.z = msg->pose.position.z;
        
        // Read the Quaternion from the Vrpn Package [Frame: Vicon[ENU]]
        q_ = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    }
    else
    {
        // Read the Drone Position from the Vrpn Package [Frame: Vicon-Y-up]  (Vicon(XYZ-NUE) to ENU frame)
        pose_msg.pose.position.x = msg->pose.position.z;
        pose_msg.pose.position.y = msg->pose.position.x;
        pose_msg.pose.position.z = msg->pose.position.y;
        
        // Read the Quaternion from the Vrpn Package [Frame: Vicon[ENU]]
        q_ = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.z, msg->pose.orientation.x, msg->pose.orientation.y);
    }

    // Transform the Quaternion to Euler Angles
    if(fabs(yaw_offset)>0.001){
    	Eigen::Vector3d euler_;
        euler_ = quaternion_to_euler(q_);
        euler_[2] += yaw_offset;
        q_ = quaternion_from_rpy(euler_);
    }
    
    pose_msg.pose.orientation.w = q_.w();
    pose_msg.pose.orientation.x = q_.x();
    pose_msg.pose.orientation.y = q_.y();
    pose_msg.pose.orientation.z = q_.z();

    pose_msg.header.stamp = msg->header.stamp;
    
    // ros::Time time_now = ros::Time::now();
    // if( (time_now - last_stamp_mocap).toSec()> 0.1)
    // {
    //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "Mocap Timeout.");
    // }
    // last_stamp_mocap = time_now;
    
    // if(interpolation){
    //     poses.push_back(pose_msg);
    //     while(poses.size()>interpolation_sample_num)
    //         poses.pop_front();
    //     updated = true;
    // }
    // else
    // {
        pose_msg.header.frame_id = "/world";
        vision_pose_pub.publish(pose_msg);
    // }
}

void gazebo_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (msg->header.frame_id == "world")
    {
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.pose.position.x = msg->pose.pose.position.x;
        pose_msg.pose.position.y = msg->pose.pose.position.y;
        pose_msg.pose.position.z = msg->pose.pose.position.z;

        Eigen::Quaterniond q_ = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        if(fabs(yaw_offset)>0.001){
        	Eigen::Vector3d euler_;
            euler_ = quaternion_to_euler(q_);
            euler_[2] += yaw_offset;
            q_ = quaternion_from_rpy(euler_);
        }
        
        pose_msg.pose.orientation.w = q_.w();
        pose_msg.pose.orientation.x = q_.x();
        pose_msg.pose.orientation.y = q_.y();
        pose_msg.pose.orientation.z = q_.z();
    
		pose_msg.header.stamp = msg->header.stamp;
		
        // if(interpolation){
        //     poses.push_back(pose_msg);
		// 	while(poses.size()>interpolation_sample_num)
		// 		poses.pop_front();
		//     updated = true;
        // }
        // else
        // {
            pose_msg.header.frame_id = "/world";
            vision_pose_pub.publish(pose_msg);
        // }

        // ros::Time time_now = ros::Time::now();
        // if( (time_now - last_stamp_gazebo).toSec()> 0.1)
        // {
        //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "Gazebo Timeout.");
        // }
        // last_stamp_gazebo = time_now;
    }
    else
    {
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "wrong gazebo ground truth frame id.");
    }
}

void vins_fusion_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (msg->header.frame_id == "world")
    {
        // geometry_msgs::PoseStamped pose_msg;
        nav_msgs::Odometry pose_msg;

        Eigen::Quaterniond q_;
        if(d435i_with_imu)
        {
            // ENU(vins) msg --> NED(efk) pose_msg
            pose_msg.pose.pose.position.y = msg->pose.pose.position.x - device_pos_[0];
            pose_msg.pose.pose.position.x = msg->pose.pose.position.y - device_pos_[1];
            pose_msg.pose.pose.position.z = -(msg->pose.pose.position.z - device_pos_[2]);

            pose_msg.twist.twist.linear.y = msg->twist.twist.linear.x;
            pose_msg.twist.twist.linear.x = msg->twist.twist.linear.y;
            pose_msg.twist.twist.linear.z = -msg->twist.twist.linear.z;

            // because init meas is (0.707, -0.707, 0.0, 0.0)
            double cnst = sqrt(2)/2;
            double q_0 = cnst*msg->pose.pose.orientation.w;
            double q_1 = cnst*msg->pose.pose.orientation.x;
            double q_2 = cnst*msg->pose.pose.orientation.y;
            double q_3 = cnst*msg->pose.pose.orientation.z;
            q_ = Eigen::Quaterniond(q_0-q_1, q_1+q_0, q_2+q_3, q_3-q_2);
        }
        else
        {
            // ENU(vins)msg --> NED(efk)pose_msg
            pose_msg.pose.pose.position.y = msg->pose.pose.position.x - device_pos_[0];
            pose_msg.pose.pose.position.x = msg->pose.pose.position.z - device_pos_[1];
            pose_msg.pose.pose.position.z = -(-msg->pose.pose.position.y - device_pos_[2]);
            
            pose_msg.twist.twist.linear.y = msg->twist.twist.linear.x;
            pose_msg.twist.twist.linear.x = msg->twist.twist.linear.z;
            pose_msg.twist.twist.linear.z = -msg->twist.twist.linear.y;

            q_ = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.z, -msg->pose.pose.orientation.y);
        }
        if(fabs(yaw_offset)>0.001){
        	Eigen::Vector3d euler_;
            euler_ = quaternion_to_euler(q_);
            euler_[2] += yaw_offset;
            q_ = quaternion_from_rpy(euler_);
        }
        
        // ENU(vins) --> NED(efk)
        pose_msg.pose.pose.orientation.w = q_.w();
        pose_msg.pose.pose.orientation.x = q_.y();
        pose_msg.pose.pose.orientation.y = q_.x();
        pose_msg.pose.pose.orientation.z = -q_.z();

        pose_msg.header.stamp = msg->header.stamp;
        
        Eigen::Vector3f pos{pose_msg.pose.pose.position.y,pose_msg.pose.pose.position.x,-pose_msg.pose.pose.position.z};
        Eigen::Vector3f vel{pose_msg.twist.twist.linear.x,pose_msg.twist.twist.linear.y,pose_msg.twist.twist.linear.z};
        static Eigen::Vector3f last_pos, last_vel;
        static int cnt = 0;
        if(armed_cur){
            if(cnt>10){
                if((cur_pos_time - msg->header.stamp).toSec() > 0.3
                || (pos - last_pos).norm()>0.5
                || (vel - last_vel).norm()>5.0
                || (cur_pos - pos).norm()>1.0){
                    return;
                }
            }else{ cnt++; }
        }else{ cnt = 0; }
        last_pos = pos;
        last_vel = vel;

        // if(interpolation){
        //     poses.push_back(pose_msg);
        //     while(poses.size()>interpolation_sample_num)
        //         poses.pop_front();
        //     updated = true;
        // }
        // else
        // {
            pose_msg.header.frame_id = "odom_ned";
            pose_msg.child_frame_id = "base_link_frd"; // should modify mavlink_receiver.cpp as Local frame by default
            vision_odom_pub.publish(pose_msg);
        // }

        // ros::Time time_now = ros::Time::now();
	    // if( (time_now - last_stamp_slam).toSec()> 0.1)
	    // {
	    //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "VINS Fusion Timeout.");
	    // }
        // last_stamp_slam = time_now;
    }
    else
    {
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "wrong VINS Fusion frame id.");
    }
}

void orb_slam3_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    if (msg->header.frame_id == "world")
    {
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.pose.position.x = msg->pose.position.x - device_pos_[0];
        pose_msg.pose.position.y = msg->pose.position.y - device_pos_[1];
        pose_msg.pose.position.z = msg->pose.position.z - device_pos_[2];
        
        Eigen::Quaterniond q_;
        if(d435i_with_imu)
        {
            // because init meas is (0.707, 0.0, 0.707, 0.0)
            double cnst = sqrt(2)/2;
            double q_0 = cnst*msg->pose.orientation.w;
            double q_1 = cnst*msg->pose.orientation.x;
            double q_2 = cnst*msg->pose.orientation.y;
            double q_3 = cnst*msg->pose.orientation.z;
            q_ = Eigen::Quaterniond(q_0+q_2, q_1+q_3, -q_0+q_2, -q_1+q_3);
        }
        else
        {
            q_ = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
        }
        q_ = Eigen::Quaterniond(q_.w(), q_.x(), q_.y(), q_.z());
        if(fabs(yaw_offset)>0.001){
        	Eigen::Vector3d euler_;
            euler_ = quaternion_to_euler(q_);
            euler_[2] += yaw_offset;
            q_ = quaternion_from_rpy(euler_);
        }
        
        pose_msg.pose.orientation.w = q_.w();
        pose_msg.pose.orientation.x = q_.x();
        pose_msg.pose.orientation.y = q_.y();
        pose_msg.pose.orientation.z = q_.z();
    
        pose_msg.header.stamp = msg->header.stamp;
		
        Eigen::Vector3f pos{pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z};
        static Eigen::Vector3f last_pos;
        static int cnt = 0;
        if(armed_cur){
            if(cnt>10){
                if((cur_pos_time - msg->header.stamp).toSec() > 0.3
                || (pos - last_pos).norm()>0.5
                || (cur_pos - pos).norm()>1.0){
                    return;
                }
            }else{ cnt++; }
        }else{ cnt = 0; }
        last_pos = pos;

        if(interpolation){
            poses.push_back(pose_msg);
            while(poses.size()>interpolation_sample_num)
                poses.pop_front();
            updated = true;
        }
        else
        {
            pose_msg.header.frame_id = "world";
            vision_pose_pub.publish(pose_msg);
        }

        // ros::Time time_now = ros::Time::now();
        // if( (time_now - last_stamp_slam).toSec()> 0.1)
        // {
        //     pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "ORB_SLAM3 Timeout.");
        // }
        // last_stamp_slam = time_now;
    }
    else
    {
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "wrong slam frame id.");
    }
}

void joy_cb(const sensor_msgs::Joy::ConstPtr &msg)
{
    if (msg->header.frame_id == "/dev/input/js0")
    {
    	for(int i=0;i<8;i++){
	        axes[i] = msg->axes[i];
            if(fabs(axes[i]) <= dead_band) axes[i] = 0.0f;
            else if(axes[i] < -dead_band) axes[i] = (axes[i]+dead_band)/(1.0f-dead_band);
            else if(axes[i] > dead_band) axes[i] = (axes[i]-dead_band)/(1.0f-dead_band);
        }
	    for(int i=0;i<13;i++)
	        buttons[i] = msg->buttons[i];
        _RCInput.data_source = drone_msgs::RCInput::DRIVER_JOYSTICK;
        _RCInput.header.stamp = ros::Time::now();
    }
    else
    {
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "the joy frame id is not /dev/input/js0.");
    }
}

void timerCallback(const ros::TimerEvent &e)
{
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "px4_transceiver");
    ros::NodeHandle nh("~");

    // 0: vicon， 1: Catographer SLAM, 2: gazebo ground truth, 3: T265, 4: VINSFusion, 5: ORBSLAM3
    nh.param<int>("input_source", input_source, 0);
    // Optitrack frame convention 0: Z-up -- 1: Y-up (See the configuration in the motive software)
    nh.param<int>("optitrack_frame", optitrack_frame, 1); 
    // Rigid body name defined in Mocap system
    nh.param<string>("object_name", object_name, "UAV");
    // VO or VIO
    nh.param<bool>("d435i_with_imu", d435i_with_imu, false);
    // pub rate
    nh.param<float>("rate_hz", rate_hz, 20);
    // fixed yaw offset from measurement
    nh.param<float>("offset_yaw", yaw_offset, 0);
    // Joystick half dead band
    nh.param<float>("dead_band", dead_band, 0.1);
    // interpolation
    nh.param<bool>("interpolation", interpolation, false);
    nh.param<float>("interpolation_rate", interpolation_rate, 50.0);
    nh.param<int>("interpolation_order", interpolation_order, 3);
    nh.param<int>("interpolation_sample_num", interpolation_sample_num, 10);

    // print
    cout<<"[perception]:input_source: "<<input_source<<endl;
    if(input_source==0) cout<<"[perception]:optitrack_frame: "<<optitrack_frame<<endl;
    cout<<"[perception]:object_name: "<<object_name<<endl;
    if(input_source>=4)cout<<"[perception]:d435i_with_imu: "<<d435i_with_imu<<endl;
    cout<<"[perception]:rate_hz: "<<rate_hz<<endl;
    cout<<"[perception]:offset_yaw: "<<yaw_offset<<endl;
    cout<<"[perception]:interpolation: "<<interpolation<<endl;
    if(interpolation){
        cout<<"[perception]:interpolation_rate: "<<interpolation_rate<<endl;
        cout<<"[perception]:interpolation_order: "<<interpolation_order<<endl;
        cout<<"[perception]:interpolation_sample_num: "<<interpolation_sample_num<<endl;
    }

    // [SUB]
    if(input_source==0) optitrack_sub = nh.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/"+ object_name + "/pose", 100, mocap_cb);
    if(input_source==1) lidar_sub = nh.subscribe<tf2_msgs::TFMessage>("/tf", 100, lidar_cb);
    if(input_source==2) gazebo_sub = nh.subscribe<nav_msgs::Odometry>("/drone_msg/ground_truth/odometry", 100, gazebo_cb);
    if(input_source==3) t265_sub = nh.subscribe<nav_msgs::Odometry>("/t265/odom/sample", 100, t265_cb);
    if(input_source==4) vins_fusion_sub = nh.subscribe<nav_msgs::Odometry>("/vins_estimator/odometry", 100, vins_fusion_cb);
    if(input_source==5) orb_slam3_sub = nh.subscribe<geometry_msgs::PoseStamped>("/orb_slam3_ros/camera", 100, orb_slam3_cb);

    joy_sub = nh.subscribe<sensor_msgs::Joy>("/joy", 100, joy_cb);

    // [PUB]
    // [mavros_extras/src/plugins/vision_pose_estimate.cpp]: MavLink message (VISION_POSITION_ESTIMATE(#102)) -> uORB message (vehicle_visual_odometry.msg)
    vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);

    // [mavros_extras/src/plugins/odom.cpp]: Mavlink message (ODOMETRY(#331)) -> uORB message (vehicle_visual_odometry.msg)
    vision_odom_pub = nh.advertise<nav_msgs::Odometry>("/mavros/odometry/out", 10);

    // Drone state
    drone_state_pub = nh.advertise<drone_msgs::DroneState>("/drone_msg/drone_state", 10);

    // Radio Control Input
    RC_input_pub = nh.advertise<drone_msgs::RCInput>("/joy/RCInput", 10);

    //　Drone odometry for Rviz
    odom_pub = nh.advertise<nav_msgs::Odometry>("/drone_msg/drone_odom", 10);

    // Drone trajectory for Rviz
    trajectory_pub = nh.advertise<nav_msgs::Path>("/drone_msg/drone_trajectory", 10);
    
    // Ground station messages
    message_pub = nh.advertise<drone_msgs::Message>("/drone_msg/message", 10);

    // Custome Timer Callback
    // ros::Timer timer = nh.createTimer(ros::Duration(10.0), timerCallback);

    // Receive Mavlink message by mavros from Autopilot
    state_from_mavros _state_from_mavros;

    // update
    ros::Rate rate(rate_hz);

    ros::Time begin = ros::Time::now();
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Main Loop<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    while (ros::ok())
    {
        ros::spinOnce();

        cur_pos_time = _state_from_mavros._DroneState.header.stamp;
        cur_pos[0] = _state_from_mavros._DroneState.position[0];
        cur_pos[1] = _state_from_mavros._DroneState.position[1];
        cur_pos[2] = _state_from_mavros._DroneState.position[2];

        armed_cur = _state_from_mavros._DroneState.armed;
        if(armed_cur > armed_last)
        {
            device_pos_[0] += cur_pos[0];  // We assume that takeoff from pos 0
            device_pos_[1] += cur_pos[1];  // We assume that takeoff from pos 0
            device_pos_[2] += cur_pos[2];  // We assume that takeoff from alt 0
        }
        armed_last = armed_cur;

        // transmit vision_pos to autopilot with interpolation
        if(interpolation) vision_pos_interpolation();

        // publish drone state from Autopilot
        pub_to_nodes(_state_from_mavros._DroneState);

        // publish RC input from Joystick or Autopilot
        if(_RCInput.data_source == drone_msgs::RCInput::DRIVER_JOYSTICK){
            // in ROS-ENU frame or Body-FLU
            _RCInput.rc_x = axes[4]; // F
            _RCInput.rc_y = axes[3]; // L
            _RCInput.rc_z = axes[1]; // U
            _RCInput.rc_r = axes[0]; // left: turn left, right turn right
            _RCInput.buttons = 0;
            for(int i=0;i<13;i++)
                if(buttons[i])
                    _RCInput.buttons += 0b1 << i;
            _RCInput.goal_enable = axes[2]<0.0f;  // L2
            if((ros::Time::now()-_RCInput.header.stamp).toSec() > 1.0)
                _RCInput.data_source = drone_msgs::RCInput::DISABLE;
        }else if(_state_from_mavros._RCInput.data_source == drone_msgs::RCInput::MAVROS_MANUAL_CONTROL){
            _RCInput = _state_from_mavros._RCInput;
            // dead band x
            if(fabs(_RCInput.rc_x) <= dead_band) _RCInput.rc_x = 0;
            else if(_RCInput.rc_x < -dead_band) _RCInput.rc_x = (_RCInput.rc_x+dead_band)/(1-dead_band);
            else if(_RCInput.rc_x > dead_band) _RCInput.rc_x = (_RCInput.rc_x-dead_band)/(1-dead_band);
            // dead band y
            if(fabs(_RCInput.rc_y) <= dead_band) _RCInput.rc_y = 0;
            else if(_RCInput.rc_y < -dead_band) _RCInput.rc_y = (_RCInput.rc_y+dead_band)/(1-dead_band);
            else if(_RCInput.rc_y > dead_band) _RCInput.rc_y = (_RCInput.rc_y-dead_band)/(1-dead_band);
            // dead band z
            if(fabs(_RCInput.rc_z) <= dead_band) _RCInput.rc_z = 0;
            else if(_RCInput.rc_z < -dead_band) _RCInput.rc_z = (_RCInput.rc_z+dead_band)/(1-dead_band);
            else if(_RCInput.rc_z > dead_band) _RCInput.rc_z = (_RCInput.rc_z-dead_band)/(1-dead_band);
            // dead band r
            if(fabs(_RCInput.rc_r) <= dead_band) _RCInput.rc_r = 0;
            else if(_RCInput.rc_r < -dead_band) _RCInput.rc_r = (_RCInput.rc_r+dead_band)/(1-dead_band);
            else if(_RCInput.rc_r > dead_band) _RCInput.rc_r = (_RCInput.rc_r-dead_band)/(1-dead_band);
            _RCInput.goal_enable = _RCInput.buttons==8; // L2  (Attention! custom defined px4 program for forwarding joy button msg from qgc)
            _RCInput.header.stamp = ros::Time::now();
        }
		
        if(_state_from_mavros._RCInput.data_source || _RCInput.data_source)
            if(fabs(_RCInput.rc_x-_RCInput_last.rc_x)>1e-3 ||
            fabs(_RCInput.rc_y-_RCInput_last.rc_y)>1e-3 ||
            fabs(_RCInput.rc_z-_RCInput_last.rc_z)>1e-3 ||
            (ros::Time::now() - begin).toSec() > 0.02)
            { // 50Hz Joystick Pub Rate
                _RCInput_last = _RCInput;
                RC_input_pub.publish(_RCInput);
                ros::Time begin = ros::Time::now();
            }
		
        rate.sleep();
    }

    return 0;
}

void vision_pos_interpolation()
{    
    static ros::Time time_ref;
    static Eigen::MatrixXf A(interpolation_sample_num,interpolation_order+1);
    static Eigen::MatrixXf b(interpolation_sample_num,7);
    static Eigen::MatrixXf X(interpolation_order+1,7);

    if(poses.size() == interpolation_sample_num)
    {
        if(updated)
        {
            updated = false;
            // Fitting
            time_ref = poses.front().header.stamp;
            std::deque<geometry_msgs::PoseStamped>::iterator it = poses.begin();
            int idx=0;
            while(it != poses.end()){
                float t = (it->header.stamp-time_ref).toSec();

                A(idx,0) = 1;
                for(int i=1;i<=interpolation_order;i++)
                {
                    A(idx,i) = t*A(idx,i-1);
                }
                b(idx,0) = it->pose.position.x;
                b(idx,1) = it->pose.position.y;
                b(idx,2) = it->pose.position.z;
                b(idx,3) = it->pose.orientation.x;
                b(idx,4) = it->pose.orientation.y;
                b(idx,5) = it->pose.orientation.z;
                b(idx,6) = it->pose.orientation.w;
                it++;
                idx++;
            }
            auto res = A.transpose() * A;
            auto val = A.transpose() * b;
            // auto  inv = res.inverse();
            X = res.llt().solve(val);
        }
        if(DEBUG){
        cout<<"A: "<<A<<endl;
        cout<<"b: "<<b<<endl;
        cout<<"X: "<<X<<endl;
        }

        while(poses.back().header.stamp >= time_stamp_header - ros::Duration(0.005)){
            if(DEBUG)  cout << "[DEB] delta time pub: " << (time_stamp_header- poses.back().header.stamp).toSec() << endl;
            // Interpolation
            geometry_msgs::PoseStamped vision;
            vision.header.stamp = time_stamp_header;// default delay
            vision.header.frame_id = "/world";
            float t = (time_stamp_header-time_ref).toSec();

            Eigen::VectorXf T(interpolation_order+1);
            T(0) = 1;
            for(int i=1;i<=interpolation_order;i++)
            {
                T(i) = t*T(i-1);
            }
    
            // Publish
            Eigen::VectorXf state(7);
            state = X.transpose()*T;

            vision.pose.position.x = state[0];
            vision.pose.position.y = state[1];
            vision.pose.position.z = state[2];

            Eigen::Quaternion<double> q = {state[6],state[3],state[4],state[5]};
            q.normalize();
            vision.pose.orientation.x = q.x();
            vision.pose.orientation.y = q.y();
            vision.pose.orientation.z = q.z();
            vision.pose.orientation.w = q.w();
            
            vision_pose_pub.publish(vision);

            time_stamp_header += ros::Duration(1.0/interpolation_rate);
        }
    }
    else if(!poses.empty())
    {
        time_stamp_header = poses.back().header.stamp;
    }
}

void pub_to_nodes(drone_msgs::DroneState State_from_fcu)
{
    // drone_msgs::DroneState
    drone_msgs::DroneState Drone_State = State_from_fcu;
    Drone_State.header.stamp = ros::Time::now();
    // if(input_source == 9 )
    // {
    //     Drone_State.position[2]  = Drone_State.rel_alt;
    // }
    drone_state_pub.publish(Drone_State);

    // Drone odometry for Rviz
    nav_msgs::Odometry Drone_odom;
    Drone_odom.header.stamp = ros::Time::now();
    Drone_odom.header.frame_id = "world";
    Drone_odom.child_frame_id = "base_link";

    Drone_odom.pose.pose.position.x = Drone_State.position[0];
    Drone_odom.pose.pose.position.y = Drone_State.position[1];
    Drone_odom.pose.pose.position.z = Drone_State.position[2];

    // if (Drone_odom.pose.pose.position.z <= 0)
    // {
    //     Drone_odom.pose.pose.position.z = 0.01;
    // }

    Drone_odom.pose.pose.orientation = Drone_State.attitude_q;
    Drone_odom.twist.twist.linear.x = Drone_State.velocity[0];
    Drone_odom.twist.twist.linear.y = Drone_State.velocity[1];
    Drone_odom.twist.twist.linear.z = Drone_State.velocity[2];
    Drone_odom.twist.twist.angular.x = Drone_State.attitude_rate[0];
    Drone_odom.twist.twist.angular.y = Drone_State.attitude_rate[1];
    Drone_odom.twist.twist.angular.z = Drone_State.attitude_rate[2];
    odom_pub.publish(Drone_odom);

    // Drone trajectory for Rviz
    geometry_msgs::PoseStamped drone_pos;
    drone_pos.header.stamp = ros::Time::now();
    drone_pos.header.frame_id = "world";
    drone_pos.pose.position.x = Drone_State.position[0];
    drone_pos.pose.position.y = Drone_State.position[1];
    drone_pos.pose.position.z = Drone_State.position[2];

    drone_pos.pose.orientation = Drone_State.attitude_q;

    posehistory_vector_.insert(posehistory_vector_.begin(), drone_pos);
    if (posehistory_vector_.size() > TRA_WINDOW)
    {
        posehistory_vector_.pop_back();
    }

    nav_msgs::Path drone_trajectory;
    drone_trajectory.header.stamp = ros::Time::now();
    drone_trajectory.header.frame_id = "world";
    drone_trajectory.poses = posehistory_vector_;
    trajectory_pub.publish(drone_trajectory);//发布
}
