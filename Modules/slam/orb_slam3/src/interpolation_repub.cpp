#include <ros/ros.h>

#include <iostream>
#include <deque>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>

using namespace std;
#define DEBUG true
//---------------------------------------外部参数-----------------------------------------------
float rate_hz;
bool d435i_with_imu;
bool interpolation;
float interpolation_delay;
int interpolation_sample_num, interpolation_order;

//---------------------------------------内部参数-----------------------------------------------
bool updated=false;
ros::Time last_stamp_slam;
geometry_msgs::PoseStamped pose;
std::deque<geometry_msgs::PoseStamped> poses;
ros::Time time_stamp_header;

//---------------------------------------发布器声明--------------------------------------------
ros::Publisher vision_pub;

//---------------------------------------接受器声明--------------------------------------------
ros::Subscriber orb_slam3_sub;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>函数声明<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void send_to_fcu();

void orb_slam3_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    if (msg->header.frame_id == "world")
    {
        pose.pose.position.x = msg->pose.position.x;
        pose.pose.position.y = msg->pose.position.y;
        pose.pose.position.z = msg->pose.position.z;
        
        Eigen::Quaterniond q_;
        if(d435i_with_imu)
        {
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
        
        pose.pose.orientation.w = q_.w();
        pose.pose.orientation.x = q_.x();
        pose.pose.orientation.y = q_.y();
        pose.pose.orientation.z = q_.z();
    
		pose.header.stamp = msg->header.stamp;
		
//		ros::Time time_now = ros::Time::now();
		
		updated = true;
//if(DEBUG) cout << "delta time sub: " << (time_now-msg->header.stamp).toSec() << endl;
    }
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>主 函 数<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
int main(int argc, char **argv)
{
    ros::init(argc, argv, "orb_slam3_interpolation_repub");
    ros::NodeHandle nh("~");

    nh.param<bool>("d435i_with_imu", d435i_with_imu, true);
    //　程序执行频率
    nh.param<float>("rate_hz", rate_hz, 50);

    // interpolation
    nh.param<bool>("interpolation", interpolation, true);
    nh.param<float>("interpolation_delay", interpolation_delay, 0.2);
    nh.param<int>("interpolation_order", interpolation_order, 2);
    nh.param<int>("interpolation_sample_num", interpolation_sample_num, 4);

    // 【订阅】ORB-SLAM3估计位姿
    orb_slam3_sub = nh.subscribe<geometry_msgs::PoseStamped>("/orb_slam3_ros/camera", 100, orb_slam3_cb);

	vision_pub = nh.advertise<geometry_msgs::PoseStamped>("/slam/pose", 10);
	
    // 频率
    ros::Rate rate(rate_hz);

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Main Loop<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    while (ros::ok())
    {
        //回调一次 更新传感器状态
        ros::spinOnce();

        // 将采集的机载设备的定位信息及偏航角信息发送至飞控，根据参数input_source选择定位信息来源
        send_to_fcu();

        rate.sleep();
    }

    return 0;
}

void send_to_fcu()
{    
    pose.header.frame_id = "/world";
    
	if(updated && !interpolation){
		vision_pub.publish(pose);
		updated = false;
		return;
	}

        static ros::Time time_ref;
        static Eigen::MatrixXf A(interpolation_sample_num,interpolation_order+1);
        static Eigen::MatrixXf b(interpolation_sample_num,7);
        static Eigen::MatrixXf X(interpolation_order+1,7);

    // interpolation
    if(poses.size()==interpolation_sample_num){
	    if(updated){
			while(poses.size()>=interpolation_sample_num)
				poses.pop_front();
			poses.push_back(pose);

           time_ref = poses.front().header.stamp;

            // Fitting
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
            X = (A.transpose() * A).llt().solve(A.transpose() * b);
        }
if(DEBUG){
cout<<"A: "<<A<<endl;
cout<<"b: "<<b<<endl;
cout<<"X: "<<X<<endl;
}
	    time_stamp_header = ros::Time::now() - ros::Duration(interpolation_delay);
if(DEBUG){
cout<<"delta_time_pub: "<<(time_stamp_header-poses.back().header.stamp).toSec()<<endl;
cout << endl;
}
	    // Interpolation
	    if(poses.back().header.stamp >= time_stamp_header){
		    geometry_msgs::PoseStamped vision;
	    	vision.header.stamp = time_stamp_header + ros::Duration(interpolation_delay);// default delay
	    	vision.header.frame_id = "/world";
	    	float t = (time_stamp_header-time_ref).toSec();

	    	Eigen::VectorXf T(interpolation_order+1);
	    	T(0) = 1;
	    	for(int i=1;i<=interpolation_order;i++)
	    	{
	    		T(i) = t*T(i-1);
	    	}
  	
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
      
    		vision_pub.publish(vision);
	    }
    }else if(updated){
	    poses.push_back(pose);
        if(poses.size()==interpolation_sample_num){
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
            X = (A.transpose() * A).llt().solve(A.transpose() * b);
        }
    }
    updated = false;
}
