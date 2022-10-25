#include <ros/ros.h>
#include <iostream>
#include "message_utils.h"

#include <geometry_msgs/PoseStamped.h>
#include <drone_msgs/DroneState.h>

//#include <mavros_msgs/State.h>
#include <geometry_msgs/Pose.h>
//#include <mavros_msgs/PositionTarget.h>
#include <eigen_conversions/eigen_msg.h>
#include <mavros/frame_tf.h>
#include <mavros_msgs/WaypointList.h>
//#include <mavros_msgs/HomePosition.h>
#include <GeographicLib/Geocentric.hpp>
#include <sensor_msgs/NavSatFix.h>

using namespace std;

drone_msgs::DroneState _DroneState;
Eigen::Matrix3f R_Body_to_ENU;   

Eigen::Matrix3f get_rotation_matrix(float phi, float theta, float psi)
{
    Eigen::Matrix3f R_Body_to_ENU;

    float r11 = cos(theta)*cos(psi);
    float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
    float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
    float r21 = cos(theta)*sin(psi);
    float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
    float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
    float r31 = - sin(theta);
    float r32 = sin(phi)*cos(theta);
    float r33 = cos(phi)*cos(theta); 
    R_Body_to_ENU << r11,r12,r13,r21,r22,r23,r31,r32,r33;

    return R_Body_to_ENU;
}

mavros_msgs::WaypointList waypoints;
void waypoints_cb(const mavros_msgs::WaypointList::ConstPtr& msg){
    waypoints = *msg;
}

Eigen::Vector3d current_gps;
void gps_cb(const sensor_msgs::NavSatFix::ConstPtr &msg)
{
    current_gps = { msg->latitude, msg->longitude, msg->altitude };
}

geometry_msgs::PoseStamped local_pos;
Eigen::Vector3d current_local_pos;
void local_pos_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    current_local_pos = mavros::ftf::to_eigen(msg->pose.position);
    local_pos = *msg;
}

void drone_state_cb(const drone_msgs::DroneState::ConstPtr &msg)
{
    _DroneState = *msg;
    R_Body_to_ENU = get_rotation_matrix(_DroneState.attitude[0], _DroneState.attitude[1], _DroneState.attitude[2]);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_goal_from_qgc");
    ros::NodeHandle nh("~");

    ros::Subscriber drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state",10,drone_state_cb);
    ros::Subscriber waypoint_sub = nh.subscribe<mavros_msgs::WaypointList>("/mavros/mission/waypoints", 100, waypoints_cb);
    ros::Subscriber local_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 100, local_pos_cb);
    ros::Subscriber gps_sub = nh.subscribe<sensor_msgs::NavSatFix>("/mavros/global_position/global",100,gps_cb);
    ros::Publisher goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 10);
    ros::Rate rate(20.0);
    printf("init ok!\n");

    while(ros::ok())
    {
        while(ros::ok() && !_DroneState.connected)
        {
            ros::spinOnce();
            rate.sleep();
        }
        printf("connected ok!\n");

        // GPS postion -> local ENU postion
        std::vector<geometry_msgs::PoseStamped> pose;
        //printf("wp size=%d\n", waypoints.waypoints.size());
        for( int index = 0; index < waypoints.waypoints.size(); index++)
        {
            //将大地坐标下的经纬度转换到地心坐标系的m，z轴指向北极，x轴零经度，零纬度。也就是ECEF坐标系
            geometry_msgs::PoseStamped p;

            //声明了一个类 earth类的实例化
            GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(),GeographicLib::Constants::WGS84_f());

            //GPS下的航点经纬高
            Eigen::Vector3d goal_gps(waypoints.waypoints[index].x_lat,waypoints.waypoints[index].y_long,0);
            
            // printf("%f %f \n", waypoints.waypoints[index].x_lat, waypoints.waypoints[index].y_long);

            //将大地坐标系转换为地心坐标系
            Eigen::Vector3d current_ecef;

            earth.Forward(current_gps.x(), current_gps.y(), current_gps.z(), current_ecef.x(), current_ecef.y(), current_ecef.z());

            Eigen::Vector3d goal_ecef;

            earth.Forward(goal_gps.x(), goal_gps.y(), goal_gps.z(), goal_ecef.x(), goal_ecef.y(), goal_ecef.z());

            Eigen::Vector3d ecef_offset = goal_ecef - current_ecef;

            Eigen::Vector3d enu_offset = mavros::ftf::transform_frame_ecef_enu(ecef_offset, current_gps);

            //仿射变换
            Eigen::Affine3d sp;

            Eigen::Quaterniond q;

            q = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ());
            
            sp.translation() = current_local_pos + enu_offset;

            sp.linear() = q.toRotationMatrix();

            //*****************往vector容器中存数据*************************//
            Eigen::Vector3d testv(sp.translation());
            p.pose.position.x = testv[0];
            p.pose.position.y = testv[1];
            printf("%f %f \n", testv[0], testv[1]);
            // printf("%f %f \n", p.pose.position.x, p.pose.position.y);
            pose.push_back(p); 
            // printf("%f %f \n", pose[index].pose.position.x, pose[index].pose.position.y);
        }

        for(int i = 0; i < pose.size(); i++)
        {
            while (ros::ok()) {
                while(ros::ok())
                {
                    ros::spinOnce();
                    if(_DroneState.armed && _DroneState.connected && _DroneState.landed)
                    break;
                    rate.sleep();
                }
                if(!_DroneState.connected)
                    break;
                
                if(fabs(local_pos.pose.position.x - pose[i].pose.position.x) < 1.0 &&
					fabs(local_pos.pose.position.y - pose[i].pose.position.y) < 1.0)
                    {
                        break;
                    }
                geometry_msgs::PoseStamped goal_get;
                goal_get.header.stamp = ros::Time::now();
                goal_get.header.frame_id = "map";
                goal_get.pose.position.x = pose[i].pose.position.x;
                goal_get.pose.position.y = pose[i].pose.position.y;
                goal_get.pose.position.z = 1;
                goal_get.pose.orientation.x = 0.0;
                goal_get.pose.orientation.y = 0.0;
                goal_get.pose.orientation.z = 0.0;
                goal_get.pose.orientation.w = 1.0;
                goal_pub.publish(goal_get);
            }
        }

        printf("waypoints upload over!\n");
        while(ros::ok())
        {
            ros::spinOnce();
            rate.sleep();
        }
    }
    return 0;
}
