#include <Eigen/Eigen>
#include <math.h>
#include <math_utils.h>
#include <drone_msgs/PositionReference.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>


using namespace std;

class Traj_gen
{
public:
    Traj_gen(void): Traj_gen_nh("~")
    {
        Traj_gen_nh.param<float>("Common/Center_x", center_x, 0.0);
        Traj_gen_nh.param<float>("Common/Center_y", center_y, 0.0);
        Traj_gen_nh.param<float>("Common/Center_z", center_z, 1.0);
        Traj_gen_nh.param<float>("Common/undulation", undulation, 1.0);
        Traj_gen_nh.param<float>("Common/radius", radius, 2.0);
        Traj_gen_nh.param<float>("Common/spin_rate", spin_rate, 0.0);
        Traj_gen_nh.param<float>("Common/spin_offset", spin_offset, 0.0);
        
        Traj_gen_nh.param<float>("Circle/circle_omega_1", circle_omega_1, 0.5);
        Traj_gen_nh.param<float>("Circle/circle_omega_2", circle_omega_2, 0.5);
        
        Traj_gen_nh.param<float>("Line/line_omega_1", line_omega_1, 0.5);
        Traj_gen_nh.param<float>("Line/line_omega_2", line_omega_2, 0.5);
        Traj_gen_nh.param<float>("Line/line_omega_3", line_omega_3, 0.5);
        Traj_gen_nh.param<float>("Line/yaw_offset", yaw_offset, 0.0);

        Traj_gen_nh.param<float>("Eight/eight_omega_1", eight_omega_1, 0.5);
        Traj_gen_nh.param<float>("Eight/eight_omega_2", eight_omega_2, 0.5);
        Traj_gen_nh.param<float>("Eight/eight_omega_3", eight_omega_3, 0.5);
        
        Traj_gen_nh.param<float>("Step/step_length", step_length, 0.0);
        Traj_gen_nh.param<float>("Step/step_interval", step_interval, 0.0);

    }

    //Printf the Traj_gen parameter
    void printf_param();

    //Traj_gen Calculation [Input: time_from_start; Output: Circle_trajectory;]
    drone_msgs::PositionReference Circle_trajectory_generation(float time_from_start);

    drone_msgs::PositionReference Eight_trajectory_generation(float time_from_start);

    drone_msgs::PositionReference Step_trajectory_generation(float time_from_start);

    drone_msgs::PositionReference Line_trajectory_generation(float time_from_start);

private:

    ros::NodeHandle Traj_gen_nh;
    // Common Parameter
    float center_x,center_y,center_z;
    float radius, undulation, spin_rate, spin_offset;

    // Circle Parameter
    float circle_omega_1, circle_omega_2;

    // Line Shape Parameter
    float line_omega_1, line_omega_2, line_omega_3, yaw_offset;

    // Eight Shape Parameter
    float eight_omega_1, eight_omega_2, eight_omega_3;
    
    // Step
    float step_length;
    float step_interval;
};


drone_msgs::PositionReference Traj_gen::Circle_trajectory_generation(float time_from_start)
{
    const float angle_1 = circle_omega_1 * time_from_start; //Angular velocity around a circle
    const float angle_2 = circle_omega_2 * time_from_start; //Angular velocity of undulation in z-axis
    const float cos_angle = cos(angle_1);
    const float sin_angle = sin(angle_1);

    drone_msgs::PositionReference Circle_trajectory;

    Circle_trajectory.header.stamp = ros::Time::now();

    Circle_trajectory.time_from_start = time_from_start;

    Circle_trajectory.Move_mode = drone_msgs::PositionReference::TRAJECTORY;

    Circle_trajectory.position_ref[0] = radius * cos_angle + center_x;
    Circle_trajectory.position_ref[1] = radius * sin_angle + center_y;
    Circle_trajectory.position_ref[2] = undulation * sin(angle_2) + center_z;

    Circle_trajectory.velocity_ref[0] = - radius * circle_omega_1 * sin_angle;
    Circle_trajectory.velocity_ref[1] = radius * circle_omega_1 * cos_angle;
    Circle_trajectory.velocity_ref[2] = undulation * circle_omega_2 * cos(angle_2);

    Circle_trajectory.yaw_ref = spin_offset + spin_rate * time_from_start;

    return Circle_trajectory;
}

drone_msgs::PositionReference Traj_gen::Line_trajectory_generation(float time_from_start)
{
    const float angle_1 = line_omega_1 * time_from_start; //Angular velocity mapping to linear velocity
    const float angle_2 = line_omega_2 * time_from_start; //Angular velocity of undulation in z-axis
    const float angle_3 = yaw_offset + line_omega_3 * time_from_start; //Rotational speed around z-axis
    const float cos_angle = cos(angle_1);
    const float sin_angle = sin(angle_1);

    drone_msgs::PositionReference Line_trajectory;

    Line_trajectory.header.stamp = ros::Time::now();

    Line_trajectory.time_from_start = time_from_start;

    Line_trajectory.Move_mode = drone_msgs::PositionReference::TRAJECTORY;
    
    Line_trajectory.position_ref[0] = cos(angle_3) * radius * sin_angle + center_x;
    Line_trajectory.position_ref[1] = sin(angle_3) * radius * sin_angle + center_y;
    Line_trajectory.position_ref[2] = undulation * sin(angle_2) + center_z;

    Line_trajectory.velocity_ref[0] = -sin(angle_3) * radius * sin_angle * line_omega_3  + cos(angle_3) * radius * cos_angle * line_omega_1;
    Line_trajectory.velocity_ref[1] = cos(angle_3) * radius * sin_angle * line_omega_3 + sin(angle_3) * radius * cos_angle * line_omega_1;
    Line_trajectory.velocity_ref[2] = undulation * cos(angle_2) * line_omega_2;

    Line_trajectory.yaw_ref = spin_offset + spin_rate * time_from_start;

    return Line_trajectory;
}


drone_msgs::PositionReference Traj_gen::Eight_trajectory_generation(float time_from_start)
{
    Eigen::Vector3f position;
    Eigen::Vector3f velocity;
    Eigen::Vector3f acceleration;
    
    const float angle_1 = eight_omega_1 * time_from_start; //Angular velocity around the figure of eight
    const float angle_2 = eight_omega_2 * time_from_start; //Angular velocity of undulation in z-axis
    const float angle_3 = eight_omega_3 * time_from_start; //Rotational speed around z-axis
    const float cos_angle = cos(angle_1);
    const float sin_angle = sin(angle_1);
    
    Eigen::Vector3f axis_1 = {1.0, 0.0, 0.0};
    Eigen::Vector3f axis_2 = {0.0, 1.0, 0.0};
    Eigen::Vector3f axis_3 = {0.0, 0.0, 1.0};
    Eigen::Matrix3f R, Omega;
    R << cos(angle_3), -sin(angle_3), 0.0,
    	sin(angle_3), cos(angle_3), 0.0,
    	0.0,			0.0,		1.0;
    Omega << -sin(angle_3), -cos(angle_3), 0.0,
    		cos(angle_3), -sin(angle_3), 0.0,
    		0.0,			0.0,		0.0;

    Eigen::Vector3f origin_{center_x,center_y,center_z};
    position = origin_ + undulation * sin(angle_2) * axis_3 +
    			R * radius/1.25 * (sin_angle * axis_1 + 2 * sin_angle * cos_angle * axis_2);

    velocity = R * (radius/1.25 * eight_omega_1 * (cos_angle * axis_1 + 2 * (cos_angle*cos_angle - sin_angle*sin_angle) * axis_2) + undulation * eight_omega_2 * cos(angle_2) * axis_3)
    		+ eight_omega_3 * Omega * (radius/1.25 * (sin_angle * axis_1 + 2 * sin_angle * cos_angle * axis_2) + undulation * sin(angle_2) * axis_3);

    acceleration << 0.0, 0.0, 0.0;

    drone_msgs::PositionReference Eight_trajectory;

    Eight_trajectory.header.stamp = ros::Time::now();

    Eight_trajectory.time_from_start = time_from_start;

    Eight_trajectory.Move_mode = drone_msgs::PositionReference::TRAJECTORY;

    Eight_trajectory.position_ref[0] = position[0];
    Eight_trajectory.position_ref[1] = position[1];
    Eight_trajectory.position_ref[2] = position[2];

    Eight_trajectory.velocity_ref[0] = velocity[0];
    Eight_trajectory.velocity_ref[1] = velocity[1];
    Eight_trajectory.velocity_ref[2] = velocity[2];

    Eight_trajectory.yaw_ref = spin_offset + spin_rate * time_from_start;

    return Eight_trajectory;
}


drone_msgs::PositionReference Traj_gen::Step_trajectory_generation(float time_from_start)
{
    drone_msgs::PositionReference Step_trajectory;

    Step_trajectory.header.stamp = ros::Time::now();

    Step_trajectory.time_from_start = time_from_start;

    Step_trajectory.Move_mode = drone_msgs::PositionReference::TRAJECTORY;

    int i = time_from_start / step_interval;

    if( i%2 == 0)
    {
        Step_trajectory.position_ref[0] = step_length;
    }else 
    {
        Step_trajectory.position_ref[0] = - step_length;
    }

    Step_trajectory.position_ref[1] = 0;
    Step_trajectory.position_ref[2] = 1.0;

    Step_trajectory.velocity_ref[0] = 0;
    Step_trajectory.velocity_ref[1] = 0;
    Step_trajectory.velocity_ref[2] = 0;

    Step_trajectory.yaw_ref = spin_offset + spin_rate * time_from_start;

    return Step_trajectory;
}

void Traj_gen::printf_param()
{
    cout <<">>>>>>>>>>>>>>>>>>>>>>>>>Traj_gen Parameter <<<<<<<<<<<<<<<<<<<<<<" <<endl;
    cout <<"Circle Shape:  " <<endl;
    cout <<"circle_center :  " << center_x <<" [m] "<< center_y <<" [m] "<< center_z <<" [m] "<<endl;
    cout <<"circle_radius :  "<< radius <<" [m] " <<"linear_vel : "<< circle_omega_1 * radius <<" [m/s] " << " undulation : " << undulation << endl;

    cout <<"Eight Shape:  " <<endl;
    cout <<"eight_origin_ :  "<< center_x <<" [m] "<< center_y <<" [m] "<< center_z <<" [m] "<<endl;
    cout <<"eight_omega_ :  "<< eight_omega_1  <<" [rad/s] " << eight_omega_2  <<" [rad/s] " << eight_omega_3  <<" [rad/s] " <<endl;
    cout <<"radius : "<< radius << " undulation : " << undulation << endl;

    cout <<"Step:  " <<endl;
    cout <<"step_length :  "<< step_length << " [m] step_interval : "<< step_interval << " [s] "<<endl;
}
