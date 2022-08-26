#include <ros/ros.h>
#include "message_utils.h"

using namespace std;
void msg_cb(const drone_msgs::Message::ConstPtr& msg)
{
    drone_msgs::Message message = *msg;
    printf_message(message);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ground_station_msg");
    ros::NodeHandle nh("~");

    ros::Subscriber message_main_sub = nh.subscribe<drone_msgs::Message>("/drone_msg/message", 10, msg_cb);

    ros::Rate rate(1.0);

    cout <<"=======================================================================" <<endl;
    cout <<"===>>>>>>>>>>>>>>>>> Ground Station Message  <<<<<<<<<<<<<<<<<<<<<<<===" <<endl;
    cout <<"=======================================================================" <<endl;
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Main Loop<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    while(ros::ok())
    {
        ros::spinOnce();

        rate.sleep();
    }

    return 0;

}
