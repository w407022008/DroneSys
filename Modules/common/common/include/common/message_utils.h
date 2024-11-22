#ifndef MESSAGE_UTILS_H
#define MESSAGE_UTILS_H

#include <string>
#include <drone_msgs/Message.h>
using namespace std;

inline void pub_message(ros::Publisher& puber, int msg_type, std::string source_node, std::string msg_content)
{
    drone_msgs::Message exect_msg;
    exect_msg.header.stamp = ros::Time::now();
    exect_msg.message_type = msg_type;
    exect_msg.source_node = source_node;
    exect_msg.content = msg_content;
    puber.publish(exect_msg);
}

inline void printf_message(const drone_msgs::Message& message)
{
    //cout <<">>>>>>>>>>>>>>>>>>>>>>>> Message <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;
    if(message.message_type == drone_msgs::Message::NORMAL)
    {
        cout << "[NORMAL]" << "["<< message.source_node << "]:" << message.content <<endl;
    }else if(message.message_type == drone_msgs::Message::WARN)
    {
        cout << "[WARN]" << "["<< message.source_node << "]:" <<message.content <<endl;
    }else if(message.message_type == drone_msgs::Message::ERROR)
    {
        cout << "[ERROR]" << "["<< message.source_node << "]:" << message.content <<endl;
    }
    
}

#endif
