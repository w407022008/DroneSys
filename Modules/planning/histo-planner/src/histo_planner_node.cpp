#include <ros/ros.h>
#include <visualization_msgs/Marker.h>


#include "histo_planning.h"

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
}

using namespace Histo_Planning;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "histo_planner_node");
  ros::NodeHandle nh("~");

  Histo_Planner histo_planning;
  histo_planning.init(nh);

  ros::spin();

  return 0;
}
