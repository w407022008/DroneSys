#include <ros/ros.h>
#include <exploration_manager/fast_exploration_fsm.h>
#include <csignal>
#include <iostream>
#include <execinfo.h>
#include <cstdlib>
#include <plan_manage/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace fast_planner;

void signalHandler(int signal){
  void *array[10];
  size_t size;

  size = backtrace(array,10);
  std::cerr << "Error: signal "<<signal<<":\n";
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char** argv) {
  signal(SIGSEGV, signalHandler);
  signal(SIGINT,signalHandler);
  ros::init(argc, argv, "exploration_node");
  ros::NodeHandle nh("~");

  FastExplorationFSM expl_fsm;
  expl_fsm.init(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
