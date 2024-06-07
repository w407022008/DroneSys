#include <position_controller.h>
#include <position_controller_params.h>

#include "autopilot/autopilot.h"

// template class autopilot::AutoPilot<position_controller::PositionController,
//                                     position_controller::PositionControllerParams>;

int main(int argc, char **argv) {
  ros::init(argc, argv, "autopilot");
  autopilot::AutoPilot<position_controller::PositionController,
                       position_controller::PositionControllerParams>
      autopilot;

  ros::spin();

  return 0;
}
