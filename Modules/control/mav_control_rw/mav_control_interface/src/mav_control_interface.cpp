/*
 * Copyright (c) 2015, Markus Achtelik, ASL, ETH Zurich, Switzerland
 * You can contact the author at <markus dot achtelik at mavt dot ethz dot ch>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mav_control_interface/mav_control_interface.h>

#include "mav_control_interface_impl.h"

namespace mav_control_interface {

MavControlInterface::MavControlInterface(ros::NodeHandle& nh, ros::NodeHandle& private_nh,
                                         std::shared_ptr<PositionControllerInterface> controller,
                                         std::shared_ptr<RcInterfaceBase> rc_interface)
{
  mav_control_interface_impl_.reset(new MavControlInterfaceImpl(nh, private_nh, controller, rc_interface));
}

MavControlInterface::~MavControlInterface()
{
}

} /* namespace mav_control_interface */
