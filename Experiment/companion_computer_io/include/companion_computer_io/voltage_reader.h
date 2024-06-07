#pragma once

#include <ros/ros.h>

#include "companion_computer_io/adc.h"

namespace companion_computer_io
{

class VoltageReader
{
public:
  VoltageReader();
  ~VoltageReader();

private:

  void readVoltage(const ros::TimerEvent& time);
  bool loadParameters();

  ros::NodeHandle nh_;

  ros::Publisher voltage_pub_;
  ros::Timer read_battery_timer_;

  ADCReader adc_reader_;

  double max_adc_value_;
  double max_adc_voltage_;

  // Parameters
  std::string board_name_;
  int adc_id_;
  double read_voltage_frequency_;
  double voltage_divider_upper_res_;
  double voltage_divider_lower_res_;
};

} // namespace companion_computer_io
