/****************************************************************************
 *
 *   Copyright (C) 2013-2022 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

#include <uORB/topics/uORBTopics.hpp>
#include <uORB/uORB.h>
#include <uORB/topics/action_request.h>
#include <uORB/topics/actuator_armed.h>
#include <uORB/topics/actuator_controls_status.h>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/actuator_outputs.h>
#include <uORB/topics/actuator_servos.h>
#include <uORB/topics/actuator_servos_trim.h>
#include <uORB/topics/actuator_test.h>
#include <uORB/topics/adc_report.h>
#include <uORB/topics/airspeed.h>
#include <uORB/topics/airspeed_validated.h>
#include <uORB/topics/airspeed_wind.h>
#include <uORB/topics/autotune_attitude_control_status.h>
#include <uORB/topics/battery_status.h>
#include <uORB/topics/button_event.h>
#include <uORB/topics/camera_capture.h>
#include <uORB/topics/camera_status.h>
#include <uORB/topics/camera_trigger.h>
#include <uORB/topics/cellular_status.h>
#include <uORB/topics/collision_constraints.h>
#include <uORB/topics/collision_report.h>
#include <uORB/topics/control_allocator_status.h>
#include <uORB/topics/cpuload.h>
#include <uORB/topics/debug_array.h>
#include <uORB/topics/debug_key_value.h>
#include <uORB/topics/debug_value.h>
#include <uORB/topics/debug_vect.h>
#include <uORB/topics/differential_pressure.h>
#include <uORB/topics/distance_sensor.h>
#include <uORB/topics/ekf2_timestamps.h>
#include <uORB/topics/esc_report.h>
#include <uORB/topics/esc_status.h>
#include <uORB/topics/estimator_aid_source1d.h>
#include <uORB/topics/estimator_aid_source2d.h>
#include <uORB/topics/estimator_aid_source3d.h>
#include <uORB/topics/estimator_bias.h>
#include <uORB/topics/estimator_bias3d.h>
#include <uORB/topics/estimator_event_flags.h>
#include <uORB/topics/estimator_gps_status.h>
#include <uORB/topics/estimator_innovations.h>
#include <uORB/topics/estimator_selector_status.h>
#include <uORB/topics/estimator_sensor_bias.h>
#include <uORB/topics/estimator_states.h>
#include <uORB/topics/estimator_status.h>
#include <uORB/topics/estimator_status_flags.h>
#include <uORB/topics/event.h>
#include <uORB/topics/failsafe_flags.h>
#include <uORB/topics/failure_detector_status.h>
#include <uORB/topics/follow_target.h>
#include <uORB/topics/follow_target_estimator.h>
#include <uORB/topics/follow_target_status.h>
#include <uORB/topics/generator_status.h>
#include <uORB/topics/geofence_result.h>
#include <uORB/topics/gimbal_controls.h>
#include <uORB/topics/gimbal_device_attitude_status.h>
#include <uORB/topics/gimbal_device_information.h>
#include <uORB/topics/gimbal_device_set_attitude.h>
#include <uORB/topics/gimbal_manager_information.h>
#include <uORB/topics/gimbal_manager_set_attitude.h>
#include <uORB/topics/gimbal_manager_set_manual_control.h>
#include <uORB/topics/gimbal_manager_status.h>
#include <uORB/topics/gps_dump.h>
#include <uORB/topics/gps_inject_data.h>
#include <uORB/topics/gripper.h>
#include <uORB/topics/health_report.h>
#include <uORB/topics/heater_status.h>
#include <uORB/topics/home_position.h>
#include <uORB/topics/hover_thrust_estimate.h>
#include <uORB/topics/input_rc.h>
#include <uORB/topics/internal_combustion_engine_status.h>
#include <uORB/topics/iridiumsbd_status.h>
#include <uORB/topics/irlock_report.h>
#include <uORB/topics/landing_gear.h>
#include <uORB/topics/landing_gear_wheel.h>
#include <uORB/topics/landing_target_innovations.h>
#include <uORB/topics/landing_target_pose.h>
#include <uORB/topics/launch_detection_status.h>
#include <uORB/topics/led_control.h>
#include <uORB/topics/log_message.h>
#include <uORB/topics/logger_status.h>
#include <uORB/topics/mag_worker_data.h>
#include <uORB/topics/magnetometer_bias_estimate.h>
#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/manual_control_switches.h>
#include <uORB/topics/mavlink_log.h>
#include <uORB/topics/mavlink_tunnel.h>
#include <uORB/topics/mission.h>
#include <uORB/topics/mission_result.h>
#include <uORB/topics/mode_completed.h>
#include <uORB/topics/mount_orientation.h>
#include <uORB/topics/navigator_mission_item.h>
#include <uORB/topics/normalized_unsigned_setpoint.h>
#include <uORB/topics/npfg_status.h>
#include <uORB/topics/obstacle_distance.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/onboard_computer_status.h>
#include <uORB/topics/orb_test.h>
#include <uORB/topics/orb_test_large.h>
#include <uORB/topics/orb_test_medium.h>
#include <uORB/topics/orbit_status.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/ping.h>
#include <uORB/topics/position_controller_landing_status.h>
#include <uORB/topics/position_controller_status.h>
#include <uORB/topics/position_setpoint.h>
#include <uORB/topics/position_setpoint_triplet.h>
#include <uORB/topics/power_button_state.h>
#include <uORB/topics/power_monitor.h>
#include <uORB/topics/pps_capture.h>
#include <uORB/topics/pwm_input.h>
#include <uORB/topics/px4io_status.h>
#include <uORB/topics/qshell_req.h>
#include <uORB/topics/qshell_retval.h>
#include <uORB/topics/radio_status.h>
#include <uORB/topics/rate_ctrl_status.h>
#include <uORB/topics/rc_channels.h>
#include <uORB/topics/rc_parameter_map.h>
#include <uORB/topics/rpm.h>
#include <uORB/topics/rtl_time_estimate.h>
#include <uORB/topics/satellite_info.h>
#include <uORB/topics/sensor_accel.h>
#include <uORB/topics/sensor_accel_fifo.h>
#include <uORB/topics/sensor_baro.h>
#include <uORB/topics/sensor_combined.h>
#include <uORB/topics/sensor_correction.h>
#include <uORB/topics/sensor_gnss_relative.h>
#include <uORB/topics/sensor_gps.h>
#include <uORB/topics/sensor_gyro.h>
#include <uORB/topics/sensor_gyro_fft.h>
#include <uORB/topics/sensor_gyro_fifo.h>
#include <uORB/topics/sensor_hygrometer.h>
#include <uORB/topics/sensor_mag.h>
#include <uORB/topics/sensor_optical_flow.h>
#include <uORB/topics/sensor_preflight_mag.h>
#include <uORB/topics/sensor_selection.h>
#include <uORB/topics/sensor_uwb.h>
#include <uORB/topics/sensors_status.h>
#include <uORB/topics/sensors_status_imu.h>
#include <uORB/topics/system_power.h>
#include <uORB/topics/takeoff_status.h>
#include <uORB/topics/task_stack_info.h>
#include <uORB/topics/tecs_status.h>
#include <uORB/topics/telemetry_status.h>
#include <uORB/topics/tiltrotor_extra_controls.h>
#include <uORB/topics/timesync_status.h>
#include <uORB/topics/trajectory_bezier.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/trajectory_waypoint.h>
#include <uORB/topics/transponder_report.h>
#include <uORB/topics/tune_control.h>
#include <uORB/topics/uavcan_parameter_request.h>
#include <uORB/topics/uavcan_parameter_value.h>
#include <uORB/topics/ulog_stream.h>
#include <uORB/topics/ulog_stream_ack.h>
#include <uORB/topics/vehicle_acceleration.h>
#include <uORB/topics/vehicle_air_data.h>
#include <uORB/topics/vehicle_angular_acceleration_setpoint.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_global_position.h>
#include <uORB/topics/vehicle_imu.h>
#include <uORB/topics/vehicle_imu_status.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>
#include <uORB/topics/vehicle_magnetometer.h>
#include <uORB/topics/vehicle_odometry.h>
#include <uORB/topics/vehicle_optical_flow.h>
#include <uORB/topics/vehicle_optical_flow_vel.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_roi.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_thrust_setpoint.h>
#include <uORB/topics/vehicle_torque_setpoint.h>
#include <uORB/topics/vehicle_trajectory_bezier.h>
#include <uORB/topics/vehicle_trajectory_waypoint.h>
#include <uORB/topics/vtol_vehicle_status.h>
#include <uORB/topics/wind.h>
#include <uORB/topics/windspeed.h>
#include <uORB/topics/yaw_estimator_status.h>


const constexpr struct orb_metadata *const uorb_topics_list[ORB_TOPICS_COUNT] = {
	ORB_ID(action_request), 
	ORB_ID(actuator_armed), 
	ORB_ID(actuator_controls_status_0), 
	ORB_ID(actuator_controls_status_1), 
	ORB_ID(actuator_motors), 
	ORB_ID(actuator_outputs), 
	ORB_ID(actuator_outputs_debug), 
	ORB_ID(actuator_outputs_sim), 
	ORB_ID(actuator_servos), 
	ORB_ID(actuator_servos_trim), 
	ORB_ID(actuator_test), 
	ORB_ID(adc_report), 
	ORB_ID(airspeed), 
	ORB_ID(airspeed_validated), 
	ORB_ID(airspeed_wind), 
	ORB_ID(autotune_attitude_control_status), 
	ORB_ID(battery_status), 
	ORB_ID(button_event), 
	ORB_ID(camera_capture), 
	ORB_ID(camera_status), 
	ORB_ID(camera_trigger), 
	ORB_ID(cellular_status), 
	ORB_ID(collision_constraints), 
	ORB_ID(collision_report), 
	ORB_ID(control_allocator_status), 
	ORB_ID(cpuload), 
	ORB_ID(debug_array), 
	ORB_ID(debug_key_value), 
	ORB_ID(debug_value), 
	ORB_ID(debug_vect), 
	ORB_ID(differential_pressure), 
	ORB_ID(distance_sensor), 
	ORB_ID(ekf2_timestamps), 
	ORB_ID(esc_report), 
	ORB_ID(esc_status), 
	ORB_ID(estimator_aid_src_airspeed), 
	ORB_ID(estimator_aid_src_aux_vel), 
	ORB_ID(estimator_aid_src_baro_hgt), 
	ORB_ID(estimator_aid_src_ev_hgt), 
	ORB_ID(estimator_aid_src_ev_pos), 
	ORB_ID(estimator_aid_src_ev_vel), 
	ORB_ID(estimator_aid_src_ev_yaw), 
	ORB_ID(estimator_aid_src_fake_hgt), 
	ORB_ID(estimator_aid_src_fake_pos), 
	ORB_ID(estimator_aid_src_gnss_hgt), 
	ORB_ID(estimator_aid_src_gnss_pos), 
	ORB_ID(estimator_aid_src_gnss_vel), 
	ORB_ID(estimator_aid_src_gnss_yaw), 
	ORB_ID(estimator_aid_src_gravity), 
	ORB_ID(estimator_aid_src_mag), 
	ORB_ID(estimator_aid_src_mag_heading), 
	ORB_ID(estimator_aid_src_optical_flow), 
	ORB_ID(estimator_aid_src_rng_hgt), 
	ORB_ID(estimator_aid_src_sideslip), 
	ORB_ID(estimator_aid_src_terrain_optical_flow), 
	ORB_ID(estimator_attitude), 
	ORB_ID(estimator_baro_bias), 
	ORB_ID(estimator_bias3d), 
	ORB_ID(estimator_ev_pos_bias), 
	ORB_ID(estimator_event_flags), 
	ORB_ID(estimator_global_position), 
	ORB_ID(estimator_gnss_hgt_bias), 
	ORB_ID(estimator_gps_status), 
	ORB_ID(estimator_innovation_test_ratios), 
	ORB_ID(estimator_innovation_variances), 
	ORB_ID(estimator_innovations), 
	ORB_ID(estimator_local_position), 
	ORB_ID(estimator_odometry), 
	ORB_ID(estimator_optical_flow_vel), 
	ORB_ID(estimator_rng_hgt_bias), 
	ORB_ID(estimator_selector_status), 
	ORB_ID(estimator_sensor_bias), 
	ORB_ID(estimator_states), 
	ORB_ID(estimator_status), 
	ORB_ID(estimator_status_flags), 
	ORB_ID(estimator_wind), 
	ORB_ID(event), 
	ORB_ID(external_ins_attitude), 
	ORB_ID(external_ins_global_position), 
	ORB_ID(external_ins_local_position), 
	ORB_ID(failsafe_flags), 
	ORB_ID(failure_detector_status), 
	ORB_ID(flaps_setpoint), 
	ORB_ID(follow_target), 
	ORB_ID(follow_target_estimator), 
	ORB_ID(follow_target_status), 
	ORB_ID(fw_virtual_attitude_setpoint), 
	ORB_ID(generator_status), 
	ORB_ID(geofence_result), 
	ORB_ID(gimbal_controls), 
	ORB_ID(gimbal_device_attitude_status), 
	ORB_ID(gimbal_device_information), 
	ORB_ID(gimbal_device_set_attitude), 
	ORB_ID(gimbal_manager_information), 
	ORB_ID(gimbal_manager_set_attitude), 
	ORB_ID(gimbal_manager_set_manual_control), 
	ORB_ID(gimbal_manager_status), 
	ORB_ID(gimbal_v1_command), 
	ORB_ID(gps_dump), 
	ORB_ID(gps_inject_data), 
	ORB_ID(gripper), 
	ORB_ID(health_report), 
	ORB_ID(heater_status), 
	ORB_ID(home_position), 
	ORB_ID(hover_thrust_estimate), 
	ORB_ID(input_rc), 
	ORB_ID(internal_combustion_engine_status), 
	ORB_ID(iridiumsbd_status), 
	ORB_ID(irlock_report), 
	ORB_ID(landing_gear), 
	ORB_ID(landing_gear_wheel), 
	ORB_ID(landing_target_innovations), 
	ORB_ID(landing_target_pose), 
	ORB_ID(launch_detection_status), 
	ORB_ID(led_control), 
	ORB_ID(log_message), 
	ORB_ID(logger_status), 
	ORB_ID(mag_worker_data), 
	ORB_ID(magnetometer_bias_estimate), 
	ORB_ID(manual_control_input), 
	ORB_ID(manual_control_setpoint), 
	ORB_ID(manual_control_switches), 
	ORB_ID(mavlink_log), 
	ORB_ID(mavlink_tunnel), 
	ORB_ID(mc_virtual_attitude_setpoint), 
	ORB_ID(mission), 
	ORB_ID(mission_result), 
	ORB_ID(mode_completed), 
	ORB_ID(mount_orientation), 
	ORB_ID(navigator_mission_item), 
	ORB_ID(npfg_status), 
	ORB_ID(obstacle_distance), 
	ORB_ID(obstacle_distance_fused), 
	ORB_ID(offboard_control_mode), 
	ORB_ID(onboard_computer_status), 
	ORB_ID(orb_multitest), 
	ORB_ID(orb_test), 
	ORB_ID(orb_test_large), 
	ORB_ID(orb_test_medium), 
	ORB_ID(orb_test_medium_multi), 
	ORB_ID(orb_test_medium_queue), 
	ORB_ID(orb_test_medium_queue_poll), 
	ORB_ID(orb_test_medium_wrap_around), 
	ORB_ID(orbit_status), 
	ORB_ID(parameter_update), 
	ORB_ID(ping), 
	ORB_ID(position_controller_landing_status), 
	ORB_ID(position_controller_status), 
	ORB_ID(position_setpoint), 
	ORB_ID(position_setpoint_triplet), 
	ORB_ID(power_button_state), 
	ORB_ID(power_monitor), 
	ORB_ID(pps_capture), 
	ORB_ID(pwm_input), 
	ORB_ID(px4io_status), 
	ORB_ID(qshell_req), 
	ORB_ID(qshell_retval), 
	ORB_ID(radio_status), 
	ORB_ID(rate_ctrl_status), 
	ORB_ID(rc_channels), 
	ORB_ID(rc_parameter_map), 
	ORB_ID(rpm), 
	ORB_ID(rtl_time_estimate), 
	ORB_ID(safety_button), 
	ORB_ID(satellite_info), 
	ORB_ID(sensor_accel), 
	ORB_ID(sensor_accel_fifo), 
	ORB_ID(sensor_baro), 
	ORB_ID(sensor_combined), 
	ORB_ID(sensor_correction), 
	ORB_ID(sensor_gnss_relative), 
	ORB_ID(sensor_gps), 
	ORB_ID(sensor_gyro), 
	ORB_ID(sensor_gyro_fft), 
	ORB_ID(sensor_gyro_fifo), 
	ORB_ID(sensor_hygrometer), 
	ORB_ID(sensor_mag), 
	ORB_ID(sensor_optical_flow), 
	ORB_ID(sensor_preflight_mag), 
	ORB_ID(sensor_selection), 
	ORB_ID(sensor_uwb), 
	ORB_ID(sensors_status_baro), 
	ORB_ID(sensors_status_imu), 
	ORB_ID(sensors_status_mag), 
	ORB_ID(spoilers_setpoint), 
	ORB_ID(system_power), 
	ORB_ID(takeoff_status), 
	ORB_ID(task_stack_info), 
	ORB_ID(tecs_status), 
	ORB_ID(telemetry_status), 
	ORB_ID(tiltrotor_extra_controls), 
	ORB_ID(timesync_status), 
	ORB_ID(trajectory_bezier), 
	ORB_ID(trajectory_setpoint), 
	ORB_ID(trajectory_waypoint), 
	ORB_ID(transponder_report), 
	ORB_ID(tune_control), 
	ORB_ID(uavcan_parameter_request), 
	ORB_ID(uavcan_parameter_value), 
	ORB_ID(ulog_stream), 
	ORB_ID(ulog_stream_ack), 
	ORB_ID(vehicle_acceleration), 
	ORB_ID(vehicle_air_data), 
	ORB_ID(vehicle_angular_acceleration_setpoint), 
	ORB_ID(vehicle_angular_velocity), 
	ORB_ID(vehicle_angular_velocity_groundtruth), 
	ORB_ID(vehicle_attitude), 
	ORB_ID(vehicle_attitude_groundtruth), 
	ORB_ID(vehicle_attitude_setpoint), 
	ORB_ID(vehicle_command), 
	ORB_ID(vehicle_command_ack), 
	ORB_ID(vehicle_constraints), 
	ORB_ID(vehicle_control_mode), 
	ORB_ID(vehicle_global_position), 
	ORB_ID(vehicle_global_position_groundtruth), 
	ORB_ID(vehicle_gps_position), 
	ORB_ID(vehicle_imu), 
	ORB_ID(vehicle_imu_status), 
	ORB_ID(vehicle_land_detected), 
	ORB_ID(vehicle_local_position), 
	ORB_ID(vehicle_local_position_groundtruth), 
	ORB_ID(vehicle_local_position_setpoint), 
	ORB_ID(vehicle_magnetometer), 
	ORB_ID(vehicle_mocap_odometry), 
	ORB_ID(vehicle_odometry), 
	ORB_ID(vehicle_optical_flow), 
	ORB_ID(vehicle_optical_flow_vel), 
	ORB_ID(vehicle_rates_setpoint), 
	ORB_ID(vehicle_roi), 
	ORB_ID(vehicle_status), 
	ORB_ID(vehicle_thrust_setpoint), 
	ORB_ID(vehicle_thrust_setpoint_virtual_fw), 
	ORB_ID(vehicle_thrust_setpoint_virtual_mc), 
	ORB_ID(vehicle_torque_setpoint), 
	ORB_ID(vehicle_torque_setpoint_virtual_fw), 
	ORB_ID(vehicle_torque_setpoint_virtual_mc), 
	ORB_ID(vehicle_trajectory_bezier), 
	ORB_ID(vehicle_trajectory_waypoint), 
	ORB_ID(vehicle_trajectory_waypoint_desired), 
	ORB_ID(vehicle_visual_odometry), 
	ORB_ID(vtol_vehicle_status), 
	ORB_ID(wind), 
	ORB_ID(windspeed), 
	ORB_ID(yaw_estimator_status), 

};

const struct orb_metadata *const *orb_get_topics()
{
	return uorb_topics_list;
}

const struct orb_metadata *get_orb_meta(ORB_ID id)
{
	if (id == ORB_ID::INVALID) {
		return nullptr;
	}

	return uorb_topics_list[static_cast<uint8_t>(id)];
}
