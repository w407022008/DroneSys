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


// auto-generated file

#pragma once

#include <ucdr/microcdr.h>
#include <string.h>
#include <uORB/topics/failsafe_flags.h>


static inline constexpr int ucdr_topic_size_failsafe_flags()
{
	return 85;
}

bool ucdr_serialize_failsafe_flags(const failsafe_flags_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.mode_req_angular_velocity) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_angular_velocity, sizeof(topic.mode_req_angular_velocity));
	buf.iterator += sizeof(topic.mode_req_angular_velocity);
	buf.offset += sizeof(topic.mode_req_angular_velocity);
	static_assert(sizeof(topic.mode_req_attitude) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_attitude, sizeof(topic.mode_req_attitude));
	buf.iterator += sizeof(topic.mode_req_attitude);
	buf.offset += sizeof(topic.mode_req_attitude);
	static_assert(sizeof(topic.mode_req_local_alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_local_alt, sizeof(topic.mode_req_local_alt));
	buf.iterator += sizeof(topic.mode_req_local_alt);
	buf.offset += sizeof(topic.mode_req_local_alt);
	static_assert(sizeof(topic.mode_req_local_position) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_local_position, sizeof(topic.mode_req_local_position));
	buf.iterator += sizeof(topic.mode_req_local_position);
	buf.offset += sizeof(topic.mode_req_local_position);
	static_assert(sizeof(topic.mode_req_local_position_relaxed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_local_position_relaxed, sizeof(topic.mode_req_local_position_relaxed));
	buf.iterator += sizeof(topic.mode_req_local_position_relaxed);
	buf.offset += sizeof(topic.mode_req_local_position_relaxed);
	static_assert(sizeof(topic.mode_req_global_position) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_global_position, sizeof(topic.mode_req_global_position));
	buf.iterator += sizeof(topic.mode_req_global_position);
	buf.offset += sizeof(topic.mode_req_global_position);
	static_assert(sizeof(topic.mode_req_mission) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_mission, sizeof(topic.mode_req_mission));
	buf.iterator += sizeof(topic.mode_req_mission);
	buf.offset += sizeof(topic.mode_req_mission);
	static_assert(sizeof(topic.mode_req_offboard_signal) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_offboard_signal, sizeof(topic.mode_req_offboard_signal));
	buf.iterator += sizeof(topic.mode_req_offboard_signal);
	buf.offset += sizeof(topic.mode_req_offboard_signal);
	static_assert(sizeof(topic.mode_req_home_position) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_home_position, sizeof(topic.mode_req_home_position));
	buf.iterator += sizeof(topic.mode_req_home_position);
	buf.offset += sizeof(topic.mode_req_home_position);
	static_assert(sizeof(topic.mode_req_wind_and_flight_time_compliance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_wind_and_flight_time_compliance, sizeof(topic.mode_req_wind_and_flight_time_compliance));
	buf.iterator += sizeof(topic.mode_req_wind_and_flight_time_compliance);
	buf.offset += sizeof(topic.mode_req_wind_and_flight_time_compliance);
	static_assert(sizeof(topic.mode_req_prevent_arming) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_prevent_arming, sizeof(topic.mode_req_prevent_arming));
	buf.iterator += sizeof(topic.mode_req_prevent_arming);
	buf.offset += sizeof(topic.mode_req_prevent_arming);
	static_assert(sizeof(topic.mode_req_manual_control) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_manual_control, sizeof(topic.mode_req_manual_control));
	buf.iterator += sizeof(topic.mode_req_manual_control);
	buf.offset += sizeof(topic.mode_req_manual_control);
	static_assert(sizeof(topic.mode_req_other) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mode_req_other, sizeof(topic.mode_req_other));
	buf.iterator += sizeof(topic.mode_req_other);
	buf.offset += sizeof(topic.mode_req_other);
	static_assert(sizeof(topic.angular_velocity_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.angular_velocity_invalid, sizeof(topic.angular_velocity_invalid));
	buf.iterator += sizeof(topic.angular_velocity_invalid);
	buf.offset += sizeof(topic.angular_velocity_invalid);
	static_assert(sizeof(topic.attitude_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.attitude_invalid, sizeof(topic.attitude_invalid));
	buf.iterator += sizeof(topic.attitude_invalid);
	buf.offset += sizeof(topic.attitude_invalid);
	static_assert(sizeof(topic.local_altitude_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.local_altitude_invalid, sizeof(topic.local_altitude_invalid));
	buf.iterator += sizeof(topic.local_altitude_invalid);
	buf.offset += sizeof(topic.local_altitude_invalid);
	static_assert(sizeof(topic.local_position_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.local_position_invalid, sizeof(topic.local_position_invalid));
	buf.iterator += sizeof(topic.local_position_invalid);
	buf.offset += sizeof(topic.local_position_invalid);
	static_assert(sizeof(topic.local_position_invalid_relaxed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.local_position_invalid_relaxed, sizeof(topic.local_position_invalid_relaxed));
	buf.iterator += sizeof(topic.local_position_invalid_relaxed);
	buf.offset += sizeof(topic.local_position_invalid_relaxed);
	static_assert(sizeof(topic.local_velocity_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.local_velocity_invalid, sizeof(topic.local_velocity_invalid));
	buf.iterator += sizeof(topic.local_velocity_invalid);
	buf.offset += sizeof(topic.local_velocity_invalid);
	static_assert(sizeof(topic.global_position_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.global_position_invalid, sizeof(topic.global_position_invalid));
	buf.iterator += sizeof(topic.global_position_invalid);
	buf.offset += sizeof(topic.global_position_invalid);
	static_assert(sizeof(topic.auto_mission_missing) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.auto_mission_missing, sizeof(topic.auto_mission_missing));
	buf.iterator += sizeof(topic.auto_mission_missing);
	buf.offset += sizeof(topic.auto_mission_missing);
	static_assert(sizeof(topic.offboard_control_signal_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.offboard_control_signal_lost, sizeof(topic.offboard_control_signal_lost));
	buf.iterator += sizeof(topic.offboard_control_signal_lost);
	buf.offset += sizeof(topic.offboard_control_signal_lost);
	static_assert(sizeof(topic.home_position_invalid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.home_position_invalid, sizeof(topic.home_position_invalid));
	buf.iterator += sizeof(topic.home_position_invalid);
	buf.offset += sizeof(topic.home_position_invalid);
	static_assert(sizeof(topic.manual_control_signal_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.manual_control_signal_lost, sizeof(topic.manual_control_signal_lost));
	buf.iterator += sizeof(topic.manual_control_signal_lost);
	buf.offset += sizeof(topic.manual_control_signal_lost);
	static_assert(sizeof(topic.gcs_connection_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gcs_connection_lost, sizeof(topic.gcs_connection_lost));
	buf.iterator += sizeof(topic.gcs_connection_lost);
	buf.offset += sizeof(topic.gcs_connection_lost);
	static_assert(sizeof(topic.battery_warning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.battery_warning, sizeof(topic.battery_warning));
	buf.iterator += sizeof(topic.battery_warning);
	buf.offset += sizeof(topic.battery_warning);
	static_assert(sizeof(topic.battery_low_remaining_time) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.battery_low_remaining_time, sizeof(topic.battery_low_remaining_time));
	buf.iterator += sizeof(topic.battery_low_remaining_time);
	buf.offset += sizeof(topic.battery_low_remaining_time);
	static_assert(sizeof(topic.battery_unhealthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.battery_unhealthy, sizeof(topic.battery_unhealthy));
	buf.iterator += sizeof(topic.battery_unhealthy);
	buf.offset += sizeof(topic.battery_unhealthy);
	static_assert(sizeof(topic.primary_geofence_breached) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.primary_geofence_breached, sizeof(topic.primary_geofence_breached));
	buf.iterator += sizeof(topic.primary_geofence_breached);
	buf.offset += sizeof(topic.primary_geofence_breached);
	static_assert(sizeof(topic.mission_failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mission_failure, sizeof(topic.mission_failure));
	buf.iterator += sizeof(topic.mission_failure);
	buf.offset += sizeof(topic.mission_failure);
	static_assert(sizeof(topic.vtol_fixed_wing_system_failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vtol_fixed_wing_system_failure, sizeof(topic.vtol_fixed_wing_system_failure));
	buf.iterator += sizeof(topic.vtol_fixed_wing_system_failure);
	buf.offset += sizeof(topic.vtol_fixed_wing_system_failure);
	static_assert(sizeof(topic.wind_limit_exceeded) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.wind_limit_exceeded, sizeof(topic.wind_limit_exceeded));
	buf.iterator += sizeof(topic.wind_limit_exceeded);
	buf.offset += sizeof(topic.wind_limit_exceeded);
	static_assert(sizeof(topic.flight_time_limit_exceeded) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flight_time_limit_exceeded, sizeof(topic.flight_time_limit_exceeded));
	buf.iterator += sizeof(topic.flight_time_limit_exceeded);
	buf.offset += sizeof(topic.flight_time_limit_exceeded);
	static_assert(sizeof(topic.local_position_accuracy_low) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.local_position_accuracy_low, sizeof(topic.local_position_accuracy_low));
	buf.iterator += sizeof(topic.local_position_accuracy_low);
	buf.offset += sizeof(topic.local_position_accuracy_low);
	static_assert(sizeof(topic.fd_critical_failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_critical_failure, sizeof(topic.fd_critical_failure));
	buf.iterator += sizeof(topic.fd_critical_failure);
	buf.offset += sizeof(topic.fd_critical_failure);
	static_assert(sizeof(topic.fd_esc_arming_failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_esc_arming_failure, sizeof(topic.fd_esc_arming_failure));
	buf.iterator += sizeof(topic.fd_esc_arming_failure);
	buf.offset += sizeof(topic.fd_esc_arming_failure);
	static_assert(sizeof(topic.fd_imbalanced_prop) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_imbalanced_prop, sizeof(topic.fd_imbalanced_prop));
	buf.iterator += sizeof(topic.fd_imbalanced_prop);
	buf.offset += sizeof(topic.fd_imbalanced_prop);
	static_assert(sizeof(topic.fd_motor_failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_motor_failure, sizeof(topic.fd_motor_failure));
	buf.iterator += sizeof(topic.fd_motor_failure);
	buf.offset += sizeof(topic.fd_motor_failure);
	return true;
}

bool ucdr_deserialize_failsafe_flags(ucdrBuffer& buf, failsafe_flags_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.mode_req_angular_velocity) == 4, "size mismatch");
	memcpy(&topic.mode_req_angular_velocity, buf.iterator, sizeof(topic.mode_req_angular_velocity));
	buf.iterator += sizeof(topic.mode_req_angular_velocity);
	buf.offset += sizeof(topic.mode_req_angular_velocity);
	static_assert(sizeof(topic.mode_req_attitude) == 4, "size mismatch");
	memcpy(&topic.mode_req_attitude, buf.iterator, sizeof(topic.mode_req_attitude));
	buf.iterator += sizeof(topic.mode_req_attitude);
	buf.offset += sizeof(topic.mode_req_attitude);
	static_assert(sizeof(topic.mode_req_local_alt) == 4, "size mismatch");
	memcpy(&topic.mode_req_local_alt, buf.iterator, sizeof(topic.mode_req_local_alt));
	buf.iterator += sizeof(topic.mode_req_local_alt);
	buf.offset += sizeof(topic.mode_req_local_alt);
	static_assert(sizeof(topic.mode_req_local_position) == 4, "size mismatch");
	memcpy(&topic.mode_req_local_position, buf.iterator, sizeof(topic.mode_req_local_position));
	buf.iterator += sizeof(topic.mode_req_local_position);
	buf.offset += sizeof(topic.mode_req_local_position);
	static_assert(sizeof(topic.mode_req_local_position_relaxed) == 4, "size mismatch");
	memcpy(&topic.mode_req_local_position_relaxed, buf.iterator, sizeof(topic.mode_req_local_position_relaxed));
	buf.iterator += sizeof(topic.mode_req_local_position_relaxed);
	buf.offset += sizeof(topic.mode_req_local_position_relaxed);
	static_assert(sizeof(topic.mode_req_global_position) == 4, "size mismatch");
	memcpy(&topic.mode_req_global_position, buf.iterator, sizeof(topic.mode_req_global_position));
	buf.iterator += sizeof(topic.mode_req_global_position);
	buf.offset += sizeof(topic.mode_req_global_position);
	static_assert(sizeof(topic.mode_req_mission) == 4, "size mismatch");
	memcpy(&topic.mode_req_mission, buf.iterator, sizeof(topic.mode_req_mission));
	buf.iterator += sizeof(topic.mode_req_mission);
	buf.offset += sizeof(topic.mode_req_mission);
	static_assert(sizeof(topic.mode_req_offboard_signal) == 4, "size mismatch");
	memcpy(&topic.mode_req_offboard_signal, buf.iterator, sizeof(topic.mode_req_offboard_signal));
	buf.iterator += sizeof(topic.mode_req_offboard_signal);
	buf.offset += sizeof(topic.mode_req_offboard_signal);
	static_assert(sizeof(topic.mode_req_home_position) == 4, "size mismatch");
	memcpy(&topic.mode_req_home_position, buf.iterator, sizeof(topic.mode_req_home_position));
	buf.iterator += sizeof(topic.mode_req_home_position);
	buf.offset += sizeof(topic.mode_req_home_position);
	static_assert(sizeof(topic.mode_req_wind_and_flight_time_compliance) == 4, "size mismatch");
	memcpy(&topic.mode_req_wind_and_flight_time_compliance, buf.iterator, sizeof(topic.mode_req_wind_and_flight_time_compliance));
	buf.iterator += sizeof(topic.mode_req_wind_and_flight_time_compliance);
	buf.offset += sizeof(topic.mode_req_wind_and_flight_time_compliance);
	static_assert(sizeof(topic.mode_req_prevent_arming) == 4, "size mismatch");
	memcpy(&topic.mode_req_prevent_arming, buf.iterator, sizeof(topic.mode_req_prevent_arming));
	buf.iterator += sizeof(topic.mode_req_prevent_arming);
	buf.offset += sizeof(topic.mode_req_prevent_arming);
	static_assert(sizeof(topic.mode_req_manual_control) == 4, "size mismatch");
	memcpy(&topic.mode_req_manual_control, buf.iterator, sizeof(topic.mode_req_manual_control));
	buf.iterator += sizeof(topic.mode_req_manual_control);
	buf.offset += sizeof(topic.mode_req_manual_control);
	static_assert(sizeof(topic.mode_req_other) == 4, "size mismatch");
	memcpy(&topic.mode_req_other, buf.iterator, sizeof(topic.mode_req_other));
	buf.iterator += sizeof(topic.mode_req_other);
	buf.offset += sizeof(topic.mode_req_other);
	static_assert(sizeof(topic.angular_velocity_invalid) == 1, "size mismatch");
	memcpy(&topic.angular_velocity_invalid, buf.iterator, sizeof(topic.angular_velocity_invalid));
	buf.iterator += sizeof(topic.angular_velocity_invalid);
	buf.offset += sizeof(topic.angular_velocity_invalid);
	static_assert(sizeof(topic.attitude_invalid) == 1, "size mismatch");
	memcpy(&topic.attitude_invalid, buf.iterator, sizeof(topic.attitude_invalid));
	buf.iterator += sizeof(topic.attitude_invalid);
	buf.offset += sizeof(topic.attitude_invalid);
	static_assert(sizeof(topic.local_altitude_invalid) == 1, "size mismatch");
	memcpy(&topic.local_altitude_invalid, buf.iterator, sizeof(topic.local_altitude_invalid));
	buf.iterator += sizeof(topic.local_altitude_invalid);
	buf.offset += sizeof(topic.local_altitude_invalid);
	static_assert(sizeof(topic.local_position_invalid) == 1, "size mismatch");
	memcpy(&topic.local_position_invalid, buf.iterator, sizeof(topic.local_position_invalid));
	buf.iterator += sizeof(topic.local_position_invalid);
	buf.offset += sizeof(topic.local_position_invalid);
	static_assert(sizeof(topic.local_position_invalid_relaxed) == 1, "size mismatch");
	memcpy(&topic.local_position_invalid_relaxed, buf.iterator, sizeof(topic.local_position_invalid_relaxed));
	buf.iterator += sizeof(topic.local_position_invalid_relaxed);
	buf.offset += sizeof(topic.local_position_invalid_relaxed);
	static_assert(sizeof(topic.local_velocity_invalid) == 1, "size mismatch");
	memcpy(&topic.local_velocity_invalid, buf.iterator, sizeof(topic.local_velocity_invalid));
	buf.iterator += sizeof(topic.local_velocity_invalid);
	buf.offset += sizeof(topic.local_velocity_invalid);
	static_assert(sizeof(topic.global_position_invalid) == 1, "size mismatch");
	memcpy(&topic.global_position_invalid, buf.iterator, sizeof(topic.global_position_invalid));
	buf.iterator += sizeof(topic.global_position_invalid);
	buf.offset += sizeof(topic.global_position_invalid);
	static_assert(sizeof(topic.auto_mission_missing) == 1, "size mismatch");
	memcpy(&topic.auto_mission_missing, buf.iterator, sizeof(topic.auto_mission_missing));
	buf.iterator += sizeof(topic.auto_mission_missing);
	buf.offset += sizeof(topic.auto_mission_missing);
	static_assert(sizeof(topic.offboard_control_signal_lost) == 1, "size mismatch");
	memcpy(&topic.offboard_control_signal_lost, buf.iterator, sizeof(topic.offboard_control_signal_lost));
	buf.iterator += sizeof(topic.offboard_control_signal_lost);
	buf.offset += sizeof(topic.offboard_control_signal_lost);
	static_assert(sizeof(topic.home_position_invalid) == 1, "size mismatch");
	memcpy(&topic.home_position_invalid, buf.iterator, sizeof(topic.home_position_invalid));
	buf.iterator += sizeof(topic.home_position_invalid);
	buf.offset += sizeof(topic.home_position_invalid);
	static_assert(sizeof(topic.manual_control_signal_lost) == 1, "size mismatch");
	memcpy(&topic.manual_control_signal_lost, buf.iterator, sizeof(topic.manual_control_signal_lost));
	buf.iterator += sizeof(topic.manual_control_signal_lost);
	buf.offset += sizeof(topic.manual_control_signal_lost);
	static_assert(sizeof(topic.gcs_connection_lost) == 1, "size mismatch");
	memcpy(&topic.gcs_connection_lost, buf.iterator, sizeof(topic.gcs_connection_lost));
	buf.iterator += sizeof(topic.gcs_connection_lost);
	buf.offset += sizeof(topic.gcs_connection_lost);
	static_assert(sizeof(topic.battery_warning) == 1, "size mismatch");
	memcpy(&topic.battery_warning, buf.iterator, sizeof(topic.battery_warning));
	buf.iterator += sizeof(topic.battery_warning);
	buf.offset += sizeof(topic.battery_warning);
	static_assert(sizeof(topic.battery_low_remaining_time) == 1, "size mismatch");
	memcpy(&topic.battery_low_remaining_time, buf.iterator, sizeof(topic.battery_low_remaining_time));
	buf.iterator += sizeof(topic.battery_low_remaining_time);
	buf.offset += sizeof(topic.battery_low_remaining_time);
	static_assert(sizeof(topic.battery_unhealthy) == 1, "size mismatch");
	memcpy(&topic.battery_unhealthy, buf.iterator, sizeof(topic.battery_unhealthy));
	buf.iterator += sizeof(topic.battery_unhealthy);
	buf.offset += sizeof(topic.battery_unhealthy);
	static_assert(sizeof(topic.primary_geofence_breached) == 1, "size mismatch");
	memcpy(&topic.primary_geofence_breached, buf.iterator, sizeof(topic.primary_geofence_breached));
	buf.iterator += sizeof(topic.primary_geofence_breached);
	buf.offset += sizeof(topic.primary_geofence_breached);
	static_assert(sizeof(topic.mission_failure) == 1, "size mismatch");
	memcpy(&topic.mission_failure, buf.iterator, sizeof(topic.mission_failure));
	buf.iterator += sizeof(topic.mission_failure);
	buf.offset += sizeof(topic.mission_failure);
	static_assert(sizeof(topic.vtol_fixed_wing_system_failure) == 1, "size mismatch");
	memcpy(&topic.vtol_fixed_wing_system_failure, buf.iterator, sizeof(topic.vtol_fixed_wing_system_failure));
	buf.iterator += sizeof(topic.vtol_fixed_wing_system_failure);
	buf.offset += sizeof(topic.vtol_fixed_wing_system_failure);
	static_assert(sizeof(topic.wind_limit_exceeded) == 1, "size mismatch");
	memcpy(&topic.wind_limit_exceeded, buf.iterator, sizeof(topic.wind_limit_exceeded));
	buf.iterator += sizeof(topic.wind_limit_exceeded);
	buf.offset += sizeof(topic.wind_limit_exceeded);
	static_assert(sizeof(topic.flight_time_limit_exceeded) == 1, "size mismatch");
	memcpy(&topic.flight_time_limit_exceeded, buf.iterator, sizeof(topic.flight_time_limit_exceeded));
	buf.iterator += sizeof(topic.flight_time_limit_exceeded);
	buf.offset += sizeof(topic.flight_time_limit_exceeded);
	static_assert(sizeof(topic.local_position_accuracy_low) == 1, "size mismatch");
	memcpy(&topic.local_position_accuracy_low, buf.iterator, sizeof(topic.local_position_accuracy_low));
	buf.iterator += sizeof(topic.local_position_accuracy_low);
	buf.offset += sizeof(topic.local_position_accuracy_low);
	static_assert(sizeof(topic.fd_critical_failure) == 1, "size mismatch");
	memcpy(&topic.fd_critical_failure, buf.iterator, sizeof(topic.fd_critical_failure));
	buf.iterator += sizeof(topic.fd_critical_failure);
	buf.offset += sizeof(topic.fd_critical_failure);
	static_assert(sizeof(topic.fd_esc_arming_failure) == 1, "size mismatch");
	memcpy(&topic.fd_esc_arming_failure, buf.iterator, sizeof(topic.fd_esc_arming_failure));
	buf.iterator += sizeof(topic.fd_esc_arming_failure);
	buf.offset += sizeof(topic.fd_esc_arming_failure);
	static_assert(sizeof(topic.fd_imbalanced_prop) == 1, "size mismatch");
	memcpy(&topic.fd_imbalanced_prop, buf.iterator, sizeof(topic.fd_imbalanced_prop));
	buf.iterator += sizeof(topic.fd_imbalanced_prop);
	buf.offset += sizeof(topic.fd_imbalanced_prop);
	static_assert(sizeof(topic.fd_motor_failure) == 1, "size mismatch");
	memcpy(&topic.fd_motor_failure, buf.iterator, sizeof(topic.fd_motor_failure));
	buf.iterator += sizeof(topic.fd_motor_failure);
	buf.offset += sizeof(topic.fd_motor_failure);
	return true;
}
