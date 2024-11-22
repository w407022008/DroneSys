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
#include <uORB/topics/vehicle_status.h>


static inline constexpr int ucdr_topic_size_vehicle_status()
{
	return 71;
}

bool ucdr_serialize_vehicle_status(const vehicle_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.armed_time) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.armed_time, sizeof(topic.armed_time));
	buf.iterator += sizeof(topic.armed_time);
	buf.offset += sizeof(topic.armed_time);
	static_assert(sizeof(topic.takeoff_time) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.takeoff_time, sizeof(topic.takeoff_time));
	buf.iterator += sizeof(topic.takeoff_time);
	buf.offset += sizeof(topic.takeoff_time);
	static_assert(sizeof(topic.arming_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_state, sizeof(topic.arming_state));
	buf.iterator += sizeof(topic.arming_state);
	buf.offset += sizeof(topic.arming_state);
	static_assert(sizeof(topic.latest_arming_reason) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.latest_arming_reason, sizeof(topic.latest_arming_reason));
	buf.iterator += sizeof(topic.latest_arming_reason);
	buf.offset += sizeof(topic.latest_arming_reason);
	static_assert(sizeof(topic.latest_disarming_reason) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.latest_disarming_reason, sizeof(topic.latest_disarming_reason));
	buf.iterator += sizeof(topic.latest_disarming_reason);
	buf.offset += sizeof(topic.latest_disarming_reason);
	buf.iterator += 5; // padding
	buf.offset += 5; // padding
	static_assert(sizeof(topic.nav_state_timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.nav_state_timestamp, sizeof(topic.nav_state_timestamp));
	buf.iterator += sizeof(topic.nav_state_timestamp);
	buf.offset += sizeof(topic.nav_state_timestamp);
	static_assert(sizeof(topic.nav_state_user_intention) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.nav_state_user_intention, sizeof(topic.nav_state_user_intention));
	buf.iterator += sizeof(topic.nav_state_user_intention);
	buf.offset += sizeof(topic.nav_state_user_intention);
	static_assert(sizeof(topic.nav_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.nav_state, sizeof(topic.nav_state));
	buf.iterator += sizeof(topic.nav_state);
	buf.offset += sizeof(topic.nav_state);
	static_assert(sizeof(topic.failure_detector_status) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.failure_detector_status, sizeof(topic.failure_detector_status));
	buf.iterator += sizeof(topic.failure_detector_status);
	buf.offset += sizeof(topic.failure_detector_status);
	static_assert(sizeof(topic.hil_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.hil_state, sizeof(topic.hil_state));
	buf.iterator += sizeof(topic.hil_state);
	buf.offset += sizeof(topic.hil_state);
	static_assert(sizeof(topic.vehicle_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vehicle_type, sizeof(topic.vehicle_type));
	buf.iterator += sizeof(topic.vehicle_type);
	buf.offset += sizeof(topic.vehicle_type);
	static_assert(sizeof(topic.failsafe) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.failsafe, sizeof(topic.failsafe));
	buf.iterator += sizeof(topic.failsafe);
	buf.offset += sizeof(topic.failsafe);
	static_assert(sizeof(topic.failsafe_and_user_took_over) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.failsafe_and_user_took_over, sizeof(topic.failsafe_and_user_took_over));
	buf.iterator += sizeof(topic.failsafe_and_user_took_over);
	buf.offset += sizeof(topic.failsafe_and_user_took_over);
	static_assert(sizeof(topic.gcs_connection_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gcs_connection_lost, sizeof(topic.gcs_connection_lost));
	buf.iterator += sizeof(topic.gcs_connection_lost);
	buf.offset += sizeof(topic.gcs_connection_lost);
	static_assert(sizeof(topic.gcs_connection_lost_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gcs_connection_lost_counter, sizeof(topic.gcs_connection_lost_counter));
	buf.iterator += sizeof(topic.gcs_connection_lost_counter);
	buf.offset += sizeof(topic.gcs_connection_lost_counter);
	static_assert(sizeof(topic.high_latency_data_link_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.high_latency_data_link_lost, sizeof(topic.high_latency_data_link_lost));
	buf.iterator += sizeof(topic.high_latency_data_link_lost);
	buf.offset += sizeof(topic.high_latency_data_link_lost);
	static_assert(sizeof(topic.is_vtol) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.is_vtol, sizeof(topic.is_vtol));
	buf.iterator += sizeof(topic.is_vtol);
	buf.offset += sizeof(topic.is_vtol);
	static_assert(sizeof(topic.is_vtol_tailsitter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.is_vtol_tailsitter, sizeof(topic.is_vtol_tailsitter));
	buf.iterator += sizeof(topic.is_vtol_tailsitter);
	buf.offset += sizeof(topic.is_vtol_tailsitter);
	static_assert(sizeof(topic.in_transition_mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.in_transition_mode, sizeof(topic.in_transition_mode));
	buf.iterator += sizeof(topic.in_transition_mode);
	buf.offset += sizeof(topic.in_transition_mode);
	static_assert(sizeof(topic.in_transition_to_fw) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.in_transition_to_fw, sizeof(topic.in_transition_to_fw));
	buf.iterator += sizeof(topic.in_transition_to_fw);
	buf.offset += sizeof(topic.in_transition_to_fw);
	static_assert(sizeof(topic.system_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.system_type, sizeof(topic.system_type));
	buf.iterator += sizeof(topic.system_type);
	buf.offset += sizeof(topic.system_type);
	static_assert(sizeof(topic.system_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.system_id, sizeof(topic.system_id));
	buf.iterator += sizeof(topic.system_id);
	buf.offset += sizeof(topic.system_id);
	static_assert(sizeof(topic.component_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.component_id, sizeof(topic.component_id));
	buf.iterator += sizeof(topic.component_id);
	buf.offset += sizeof(topic.component_id);
	static_assert(sizeof(topic.safety_button_available) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.safety_button_available, sizeof(topic.safety_button_available));
	buf.iterator += sizeof(topic.safety_button_available);
	buf.offset += sizeof(topic.safety_button_available);
	static_assert(sizeof(topic.safety_off) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.safety_off, sizeof(topic.safety_off));
	buf.iterator += sizeof(topic.safety_off);
	buf.offset += sizeof(topic.safety_off);
	static_assert(sizeof(topic.power_input_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.power_input_valid, sizeof(topic.power_input_valid));
	buf.iterator += sizeof(topic.power_input_valid);
	buf.offset += sizeof(topic.power_input_valid);
	static_assert(sizeof(topic.usb_connected) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.usb_connected, sizeof(topic.usb_connected));
	buf.iterator += sizeof(topic.usb_connected);
	buf.offset += sizeof(topic.usb_connected);
	static_assert(sizeof(topic.open_drone_id_system_present) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.open_drone_id_system_present, sizeof(topic.open_drone_id_system_present));
	buf.iterator += sizeof(topic.open_drone_id_system_present);
	buf.offset += sizeof(topic.open_drone_id_system_present);
	static_assert(sizeof(topic.open_drone_id_system_healthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.open_drone_id_system_healthy, sizeof(topic.open_drone_id_system_healthy));
	buf.iterator += sizeof(topic.open_drone_id_system_healthy);
	buf.offset += sizeof(topic.open_drone_id_system_healthy);
	static_assert(sizeof(topic.parachute_system_present) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.parachute_system_present, sizeof(topic.parachute_system_present));
	buf.iterator += sizeof(topic.parachute_system_present);
	buf.offset += sizeof(topic.parachute_system_present);
	static_assert(sizeof(topic.parachute_system_healthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.parachute_system_healthy, sizeof(topic.parachute_system_healthy));
	buf.iterator += sizeof(topic.parachute_system_healthy);
	buf.offset += sizeof(topic.parachute_system_healthy);
	static_assert(sizeof(topic.avoidance_system_required) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.avoidance_system_required, sizeof(topic.avoidance_system_required));
	buf.iterator += sizeof(topic.avoidance_system_required);
	buf.offset += sizeof(topic.avoidance_system_required);
	static_assert(sizeof(topic.avoidance_system_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.avoidance_system_valid, sizeof(topic.avoidance_system_valid));
	buf.iterator += sizeof(topic.avoidance_system_valid);
	buf.offset += sizeof(topic.avoidance_system_valid);
	static_assert(sizeof(topic.rc_calibration_in_progress) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rc_calibration_in_progress, sizeof(topic.rc_calibration_in_progress));
	buf.iterator += sizeof(topic.rc_calibration_in_progress);
	buf.offset += sizeof(topic.rc_calibration_in_progress);
	static_assert(sizeof(topic.calibration_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.calibration_enabled, sizeof(topic.calibration_enabled));
	buf.iterator += sizeof(topic.calibration_enabled);
	buf.offset += sizeof(topic.calibration_enabled);
	static_assert(sizeof(topic.pre_flight_checks_pass) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flight_checks_pass, sizeof(topic.pre_flight_checks_pass));
	buf.iterator += sizeof(topic.pre_flight_checks_pass);
	buf.offset += sizeof(topic.pre_flight_checks_pass);
	return true;
}

bool ucdr_deserialize_vehicle_status(ucdrBuffer& buf, vehicle_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.armed_time) == 8, "size mismatch");
	memcpy(&topic.armed_time, buf.iterator, sizeof(topic.armed_time));
	buf.iterator += sizeof(topic.armed_time);
	buf.offset += sizeof(topic.armed_time);
	static_assert(sizeof(topic.takeoff_time) == 8, "size mismatch");
	memcpy(&topic.takeoff_time, buf.iterator, sizeof(topic.takeoff_time));
	buf.iterator += sizeof(topic.takeoff_time);
	buf.offset += sizeof(topic.takeoff_time);
	static_assert(sizeof(topic.arming_state) == 1, "size mismatch");
	memcpy(&topic.arming_state, buf.iterator, sizeof(topic.arming_state));
	buf.iterator += sizeof(topic.arming_state);
	buf.offset += sizeof(topic.arming_state);
	static_assert(sizeof(topic.latest_arming_reason) == 1, "size mismatch");
	memcpy(&topic.latest_arming_reason, buf.iterator, sizeof(topic.latest_arming_reason));
	buf.iterator += sizeof(topic.latest_arming_reason);
	buf.offset += sizeof(topic.latest_arming_reason);
	static_assert(sizeof(topic.latest_disarming_reason) == 1, "size mismatch");
	memcpy(&topic.latest_disarming_reason, buf.iterator, sizeof(topic.latest_disarming_reason));
	buf.iterator += sizeof(topic.latest_disarming_reason);
	buf.offset += sizeof(topic.latest_disarming_reason);
	buf.iterator += 5; // padding
	buf.offset += 5; // padding
	static_assert(sizeof(topic.nav_state_timestamp) == 8, "size mismatch");
	memcpy(&topic.nav_state_timestamp, buf.iterator, sizeof(topic.nav_state_timestamp));
	buf.iterator += sizeof(topic.nav_state_timestamp);
	buf.offset += sizeof(topic.nav_state_timestamp);
	static_assert(sizeof(topic.nav_state_user_intention) == 1, "size mismatch");
	memcpy(&topic.nav_state_user_intention, buf.iterator, sizeof(topic.nav_state_user_intention));
	buf.iterator += sizeof(topic.nav_state_user_intention);
	buf.offset += sizeof(topic.nav_state_user_intention);
	static_assert(sizeof(topic.nav_state) == 1, "size mismatch");
	memcpy(&topic.nav_state, buf.iterator, sizeof(topic.nav_state));
	buf.iterator += sizeof(topic.nav_state);
	buf.offset += sizeof(topic.nav_state);
	static_assert(sizeof(topic.failure_detector_status) == 2, "size mismatch");
	memcpy(&topic.failure_detector_status, buf.iterator, sizeof(topic.failure_detector_status));
	buf.iterator += sizeof(topic.failure_detector_status);
	buf.offset += sizeof(topic.failure_detector_status);
	static_assert(sizeof(topic.hil_state) == 1, "size mismatch");
	memcpy(&topic.hil_state, buf.iterator, sizeof(topic.hil_state));
	buf.iterator += sizeof(topic.hil_state);
	buf.offset += sizeof(topic.hil_state);
	static_assert(sizeof(topic.vehicle_type) == 1, "size mismatch");
	memcpy(&topic.vehicle_type, buf.iterator, sizeof(topic.vehicle_type));
	buf.iterator += sizeof(topic.vehicle_type);
	buf.offset += sizeof(topic.vehicle_type);
	static_assert(sizeof(topic.failsafe) == 1, "size mismatch");
	memcpy(&topic.failsafe, buf.iterator, sizeof(topic.failsafe));
	buf.iterator += sizeof(topic.failsafe);
	buf.offset += sizeof(topic.failsafe);
	static_assert(sizeof(topic.failsafe_and_user_took_over) == 1, "size mismatch");
	memcpy(&topic.failsafe_and_user_took_over, buf.iterator, sizeof(topic.failsafe_and_user_took_over));
	buf.iterator += sizeof(topic.failsafe_and_user_took_over);
	buf.offset += sizeof(topic.failsafe_and_user_took_over);
	static_assert(sizeof(topic.gcs_connection_lost) == 1, "size mismatch");
	memcpy(&topic.gcs_connection_lost, buf.iterator, sizeof(topic.gcs_connection_lost));
	buf.iterator += sizeof(topic.gcs_connection_lost);
	buf.offset += sizeof(topic.gcs_connection_lost);
	static_assert(sizeof(topic.gcs_connection_lost_counter) == 1, "size mismatch");
	memcpy(&topic.gcs_connection_lost_counter, buf.iterator, sizeof(topic.gcs_connection_lost_counter));
	buf.iterator += sizeof(topic.gcs_connection_lost_counter);
	buf.offset += sizeof(topic.gcs_connection_lost_counter);
	static_assert(sizeof(topic.high_latency_data_link_lost) == 1, "size mismatch");
	memcpy(&topic.high_latency_data_link_lost, buf.iterator, sizeof(topic.high_latency_data_link_lost));
	buf.iterator += sizeof(topic.high_latency_data_link_lost);
	buf.offset += sizeof(topic.high_latency_data_link_lost);
	static_assert(sizeof(topic.is_vtol) == 1, "size mismatch");
	memcpy(&topic.is_vtol, buf.iterator, sizeof(topic.is_vtol));
	buf.iterator += sizeof(topic.is_vtol);
	buf.offset += sizeof(topic.is_vtol);
	static_assert(sizeof(topic.is_vtol_tailsitter) == 1, "size mismatch");
	memcpy(&topic.is_vtol_tailsitter, buf.iterator, sizeof(topic.is_vtol_tailsitter));
	buf.iterator += sizeof(topic.is_vtol_tailsitter);
	buf.offset += sizeof(topic.is_vtol_tailsitter);
	static_assert(sizeof(topic.in_transition_mode) == 1, "size mismatch");
	memcpy(&topic.in_transition_mode, buf.iterator, sizeof(topic.in_transition_mode));
	buf.iterator += sizeof(topic.in_transition_mode);
	buf.offset += sizeof(topic.in_transition_mode);
	static_assert(sizeof(topic.in_transition_to_fw) == 1, "size mismatch");
	memcpy(&topic.in_transition_to_fw, buf.iterator, sizeof(topic.in_transition_to_fw));
	buf.iterator += sizeof(topic.in_transition_to_fw);
	buf.offset += sizeof(topic.in_transition_to_fw);
	static_assert(sizeof(topic.system_type) == 1, "size mismatch");
	memcpy(&topic.system_type, buf.iterator, sizeof(topic.system_type));
	buf.iterator += sizeof(topic.system_type);
	buf.offset += sizeof(topic.system_type);
	static_assert(sizeof(topic.system_id) == 1, "size mismatch");
	memcpy(&topic.system_id, buf.iterator, sizeof(topic.system_id));
	buf.iterator += sizeof(topic.system_id);
	buf.offset += sizeof(topic.system_id);
	static_assert(sizeof(topic.component_id) == 1, "size mismatch");
	memcpy(&topic.component_id, buf.iterator, sizeof(topic.component_id));
	buf.iterator += sizeof(topic.component_id);
	buf.offset += sizeof(topic.component_id);
	static_assert(sizeof(topic.safety_button_available) == 1, "size mismatch");
	memcpy(&topic.safety_button_available, buf.iterator, sizeof(topic.safety_button_available));
	buf.iterator += sizeof(topic.safety_button_available);
	buf.offset += sizeof(topic.safety_button_available);
	static_assert(sizeof(topic.safety_off) == 1, "size mismatch");
	memcpy(&topic.safety_off, buf.iterator, sizeof(topic.safety_off));
	buf.iterator += sizeof(topic.safety_off);
	buf.offset += sizeof(topic.safety_off);
	static_assert(sizeof(topic.power_input_valid) == 1, "size mismatch");
	memcpy(&topic.power_input_valid, buf.iterator, sizeof(topic.power_input_valid));
	buf.iterator += sizeof(topic.power_input_valid);
	buf.offset += sizeof(topic.power_input_valid);
	static_assert(sizeof(topic.usb_connected) == 1, "size mismatch");
	memcpy(&topic.usb_connected, buf.iterator, sizeof(topic.usb_connected));
	buf.iterator += sizeof(topic.usb_connected);
	buf.offset += sizeof(topic.usb_connected);
	static_assert(sizeof(topic.open_drone_id_system_present) == 1, "size mismatch");
	memcpy(&topic.open_drone_id_system_present, buf.iterator, sizeof(topic.open_drone_id_system_present));
	buf.iterator += sizeof(topic.open_drone_id_system_present);
	buf.offset += sizeof(topic.open_drone_id_system_present);
	static_assert(sizeof(topic.open_drone_id_system_healthy) == 1, "size mismatch");
	memcpy(&topic.open_drone_id_system_healthy, buf.iterator, sizeof(topic.open_drone_id_system_healthy));
	buf.iterator += sizeof(topic.open_drone_id_system_healthy);
	buf.offset += sizeof(topic.open_drone_id_system_healthy);
	static_assert(sizeof(topic.parachute_system_present) == 1, "size mismatch");
	memcpy(&topic.parachute_system_present, buf.iterator, sizeof(topic.parachute_system_present));
	buf.iterator += sizeof(topic.parachute_system_present);
	buf.offset += sizeof(topic.parachute_system_present);
	static_assert(sizeof(topic.parachute_system_healthy) == 1, "size mismatch");
	memcpy(&topic.parachute_system_healthy, buf.iterator, sizeof(topic.parachute_system_healthy));
	buf.iterator += sizeof(topic.parachute_system_healthy);
	buf.offset += sizeof(topic.parachute_system_healthy);
	static_assert(sizeof(topic.avoidance_system_required) == 1, "size mismatch");
	memcpy(&topic.avoidance_system_required, buf.iterator, sizeof(topic.avoidance_system_required));
	buf.iterator += sizeof(topic.avoidance_system_required);
	buf.offset += sizeof(topic.avoidance_system_required);
	static_assert(sizeof(topic.avoidance_system_valid) == 1, "size mismatch");
	memcpy(&topic.avoidance_system_valid, buf.iterator, sizeof(topic.avoidance_system_valid));
	buf.iterator += sizeof(topic.avoidance_system_valid);
	buf.offset += sizeof(topic.avoidance_system_valid);
	static_assert(sizeof(topic.rc_calibration_in_progress) == 1, "size mismatch");
	memcpy(&topic.rc_calibration_in_progress, buf.iterator, sizeof(topic.rc_calibration_in_progress));
	buf.iterator += sizeof(topic.rc_calibration_in_progress);
	buf.offset += sizeof(topic.rc_calibration_in_progress);
	static_assert(sizeof(topic.calibration_enabled) == 1, "size mismatch");
	memcpy(&topic.calibration_enabled, buf.iterator, sizeof(topic.calibration_enabled));
	buf.iterator += sizeof(topic.calibration_enabled);
	buf.offset += sizeof(topic.calibration_enabled);
	static_assert(sizeof(topic.pre_flight_checks_pass) == 1, "size mismatch");
	memcpy(&topic.pre_flight_checks_pass, buf.iterator, sizeof(topic.pre_flight_checks_pass));
	buf.iterator += sizeof(topic.pre_flight_checks_pass);
	buf.offset += sizeof(topic.pre_flight_checks_pass);
	return true;
}
