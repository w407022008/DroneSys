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

/* Auto-generated by genmsg_cpp from file /home/sique/src/PX4_v1.14.2/msg/FailsafeFlags.msg */


#pragma once


#include <uORB/uORB.h>


#ifndef __cplusplus

#endif


#ifdef __cplusplus
struct __EXPORT failsafe_flags_s {
#else
struct failsafe_flags_s {
#endif
	uint64_t timestamp;
	uint32_t mode_req_angular_velocity;
	uint32_t mode_req_attitude;
	uint32_t mode_req_local_alt;
	uint32_t mode_req_local_position;
	uint32_t mode_req_local_position_relaxed;
	uint32_t mode_req_global_position;
	uint32_t mode_req_mission;
	uint32_t mode_req_offboard_signal;
	uint32_t mode_req_home_position;
	uint32_t mode_req_wind_and_flight_time_compliance;
	uint32_t mode_req_prevent_arming;
	uint32_t mode_req_manual_control;
	uint32_t mode_req_other;
	bool angular_velocity_invalid;
	bool attitude_invalid;
	bool local_altitude_invalid;
	bool local_position_invalid;
	bool local_position_invalid_relaxed;
	bool local_velocity_invalid;
	bool global_position_invalid;
	bool auto_mission_missing;
	bool offboard_control_signal_lost;
	bool home_position_invalid;
	bool manual_control_signal_lost;
	bool gcs_connection_lost;
	uint8_t battery_warning;
	bool battery_low_remaining_time;
	bool battery_unhealthy;
	bool primary_geofence_breached;
	bool mission_failure;
	bool vtol_fixed_wing_system_failure;
	bool wind_limit_exceeded;
	bool flight_time_limit_exceeded;
	bool local_position_accuracy_low;
	bool fd_critical_failure;
	bool fd_esc_arming_failure;
	bool fd_imbalanced_prop;
	bool fd_motor_failure;
	uint8_t _padding0[3]; // required for logger


#ifdef __cplusplus

#endif
};

#ifdef __cplusplus
namespace px4 {
	namespace msg {
		using FailsafeFlags = failsafe_flags_s;
	} // namespace msg
} // namespace px4
#endif

/* register this as object request broker structure */
ORB_DECLARE(failsafe_flags);


#ifdef __cplusplus
void print_message(const orb_metadata *meta, const failsafe_flags_s& message);
#endif
