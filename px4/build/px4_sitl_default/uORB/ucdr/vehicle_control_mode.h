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
#include <uORB/topics/vehicle_control_mode.h>


static inline constexpr int ucdr_topic_size_vehicle_control_mode()
{
	return 21;
}

bool ucdr_serialize_vehicle_control_mode(const vehicle_control_mode_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.flag_armed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_armed, sizeof(topic.flag_armed));
	buf.iterator += sizeof(topic.flag_armed);
	buf.offset += sizeof(topic.flag_armed);
	static_assert(sizeof(topic.flag_multicopter_position_control_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_multicopter_position_control_enabled, sizeof(topic.flag_multicopter_position_control_enabled));
	buf.iterator += sizeof(topic.flag_multicopter_position_control_enabled);
	buf.offset += sizeof(topic.flag_multicopter_position_control_enabled);
	static_assert(sizeof(topic.flag_control_manual_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_manual_enabled, sizeof(topic.flag_control_manual_enabled));
	buf.iterator += sizeof(topic.flag_control_manual_enabled);
	buf.offset += sizeof(topic.flag_control_manual_enabled);
	static_assert(sizeof(topic.flag_control_auto_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_auto_enabled, sizeof(topic.flag_control_auto_enabled));
	buf.iterator += sizeof(topic.flag_control_auto_enabled);
	buf.offset += sizeof(topic.flag_control_auto_enabled);
	static_assert(sizeof(topic.flag_control_offboard_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_offboard_enabled, sizeof(topic.flag_control_offboard_enabled));
	buf.iterator += sizeof(topic.flag_control_offboard_enabled);
	buf.offset += sizeof(topic.flag_control_offboard_enabled);
	static_assert(sizeof(topic.flag_control_rates_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_rates_enabled, sizeof(topic.flag_control_rates_enabled));
	buf.iterator += sizeof(topic.flag_control_rates_enabled);
	buf.offset += sizeof(topic.flag_control_rates_enabled);
	static_assert(sizeof(topic.flag_control_attitude_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_attitude_enabled, sizeof(topic.flag_control_attitude_enabled));
	buf.iterator += sizeof(topic.flag_control_attitude_enabled);
	buf.offset += sizeof(topic.flag_control_attitude_enabled);
	static_assert(sizeof(topic.flag_control_acceleration_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_acceleration_enabled, sizeof(topic.flag_control_acceleration_enabled));
	buf.iterator += sizeof(topic.flag_control_acceleration_enabled);
	buf.offset += sizeof(topic.flag_control_acceleration_enabled);
	static_assert(sizeof(topic.flag_control_velocity_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_velocity_enabled, sizeof(topic.flag_control_velocity_enabled));
	buf.iterator += sizeof(topic.flag_control_velocity_enabled);
	buf.offset += sizeof(topic.flag_control_velocity_enabled);
	static_assert(sizeof(topic.flag_control_position_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_position_enabled, sizeof(topic.flag_control_position_enabled));
	buf.iterator += sizeof(topic.flag_control_position_enabled);
	buf.offset += sizeof(topic.flag_control_position_enabled);
	static_assert(sizeof(topic.flag_control_altitude_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_altitude_enabled, sizeof(topic.flag_control_altitude_enabled));
	buf.iterator += sizeof(topic.flag_control_altitude_enabled);
	buf.offset += sizeof(topic.flag_control_altitude_enabled);
	static_assert(sizeof(topic.flag_control_climb_rate_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_climb_rate_enabled, sizeof(topic.flag_control_climb_rate_enabled));
	buf.iterator += sizeof(topic.flag_control_climb_rate_enabled);
	buf.offset += sizeof(topic.flag_control_climb_rate_enabled);
	static_assert(sizeof(topic.flag_control_termination_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flag_control_termination_enabled, sizeof(topic.flag_control_termination_enabled));
	buf.iterator += sizeof(topic.flag_control_termination_enabled);
	buf.offset += sizeof(topic.flag_control_termination_enabled);
	return true;
}

bool ucdr_deserialize_vehicle_control_mode(ucdrBuffer& buf, vehicle_control_mode_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.flag_armed) == 1, "size mismatch");
	memcpy(&topic.flag_armed, buf.iterator, sizeof(topic.flag_armed));
	buf.iterator += sizeof(topic.flag_armed);
	buf.offset += sizeof(topic.flag_armed);
	static_assert(sizeof(topic.flag_multicopter_position_control_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_multicopter_position_control_enabled, buf.iterator, sizeof(topic.flag_multicopter_position_control_enabled));
	buf.iterator += sizeof(topic.flag_multicopter_position_control_enabled);
	buf.offset += sizeof(topic.flag_multicopter_position_control_enabled);
	static_assert(sizeof(topic.flag_control_manual_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_manual_enabled, buf.iterator, sizeof(topic.flag_control_manual_enabled));
	buf.iterator += sizeof(topic.flag_control_manual_enabled);
	buf.offset += sizeof(topic.flag_control_manual_enabled);
	static_assert(sizeof(topic.flag_control_auto_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_auto_enabled, buf.iterator, sizeof(topic.flag_control_auto_enabled));
	buf.iterator += sizeof(topic.flag_control_auto_enabled);
	buf.offset += sizeof(topic.flag_control_auto_enabled);
	static_assert(sizeof(topic.flag_control_offboard_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_offboard_enabled, buf.iterator, sizeof(topic.flag_control_offboard_enabled));
	buf.iterator += sizeof(topic.flag_control_offboard_enabled);
	buf.offset += sizeof(topic.flag_control_offboard_enabled);
	static_assert(sizeof(topic.flag_control_rates_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_rates_enabled, buf.iterator, sizeof(topic.flag_control_rates_enabled));
	buf.iterator += sizeof(topic.flag_control_rates_enabled);
	buf.offset += sizeof(topic.flag_control_rates_enabled);
	static_assert(sizeof(topic.flag_control_attitude_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_attitude_enabled, buf.iterator, sizeof(topic.flag_control_attitude_enabled));
	buf.iterator += sizeof(topic.flag_control_attitude_enabled);
	buf.offset += sizeof(topic.flag_control_attitude_enabled);
	static_assert(sizeof(topic.flag_control_acceleration_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_acceleration_enabled, buf.iterator, sizeof(topic.flag_control_acceleration_enabled));
	buf.iterator += sizeof(topic.flag_control_acceleration_enabled);
	buf.offset += sizeof(topic.flag_control_acceleration_enabled);
	static_assert(sizeof(topic.flag_control_velocity_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_velocity_enabled, buf.iterator, sizeof(topic.flag_control_velocity_enabled));
	buf.iterator += sizeof(topic.flag_control_velocity_enabled);
	buf.offset += sizeof(topic.flag_control_velocity_enabled);
	static_assert(sizeof(topic.flag_control_position_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_position_enabled, buf.iterator, sizeof(topic.flag_control_position_enabled));
	buf.iterator += sizeof(topic.flag_control_position_enabled);
	buf.offset += sizeof(topic.flag_control_position_enabled);
	static_assert(sizeof(topic.flag_control_altitude_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_altitude_enabled, buf.iterator, sizeof(topic.flag_control_altitude_enabled));
	buf.iterator += sizeof(topic.flag_control_altitude_enabled);
	buf.offset += sizeof(topic.flag_control_altitude_enabled);
	static_assert(sizeof(topic.flag_control_climb_rate_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_climb_rate_enabled, buf.iterator, sizeof(topic.flag_control_climb_rate_enabled));
	buf.iterator += sizeof(topic.flag_control_climb_rate_enabled);
	buf.offset += sizeof(topic.flag_control_climb_rate_enabled);
	static_assert(sizeof(topic.flag_control_termination_enabled) == 1, "size mismatch");
	memcpy(&topic.flag_control_termination_enabled, buf.iterator, sizeof(topic.flag_control_termination_enabled));
	buf.iterator += sizeof(topic.flag_control_termination_enabled);
	buf.offset += sizeof(topic.flag_control_termination_enabled);
	return true;
}
