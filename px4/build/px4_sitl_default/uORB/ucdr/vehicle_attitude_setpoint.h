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
#include <uORB/topics/vehicle_attitude_setpoint.h>


static inline constexpr int ucdr_topic_size_vehicle_attitude_setpoint()
{
	return 54;
}

bool ucdr_serialize_vehicle_attitude_setpoint(const vehicle_attitude_setpoint_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.roll_body) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.roll_body, sizeof(topic.roll_body));
	buf.iterator += sizeof(topic.roll_body);
	buf.offset += sizeof(topic.roll_body);
	static_assert(sizeof(topic.pitch_body) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_body, sizeof(topic.pitch_body));
	buf.iterator += sizeof(topic.pitch_body);
	buf.offset += sizeof(topic.pitch_body);
	static_assert(sizeof(topic.yaw_body) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_body, sizeof(topic.yaw_body));
	buf.iterator += sizeof(topic.yaw_body);
	buf.offset += sizeof(topic.yaw_body);
	static_assert(sizeof(topic.yaw_sp_move_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_sp_move_rate, sizeof(topic.yaw_sp_move_rate));
	buf.iterator += sizeof(topic.yaw_sp_move_rate);
	buf.offset += sizeof(topic.yaw_sp_move_rate);
	static_assert(sizeof(topic.q_d) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.q_d, sizeof(topic.q_d));
	buf.iterator += sizeof(topic.q_d);
	buf.offset += sizeof(topic.q_d);
	static_assert(sizeof(topic.thrust_body) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.thrust_body, sizeof(topic.thrust_body));
	buf.iterator += sizeof(topic.thrust_body);
	buf.offset += sizeof(topic.thrust_body);
	static_assert(sizeof(topic.reset_integral) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_integral, sizeof(topic.reset_integral));
	buf.iterator += sizeof(topic.reset_integral);
	buf.offset += sizeof(topic.reset_integral);
	static_assert(sizeof(topic.fw_control_yaw_wheel) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fw_control_yaw_wheel, sizeof(topic.fw_control_yaw_wheel));
	buf.iterator += sizeof(topic.fw_control_yaw_wheel);
	buf.offset += sizeof(topic.fw_control_yaw_wheel);
	return true;
}

bool ucdr_deserialize_vehicle_attitude_setpoint(ucdrBuffer& buf, vehicle_attitude_setpoint_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.roll_body) == 4, "size mismatch");
	memcpy(&topic.roll_body, buf.iterator, sizeof(topic.roll_body));
	buf.iterator += sizeof(topic.roll_body);
	buf.offset += sizeof(topic.roll_body);
	static_assert(sizeof(topic.pitch_body) == 4, "size mismatch");
	memcpy(&topic.pitch_body, buf.iterator, sizeof(topic.pitch_body));
	buf.iterator += sizeof(topic.pitch_body);
	buf.offset += sizeof(topic.pitch_body);
	static_assert(sizeof(topic.yaw_body) == 4, "size mismatch");
	memcpy(&topic.yaw_body, buf.iterator, sizeof(topic.yaw_body));
	buf.iterator += sizeof(topic.yaw_body);
	buf.offset += sizeof(topic.yaw_body);
	static_assert(sizeof(topic.yaw_sp_move_rate) == 4, "size mismatch");
	memcpy(&topic.yaw_sp_move_rate, buf.iterator, sizeof(topic.yaw_sp_move_rate));
	buf.iterator += sizeof(topic.yaw_sp_move_rate);
	buf.offset += sizeof(topic.yaw_sp_move_rate);
	static_assert(sizeof(topic.q_d) == 16, "size mismatch");
	memcpy(&topic.q_d, buf.iterator, sizeof(topic.q_d));
	buf.iterator += sizeof(topic.q_d);
	buf.offset += sizeof(topic.q_d);
	static_assert(sizeof(topic.thrust_body) == 12, "size mismatch");
	memcpy(&topic.thrust_body, buf.iterator, sizeof(topic.thrust_body));
	buf.iterator += sizeof(topic.thrust_body);
	buf.offset += sizeof(topic.thrust_body);
	static_assert(sizeof(topic.reset_integral) == 1, "size mismatch");
	memcpy(&topic.reset_integral, buf.iterator, sizeof(topic.reset_integral));
	buf.iterator += sizeof(topic.reset_integral);
	buf.offset += sizeof(topic.reset_integral);
	static_assert(sizeof(topic.fw_control_yaw_wheel) == 1, "size mismatch");
	memcpy(&topic.fw_control_yaw_wheel, buf.iterator, sizeof(topic.fw_control_yaw_wheel));
	buf.iterator += sizeof(topic.fw_control_yaw_wheel);
	buf.offset += sizeof(topic.fw_control_yaw_wheel);
	return true;
}
