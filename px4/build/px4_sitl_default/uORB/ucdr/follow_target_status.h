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
#include <uORB/topics/follow_target_status.h>


static inline constexpr int ucdr_topic_size_follow_target_status()
{
	return 44;
}

bool ucdr_serialize_follow_target_status(const follow_target_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.tracked_target_course) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tracked_target_course, sizeof(topic.tracked_target_course));
	buf.iterator += sizeof(topic.tracked_target_course);
	buf.offset += sizeof(topic.tracked_target_course);
	static_assert(sizeof(topic.follow_angle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.follow_angle, sizeof(topic.follow_angle));
	buf.iterator += sizeof(topic.follow_angle);
	buf.offset += sizeof(topic.follow_angle);
	static_assert(sizeof(topic.orbit_angle_setpoint) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.orbit_angle_setpoint, sizeof(topic.orbit_angle_setpoint));
	buf.iterator += sizeof(topic.orbit_angle_setpoint);
	buf.offset += sizeof(topic.orbit_angle_setpoint);
	static_assert(sizeof(topic.angular_rate_setpoint) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.angular_rate_setpoint, sizeof(topic.angular_rate_setpoint));
	buf.iterator += sizeof(topic.angular_rate_setpoint);
	buf.offset += sizeof(topic.angular_rate_setpoint);
	static_assert(sizeof(topic.desired_position_raw) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.desired_position_raw, sizeof(topic.desired_position_raw));
	buf.iterator += sizeof(topic.desired_position_raw);
	buf.offset += sizeof(topic.desired_position_raw);
	static_assert(sizeof(topic.in_emergency_ascent) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.in_emergency_ascent, sizeof(topic.in_emergency_ascent));
	buf.iterator += sizeof(topic.in_emergency_ascent);
	buf.offset += sizeof(topic.in_emergency_ascent);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.gimbal_pitch) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gimbal_pitch, sizeof(topic.gimbal_pitch));
	buf.iterator += sizeof(topic.gimbal_pitch);
	buf.offset += sizeof(topic.gimbal_pitch);
	return true;
}

bool ucdr_deserialize_follow_target_status(ucdrBuffer& buf, follow_target_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.tracked_target_course) == 4, "size mismatch");
	memcpy(&topic.tracked_target_course, buf.iterator, sizeof(topic.tracked_target_course));
	buf.iterator += sizeof(topic.tracked_target_course);
	buf.offset += sizeof(topic.tracked_target_course);
	static_assert(sizeof(topic.follow_angle) == 4, "size mismatch");
	memcpy(&topic.follow_angle, buf.iterator, sizeof(topic.follow_angle));
	buf.iterator += sizeof(topic.follow_angle);
	buf.offset += sizeof(topic.follow_angle);
	static_assert(sizeof(topic.orbit_angle_setpoint) == 4, "size mismatch");
	memcpy(&topic.orbit_angle_setpoint, buf.iterator, sizeof(topic.orbit_angle_setpoint));
	buf.iterator += sizeof(topic.orbit_angle_setpoint);
	buf.offset += sizeof(topic.orbit_angle_setpoint);
	static_assert(sizeof(topic.angular_rate_setpoint) == 4, "size mismatch");
	memcpy(&topic.angular_rate_setpoint, buf.iterator, sizeof(topic.angular_rate_setpoint));
	buf.iterator += sizeof(topic.angular_rate_setpoint);
	buf.offset += sizeof(topic.angular_rate_setpoint);
	static_assert(sizeof(topic.desired_position_raw) == 12, "size mismatch");
	memcpy(&topic.desired_position_raw, buf.iterator, sizeof(topic.desired_position_raw));
	buf.iterator += sizeof(topic.desired_position_raw);
	buf.offset += sizeof(topic.desired_position_raw);
	static_assert(sizeof(topic.in_emergency_ascent) == 1, "size mismatch");
	memcpy(&topic.in_emergency_ascent, buf.iterator, sizeof(topic.in_emergency_ascent));
	buf.iterator += sizeof(topic.in_emergency_ascent);
	buf.offset += sizeof(topic.in_emergency_ascent);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.gimbal_pitch) == 4, "size mismatch");
	memcpy(&topic.gimbal_pitch, buf.iterator, sizeof(topic.gimbal_pitch));
	buf.iterator += sizeof(topic.gimbal_pitch);
	buf.offset += sizeof(topic.gimbal_pitch);
	return true;
}
