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
#include <uORB/topics/trajectory_waypoint.h>


static inline constexpr int ucdr_topic_size_trajectory_waypoint()
{
	return 54;
}

bool ucdr_serialize_trajectory_waypoint(const trajectory_waypoint_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.position, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.velocity, sizeof(topic.velocity));
	buf.iterator += sizeof(topic.velocity);
	buf.offset += sizeof(topic.velocity);
	static_assert(sizeof(topic.acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.acceleration, sizeof(topic.acceleration));
	buf.iterator += sizeof(topic.acceleration);
	buf.offset += sizeof(topic.acceleration);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_speed, sizeof(topic.yaw_speed));
	buf.iterator += sizeof(topic.yaw_speed);
	buf.offset += sizeof(topic.yaw_speed);
	static_assert(sizeof(topic.point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.point_valid, sizeof(topic.point_valid));
	buf.iterator += sizeof(topic.point_valid);
	buf.offset += sizeof(topic.point_valid);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	return true;
}

bool ucdr_deserialize_trajectory_waypoint(ucdrBuffer& buf, trajectory_waypoint_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(&topic.position, buf.iterator, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.velocity) == 12, "size mismatch");
	memcpy(&topic.velocity, buf.iterator, sizeof(topic.velocity));
	buf.iterator += sizeof(topic.velocity);
	buf.offset += sizeof(topic.velocity);
	static_assert(sizeof(topic.acceleration) == 12, "size mismatch");
	memcpy(&topic.acceleration, buf.iterator, sizeof(topic.acceleration));
	buf.iterator += sizeof(topic.acceleration);
	buf.offset += sizeof(topic.acceleration);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.yaw_speed) == 4, "size mismatch");
	memcpy(&topic.yaw_speed, buf.iterator, sizeof(topic.yaw_speed));
	buf.iterator += sizeof(topic.yaw_speed);
	buf.offset += sizeof(topic.yaw_speed);
	static_assert(sizeof(topic.point_valid) == 1, "size mismatch");
	memcpy(&topic.point_valid, buf.iterator, sizeof(topic.point_valid));
	buf.iterator += sizeof(topic.point_valid);
	buf.offset += sizeof(topic.point_valid);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	return true;
}
