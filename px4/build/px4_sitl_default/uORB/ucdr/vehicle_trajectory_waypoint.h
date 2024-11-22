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
#include <uORB/topics/vehicle_trajectory_waypoint.h>

#include <uORB/ucdr/trajectory_waypoint.h>

static inline constexpr int ucdr_topic_size_vehicle_trajectory_waypoint()
{
	return 294;
}

bool ucdr_serialize_vehicle_trajectory_waypoint(const vehicle_trajectory_waypoint_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 7; // padding
	buf.offset += 7; // padding
	static_assert(sizeof(topic.waypoints[0].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].timestamp, sizeof(topic.waypoints[0].timestamp));
	buf.iterator += sizeof(topic.waypoints[0].timestamp);
	buf.offset += sizeof(topic.waypoints[0].timestamp);
	static_assert(sizeof(topic.waypoints[0].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].position, sizeof(topic.waypoints[0].position));
	buf.iterator += sizeof(topic.waypoints[0].position);
	buf.offset += sizeof(topic.waypoints[0].position);
	static_assert(sizeof(topic.waypoints[0].velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].velocity, sizeof(topic.waypoints[0].velocity));
	buf.iterator += sizeof(topic.waypoints[0].velocity);
	buf.offset += sizeof(topic.waypoints[0].velocity);
	static_assert(sizeof(topic.waypoints[0].acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].acceleration, sizeof(topic.waypoints[0].acceleration));
	buf.iterator += sizeof(topic.waypoints[0].acceleration);
	buf.offset += sizeof(topic.waypoints[0].acceleration);
	static_assert(sizeof(topic.waypoints[0].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].yaw, sizeof(topic.waypoints[0].yaw));
	buf.iterator += sizeof(topic.waypoints[0].yaw);
	buf.offset += sizeof(topic.waypoints[0].yaw);
	static_assert(sizeof(topic.waypoints[0].yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].yaw_speed, sizeof(topic.waypoints[0].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[0].yaw_speed);
	buf.offset += sizeof(topic.waypoints[0].yaw_speed);
	static_assert(sizeof(topic.waypoints[0].point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].point_valid, sizeof(topic.waypoints[0].point_valid));
	buf.iterator += sizeof(topic.waypoints[0].point_valid);
	buf.offset += sizeof(topic.waypoints[0].point_valid);
	static_assert(sizeof(topic.waypoints[0].type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[0].type, sizeof(topic.waypoints[0].type));
	buf.iterator += sizeof(topic.waypoints[0].type);
	buf.offset += sizeof(topic.waypoints[0].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[1].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].timestamp, sizeof(topic.waypoints[1].timestamp));
	buf.iterator += sizeof(topic.waypoints[1].timestamp);
	buf.offset += sizeof(topic.waypoints[1].timestamp);
	static_assert(sizeof(topic.waypoints[1].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].position, sizeof(topic.waypoints[1].position));
	buf.iterator += sizeof(topic.waypoints[1].position);
	buf.offset += sizeof(topic.waypoints[1].position);
	static_assert(sizeof(topic.waypoints[1].velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].velocity, sizeof(topic.waypoints[1].velocity));
	buf.iterator += sizeof(topic.waypoints[1].velocity);
	buf.offset += sizeof(topic.waypoints[1].velocity);
	static_assert(sizeof(topic.waypoints[1].acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].acceleration, sizeof(topic.waypoints[1].acceleration));
	buf.iterator += sizeof(topic.waypoints[1].acceleration);
	buf.offset += sizeof(topic.waypoints[1].acceleration);
	static_assert(sizeof(topic.waypoints[1].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].yaw, sizeof(topic.waypoints[1].yaw));
	buf.iterator += sizeof(topic.waypoints[1].yaw);
	buf.offset += sizeof(topic.waypoints[1].yaw);
	static_assert(sizeof(topic.waypoints[1].yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].yaw_speed, sizeof(topic.waypoints[1].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[1].yaw_speed);
	buf.offset += sizeof(topic.waypoints[1].yaw_speed);
	static_assert(sizeof(topic.waypoints[1].point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].point_valid, sizeof(topic.waypoints[1].point_valid));
	buf.iterator += sizeof(topic.waypoints[1].point_valid);
	buf.offset += sizeof(topic.waypoints[1].point_valid);
	static_assert(sizeof(topic.waypoints[1].type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[1].type, sizeof(topic.waypoints[1].type));
	buf.iterator += sizeof(topic.waypoints[1].type);
	buf.offset += sizeof(topic.waypoints[1].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[2].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].timestamp, sizeof(topic.waypoints[2].timestamp));
	buf.iterator += sizeof(topic.waypoints[2].timestamp);
	buf.offset += sizeof(topic.waypoints[2].timestamp);
	static_assert(sizeof(topic.waypoints[2].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].position, sizeof(topic.waypoints[2].position));
	buf.iterator += sizeof(topic.waypoints[2].position);
	buf.offset += sizeof(topic.waypoints[2].position);
	static_assert(sizeof(topic.waypoints[2].velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].velocity, sizeof(topic.waypoints[2].velocity));
	buf.iterator += sizeof(topic.waypoints[2].velocity);
	buf.offset += sizeof(topic.waypoints[2].velocity);
	static_assert(sizeof(topic.waypoints[2].acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].acceleration, sizeof(topic.waypoints[2].acceleration));
	buf.iterator += sizeof(topic.waypoints[2].acceleration);
	buf.offset += sizeof(topic.waypoints[2].acceleration);
	static_assert(sizeof(topic.waypoints[2].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].yaw, sizeof(topic.waypoints[2].yaw));
	buf.iterator += sizeof(topic.waypoints[2].yaw);
	buf.offset += sizeof(topic.waypoints[2].yaw);
	static_assert(sizeof(topic.waypoints[2].yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].yaw_speed, sizeof(topic.waypoints[2].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[2].yaw_speed);
	buf.offset += sizeof(topic.waypoints[2].yaw_speed);
	static_assert(sizeof(topic.waypoints[2].point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].point_valid, sizeof(topic.waypoints[2].point_valid));
	buf.iterator += sizeof(topic.waypoints[2].point_valid);
	buf.offset += sizeof(topic.waypoints[2].point_valid);
	static_assert(sizeof(topic.waypoints[2].type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[2].type, sizeof(topic.waypoints[2].type));
	buf.iterator += sizeof(topic.waypoints[2].type);
	buf.offset += sizeof(topic.waypoints[2].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[3].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].timestamp, sizeof(topic.waypoints[3].timestamp));
	buf.iterator += sizeof(topic.waypoints[3].timestamp);
	buf.offset += sizeof(topic.waypoints[3].timestamp);
	static_assert(sizeof(topic.waypoints[3].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].position, sizeof(topic.waypoints[3].position));
	buf.iterator += sizeof(topic.waypoints[3].position);
	buf.offset += sizeof(topic.waypoints[3].position);
	static_assert(sizeof(topic.waypoints[3].velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].velocity, sizeof(topic.waypoints[3].velocity));
	buf.iterator += sizeof(topic.waypoints[3].velocity);
	buf.offset += sizeof(topic.waypoints[3].velocity);
	static_assert(sizeof(topic.waypoints[3].acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].acceleration, sizeof(topic.waypoints[3].acceleration));
	buf.iterator += sizeof(topic.waypoints[3].acceleration);
	buf.offset += sizeof(topic.waypoints[3].acceleration);
	static_assert(sizeof(topic.waypoints[3].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].yaw, sizeof(topic.waypoints[3].yaw));
	buf.iterator += sizeof(topic.waypoints[3].yaw);
	buf.offset += sizeof(topic.waypoints[3].yaw);
	static_assert(sizeof(topic.waypoints[3].yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].yaw_speed, sizeof(topic.waypoints[3].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[3].yaw_speed);
	buf.offset += sizeof(topic.waypoints[3].yaw_speed);
	static_assert(sizeof(topic.waypoints[3].point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].point_valid, sizeof(topic.waypoints[3].point_valid));
	buf.iterator += sizeof(topic.waypoints[3].point_valid);
	buf.offset += sizeof(topic.waypoints[3].point_valid);
	static_assert(sizeof(topic.waypoints[3].type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[3].type, sizeof(topic.waypoints[3].type));
	buf.iterator += sizeof(topic.waypoints[3].type);
	buf.offset += sizeof(topic.waypoints[3].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[4].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].timestamp, sizeof(topic.waypoints[4].timestamp));
	buf.iterator += sizeof(topic.waypoints[4].timestamp);
	buf.offset += sizeof(topic.waypoints[4].timestamp);
	static_assert(sizeof(topic.waypoints[4].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].position, sizeof(topic.waypoints[4].position));
	buf.iterator += sizeof(topic.waypoints[4].position);
	buf.offset += sizeof(topic.waypoints[4].position);
	static_assert(sizeof(topic.waypoints[4].velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].velocity, sizeof(topic.waypoints[4].velocity));
	buf.iterator += sizeof(topic.waypoints[4].velocity);
	buf.offset += sizeof(topic.waypoints[4].velocity);
	static_assert(sizeof(topic.waypoints[4].acceleration) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].acceleration, sizeof(topic.waypoints[4].acceleration));
	buf.iterator += sizeof(topic.waypoints[4].acceleration);
	buf.offset += sizeof(topic.waypoints[4].acceleration);
	static_assert(sizeof(topic.waypoints[4].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].yaw, sizeof(topic.waypoints[4].yaw));
	buf.iterator += sizeof(topic.waypoints[4].yaw);
	buf.offset += sizeof(topic.waypoints[4].yaw);
	static_assert(sizeof(topic.waypoints[4].yaw_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].yaw_speed, sizeof(topic.waypoints[4].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[4].yaw_speed);
	buf.offset += sizeof(topic.waypoints[4].yaw_speed);
	static_assert(sizeof(topic.waypoints[4].point_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].point_valid, sizeof(topic.waypoints[4].point_valid));
	buf.iterator += sizeof(topic.waypoints[4].point_valid);
	buf.offset += sizeof(topic.waypoints[4].point_valid);
	static_assert(sizeof(topic.waypoints[4].type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.waypoints[4].type, sizeof(topic.waypoints[4].type));
	buf.iterator += sizeof(topic.waypoints[4].type);
	buf.offset += sizeof(topic.waypoints[4].type);
	return true;
}

bool ucdr_deserialize_vehicle_trajectory_waypoint(ucdrBuffer& buf, vehicle_trajectory_waypoint_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 7; // padding
	buf.offset += 7; // padding
	static_assert(sizeof(topic.waypoints[0].timestamp) == 8, "size mismatch");
	memcpy(&topic.waypoints[0].timestamp, buf.iterator, sizeof(topic.waypoints[0].timestamp));
	buf.iterator += sizeof(topic.waypoints[0].timestamp);
	buf.offset += sizeof(topic.waypoints[0].timestamp);
	static_assert(sizeof(topic.waypoints[0].position) == 12, "size mismatch");
	memcpy(&topic.waypoints[0].position, buf.iterator, sizeof(topic.waypoints[0].position));
	buf.iterator += sizeof(topic.waypoints[0].position);
	buf.offset += sizeof(topic.waypoints[0].position);
	static_assert(sizeof(topic.waypoints[0].velocity) == 12, "size mismatch");
	memcpy(&topic.waypoints[0].velocity, buf.iterator, sizeof(topic.waypoints[0].velocity));
	buf.iterator += sizeof(topic.waypoints[0].velocity);
	buf.offset += sizeof(topic.waypoints[0].velocity);
	static_assert(sizeof(topic.waypoints[0].acceleration) == 12, "size mismatch");
	memcpy(&topic.waypoints[0].acceleration, buf.iterator, sizeof(topic.waypoints[0].acceleration));
	buf.iterator += sizeof(topic.waypoints[0].acceleration);
	buf.offset += sizeof(topic.waypoints[0].acceleration);
	static_assert(sizeof(topic.waypoints[0].yaw) == 4, "size mismatch");
	memcpy(&topic.waypoints[0].yaw, buf.iterator, sizeof(topic.waypoints[0].yaw));
	buf.iterator += sizeof(topic.waypoints[0].yaw);
	buf.offset += sizeof(topic.waypoints[0].yaw);
	static_assert(sizeof(topic.waypoints[0].yaw_speed) == 4, "size mismatch");
	memcpy(&topic.waypoints[0].yaw_speed, buf.iterator, sizeof(topic.waypoints[0].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[0].yaw_speed);
	buf.offset += sizeof(topic.waypoints[0].yaw_speed);
	static_assert(sizeof(topic.waypoints[0].point_valid) == 1, "size mismatch");
	memcpy(&topic.waypoints[0].point_valid, buf.iterator, sizeof(topic.waypoints[0].point_valid));
	buf.iterator += sizeof(topic.waypoints[0].point_valid);
	buf.offset += sizeof(topic.waypoints[0].point_valid);
	static_assert(sizeof(topic.waypoints[0].type) == 1, "size mismatch");
	memcpy(&topic.waypoints[0].type, buf.iterator, sizeof(topic.waypoints[0].type));
	buf.iterator += sizeof(topic.waypoints[0].type);
	buf.offset += sizeof(topic.waypoints[0].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[1].timestamp) == 8, "size mismatch");
	memcpy(&topic.waypoints[1].timestamp, buf.iterator, sizeof(topic.waypoints[1].timestamp));
	buf.iterator += sizeof(topic.waypoints[1].timestamp);
	buf.offset += sizeof(topic.waypoints[1].timestamp);
	static_assert(sizeof(topic.waypoints[1].position) == 12, "size mismatch");
	memcpy(&topic.waypoints[1].position, buf.iterator, sizeof(topic.waypoints[1].position));
	buf.iterator += sizeof(topic.waypoints[1].position);
	buf.offset += sizeof(topic.waypoints[1].position);
	static_assert(sizeof(topic.waypoints[1].velocity) == 12, "size mismatch");
	memcpy(&topic.waypoints[1].velocity, buf.iterator, sizeof(topic.waypoints[1].velocity));
	buf.iterator += sizeof(topic.waypoints[1].velocity);
	buf.offset += sizeof(topic.waypoints[1].velocity);
	static_assert(sizeof(topic.waypoints[1].acceleration) == 12, "size mismatch");
	memcpy(&topic.waypoints[1].acceleration, buf.iterator, sizeof(topic.waypoints[1].acceleration));
	buf.iterator += sizeof(topic.waypoints[1].acceleration);
	buf.offset += sizeof(topic.waypoints[1].acceleration);
	static_assert(sizeof(topic.waypoints[1].yaw) == 4, "size mismatch");
	memcpy(&topic.waypoints[1].yaw, buf.iterator, sizeof(topic.waypoints[1].yaw));
	buf.iterator += sizeof(topic.waypoints[1].yaw);
	buf.offset += sizeof(topic.waypoints[1].yaw);
	static_assert(sizeof(topic.waypoints[1].yaw_speed) == 4, "size mismatch");
	memcpy(&topic.waypoints[1].yaw_speed, buf.iterator, sizeof(topic.waypoints[1].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[1].yaw_speed);
	buf.offset += sizeof(topic.waypoints[1].yaw_speed);
	static_assert(sizeof(topic.waypoints[1].point_valid) == 1, "size mismatch");
	memcpy(&topic.waypoints[1].point_valid, buf.iterator, sizeof(topic.waypoints[1].point_valid));
	buf.iterator += sizeof(topic.waypoints[1].point_valid);
	buf.offset += sizeof(topic.waypoints[1].point_valid);
	static_assert(sizeof(topic.waypoints[1].type) == 1, "size mismatch");
	memcpy(&topic.waypoints[1].type, buf.iterator, sizeof(topic.waypoints[1].type));
	buf.iterator += sizeof(topic.waypoints[1].type);
	buf.offset += sizeof(topic.waypoints[1].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[2].timestamp) == 8, "size mismatch");
	memcpy(&topic.waypoints[2].timestamp, buf.iterator, sizeof(topic.waypoints[2].timestamp));
	buf.iterator += sizeof(topic.waypoints[2].timestamp);
	buf.offset += sizeof(topic.waypoints[2].timestamp);
	static_assert(sizeof(topic.waypoints[2].position) == 12, "size mismatch");
	memcpy(&topic.waypoints[2].position, buf.iterator, sizeof(topic.waypoints[2].position));
	buf.iterator += sizeof(topic.waypoints[2].position);
	buf.offset += sizeof(topic.waypoints[2].position);
	static_assert(sizeof(topic.waypoints[2].velocity) == 12, "size mismatch");
	memcpy(&topic.waypoints[2].velocity, buf.iterator, sizeof(topic.waypoints[2].velocity));
	buf.iterator += sizeof(topic.waypoints[2].velocity);
	buf.offset += sizeof(topic.waypoints[2].velocity);
	static_assert(sizeof(topic.waypoints[2].acceleration) == 12, "size mismatch");
	memcpy(&topic.waypoints[2].acceleration, buf.iterator, sizeof(topic.waypoints[2].acceleration));
	buf.iterator += sizeof(topic.waypoints[2].acceleration);
	buf.offset += sizeof(topic.waypoints[2].acceleration);
	static_assert(sizeof(topic.waypoints[2].yaw) == 4, "size mismatch");
	memcpy(&topic.waypoints[2].yaw, buf.iterator, sizeof(topic.waypoints[2].yaw));
	buf.iterator += sizeof(topic.waypoints[2].yaw);
	buf.offset += sizeof(topic.waypoints[2].yaw);
	static_assert(sizeof(topic.waypoints[2].yaw_speed) == 4, "size mismatch");
	memcpy(&topic.waypoints[2].yaw_speed, buf.iterator, sizeof(topic.waypoints[2].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[2].yaw_speed);
	buf.offset += sizeof(topic.waypoints[2].yaw_speed);
	static_assert(sizeof(topic.waypoints[2].point_valid) == 1, "size mismatch");
	memcpy(&topic.waypoints[2].point_valid, buf.iterator, sizeof(topic.waypoints[2].point_valid));
	buf.iterator += sizeof(topic.waypoints[2].point_valid);
	buf.offset += sizeof(topic.waypoints[2].point_valid);
	static_assert(sizeof(topic.waypoints[2].type) == 1, "size mismatch");
	memcpy(&topic.waypoints[2].type, buf.iterator, sizeof(topic.waypoints[2].type));
	buf.iterator += sizeof(topic.waypoints[2].type);
	buf.offset += sizeof(topic.waypoints[2].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[3].timestamp) == 8, "size mismatch");
	memcpy(&topic.waypoints[3].timestamp, buf.iterator, sizeof(topic.waypoints[3].timestamp));
	buf.iterator += sizeof(topic.waypoints[3].timestamp);
	buf.offset += sizeof(topic.waypoints[3].timestamp);
	static_assert(sizeof(topic.waypoints[3].position) == 12, "size mismatch");
	memcpy(&topic.waypoints[3].position, buf.iterator, sizeof(topic.waypoints[3].position));
	buf.iterator += sizeof(topic.waypoints[3].position);
	buf.offset += sizeof(topic.waypoints[3].position);
	static_assert(sizeof(topic.waypoints[3].velocity) == 12, "size mismatch");
	memcpy(&topic.waypoints[3].velocity, buf.iterator, sizeof(topic.waypoints[3].velocity));
	buf.iterator += sizeof(topic.waypoints[3].velocity);
	buf.offset += sizeof(topic.waypoints[3].velocity);
	static_assert(sizeof(topic.waypoints[3].acceleration) == 12, "size mismatch");
	memcpy(&topic.waypoints[3].acceleration, buf.iterator, sizeof(topic.waypoints[3].acceleration));
	buf.iterator += sizeof(topic.waypoints[3].acceleration);
	buf.offset += sizeof(topic.waypoints[3].acceleration);
	static_assert(sizeof(topic.waypoints[3].yaw) == 4, "size mismatch");
	memcpy(&topic.waypoints[3].yaw, buf.iterator, sizeof(topic.waypoints[3].yaw));
	buf.iterator += sizeof(topic.waypoints[3].yaw);
	buf.offset += sizeof(topic.waypoints[3].yaw);
	static_assert(sizeof(topic.waypoints[3].yaw_speed) == 4, "size mismatch");
	memcpy(&topic.waypoints[3].yaw_speed, buf.iterator, sizeof(topic.waypoints[3].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[3].yaw_speed);
	buf.offset += sizeof(topic.waypoints[3].yaw_speed);
	static_assert(sizeof(topic.waypoints[3].point_valid) == 1, "size mismatch");
	memcpy(&topic.waypoints[3].point_valid, buf.iterator, sizeof(topic.waypoints[3].point_valid));
	buf.iterator += sizeof(topic.waypoints[3].point_valid);
	buf.offset += sizeof(topic.waypoints[3].point_valid);
	static_assert(sizeof(topic.waypoints[3].type) == 1, "size mismatch");
	memcpy(&topic.waypoints[3].type, buf.iterator, sizeof(topic.waypoints[3].type));
	buf.iterator += sizeof(topic.waypoints[3].type);
	buf.offset += sizeof(topic.waypoints[3].type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.waypoints[4].timestamp) == 8, "size mismatch");
	memcpy(&topic.waypoints[4].timestamp, buf.iterator, sizeof(topic.waypoints[4].timestamp));
	buf.iterator += sizeof(topic.waypoints[4].timestamp);
	buf.offset += sizeof(topic.waypoints[4].timestamp);
	static_assert(sizeof(topic.waypoints[4].position) == 12, "size mismatch");
	memcpy(&topic.waypoints[4].position, buf.iterator, sizeof(topic.waypoints[4].position));
	buf.iterator += sizeof(topic.waypoints[4].position);
	buf.offset += sizeof(topic.waypoints[4].position);
	static_assert(sizeof(topic.waypoints[4].velocity) == 12, "size mismatch");
	memcpy(&topic.waypoints[4].velocity, buf.iterator, sizeof(topic.waypoints[4].velocity));
	buf.iterator += sizeof(topic.waypoints[4].velocity);
	buf.offset += sizeof(topic.waypoints[4].velocity);
	static_assert(sizeof(topic.waypoints[4].acceleration) == 12, "size mismatch");
	memcpy(&topic.waypoints[4].acceleration, buf.iterator, sizeof(topic.waypoints[4].acceleration));
	buf.iterator += sizeof(topic.waypoints[4].acceleration);
	buf.offset += sizeof(topic.waypoints[4].acceleration);
	static_assert(sizeof(topic.waypoints[4].yaw) == 4, "size mismatch");
	memcpy(&topic.waypoints[4].yaw, buf.iterator, sizeof(topic.waypoints[4].yaw));
	buf.iterator += sizeof(topic.waypoints[4].yaw);
	buf.offset += sizeof(topic.waypoints[4].yaw);
	static_assert(sizeof(topic.waypoints[4].yaw_speed) == 4, "size mismatch");
	memcpy(&topic.waypoints[4].yaw_speed, buf.iterator, sizeof(topic.waypoints[4].yaw_speed));
	buf.iterator += sizeof(topic.waypoints[4].yaw_speed);
	buf.offset += sizeof(topic.waypoints[4].yaw_speed);
	static_assert(sizeof(topic.waypoints[4].point_valid) == 1, "size mismatch");
	memcpy(&topic.waypoints[4].point_valid, buf.iterator, sizeof(topic.waypoints[4].point_valid));
	buf.iterator += sizeof(topic.waypoints[4].point_valid);
	buf.offset += sizeof(topic.waypoints[4].point_valid);
	static_assert(sizeof(topic.waypoints[4].type) == 1, "size mismatch");
	memcpy(&topic.waypoints[4].type, buf.iterator, sizeof(topic.waypoints[4].type));
	buf.iterator += sizeof(topic.waypoints[4].type);
	buf.offset += sizeof(topic.waypoints[4].type);
	return true;
}
