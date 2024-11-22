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
#include <uORB/topics/position_setpoint.h>


static inline constexpr int ucdr_topic_size_position_setpoint()
{
	return 85;
}

bool ucdr_serialize_position_setpoint(const position_setpoint_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.vx) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vx, sizeof(topic.vx));
	buf.iterator += sizeof(topic.vx);
	buf.offset += sizeof(topic.vx);
	static_assert(sizeof(topic.vy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vy, sizeof(topic.vy));
	buf.iterator += sizeof(topic.vy);
	buf.offset += sizeof(topic.vy);
	static_assert(sizeof(topic.vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vz, sizeof(topic.vz));
	buf.iterator += sizeof(topic.vz);
	buf.offset += sizeof(topic.vz);
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lat, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lon, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.alt, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.yaw_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_valid, sizeof(topic.yaw_valid));
	buf.iterator += sizeof(topic.yaw_valid);
	buf.offset += sizeof(topic.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.yawspeed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yawspeed, sizeof(topic.yawspeed));
	buf.iterator += sizeof(topic.yawspeed);
	buf.offset += sizeof(topic.yawspeed);
	static_assert(sizeof(topic.yawspeed_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.yawspeed_valid, sizeof(topic.yawspeed_valid));
	buf.iterator += sizeof(topic.yawspeed_valid);
	buf.offset += sizeof(topic.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.loiter_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.loiter_radius, sizeof(topic.loiter_radius));
	buf.iterator += sizeof(topic.loiter_radius);
	buf.offset += sizeof(topic.loiter_radius);
	static_assert(sizeof(topic.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.loiter_direction_counter_clockwise, sizeof(topic.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.acceptance_radius, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.cruising_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cruising_speed, sizeof(topic.cruising_speed));
	buf.iterator += sizeof(topic.cruising_speed);
	buf.offset += sizeof(topic.cruising_speed);
	static_assert(sizeof(topic.gliding_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gliding_enabled, sizeof(topic.gliding_enabled));
	buf.iterator += sizeof(topic.gliding_enabled);
	buf.offset += sizeof(topic.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.cruising_throttle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cruising_throttle, sizeof(topic.cruising_throttle));
	buf.iterator += sizeof(topic.cruising_throttle);
	buf.offset += sizeof(topic.cruising_throttle);
	static_assert(sizeof(topic.disable_weather_vane) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.disable_weather_vane, sizeof(topic.disable_weather_vane));
	buf.iterator += sizeof(topic.disable_weather_vane);
	buf.offset += sizeof(topic.disable_weather_vane);
	return true;
}

bool ucdr_deserialize_position_setpoint(ucdrBuffer& buf, position_setpoint_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(&topic.valid, buf.iterator, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.vx) == 4, "size mismatch");
	memcpy(&topic.vx, buf.iterator, sizeof(topic.vx));
	buf.iterator += sizeof(topic.vx);
	buf.offset += sizeof(topic.vx);
	static_assert(sizeof(topic.vy) == 4, "size mismatch");
	memcpy(&topic.vy, buf.iterator, sizeof(topic.vy));
	buf.iterator += sizeof(topic.vy);
	buf.offset += sizeof(topic.vy);
	static_assert(sizeof(topic.vz) == 4, "size mismatch");
	memcpy(&topic.vz, buf.iterator, sizeof(topic.vz));
	buf.iterator += sizeof(topic.vz);
	buf.offset += sizeof(topic.vz);
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(&topic.lat, buf.iterator, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(&topic.lon, buf.iterator, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(&topic.alt, buf.iterator, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.yaw_valid) == 1, "size mismatch");
	memcpy(&topic.yaw_valid, buf.iterator, sizeof(topic.yaw_valid));
	buf.iterator += sizeof(topic.yaw_valid);
	buf.offset += sizeof(topic.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.yawspeed) == 4, "size mismatch");
	memcpy(&topic.yawspeed, buf.iterator, sizeof(topic.yawspeed));
	buf.iterator += sizeof(topic.yawspeed);
	buf.offset += sizeof(topic.yawspeed);
	static_assert(sizeof(topic.yawspeed_valid) == 1, "size mismatch");
	memcpy(&topic.yawspeed_valid, buf.iterator, sizeof(topic.yawspeed_valid));
	buf.iterator += sizeof(topic.yawspeed_valid);
	buf.offset += sizeof(topic.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.loiter_radius) == 4, "size mismatch");
	memcpy(&topic.loiter_radius, buf.iterator, sizeof(topic.loiter_radius));
	buf.iterator += sizeof(topic.loiter_radius);
	buf.offset += sizeof(topic.loiter_radius);
	static_assert(sizeof(topic.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(&topic.loiter_direction_counter_clockwise, buf.iterator, sizeof(topic.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.acceptance_radius, buf.iterator, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.cruising_speed) == 4, "size mismatch");
	memcpy(&topic.cruising_speed, buf.iterator, sizeof(topic.cruising_speed));
	buf.iterator += sizeof(topic.cruising_speed);
	buf.offset += sizeof(topic.cruising_speed);
	static_assert(sizeof(topic.gliding_enabled) == 1, "size mismatch");
	memcpy(&topic.gliding_enabled, buf.iterator, sizeof(topic.gliding_enabled));
	buf.iterator += sizeof(topic.gliding_enabled);
	buf.offset += sizeof(topic.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.cruising_throttle) == 4, "size mismatch");
	memcpy(&topic.cruising_throttle, buf.iterator, sizeof(topic.cruising_throttle));
	buf.iterator += sizeof(topic.cruising_throttle);
	buf.offset += sizeof(topic.cruising_throttle);
	static_assert(sizeof(topic.disable_weather_vane) == 1, "size mismatch");
	memcpy(&topic.disable_weather_vane, buf.iterator, sizeof(topic.disable_weather_vane));
	buf.iterator += sizeof(topic.disable_weather_vane);
	buf.offset += sizeof(topic.disable_weather_vane);
	return true;
}
