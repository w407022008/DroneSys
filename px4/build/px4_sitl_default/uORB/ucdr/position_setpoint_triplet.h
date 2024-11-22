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
#include <uORB/topics/position_setpoint_triplet.h>

#include <uORB/ucdr/position_setpoint.h>
#include <uORB/ucdr/position_setpoint.h>
#include <uORB/ucdr/position_setpoint.h>

static inline constexpr int ucdr_topic_size_position_setpoint_triplet()
{
	return 269;
}

bool ucdr_serialize_position_setpoint_triplet(const position_setpoint_triplet_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.previous.timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.previous.timestamp, sizeof(topic.previous.timestamp));
	buf.iterator += sizeof(topic.previous.timestamp);
	buf.offset += sizeof(topic.previous.timestamp);
	static_assert(sizeof(topic.previous.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.valid, sizeof(topic.previous.valid));
	buf.iterator += sizeof(topic.previous.valid);
	buf.offset += sizeof(topic.previous.valid);
	static_assert(sizeof(topic.previous.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.type, sizeof(topic.previous.type));
	buf.iterator += sizeof(topic.previous.type);
	buf.offset += sizeof(topic.previous.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.previous.vx) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.vx, sizeof(topic.previous.vx));
	buf.iterator += sizeof(topic.previous.vx);
	buf.offset += sizeof(topic.previous.vx);
	static_assert(sizeof(topic.previous.vy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.vy, sizeof(topic.previous.vy));
	buf.iterator += sizeof(topic.previous.vy);
	buf.offset += sizeof(topic.previous.vy);
	static_assert(sizeof(topic.previous.vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.vz, sizeof(topic.previous.vz));
	buf.iterator += sizeof(topic.previous.vz);
	buf.offset += sizeof(topic.previous.vz);
	static_assert(sizeof(topic.previous.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.previous.lat, sizeof(topic.previous.lat));
	buf.iterator += sizeof(topic.previous.lat);
	buf.offset += sizeof(topic.previous.lat);
	static_assert(sizeof(topic.previous.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.previous.lon, sizeof(topic.previous.lon));
	buf.iterator += sizeof(topic.previous.lon);
	buf.offset += sizeof(topic.previous.lon);
	static_assert(sizeof(topic.previous.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.alt, sizeof(topic.previous.alt));
	buf.iterator += sizeof(topic.previous.alt);
	buf.offset += sizeof(topic.previous.alt);
	static_assert(sizeof(topic.previous.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.yaw, sizeof(topic.previous.yaw));
	buf.iterator += sizeof(topic.previous.yaw);
	buf.offset += sizeof(topic.previous.yaw);
	static_assert(sizeof(topic.previous.yaw_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.yaw_valid, sizeof(topic.previous.yaw_valid));
	buf.iterator += sizeof(topic.previous.yaw_valid);
	buf.offset += sizeof(topic.previous.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.yawspeed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.yawspeed, sizeof(topic.previous.yawspeed));
	buf.iterator += sizeof(topic.previous.yawspeed);
	buf.offset += sizeof(topic.previous.yawspeed);
	static_assert(sizeof(topic.previous.yawspeed_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.yawspeed_valid, sizeof(topic.previous.yawspeed_valid));
	buf.iterator += sizeof(topic.previous.yawspeed_valid);
	buf.offset += sizeof(topic.previous.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.loiter_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.loiter_radius, sizeof(topic.previous.loiter_radius));
	buf.iterator += sizeof(topic.previous.loiter_radius);
	buf.offset += sizeof(topic.previous.loiter_radius);
	static_assert(sizeof(topic.previous.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.loiter_direction_counter_clockwise, sizeof(topic.previous.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.previous.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.previous.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.acceptance_radius, sizeof(topic.previous.acceptance_radius));
	buf.iterator += sizeof(topic.previous.acceptance_radius);
	buf.offset += sizeof(topic.previous.acceptance_radius);
	static_assert(sizeof(topic.previous.cruising_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.cruising_speed, sizeof(topic.previous.cruising_speed));
	buf.iterator += sizeof(topic.previous.cruising_speed);
	buf.offset += sizeof(topic.previous.cruising_speed);
	static_assert(sizeof(topic.previous.gliding_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.gliding_enabled, sizeof(topic.previous.gliding_enabled));
	buf.iterator += sizeof(topic.previous.gliding_enabled);
	buf.offset += sizeof(topic.previous.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.cruising_throttle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.previous.cruising_throttle, sizeof(topic.previous.cruising_throttle));
	buf.iterator += sizeof(topic.previous.cruising_throttle);
	buf.offset += sizeof(topic.previous.cruising_throttle);
	static_assert(sizeof(topic.previous.disable_weather_vane) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.previous.disable_weather_vane, sizeof(topic.previous.disable_weather_vane));
	buf.iterator += sizeof(topic.previous.disable_weather_vane);
	buf.offset += sizeof(topic.previous.disable_weather_vane);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.current.timestamp, sizeof(topic.current.timestamp));
	buf.iterator += sizeof(topic.current.timestamp);
	buf.offset += sizeof(topic.current.timestamp);
	static_assert(sizeof(topic.current.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.valid, sizeof(topic.current.valid));
	buf.iterator += sizeof(topic.current.valid);
	buf.offset += sizeof(topic.current.valid);
	static_assert(sizeof(topic.current.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.type, sizeof(topic.current.type));
	buf.iterator += sizeof(topic.current.type);
	buf.offset += sizeof(topic.current.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.current.vx) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.vx, sizeof(topic.current.vx));
	buf.iterator += sizeof(topic.current.vx);
	buf.offset += sizeof(topic.current.vx);
	static_assert(sizeof(topic.current.vy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.vy, sizeof(topic.current.vy));
	buf.iterator += sizeof(topic.current.vy);
	buf.offset += sizeof(topic.current.vy);
	static_assert(sizeof(topic.current.vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.vz, sizeof(topic.current.vz));
	buf.iterator += sizeof(topic.current.vz);
	buf.offset += sizeof(topic.current.vz);
	static_assert(sizeof(topic.current.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.current.lat, sizeof(topic.current.lat));
	buf.iterator += sizeof(topic.current.lat);
	buf.offset += sizeof(topic.current.lat);
	static_assert(sizeof(topic.current.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.current.lon, sizeof(topic.current.lon));
	buf.iterator += sizeof(topic.current.lon);
	buf.offset += sizeof(topic.current.lon);
	static_assert(sizeof(topic.current.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.alt, sizeof(topic.current.alt));
	buf.iterator += sizeof(topic.current.alt);
	buf.offset += sizeof(topic.current.alt);
	static_assert(sizeof(topic.current.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.yaw, sizeof(topic.current.yaw));
	buf.iterator += sizeof(topic.current.yaw);
	buf.offset += sizeof(topic.current.yaw);
	static_assert(sizeof(topic.current.yaw_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.yaw_valid, sizeof(topic.current.yaw_valid));
	buf.iterator += sizeof(topic.current.yaw_valid);
	buf.offset += sizeof(topic.current.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.yawspeed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.yawspeed, sizeof(topic.current.yawspeed));
	buf.iterator += sizeof(topic.current.yawspeed);
	buf.offset += sizeof(topic.current.yawspeed);
	static_assert(sizeof(topic.current.yawspeed_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.yawspeed_valid, sizeof(topic.current.yawspeed_valid));
	buf.iterator += sizeof(topic.current.yawspeed_valid);
	buf.offset += sizeof(topic.current.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.loiter_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.loiter_radius, sizeof(topic.current.loiter_radius));
	buf.iterator += sizeof(topic.current.loiter_radius);
	buf.offset += sizeof(topic.current.loiter_radius);
	static_assert(sizeof(topic.current.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.loiter_direction_counter_clockwise, sizeof(topic.current.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.current.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.current.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.acceptance_radius, sizeof(topic.current.acceptance_radius));
	buf.iterator += sizeof(topic.current.acceptance_radius);
	buf.offset += sizeof(topic.current.acceptance_radius);
	static_assert(sizeof(topic.current.cruising_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.cruising_speed, sizeof(topic.current.cruising_speed));
	buf.iterator += sizeof(topic.current.cruising_speed);
	buf.offset += sizeof(topic.current.cruising_speed);
	static_assert(sizeof(topic.current.gliding_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.gliding_enabled, sizeof(topic.current.gliding_enabled));
	buf.iterator += sizeof(topic.current.gliding_enabled);
	buf.offset += sizeof(topic.current.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.cruising_throttle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current.cruising_throttle, sizeof(topic.current.cruising_throttle));
	buf.iterator += sizeof(topic.current.cruising_throttle);
	buf.offset += sizeof(topic.current.cruising_throttle);
	static_assert(sizeof(topic.current.disable_weather_vane) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.current.disable_weather_vane, sizeof(topic.current.disable_weather_vane));
	buf.iterator += sizeof(topic.current.disable_weather_vane);
	buf.offset += sizeof(topic.current.disable_weather_vane);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.next.timestamp, sizeof(topic.next.timestamp));
	buf.iterator += sizeof(topic.next.timestamp);
	buf.offset += sizeof(topic.next.timestamp);
	static_assert(sizeof(topic.next.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.valid, sizeof(topic.next.valid));
	buf.iterator += sizeof(topic.next.valid);
	buf.offset += sizeof(topic.next.valid);
	static_assert(sizeof(topic.next.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.type, sizeof(topic.next.type));
	buf.iterator += sizeof(topic.next.type);
	buf.offset += sizeof(topic.next.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.next.vx) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.vx, sizeof(topic.next.vx));
	buf.iterator += sizeof(topic.next.vx);
	buf.offset += sizeof(topic.next.vx);
	static_assert(sizeof(topic.next.vy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.vy, sizeof(topic.next.vy));
	buf.iterator += sizeof(topic.next.vy);
	buf.offset += sizeof(topic.next.vy);
	static_assert(sizeof(topic.next.vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.vz, sizeof(topic.next.vz));
	buf.iterator += sizeof(topic.next.vz);
	buf.offset += sizeof(topic.next.vz);
	static_assert(sizeof(topic.next.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.next.lat, sizeof(topic.next.lat));
	buf.iterator += sizeof(topic.next.lat);
	buf.offset += sizeof(topic.next.lat);
	static_assert(sizeof(topic.next.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.next.lon, sizeof(topic.next.lon));
	buf.iterator += sizeof(topic.next.lon);
	buf.offset += sizeof(topic.next.lon);
	static_assert(sizeof(topic.next.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.alt, sizeof(topic.next.alt));
	buf.iterator += sizeof(topic.next.alt);
	buf.offset += sizeof(topic.next.alt);
	static_assert(sizeof(topic.next.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.yaw, sizeof(topic.next.yaw));
	buf.iterator += sizeof(topic.next.yaw);
	buf.offset += sizeof(topic.next.yaw);
	static_assert(sizeof(topic.next.yaw_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.yaw_valid, sizeof(topic.next.yaw_valid));
	buf.iterator += sizeof(topic.next.yaw_valid);
	buf.offset += sizeof(topic.next.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.yawspeed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.yawspeed, sizeof(topic.next.yawspeed));
	buf.iterator += sizeof(topic.next.yawspeed);
	buf.offset += sizeof(topic.next.yawspeed);
	static_assert(sizeof(topic.next.yawspeed_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.yawspeed_valid, sizeof(topic.next.yawspeed_valid));
	buf.iterator += sizeof(topic.next.yawspeed_valid);
	buf.offset += sizeof(topic.next.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.loiter_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.loiter_radius, sizeof(topic.next.loiter_radius));
	buf.iterator += sizeof(topic.next.loiter_radius);
	buf.offset += sizeof(topic.next.loiter_radius);
	static_assert(sizeof(topic.next.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.loiter_direction_counter_clockwise, sizeof(topic.next.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.next.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.next.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.acceptance_radius, sizeof(topic.next.acceptance_radius));
	buf.iterator += sizeof(topic.next.acceptance_radius);
	buf.offset += sizeof(topic.next.acceptance_radius);
	static_assert(sizeof(topic.next.cruising_speed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.cruising_speed, sizeof(topic.next.cruising_speed));
	buf.iterator += sizeof(topic.next.cruising_speed);
	buf.offset += sizeof(topic.next.cruising_speed);
	static_assert(sizeof(topic.next.gliding_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.gliding_enabled, sizeof(topic.next.gliding_enabled));
	buf.iterator += sizeof(topic.next.gliding_enabled);
	buf.offset += sizeof(topic.next.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.cruising_throttle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.next.cruising_throttle, sizeof(topic.next.cruising_throttle));
	buf.iterator += sizeof(topic.next.cruising_throttle);
	buf.offset += sizeof(topic.next.cruising_throttle);
	static_assert(sizeof(topic.next.disable_weather_vane) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.next.disable_weather_vane, sizeof(topic.next.disable_weather_vane));
	buf.iterator += sizeof(topic.next.disable_weather_vane);
	buf.offset += sizeof(topic.next.disable_weather_vane);
	return true;
}

bool ucdr_deserialize_position_setpoint_triplet(ucdrBuffer& buf, position_setpoint_triplet_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.previous.timestamp) == 8, "size mismatch");
	memcpy(&topic.previous.timestamp, buf.iterator, sizeof(topic.previous.timestamp));
	buf.iterator += sizeof(topic.previous.timestamp);
	buf.offset += sizeof(topic.previous.timestamp);
	static_assert(sizeof(topic.previous.valid) == 1, "size mismatch");
	memcpy(&topic.previous.valid, buf.iterator, sizeof(topic.previous.valid));
	buf.iterator += sizeof(topic.previous.valid);
	buf.offset += sizeof(topic.previous.valid);
	static_assert(sizeof(topic.previous.type) == 1, "size mismatch");
	memcpy(&topic.previous.type, buf.iterator, sizeof(topic.previous.type));
	buf.iterator += sizeof(topic.previous.type);
	buf.offset += sizeof(topic.previous.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.previous.vx) == 4, "size mismatch");
	memcpy(&topic.previous.vx, buf.iterator, sizeof(topic.previous.vx));
	buf.iterator += sizeof(topic.previous.vx);
	buf.offset += sizeof(topic.previous.vx);
	static_assert(sizeof(topic.previous.vy) == 4, "size mismatch");
	memcpy(&topic.previous.vy, buf.iterator, sizeof(topic.previous.vy));
	buf.iterator += sizeof(topic.previous.vy);
	buf.offset += sizeof(topic.previous.vy);
	static_assert(sizeof(topic.previous.vz) == 4, "size mismatch");
	memcpy(&topic.previous.vz, buf.iterator, sizeof(topic.previous.vz));
	buf.iterator += sizeof(topic.previous.vz);
	buf.offset += sizeof(topic.previous.vz);
	static_assert(sizeof(topic.previous.lat) == 8, "size mismatch");
	memcpy(&topic.previous.lat, buf.iterator, sizeof(topic.previous.lat));
	buf.iterator += sizeof(topic.previous.lat);
	buf.offset += sizeof(topic.previous.lat);
	static_assert(sizeof(topic.previous.lon) == 8, "size mismatch");
	memcpy(&topic.previous.lon, buf.iterator, sizeof(topic.previous.lon));
	buf.iterator += sizeof(topic.previous.lon);
	buf.offset += sizeof(topic.previous.lon);
	static_assert(sizeof(topic.previous.alt) == 4, "size mismatch");
	memcpy(&topic.previous.alt, buf.iterator, sizeof(topic.previous.alt));
	buf.iterator += sizeof(topic.previous.alt);
	buf.offset += sizeof(topic.previous.alt);
	static_assert(sizeof(topic.previous.yaw) == 4, "size mismatch");
	memcpy(&topic.previous.yaw, buf.iterator, sizeof(topic.previous.yaw));
	buf.iterator += sizeof(topic.previous.yaw);
	buf.offset += sizeof(topic.previous.yaw);
	static_assert(sizeof(topic.previous.yaw_valid) == 1, "size mismatch");
	memcpy(&topic.previous.yaw_valid, buf.iterator, sizeof(topic.previous.yaw_valid));
	buf.iterator += sizeof(topic.previous.yaw_valid);
	buf.offset += sizeof(topic.previous.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.yawspeed) == 4, "size mismatch");
	memcpy(&topic.previous.yawspeed, buf.iterator, sizeof(topic.previous.yawspeed));
	buf.iterator += sizeof(topic.previous.yawspeed);
	buf.offset += sizeof(topic.previous.yawspeed);
	static_assert(sizeof(topic.previous.yawspeed_valid) == 1, "size mismatch");
	memcpy(&topic.previous.yawspeed_valid, buf.iterator, sizeof(topic.previous.yawspeed_valid));
	buf.iterator += sizeof(topic.previous.yawspeed_valid);
	buf.offset += sizeof(topic.previous.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.loiter_radius) == 4, "size mismatch");
	memcpy(&topic.previous.loiter_radius, buf.iterator, sizeof(topic.previous.loiter_radius));
	buf.iterator += sizeof(topic.previous.loiter_radius);
	buf.offset += sizeof(topic.previous.loiter_radius);
	static_assert(sizeof(topic.previous.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(&topic.previous.loiter_direction_counter_clockwise, buf.iterator, sizeof(topic.previous.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.previous.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.previous.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.previous.acceptance_radius, buf.iterator, sizeof(topic.previous.acceptance_radius));
	buf.iterator += sizeof(topic.previous.acceptance_radius);
	buf.offset += sizeof(topic.previous.acceptance_radius);
	static_assert(sizeof(topic.previous.cruising_speed) == 4, "size mismatch");
	memcpy(&topic.previous.cruising_speed, buf.iterator, sizeof(topic.previous.cruising_speed));
	buf.iterator += sizeof(topic.previous.cruising_speed);
	buf.offset += sizeof(topic.previous.cruising_speed);
	static_assert(sizeof(topic.previous.gliding_enabled) == 1, "size mismatch");
	memcpy(&topic.previous.gliding_enabled, buf.iterator, sizeof(topic.previous.gliding_enabled));
	buf.iterator += sizeof(topic.previous.gliding_enabled);
	buf.offset += sizeof(topic.previous.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.previous.cruising_throttle) == 4, "size mismatch");
	memcpy(&topic.previous.cruising_throttle, buf.iterator, sizeof(topic.previous.cruising_throttle));
	buf.iterator += sizeof(topic.previous.cruising_throttle);
	buf.offset += sizeof(topic.previous.cruising_throttle);
	static_assert(sizeof(topic.previous.disable_weather_vane) == 1, "size mismatch");
	memcpy(&topic.previous.disable_weather_vane, buf.iterator, sizeof(topic.previous.disable_weather_vane));
	buf.iterator += sizeof(topic.previous.disable_weather_vane);
	buf.offset += sizeof(topic.previous.disable_weather_vane);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.timestamp) == 8, "size mismatch");
	memcpy(&topic.current.timestamp, buf.iterator, sizeof(topic.current.timestamp));
	buf.iterator += sizeof(topic.current.timestamp);
	buf.offset += sizeof(topic.current.timestamp);
	static_assert(sizeof(topic.current.valid) == 1, "size mismatch");
	memcpy(&topic.current.valid, buf.iterator, sizeof(topic.current.valid));
	buf.iterator += sizeof(topic.current.valid);
	buf.offset += sizeof(topic.current.valid);
	static_assert(sizeof(topic.current.type) == 1, "size mismatch");
	memcpy(&topic.current.type, buf.iterator, sizeof(topic.current.type));
	buf.iterator += sizeof(topic.current.type);
	buf.offset += sizeof(topic.current.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.current.vx) == 4, "size mismatch");
	memcpy(&topic.current.vx, buf.iterator, sizeof(topic.current.vx));
	buf.iterator += sizeof(topic.current.vx);
	buf.offset += sizeof(topic.current.vx);
	static_assert(sizeof(topic.current.vy) == 4, "size mismatch");
	memcpy(&topic.current.vy, buf.iterator, sizeof(topic.current.vy));
	buf.iterator += sizeof(topic.current.vy);
	buf.offset += sizeof(topic.current.vy);
	static_assert(sizeof(topic.current.vz) == 4, "size mismatch");
	memcpy(&topic.current.vz, buf.iterator, sizeof(topic.current.vz));
	buf.iterator += sizeof(topic.current.vz);
	buf.offset += sizeof(topic.current.vz);
	static_assert(sizeof(topic.current.lat) == 8, "size mismatch");
	memcpy(&topic.current.lat, buf.iterator, sizeof(topic.current.lat));
	buf.iterator += sizeof(topic.current.lat);
	buf.offset += sizeof(topic.current.lat);
	static_assert(sizeof(topic.current.lon) == 8, "size mismatch");
	memcpy(&topic.current.lon, buf.iterator, sizeof(topic.current.lon));
	buf.iterator += sizeof(topic.current.lon);
	buf.offset += sizeof(topic.current.lon);
	static_assert(sizeof(topic.current.alt) == 4, "size mismatch");
	memcpy(&topic.current.alt, buf.iterator, sizeof(topic.current.alt));
	buf.iterator += sizeof(topic.current.alt);
	buf.offset += sizeof(topic.current.alt);
	static_assert(sizeof(topic.current.yaw) == 4, "size mismatch");
	memcpy(&topic.current.yaw, buf.iterator, sizeof(topic.current.yaw));
	buf.iterator += sizeof(topic.current.yaw);
	buf.offset += sizeof(topic.current.yaw);
	static_assert(sizeof(topic.current.yaw_valid) == 1, "size mismatch");
	memcpy(&topic.current.yaw_valid, buf.iterator, sizeof(topic.current.yaw_valid));
	buf.iterator += sizeof(topic.current.yaw_valid);
	buf.offset += sizeof(topic.current.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.yawspeed) == 4, "size mismatch");
	memcpy(&topic.current.yawspeed, buf.iterator, sizeof(topic.current.yawspeed));
	buf.iterator += sizeof(topic.current.yawspeed);
	buf.offset += sizeof(topic.current.yawspeed);
	static_assert(sizeof(topic.current.yawspeed_valid) == 1, "size mismatch");
	memcpy(&topic.current.yawspeed_valid, buf.iterator, sizeof(topic.current.yawspeed_valid));
	buf.iterator += sizeof(topic.current.yawspeed_valid);
	buf.offset += sizeof(topic.current.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.loiter_radius) == 4, "size mismatch");
	memcpy(&topic.current.loiter_radius, buf.iterator, sizeof(topic.current.loiter_radius));
	buf.iterator += sizeof(topic.current.loiter_radius);
	buf.offset += sizeof(topic.current.loiter_radius);
	static_assert(sizeof(topic.current.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(&topic.current.loiter_direction_counter_clockwise, buf.iterator, sizeof(topic.current.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.current.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.current.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.current.acceptance_radius, buf.iterator, sizeof(topic.current.acceptance_radius));
	buf.iterator += sizeof(topic.current.acceptance_radius);
	buf.offset += sizeof(topic.current.acceptance_radius);
	static_assert(sizeof(topic.current.cruising_speed) == 4, "size mismatch");
	memcpy(&topic.current.cruising_speed, buf.iterator, sizeof(topic.current.cruising_speed));
	buf.iterator += sizeof(topic.current.cruising_speed);
	buf.offset += sizeof(topic.current.cruising_speed);
	static_assert(sizeof(topic.current.gliding_enabled) == 1, "size mismatch");
	memcpy(&topic.current.gliding_enabled, buf.iterator, sizeof(topic.current.gliding_enabled));
	buf.iterator += sizeof(topic.current.gliding_enabled);
	buf.offset += sizeof(topic.current.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.current.cruising_throttle) == 4, "size mismatch");
	memcpy(&topic.current.cruising_throttle, buf.iterator, sizeof(topic.current.cruising_throttle));
	buf.iterator += sizeof(topic.current.cruising_throttle);
	buf.offset += sizeof(topic.current.cruising_throttle);
	static_assert(sizeof(topic.current.disable_weather_vane) == 1, "size mismatch");
	memcpy(&topic.current.disable_weather_vane, buf.iterator, sizeof(topic.current.disable_weather_vane));
	buf.iterator += sizeof(topic.current.disable_weather_vane);
	buf.offset += sizeof(topic.current.disable_weather_vane);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.timestamp) == 8, "size mismatch");
	memcpy(&topic.next.timestamp, buf.iterator, sizeof(topic.next.timestamp));
	buf.iterator += sizeof(topic.next.timestamp);
	buf.offset += sizeof(topic.next.timestamp);
	static_assert(sizeof(topic.next.valid) == 1, "size mismatch");
	memcpy(&topic.next.valid, buf.iterator, sizeof(topic.next.valid));
	buf.iterator += sizeof(topic.next.valid);
	buf.offset += sizeof(topic.next.valid);
	static_assert(sizeof(topic.next.type) == 1, "size mismatch");
	memcpy(&topic.next.type, buf.iterator, sizeof(topic.next.type));
	buf.iterator += sizeof(topic.next.type);
	buf.offset += sizeof(topic.next.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.next.vx) == 4, "size mismatch");
	memcpy(&topic.next.vx, buf.iterator, sizeof(topic.next.vx));
	buf.iterator += sizeof(topic.next.vx);
	buf.offset += sizeof(topic.next.vx);
	static_assert(sizeof(topic.next.vy) == 4, "size mismatch");
	memcpy(&topic.next.vy, buf.iterator, sizeof(topic.next.vy));
	buf.iterator += sizeof(topic.next.vy);
	buf.offset += sizeof(topic.next.vy);
	static_assert(sizeof(topic.next.vz) == 4, "size mismatch");
	memcpy(&topic.next.vz, buf.iterator, sizeof(topic.next.vz));
	buf.iterator += sizeof(topic.next.vz);
	buf.offset += sizeof(topic.next.vz);
	static_assert(sizeof(topic.next.lat) == 8, "size mismatch");
	memcpy(&topic.next.lat, buf.iterator, sizeof(topic.next.lat));
	buf.iterator += sizeof(topic.next.lat);
	buf.offset += sizeof(topic.next.lat);
	static_assert(sizeof(topic.next.lon) == 8, "size mismatch");
	memcpy(&topic.next.lon, buf.iterator, sizeof(topic.next.lon));
	buf.iterator += sizeof(topic.next.lon);
	buf.offset += sizeof(topic.next.lon);
	static_assert(sizeof(topic.next.alt) == 4, "size mismatch");
	memcpy(&topic.next.alt, buf.iterator, sizeof(topic.next.alt));
	buf.iterator += sizeof(topic.next.alt);
	buf.offset += sizeof(topic.next.alt);
	static_assert(sizeof(topic.next.yaw) == 4, "size mismatch");
	memcpy(&topic.next.yaw, buf.iterator, sizeof(topic.next.yaw));
	buf.iterator += sizeof(topic.next.yaw);
	buf.offset += sizeof(topic.next.yaw);
	static_assert(sizeof(topic.next.yaw_valid) == 1, "size mismatch");
	memcpy(&topic.next.yaw_valid, buf.iterator, sizeof(topic.next.yaw_valid));
	buf.iterator += sizeof(topic.next.yaw_valid);
	buf.offset += sizeof(topic.next.yaw_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.yawspeed) == 4, "size mismatch");
	memcpy(&topic.next.yawspeed, buf.iterator, sizeof(topic.next.yawspeed));
	buf.iterator += sizeof(topic.next.yawspeed);
	buf.offset += sizeof(topic.next.yawspeed);
	static_assert(sizeof(topic.next.yawspeed_valid) == 1, "size mismatch");
	memcpy(&topic.next.yawspeed_valid, buf.iterator, sizeof(topic.next.yawspeed_valid));
	buf.iterator += sizeof(topic.next.yawspeed_valid);
	buf.offset += sizeof(topic.next.yawspeed_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.loiter_radius) == 4, "size mismatch");
	memcpy(&topic.next.loiter_radius, buf.iterator, sizeof(topic.next.loiter_radius));
	buf.iterator += sizeof(topic.next.loiter_radius);
	buf.offset += sizeof(topic.next.loiter_radius);
	static_assert(sizeof(topic.next.loiter_direction_counter_clockwise) == 1, "size mismatch");
	memcpy(&topic.next.loiter_direction_counter_clockwise, buf.iterator, sizeof(topic.next.loiter_direction_counter_clockwise));
	buf.iterator += sizeof(topic.next.loiter_direction_counter_clockwise);
	buf.offset += sizeof(topic.next.loiter_direction_counter_clockwise);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.next.acceptance_radius, buf.iterator, sizeof(topic.next.acceptance_radius));
	buf.iterator += sizeof(topic.next.acceptance_radius);
	buf.offset += sizeof(topic.next.acceptance_radius);
	static_assert(sizeof(topic.next.cruising_speed) == 4, "size mismatch");
	memcpy(&topic.next.cruising_speed, buf.iterator, sizeof(topic.next.cruising_speed));
	buf.iterator += sizeof(topic.next.cruising_speed);
	buf.offset += sizeof(topic.next.cruising_speed);
	static_assert(sizeof(topic.next.gliding_enabled) == 1, "size mismatch");
	memcpy(&topic.next.gliding_enabled, buf.iterator, sizeof(topic.next.gliding_enabled));
	buf.iterator += sizeof(topic.next.gliding_enabled);
	buf.offset += sizeof(topic.next.gliding_enabled);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.next.cruising_throttle) == 4, "size mismatch");
	memcpy(&topic.next.cruising_throttle, buf.iterator, sizeof(topic.next.cruising_throttle));
	buf.iterator += sizeof(topic.next.cruising_throttle);
	buf.offset += sizeof(topic.next.cruising_throttle);
	static_assert(sizeof(topic.next.disable_weather_vane) == 1, "size mismatch");
	memcpy(&topic.next.disable_weather_vane, buf.iterator, sizeof(topic.next.disable_weather_vane));
	buf.iterator += sizeof(topic.next.disable_weather_vane);
	buf.offset += sizeof(topic.next.disable_weather_vane);
	return true;
}
