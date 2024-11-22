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
#include <uORB/topics/navigator_mission_item.h>


static inline constexpr int ucdr_topic_size_navigator_mission_item()
{
	return 51;
}

bool ucdr_serialize_navigator_mission_item(const navigator_mission_item_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.instance_count, sizeof(topic.instance_count));
	buf.iterator += sizeof(topic.instance_count);
	buf.offset += sizeof(topic.instance_count);
	static_assert(sizeof(topic.sequence_current) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.sequence_current, sizeof(topic.sequence_current));
	buf.iterator += sizeof(topic.sequence_current);
	buf.offset += sizeof(topic.sequence_current);
	static_assert(sizeof(topic.nav_cmd) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.nav_cmd, sizeof(topic.nav_cmd));
	buf.iterator += sizeof(topic.nav_cmd);
	buf.offset += sizeof(topic.nav_cmd);
	static_assert(sizeof(topic.latitude) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.latitude, sizeof(topic.latitude));
	buf.iterator += sizeof(topic.latitude);
	buf.offset += sizeof(topic.latitude);
	static_assert(sizeof(topic.longitude) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.longitude, sizeof(topic.longitude));
	buf.iterator += sizeof(topic.longitude);
	buf.offset += sizeof(topic.longitude);
	static_assert(sizeof(topic.time_inside) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.time_inside, sizeof(topic.time_inside));
	buf.iterator += sizeof(topic.time_inside);
	buf.offset += sizeof(topic.time_inside);
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.acceptance_radius, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.loiter_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.loiter_radius, sizeof(topic.loiter_radius));
	buf.iterator += sizeof(topic.loiter_radius);
	buf.offset += sizeof(topic.loiter_radius);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.altitude) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.altitude, sizeof(topic.altitude));
	buf.iterator += sizeof(topic.altitude);
	buf.offset += sizeof(topic.altitude);
	static_assert(sizeof(topic.frame) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.frame, sizeof(topic.frame));
	buf.iterator += sizeof(topic.frame);
	buf.offset += sizeof(topic.frame);
	static_assert(sizeof(topic.origin) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.origin, sizeof(topic.origin));
	buf.iterator += sizeof(topic.origin);
	buf.offset += sizeof(topic.origin);
	static_assert(sizeof(topic.loiter_exit_xtrack) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.loiter_exit_xtrack, sizeof(topic.loiter_exit_xtrack));
	buf.iterator += sizeof(topic.loiter_exit_xtrack);
	buf.offset += sizeof(topic.loiter_exit_xtrack);
	static_assert(sizeof(topic.force_heading) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.force_heading, sizeof(topic.force_heading));
	buf.iterator += sizeof(topic.force_heading);
	buf.offset += sizeof(topic.force_heading);
	static_assert(sizeof(topic.altitude_is_relative) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.altitude_is_relative, sizeof(topic.altitude_is_relative));
	buf.iterator += sizeof(topic.altitude_is_relative);
	buf.offset += sizeof(topic.altitude_is_relative);
	static_assert(sizeof(topic.autocontinue) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.autocontinue, sizeof(topic.autocontinue));
	buf.iterator += sizeof(topic.autocontinue);
	buf.offset += sizeof(topic.autocontinue);
	static_assert(sizeof(topic.vtol_back_transition) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vtol_back_transition, sizeof(topic.vtol_back_transition));
	buf.iterator += sizeof(topic.vtol_back_transition);
	buf.offset += sizeof(topic.vtol_back_transition);
	return true;
}

bool ucdr_deserialize_navigator_mission_item(ucdrBuffer& buf, navigator_mission_item_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance_count) == 4, "size mismatch");
	memcpy(&topic.instance_count, buf.iterator, sizeof(topic.instance_count));
	buf.iterator += sizeof(topic.instance_count);
	buf.offset += sizeof(topic.instance_count);
	static_assert(sizeof(topic.sequence_current) == 2, "size mismatch");
	memcpy(&topic.sequence_current, buf.iterator, sizeof(topic.sequence_current));
	buf.iterator += sizeof(topic.sequence_current);
	buf.offset += sizeof(topic.sequence_current);
	static_assert(sizeof(topic.nav_cmd) == 2, "size mismatch");
	memcpy(&topic.nav_cmd, buf.iterator, sizeof(topic.nav_cmd));
	buf.iterator += sizeof(topic.nav_cmd);
	buf.offset += sizeof(topic.nav_cmd);
	static_assert(sizeof(topic.latitude) == 4, "size mismatch");
	memcpy(&topic.latitude, buf.iterator, sizeof(topic.latitude));
	buf.iterator += sizeof(topic.latitude);
	buf.offset += sizeof(topic.latitude);
	static_assert(sizeof(topic.longitude) == 4, "size mismatch");
	memcpy(&topic.longitude, buf.iterator, sizeof(topic.longitude));
	buf.iterator += sizeof(topic.longitude);
	buf.offset += sizeof(topic.longitude);
	static_assert(sizeof(topic.time_inside) == 4, "size mismatch");
	memcpy(&topic.time_inside, buf.iterator, sizeof(topic.time_inside));
	buf.iterator += sizeof(topic.time_inside);
	buf.offset += sizeof(topic.time_inside);
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.acceptance_radius, buf.iterator, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.loiter_radius) == 4, "size mismatch");
	memcpy(&topic.loiter_radius, buf.iterator, sizeof(topic.loiter_radius));
	buf.iterator += sizeof(topic.loiter_radius);
	buf.offset += sizeof(topic.loiter_radius);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.altitude) == 4, "size mismatch");
	memcpy(&topic.altitude, buf.iterator, sizeof(topic.altitude));
	buf.iterator += sizeof(topic.altitude);
	buf.offset += sizeof(topic.altitude);
	static_assert(sizeof(topic.frame) == 1, "size mismatch");
	memcpy(&topic.frame, buf.iterator, sizeof(topic.frame));
	buf.iterator += sizeof(topic.frame);
	buf.offset += sizeof(topic.frame);
	static_assert(sizeof(topic.origin) == 1, "size mismatch");
	memcpy(&topic.origin, buf.iterator, sizeof(topic.origin));
	buf.iterator += sizeof(topic.origin);
	buf.offset += sizeof(topic.origin);
	static_assert(sizeof(topic.loiter_exit_xtrack) == 1, "size mismatch");
	memcpy(&topic.loiter_exit_xtrack, buf.iterator, sizeof(topic.loiter_exit_xtrack));
	buf.iterator += sizeof(topic.loiter_exit_xtrack);
	buf.offset += sizeof(topic.loiter_exit_xtrack);
	static_assert(sizeof(topic.force_heading) == 1, "size mismatch");
	memcpy(&topic.force_heading, buf.iterator, sizeof(topic.force_heading));
	buf.iterator += sizeof(topic.force_heading);
	buf.offset += sizeof(topic.force_heading);
	static_assert(sizeof(topic.altitude_is_relative) == 1, "size mismatch");
	memcpy(&topic.altitude_is_relative, buf.iterator, sizeof(topic.altitude_is_relative));
	buf.iterator += sizeof(topic.altitude_is_relative);
	buf.offset += sizeof(topic.altitude_is_relative);
	static_assert(sizeof(topic.autocontinue) == 1, "size mismatch");
	memcpy(&topic.autocontinue, buf.iterator, sizeof(topic.autocontinue));
	buf.iterator += sizeof(topic.autocontinue);
	buf.offset += sizeof(topic.autocontinue);
	static_assert(sizeof(topic.vtol_back_transition) == 1, "size mismatch");
	memcpy(&topic.vtol_back_transition, buf.iterator, sizeof(topic.vtol_back_transition));
	buf.iterator += sizeof(topic.vtol_back_transition);
	buf.offset += sizeof(topic.vtol_back_transition);
	return true;
}
