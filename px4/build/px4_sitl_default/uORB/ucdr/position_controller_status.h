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
#include <uORB/topics/position_controller_status.h>


static inline constexpr int ucdr_topic_size_position_controller_status()
{
	return 45;
}

bool ucdr_serialize_position_controller_status(const position_controller_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.nav_roll) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.nav_roll, sizeof(topic.nav_roll));
	buf.iterator += sizeof(topic.nav_roll);
	buf.offset += sizeof(topic.nav_roll);
	static_assert(sizeof(topic.nav_pitch) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.nav_pitch, sizeof(topic.nav_pitch));
	buf.iterator += sizeof(topic.nav_pitch);
	buf.offset += sizeof(topic.nav_pitch);
	static_assert(sizeof(topic.nav_bearing) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.nav_bearing, sizeof(topic.nav_bearing));
	buf.iterator += sizeof(topic.nav_bearing);
	buf.offset += sizeof(topic.nav_bearing);
	static_assert(sizeof(topic.target_bearing) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.target_bearing, sizeof(topic.target_bearing));
	buf.iterator += sizeof(topic.target_bearing);
	buf.offset += sizeof(topic.target_bearing);
	static_assert(sizeof(topic.xtrack_error) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.xtrack_error, sizeof(topic.xtrack_error));
	buf.iterator += sizeof(topic.xtrack_error);
	buf.offset += sizeof(topic.xtrack_error);
	static_assert(sizeof(topic.wp_dist) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.wp_dist, sizeof(topic.wp_dist));
	buf.iterator += sizeof(topic.wp_dist);
	buf.offset += sizeof(topic.wp_dist);
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.acceptance_radius, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.yaw_acceptance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_acceptance, sizeof(topic.yaw_acceptance));
	buf.iterator += sizeof(topic.yaw_acceptance);
	buf.offset += sizeof(topic.yaw_acceptance);
	static_assert(sizeof(topic.altitude_acceptance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.altitude_acceptance, sizeof(topic.altitude_acceptance));
	buf.iterator += sizeof(topic.altitude_acceptance);
	buf.offset += sizeof(topic.altitude_acceptance);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	return true;
}

bool ucdr_deserialize_position_controller_status(ucdrBuffer& buf, position_controller_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.nav_roll) == 4, "size mismatch");
	memcpy(&topic.nav_roll, buf.iterator, sizeof(topic.nav_roll));
	buf.iterator += sizeof(topic.nav_roll);
	buf.offset += sizeof(topic.nav_roll);
	static_assert(sizeof(topic.nav_pitch) == 4, "size mismatch");
	memcpy(&topic.nav_pitch, buf.iterator, sizeof(topic.nav_pitch));
	buf.iterator += sizeof(topic.nav_pitch);
	buf.offset += sizeof(topic.nav_pitch);
	static_assert(sizeof(topic.nav_bearing) == 4, "size mismatch");
	memcpy(&topic.nav_bearing, buf.iterator, sizeof(topic.nav_bearing));
	buf.iterator += sizeof(topic.nav_bearing);
	buf.offset += sizeof(topic.nav_bearing);
	static_assert(sizeof(topic.target_bearing) == 4, "size mismatch");
	memcpy(&topic.target_bearing, buf.iterator, sizeof(topic.target_bearing));
	buf.iterator += sizeof(topic.target_bearing);
	buf.offset += sizeof(topic.target_bearing);
	static_assert(sizeof(topic.xtrack_error) == 4, "size mismatch");
	memcpy(&topic.xtrack_error, buf.iterator, sizeof(topic.xtrack_error));
	buf.iterator += sizeof(topic.xtrack_error);
	buf.offset += sizeof(topic.xtrack_error);
	static_assert(sizeof(topic.wp_dist) == 4, "size mismatch");
	memcpy(&topic.wp_dist, buf.iterator, sizeof(topic.wp_dist));
	buf.iterator += sizeof(topic.wp_dist);
	buf.offset += sizeof(topic.wp_dist);
	static_assert(sizeof(topic.acceptance_radius) == 4, "size mismatch");
	memcpy(&topic.acceptance_radius, buf.iterator, sizeof(topic.acceptance_radius));
	buf.iterator += sizeof(topic.acceptance_radius);
	buf.offset += sizeof(topic.acceptance_radius);
	static_assert(sizeof(topic.yaw_acceptance) == 4, "size mismatch");
	memcpy(&topic.yaw_acceptance, buf.iterator, sizeof(topic.yaw_acceptance));
	buf.iterator += sizeof(topic.yaw_acceptance);
	buf.offset += sizeof(topic.yaw_acceptance);
	static_assert(sizeof(topic.altitude_acceptance) == 4, "size mismatch");
	memcpy(&topic.altitude_acceptance, buf.iterator, sizeof(topic.altitude_acceptance));
	buf.iterator += sizeof(topic.altitude_acceptance);
	buf.offset += sizeof(topic.altitude_acceptance);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	return true;
}
