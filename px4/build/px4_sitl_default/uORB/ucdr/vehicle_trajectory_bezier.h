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
#include <uORB/topics/vehicle_trajectory_bezier.h>

#include <uORB/ucdr/trajectory_bezier.h>

static inline constexpr int ucdr_topic_size_vehicle_trajectory_bezier()
{
	return 165;
}

bool ucdr_serialize_vehicle_trajectory_bezier(const vehicle_trajectory_bezier_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.control_points[0].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[0].timestamp, sizeof(topic.control_points[0].timestamp));
	buf.iterator += sizeof(topic.control_points[0].timestamp);
	buf.offset += sizeof(topic.control_points[0].timestamp);
	static_assert(sizeof(topic.control_points[0].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[0].position, sizeof(topic.control_points[0].position));
	buf.iterator += sizeof(topic.control_points[0].position);
	buf.offset += sizeof(topic.control_points[0].position);
	static_assert(sizeof(topic.control_points[0].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[0].yaw, sizeof(topic.control_points[0].yaw));
	buf.iterator += sizeof(topic.control_points[0].yaw);
	buf.offset += sizeof(topic.control_points[0].yaw);
	static_assert(sizeof(topic.control_points[0].delta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[0].delta, sizeof(topic.control_points[0].delta));
	buf.iterator += sizeof(topic.control_points[0].delta);
	buf.offset += sizeof(topic.control_points[0].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[1].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[1].timestamp, sizeof(topic.control_points[1].timestamp));
	buf.iterator += sizeof(topic.control_points[1].timestamp);
	buf.offset += sizeof(topic.control_points[1].timestamp);
	static_assert(sizeof(topic.control_points[1].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[1].position, sizeof(topic.control_points[1].position));
	buf.iterator += sizeof(topic.control_points[1].position);
	buf.offset += sizeof(topic.control_points[1].position);
	static_assert(sizeof(topic.control_points[1].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[1].yaw, sizeof(topic.control_points[1].yaw));
	buf.iterator += sizeof(topic.control_points[1].yaw);
	buf.offset += sizeof(topic.control_points[1].yaw);
	static_assert(sizeof(topic.control_points[1].delta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[1].delta, sizeof(topic.control_points[1].delta));
	buf.iterator += sizeof(topic.control_points[1].delta);
	buf.offset += sizeof(topic.control_points[1].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[2].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[2].timestamp, sizeof(topic.control_points[2].timestamp));
	buf.iterator += sizeof(topic.control_points[2].timestamp);
	buf.offset += sizeof(topic.control_points[2].timestamp);
	static_assert(sizeof(topic.control_points[2].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[2].position, sizeof(topic.control_points[2].position));
	buf.iterator += sizeof(topic.control_points[2].position);
	buf.offset += sizeof(topic.control_points[2].position);
	static_assert(sizeof(topic.control_points[2].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[2].yaw, sizeof(topic.control_points[2].yaw));
	buf.iterator += sizeof(topic.control_points[2].yaw);
	buf.offset += sizeof(topic.control_points[2].yaw);
	static_assert(sizeof(topic.control_points[2].delta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[2].delta, sizeof(topic.control_points[2].delta));
	buf.iterator += sizeof(topic.control_points[2].delta);
	buf.offset += sizeof(topic.control_points[2].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[3].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[3].timestamp, sizeof(topic.control_points[3].timestamp));
	buf.iterator += sizeof(topic.control_points[3].timestamp);
	buf.offset += sizeof(topic.control_points[3].timestamp);
	static_assert(sizeof(topic.control_points[3].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[3].position, sizeof(topic.control_points[3].position));
	buf.iterator += sizeof(topic.control_points[3].position);
	buf.offset += sizeof(topic.control_points[3].position);
	static_assert(sizeof(topic.control_points[3].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[3].yaw, sizeof(topic.control_points[3].yaw));
	buf.iterator += sizeof(topic.control_points[3].yaw);
	buf.offset += sizeof(topic.control_points[3].yaw);
	static_assert(sizeof(topic.control_points[3].delta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[3].delta, sizeof(topic.control_points[3].delta));
	buf.iterator += sizeof(topic.control_points[3].delta);
	buf.offset += sizeof(topic.control_points[3].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[4].timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[4].timestamp, sizeof(topic.control_points[4].timestamp));
	buf.iterator += sizeof(topic.control_points[4].timestamp);
	buf.offset += sizeof(topic.control_points[4].timestamp);
	static_assert(sizeof(topic.control_points[4].position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[4].position, sizeof(topic.control_points[4].position));
	buf.iterator += sizeof(topic.control_points[4].position);
	buf.offset += sizeof(topic.control_points[4].position);
	static_assert(sizeof(topic.control_points[4].yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[4].yaw, sizeof(topic.control_points[4].yaw));
	buf.iterator += sizeof(topic.control_points[4].yaw);
	buf.offset += sizeof(topic.control_points[4].yaw);
	static_assert(sizeof(topic.control_points[4].delta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_points[4].delta, sizeof(topic.control_points[4].delta));
	buf.iterator += sizeof(topic.control_points[4].delta);
	buf.offset += sizeof(topic.control_points[4].delta);
	static_assert(sizeof(topic.bezier_order) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.bezier_order, sizeof(topic.bezier_order));
	buf.iterator += sizeof(topic.bezier_order);
	buf.offset += sizeof(topic.bezier_order);
	return true;
}

bool ucdr_deserialize_vehicle_trajectory_bezier(ucdrBuffer& buf, vehicle_trajectory_bezier_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.control_points[0].timestamp) == 8, "size mismatch");
	memcpy(&topic.control_points[0].timestamp, buf.iterator, sizeof(topic.control_points[0].timestamp));
	buf.iterator += sizeof(topic.control_points[0].timestamp);
	buf.offset += sizeof(topic.control_points[0].timestamp);
	static_assert(sizeof(topic.control_points[0].position) == 12, "size mismatch");
	memcpy(&topic.control_points[0].position, buf.iterator, sizeof(topic.control_points[0].position));
	buf.iterator += sizeof(topic.control_points[0].position);
	buf.offset += sizeof(topic.control_points[0].position);
	static_assert(sizeof(topic.control_points[0].yaw) == 4, "size mismatch");
	memcpy(&topic.control_points[0].yaw, buf.iterator, sizeof(topic.control_points[0].yaw));
	buf.iterator += sizeof(topic.control_points[0].yaw);
	buf.offset += sizeof(topic.control_points[0].yaw);
	static_assert(sizeof(topic.control_points[0].delta) == 4, "size mismatch");
	memcpy(&topic.control_points[0].delta, buf.iterator, sizeof(topic.control_points[0].delta));
	buf.iterator += sizeof(topic.control_points[0].delta);
	buf.offset += sizeof(topic.control_points[0].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[1].timestamp) == 8, "size mismatch");
	memcpy(&topic.control_points[1].timestamp, buf.iterator, sizeof(topic.control_points[1].timestamp));
	buf.iterator += sizeof(topic.control_points[1].timestamp);
	buf.offset += sizeof(topic.control_points[1].timestamp);
	static_assert(sizeof(topic.control_points[1].position) == 12, "size mismatch");
	memcpy(&topic.control_points[1].position, buf.iterator, sizeof(topic.control_points[1].position));
	buf.iterator += sizeof(topic.control_points[1].position);
	buf.offset += sizeof(topic.control_points[1].position);
	static_assert(sizeof(topic.control_points[1].yaw) == 4, "size mismatch");
	memcpy(&topic.control_points[1].yaw, buf.iterator, sizeof(topic.control_points[1].yaw));
	buf.iterator += sizeof(topic.control_points[1].yaw);
	buf.offset += sizeof(topic.control_points[1].yaw);
	static_assert(sizeof(topic.control_points[1].delta) == 4, "size mismatch");
	memcpy(&topic.control_points[1].delta, buf.iterator, sizeof(topic.control_points[1].delta));
	buf.iterator += sizeof(topic.control_points[1].delta);
	buf.offset += sizeof(topic.control_points[1].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[2].timestamp) == 8, "size mismatch");
	memcpy(&topic.control_points[2].timestamp, buf.iterator, sizeof(topic.control_points[2].timestamp));
	buf.iterator += sizeof(topic.control_points[2].timestamp);
	buf.offset += sizeof(topic.control_points[2].timestamp);
	static_assert(sizeof(topic.control_points[2].position) == 12, "size mismatch");
	memcpy(&topic.control_points[2].position, buf.iterator, sizeof(topic.control_points[2].position));
	buf.iterator += sizeof(topic.control_points[2].position);
	buf.offset += sizeof(topic.control_points[2].position);
	static_assert(sizeof(topic.control_points[2].yaw) == 4, "size mismatch");
	memcpy(&topic.control_points[2].yaw, buf.iterator, sizeof(topic.control_points[2].yaw));
	buf.iterator += sizeof(topic.control_points[2].yaw);
	buf.offset += sizeof(topic.control_points[2].yaw);
	static_assert(sizeof(topic.control_points[2].delta) == 4, "size mismatch");
	memcpy(&topic.control_points[2].delta, buf.iterator, sizeof(topic.control_points[2].delta));
	buf.iterator += sizeof(topic.control_points[2].delta);
	buf.offset += sizeof(topic.control_points[2].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[3].timestamp) == 8, "size mismatch");
	memcpy(&topic.control_points[3].timestamp, buf.iterator, sizeof(topic.control_points[3].timestamp));
	buf.iterator += sizeof(topic.control_points[3].timestamp);
	buf.offset += sizeof(topic.control_points[3].timestamp);
	static_assert(sizeof(topic.control_points[3].position) == 12, "size mismatch");
	memcpy(&topic.control_points[3].position, buf.iterator, sizeof(topic.control_points[3].position));
	buf.iterator += sizeof(topic.control_points[3].position);
	buf.offset += sizeof(topic.control_points[3].position);
	static_assert(sizeof(topic.control_points[3].yaw) == 4, "size mismatch");
	memcpy(&topic.control_points[3].yaw, buf.iterator, sizeof(topic.control_points[3].yaw));
	buf.iterator += sizeof(topic.control_points[3].yaw);
	buf.offset += sizeof(topic.control_points[3].yaw);
	static_assert(sizeof(topic.control_points[3].delta) == 4, "size mismatch");
	memcpy(&topic.control_points[3].delta, buf.iterator, sizeof(topic.control_points[3].delta));
	buf.iterator += sizeof(topic.control_points[3].delta);
	buf.offset += sizeof(topic.control_points[3].delta);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.control_points[4].timestamp) == 8, "size mismatch");
	memcpy(&topic.control_points[4].timestamp, buf.iterator, sizeof(topic.control_points[4].timestamp));
	buf.iterator += sizeof(topic.control_points[4].timestamp);
	buf.offset += sizeof(topic.control_points[4].timestamp);
	static_assert(sizeof(topic.control_points[4].position) == 12, "size mismatch");
	memcpy(&topic.control_points[4].position, buf.iterator, sizeof(topic.control_points[4].position));
	buf.iterator += sizeof(topic.control_points[4].position);
	buf.offset += sizeof(topic.control_points[4].position);
	static_assert(sizeof(topic.control_points[4].yaw) == 4, "size mismatch");
	memcpy(&topic.control_points[4].yaw, buf.iterator, sizeof(topic.control_points[4].yaw));
	buf.iterator += sizeof(topic.control_points[4].yaw);
	buf.offset += sizeof(topic.control_points[4].yaw);
	static_assert(sizeof(topic.control_points[4].delta) == 4, "size mismatch");
	memcpy(&topic.control_points[4].delta, buf.iterator, sizeof(topic.control_points[4].delta));
	buf.iterator += sizeof(topic.control_points[4].delta);
	buf.offset += sizeof(topic.control_points[4].delta);
	static_assert(sizeof(topic.bezier_order) == 1, "size mismatch");
	memcpy(&topic.bezier_order, buf.iterator, sizeof(topic.bezier_order));
	buf.iterator += sizeof(topic.bezier_order);
	buf.offset += sizeof(topic.bezier_order);
	return true;
}
