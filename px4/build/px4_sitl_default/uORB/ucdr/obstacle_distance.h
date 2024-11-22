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
#include <uORB/topics/obstacle_distance.h>


static inline constexpr int ucdr_topic_size_obstacle_distance()
{
	return 168;
}

bool ucdr_serialize_obstacle_distance(const obstacle_distance_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.frame) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.frame, sizeof(topic.frame));
	buf.iterator += sizeof(topic.frame);
	buf.offset += sizeof(topic.frame);
	static_assert(sizeof(topic.sensor_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.sensor_type, sizeof(topic.sensor_type));
	buf.iterator += sizeof(topic.sensor_type);
	buf.offset += sizeof(topic.sensor_type);
	static_assert(sizeof(topic.distances) == 144, "size mismatch");
	memcpy(buf.iterator, &topic.distances, sizeof(topic.distances));
	buf.iterator += sizeof(topic.distances);
	buf.offset += sizeof(topic.distances);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.increment) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.increment, sizeof(topic.increment));
	buf.iterator += sizeof(topic.increment);
	buf.offset += sizeof(topic.increment);
	static_assert(sizeof(topic.min_distance) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.min_distance, sizeof(topic.min_distance));
	buf.iterator += sizeof(topic.min_distance);
	buf.offset += sizeof(topic.min_distance);
	static_assert(sizeof(topic.max_distance) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.max_distance, sizeof(topic.max_distance));
	buf.iterator += sizeof(topic.max_distance);
	buf.offset += sizeof(topic.max_distance);
	static_assert(sizeof(topic.angle_offset) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.angle_offset, sizeof(topic.angle_offset));
	buf.iterator += sizeof(topic.angle_offset);
	buf.offset += sizeof(topic.angle_offset);
	return true;
}

bool ucdr_deserialize_obstacle_distance(ucdrBuffer& buf, obstacle_distance_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.frame) == 1, "size mismatch");
	memcpy(&topic.frame, buf.iterator, sizeof(topic.frame));
	buf.iterator += sizeof(topic.frame);
	buf.offset += sizeof(topic.frame);
	static_assert(sizeof(topic.sensor_type) == 1, "size mismatch");
	memcpy(&topic.sensor_type, buf.iterator, sizeof(topic.sensor_type));
	buf.iterator += sizeof(topic.sensor_type);
	buf.offset += sizeof(topic.sensor_type);
	static_assert(sizeof(topic.distances) == 144, "size mismatch");
	memcpy(&topic.distances, buf.iterator, sizeof(topic.distances));
	buf.iterator += sizeof(topic.distances);
	buf.offset += sizeof(topic.distances);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.increment) == 4, "size mismatch");
	memcpy(&topic.increment, buf.iterator, sizeof(topic.increment));
	buf.iterator += sizeof(topic.increment);
	buf.offset += sizeof(topic.increment);
	static_assert(sizeof(topic.min_distance) == 2, "size mismatch");
	memcpy(&topic.min_distance, buf.iterator, sizeof(topic.min_distance));
	buf.iterator += sizeof(topic.min_distance);
	buf.offset += sizeof(topic.min_distance);
	static_assert(sizeof(topic.max_distance) == 2, "size mismatch");
	memcpy(&topic.max_distance, buf.iterator, sizeof(topic.max_distance));
	buf.iterator += sizeof(topic.max_distance);
	buf.offset += sizeof(topic.max_distance);
	static_assert(sizeof(topic.angle_offset) == 4, "size mismatch");
	memcpy(&topic.angle_offset, buf.iterator, sizeof(topic.angle_offset));
	buf.iterator += sizeof(topic.angle_offset);
	buf.offset += sizeof(topic.angle_offset);
	return true;
}
