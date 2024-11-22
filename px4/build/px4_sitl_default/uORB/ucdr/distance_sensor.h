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
#include <uORB/topics/distance_sensor.h>


static inline constexpr int ucdr_topic_size_distance_sensor()
{
	return 57;
}

bool ucdr_serialize_distance_sensor(const distance_sensor_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.device_id, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.min_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.min_distance, sizeof(topic.min_distance));
	buf.iterator += sizeof(topic.min_distance);
	buf.offset += sizeof(topic.min_distance);
	static_assert(sizeof(topic.max_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.max_distance, sizeof(topic.max_distance));
	buf.iterator += sizeof(topic.max_distance);
	buf.offset += sizeof(topic.max_distance);
	static_assert(sizeof(topic.current_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current_distance, sizeof(topic.current_distance));
	buf.iterator += sizeof(topic.current_distance);
	buf.offset += sizeof(topic.current_distance);
	static_assert(sizeof(topic.variance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.variance, sizeof(topic.variance));
	buf.iterator += sizeof(topic.variance);
	buf.offset += sizeof(topic.variance);
	static_assert(sizeof(topic.signal_quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.signal_quality, sizeof(topic.signal_quality));
	buf.iterator += sizeof(topic.signal_quality);
	buf.offset += sizeof(topic.signal_quality);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.h_fov) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.h_fov, sizeof(topic.h_fov));
	buf.iterator += sizeof(topic.h_fov);
	buf.offset += sizeof(topic.h_fov);
	static_assert(sizeof(topic.v_fov) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.v_fov, sizeof(topic.v_fov));
	buf.iterator += sizeof(topic.v_fov);
	buf.offset += sizeof(topic.v_fov);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.q, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.orientation, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	return true;
}

bool ucdr_deserialize_distance_sensor(ucdrBuffer& buf, distance_sensor_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(&topic.device_id, buf.iterator, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.min_distance) == 4, "size mismatch");
	memcpy(&topic.min_distance, buf.iterator, sizeof(topic.min_distance));
	buf.iterator += sizeof(topic.min_distance);
	buf.offset += sizeof(topic.min_distance);
	static_assert(sizeof(topic.max_distance) == 4, "size mismatch");
	memcpy(&topic.max_distance, buf.iterator, sizeof(topic.max_distance));
	buf.iterator += sizeof(topic.max_distance);
	buf.offset += sizeof(topic.max_distance);
	static_assert(sizeof(topic.current_distance) == 4, "size mismatch");
	memcpy(&topic.current_distance, buf.iterator, sizeof(topic.current_distance));
	buf.iterator += sizeof(topic.current_distance);
	buf.offset += sizeof(topic.current_distance);
	static_assert(sizeof(topic.variance) == 4, "size mismatch");
	memcpy(&topic.variance, buf.iterator, sizeof(topic.variance));
	buf.iterator += sizeof(topic.variance);
	buf.offset += sizeof(topic.variance);
	static_assert(sizeof(topic.signal_quality) == 1, "size mismatch");
	memcpy(&topic.signal_quality, buf.iterator, sizeof(topic.signal_quality));
	buf.iterator += sizeof(topic.signal_quality);
	buf.offset += sizeof(topic.signal_quality);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.h_fov) == 4, "size mismatch");
	memcpy(&topic.h_fov, buf.iterator, sizeof(topic.h_fov));
	buf.iterator += sizeof(topic.h_fov);
	buf.offset += sizeof(topic.h_fov);
	static_assert(sizeof(topic.v_fov) == 4, "size mismatch");
	memcpy(&topic.v_fov, buf.iterator, sizeof(topic.v_fov));
	buf.iterator += sizeof(topic.v_fov);
	buf.offset += sizeof(topic.v_fov);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(&topic.q, buf.iterator, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(&topic.orientation, buf.iterator, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	return true;
}
