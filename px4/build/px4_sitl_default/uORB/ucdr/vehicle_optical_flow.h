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
#include <uORB/topics/vehicle_optical_flow.h>


static inline constexpr int ucdr_topic_size_vehicle_optical_flow()
{
	return 64;
}

bool ucdr_serialize_vehicle_optical_flow(const vehicle_optical_flow_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_sample) == 8, "size mismatch");
	const uint64_t timestamp_sample_adjusted = topic.timestamp_sample + time_offset;
	memcpy(buf.iterator, &timestamp_sample_adjusted, sizeof(topic.timestamp_sample));
	buf.iterator += sizeof(topic.timestamp_sample);
	buf.offset += sizeof(topic.timestamp_sample);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.device_id, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.pixel_flow) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.pixel_flow, sizeof(topic.pixel_flow));
	buf.iterator += sizeof(topic.pixel_flow);
	buf.offset += sizeof(topic.pixel_flow);
	static_assert(sizeof(topic.delta_angle) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.delta_angle, sizeof(topic.delta_angle));
	buf.iterator += sizeof(topic.delta_angle);
	buf.offset += sizeof(topic.delta_angle);
	static_assert(sizeof(topic.distance_m) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.distance_m, sizeof(topic.distance_m));
	buf.iterator += sizeof(topic.distance_m);
	buf.offset += sizeof(topic.distance_m);
	static_assert(sizeof(topic.integration_timespan_us) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.integration_timespan_us, sizeof(topic.integration_timespan_us));
	buf.iterator += sizeof(topic.integration_timespan_us);
	buf.offset += sizeof(topic.integration_timespan_us);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.quality, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.max_flow_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.max_flow_rate, sizeof(topic.max_flow_rate));
	buf.iterator += sizeof(topic.max_flow_rate);
	buf.offset += sizeof(topic.max_flow_rate);
	static_assert(sizeof(topic.min_ground_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.min_ground_distance, sizeof(topic.min_ground_distance));
	buf.iterator += sizeof(topic.min_ground_distance);
	buf.offset += sizeof(topic.min_ground_distance);
	static_assert(sizeof(topic.max_ground_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.max_ground_distance, sizeof(topic.max_ground_distance));
	buf.iterator += sizeof(topic.max_ground_distance);
	buf.offset += sizeof(topic.max_ground_distance);
	return true;
}

bool ucdr_deserialize_vehicle_optical_flow(ucdrBuffer& buf, vehicle_optical_flow_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_sample) == 8, "size mismatch");
	memcpy(&topic.timestamp_sample, buf.iterator, sizeof(topic.timestamp_sample));
	if (topic.timestamp_sample == 0) topic.timestamp_sample = hrt_absolute_time();
	else topic.timestamp_sample = math::min(topic.timestamp_sample - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp_sample);
	buf.offset += sizeof(topic.timestamp_sample);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(&topic.device_id, buf.iterator, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.pixel_flow) == 8, "size mismatch");
	memcpy(&topic.pixel_flow, buf.iterator, sizeof(topic.pixel_flow));
	buf.iterator += sizeof(topic.pixel_flow);
	buf.offset += sizeof(topic.pixel_flow);
	static_assert(sizeof(topic.delta_angle) == 12, "size mismatch");
	memcpy(&topic.delta_angle, buf.iterator, sizeof(topic.delta_angle));
	buf.iterator += sizeof(topic.delta_angle);
	buf.offset += sizeof(topic.delta_angle);
	static_assert(sizeof(topic.distance_m) == 4, "size mismatch");
	memcpy(&topic.distance_m, buf.iterator, sizeof(topic.distance_m));
	buf.iterator += sizeof(topic.distance_m);
	buf.offset += sizeof(topic.distance_m);
	static_assert(sizeof(topic.integration_timespan_us) == 4, "size mismatch");
	memcpy(&topic.integration_timespan_us, buf.iterator, sizeof(topic.integration_timespan_us));
	buf.iterator += sizeof(topic.integration_timespan_us);
	buf.offset += sizeof(topic.integration_timespan_us);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(&topic.quality, buf.iterator, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.max_flow_rate) == 4, "size mismatch");
	memcpy(&topic.max_flow_rate, buf.iterator, sizeof(topic.max_flow_rate));
	buf.iterator += sizeof(topic.max_flow_rate);
	buf.offset += sizeof(topic.max_flow_rate);
	static_assert(sizeof(topic.min_ground_distance) == 4, "size mismatch");
	memcpy(&topic.min_ground_distance, buf.iterator, sizeof(topic.min_ground_distance));
	buf.iterator += sizeof(topic.min_ground_distance);
	buf.offset += sizeof(topic.min_ground_distance);
	static_assert(sizeof(topic.max_ground_distance) == 4, "size mismatch");
	memcpy(&topic.max_ground_distance, buf.iterator, sizeof(topic.max_ground_distance));
	buf.iterator += sizeof(topic.max_ground_distance);
	buf.offset += sizeof(topic.max_ground_distance);
	return true;
}
