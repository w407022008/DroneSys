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
#include <uORB/topics/ekf2_timestamps.h>


static inline constexpr int ucdr_topic_size_ekf2_timestamps()
{
	return 20;
}

bool ucdr_serialize_ekf2_timestamps(const ekf2_timestamps_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.airspeed_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.airspeed_timestamp_rel, sizeof(topic.airspeed_timestamp_rel));
	buf.iterator += sizeof(topic.airspeed_timestamp_rel);
	buf.offset += sizeof(topic.airspeed_timestamp_rel);
	static_assert(sizeof(topic.distance_sensor_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.distance_sensor_timestamp_rel, sizeof(topic.distance_sensor_timestamp_rel));
	buf.iterator += sizeof(topic.distance_sensor_timestamp_rel);
	buf.offset += sizeof(topic.distance_sensor_timestamp_rel);
	static_assert(sizeof(topic.optical_flow_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.optical_flow_timestamp_rel, sizeof(topic.optical_flow_timestamp_rel));
	buf.iterator += sizeof(topic.optical_flow_timestamp_rel);
	buf.offset += sizeof(topic.optical_flow_timestamp_rel);
	static_assert(sizeof(topic.vehicle_air_data_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.vehicle_air_data_timestamp_rel, sizeof(topic.vehicle_air_data_timestamp_rel));
	buf.iterator += sizeof(topic.vehicle_air_data_timestamp_rel);
	buf.offset += sizeof(topic.vehicle_air_data_timestamp_rel);
	static_assert(sizeof(topic.vehicle_magnetometer_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.vehicle_magnetometer_timestamp_rel, sizeof(topic.vehicle_magnetometer_timestamp_rel));
	buf.iterator += sizeof(topic.vehicle_magnetometer_timestamp_rel);
	buf.offset += sizeof(topic.vehicle_magnetometer_timestamp_rel);
	static_assert(sizeof(topic.visual_odometry_timestamp_rel) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.visual_odometry_timestamp_rel, sizeof(topic.visual_odometry_timestamp_rel));
	buf.iterator += sizeof(topic.visual_odometry_timestamp_rel);
	buf.offset += sizeof(topic.visual_odometry_timestamp_rel);
	return true;
}

bool ucdr_deserialize_ekf2_timestamps(ucdrBuffer& buf, ekf2_timestamps_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.airspeed_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.airspeed_timestamp_rel, buf.iterator, sizeof(topic.airspeed_timestamp_rel));
	buf.iterator += sizeof(topic.airspeed_timestamp_rel);
	buf.offset += sizeof(topic.airspeed_timestamp_rel);
	static_assert(sizeof(topic.distance_sensor_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.distance_sensor_timestamp_rel, buf.iterator, sizeof(topic.distance_sensor_timestamp_rel));
	buf.iterator += sizeof(topic.distance_sensor_timestamp_rel);
	buf.offset += sizeof(topic.distance_sensor_timestamp_rel);
	static_assert(sizeof(topic.optical_flow_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.optical_flow_timestamp_rel, buf.iterator, sizeof(topic.optical_flow_timestamp_rel));
	buf.iterator += sizeof(topic.optical_flow_timestamp_rel);
	buf.offset += sizeof(topic.optical_flow_timestamp_rel);
	static_assert(sizeof(topic.vehicle_air_data_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.vehicle_air_data_timestamp_rel, buf.iterator, sizeof(topic.vehicle_air_data_timestamp_rel));
	buf.iterator += sizeof(topic.vehicle_air_data_timestamp_rel);
	buf.offset += sizeof(topic.vehicle_air_data_timestamp_rel);
	static_assert(sizeof(topic.vehicle_magnetometer_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.vehicle_magnetometer_timestamp_rel, buf.iterator, sizeof(topic.vehicle_magnetometer_timestamp_rel));
	buf.iterator += sizeof(topic.vehicle_magnetometer_timestamp_rel);
	buf.offset += sizeof(topic.vehicle_magnetometer_timestamp_rel);
	static_assert(sizeof(topic.visual_odometry_timestamp_rel) == 2, "size mismatch");
	memcpy(&topic.visual_odometry_timestamp_rel, buf.iterator, sizeof(topic.visual_odometry_timestamp_rel));
	buf.iterator += sizeof(topic.visual_odometry_timestamp_rel);
	buf.offset += sizeof(topic.visual_odometry_timestamp_rel);
	return true;
}
