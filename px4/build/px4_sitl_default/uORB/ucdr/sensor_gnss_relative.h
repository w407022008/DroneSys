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
#include <uORB/topics/sensor_gnss_relative.h>


static inline constexpr int ucdr_topic_size_sensor_gnss_relative()
{
	return 86;
}

bool ucdr_serialize_sensor_gnss_relative(const sensor_gnss_relative_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.time_utc_usec) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.time_utc_usec, sizeof(topic.time_utc_usec));
	buf.iterator += sizeof(topic.time_utc_usec);
	buf.offset += sizeof(topic.time_utc_usec);
	static_assert(sizeof(topic.reference_station_id) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.reference_station_id, sizeof(topic.reference_station_id));
	buf.iterator += sizeof(topic.reference_station_id);
	buf.offset += sizeof(topic.reference_station_id);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.position, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.position_accuracy) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.position_accuracy, sizeof(topic.position_accuracy));
	buf.iterator += sizeof(topic.position_accuracy);
	buf.offset += sizeof(topic.position_accuracy);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.heading_accuracy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading_accuracy, sizeof(topic.heading_accuracy));
	buf.iterator += sizeof(topic.heading_accuracy);
	buf.offset += sizeof(topic.heading_accuracy);
	static_assert(sizeof(topic.position_length) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.position_length, sizeof(topic.position_length));
	buf.iterator += sizeof(topic.position_length);
	buf.offset += sizeof(topic.position_length);
	static_assert(sizeof(topic.accuracy_length) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accuracy_length, sizeof(topic.accuracy_length));
	buf.iterator += sizeof(topic.accuracy_length);
	buf.offset += sizeof(topic.accuracy_length);
	static_assert(sizeof(topic.gnss_fix_ok) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gnss_fix_ok, sizeof(topic.gnss_fix_ok));
	buf.iterator += sizeof(topic.gnss_fix_ok);
	buf.offset += sizeof(topic.gnss_fix_ok);
	static_assert(sizeof(topic.differential_solution) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.differential_solution, sizeof(topic.differential_solution));
	buf.iterator += sizeof(topic.differential_solution);
	buf.offset += sizeof(topic.differential_solution);
	static_assert(sizeof(topic.relative_position_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.relative_position_valid, sizeof(topic.relative_position_valid));
	buf.iterator += sizeof(topic.relative_position_valid);
	buf.offset += sizeof(topic.relative_position_valid);
	static_assert(sizeof(topic.carrier_solution_floating) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.carrier_solution_floating, sizeof(topic.carrier_solution_floating));
	buf.iterator += sizeof(topic.carrier_solution_floating);
	buf.offset += sizeof(topic.carrier_solution_floating);
	static_assert(sizeof(topic.carrier_solution_fixed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.carrier_solution_fixed, sizeof(topic.carrier_solution_fixed));
	buf.iterator += sizeof(topic.carrier_solution_fixed);
	buf.offset += sizeof(topic.carrier_solution_fixed);
	static_assert(sizeof(topic.moving_base_mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.moving_base_mode, sizeof(topic.moving_base_mode));
	buf.iterator += sizeof(topic.moving_base_mode);
	buf.offset += sizeof(topic.moving_base_mode);
	static_assert(sizeof(topic.reference_position_miss) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reference_position_miss, sizeof(topic.reference_position_miss));
	buf.iterator += sizeof(topic.reference_position_miss);
	buf.offset += sizeof(topic.reference_position_miss);
	static_assert(sizeof(topic.reference_observations_miss) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reference_observations_miss, sizeof(topic.reference_observations_miss));
	buf.iterator += sizeof(topic.reference_observations_miss);
	buf.offset += sizeof(topic.reference_observations_miss);
	static_assert(sizeof(topic.heading_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heading_valid, sizeof(topic.heading_valid));
	buf.iterator += sizeof(topic.heading_valid);
	buf.offset += sizeof(topic.heading_valid);
	static_assert(sizeof(topic.relative_position_normalized) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.relative_position_normalized, sizeof(topic.relative_position_normalized));
	buf.iterator += sizeof(topic.relative_position_normalized);
	buf.offset += sizeof(topic.relative_position_normalized);
	return true;
}

bool ucdr_deserialize_sensor_gnss_relative(ucdrBuffer& buf, sensor_gnss_relative_s& topic, int64_t time_offset = 0)
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
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.time_utc_usec) == 8, "size mismatch");
	memcpy(&topic.time_utc_usec, buf.iterator, sizeof(topic.time_utc_usec));
	buf.iterator += sizeof(topic.time_utc_usec);
	buf.offset += sizeof(topic.time_utc_usec);
	static_assert(sizeof(topic.reference_station_id) == 2, "size mismatch");
	memcpy(&topic.reference_station_id, buf.iterator, sizeof(topic.reference_station_id));
	buf.iterator += sizeof(topic.reference_station_id);
	buf.offset += sizeof(topic.reference_station_id);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(&topic.position, buf.iterator, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.position_accuracy) == 12, "size mismatch");
	memcpy(&topic.position_accuracy, buf.iterator, sizeof(topic.position_accuracy));
	buf.iterator += sizeof(topic.position_accuracy);
	buf.offset += sizeof(topic.position_accuracy);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(&topic.heading, buf.iterator, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.heading_accuracy) == 4, "size mismatch");
	memcpy(&topic.heading_accuracy, buf.iterator, sizeof(topic.heading_accuracy));
	buf.iterator += sizeof(topic.heading_accuracy);
	buf.offset += sizeof(topic.heading_accuracy);
	static_assert(sizeof(topic.position_length) == 4, "size mismatch");
	memcpy(&topic.position_length, buf.iterator, sizeof(topic.position_length));
	buf.iterator += sizeof(topic.position_length);
	buf.offset += sizeof(topic.position_length);
	static_assert(sizeof(topic.accuracy_length) == 4, "size mismatch");
	memcpy(&topic.accuracy_length, buf.iterator, sizeof(topic.accuracy_length));
	buf.iterator += sizeof(topic.accuracy_length);
	buf.offset += sizeof(topic.accuracy_length);
	static_assert(sizeof(topic.gnss_fix_ok) == 1, "size mismatch");
	memcpy(&topic.gnss_fix_ok, buf.iterator, sizeof(topic.gnss_fix_ok));
	buf.iterator += sizeof(topic.gnss_fix_ok);
	buf.offset += sizeof(topic.gnss_fix_ok);
	static_assert(sizeof(topic.differential_solution) == 1, "size mismatch");
	memcpy(&topic.differential_solution, buf.iterator, sizeof(topic.differential_solution));
	buf.iterator += sizeof(topic.differential_solution);
	buf.offset += sizeof(topic.differential_solution);
	static_assert(sizeof(topic.relative_position_valid) == 1, "size mismatch");
	memcpy(&topic.relative_position_valid, buf.iterator, sizeof(topic.relative_position_valid));
	buf.iterator += sizeof(topic.relative_position_valid);
	buf.offset += sizeof(topic.relative_position_valid);
	static_assert(sizeof(topic.carrier_solution_floating) == 1, "size mismatch");
	memcpy(&topic.carrier_solution_floating, buf.iterator, sizeof(topic.carrier_solution_floating));
	buf.iterator += sizeof(topic.carrier_solution_floating);
	buf.offset += sizeof(topic.carrier_solution_floating);
	static_assert(sizeof(topic.carrier_solution_fixed) == 1, "size mismatch");
	memcpy(&topic.carrier_solution_fixed, buf.iterator, sizeof(topic.carrier_solution_fixed));
	buf.iterator += sizeof(topic.carrier_solution_fixed);
	buf.offset += sizeof(topic.carrier_solution_fixed);
	static_assert(sizeof(topic.moving_base_mode) == 1, "size mismatch");
	memcpy(&topic.moving_base_mode, buf.iterator, sizeof(topic.moving_base_mode));
	buf.iterator += sizeof(topic.moving_base_mode);
	buf.offset += sizeof(topic.moving_base_mode);
	static_assert(sizeof(topic.reference_position_miss) == 1, "size mismatch");
	memcpy(&topic.reference_position_miss, buf.iterator, sizeof(topic.reference_position_miss));
	buf.iterator += sizeof(topic.reference_position_miss);
	buf.offset += sizeof(topic.reference_position_miss);
	static_assert(sizeof(topic.reference_observations_miss) == 1, "size mismatch");
	memcpy(&topic.reference_observations_miss, buf.iterator, sizeof(topic.reference_observations_miss));
	buf.iterator += sizeof(topic.reference_observations_miss);
	buf.offset += sizeof(topic.reference_observations_miss);
	static_assert(sizeof(topic.heading_valid) == 1, "size mismatch");
	memcpy(&topic.heading_valid, buf.iterator, sizeof(topic.heading_valid));
	buf.iterator += sizeof(topic.heading_valid);
	buf.offset += sizeof(topic.heading_valid);
	static_assert(sizeof(topic.relative_position_normalized) == 1, "size mismatch");
	memcpy(&topic.relative_position_normalized, buf.iterator, sizeof(topic.relative_position_normalized));
	buf.iterator += sizeof(topic.relative_position_normalized);
	buf.offset += sizeof(topic.relative_position_normalized);
	return true;
}
