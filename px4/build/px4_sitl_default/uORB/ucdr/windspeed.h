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
#include <uORB/topics/windspeed.h>


static inline constexpr int ucdr_topic_size_windspeed()
{
	return 73;
}

bool ucdr_serialize_windspeed(const windspeed_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.id, sizeof(topic.id));
	buf.iterator += sizeof(topic.id);
	buf.offset += sizeof(topic.id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.measurement_windspeed_x_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.measurement_windspeed_x_m_s, sizeof(topic.measurement_windspeed_x_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_x_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_x_m_s);
	static_assert(sizeof(topic.measurement_windspeed_y_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.measurement_windspeed_y_m_s, sizeof(topic.measurement_windspeed_y_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_y_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_y_m_s);
	static_assert(sizeof(topic.measurement_windspeed_z_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.measurement_windspeed_z_m_s, sizeof(topic.measurement_windspeed_z_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_z_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_z_m_s);
	static_assert(sizeof(topic.measurement_windspeed_zs_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.measurement_windspeed_zs_m_s, sizeof(topic.measurement_windspeed_zs_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_zs_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_zs_m_s);
	static_assert(sizeof(topic.filtered_windspeed_x_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filtered_windspeed_x_m_s, sizeof(topic.filtered_windspeed_x_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_x_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_x_m_s);
	static_assert(sizeof(topic.filtered_windspeed_y_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filtered_windspeed_y_m_s, sizeof(topic.filtered_windspeed_y_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_y_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_y_m_s);
	static_assert(sizeof(topic.filtered_windspeed_z_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filtered_windspeed_z_m_s, sizeof(topic.filtered_windspeed_z_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_z_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_z_m_s);
	static_assert(sizeof(topic.filtered_windspeed_zs_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filtered_windspeed_zs_m_s, sizeof(topic.filtered_windspeed_zs_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_zs_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_zs_m_s);
	static_assert(sizeof(topic.air_temperature_celsius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.air_temperature_celsius, sizeof(topic.air_temperature_celsius));
	buf.iterator += sizeof(topic.air_temperature_celsius);
	buf.offset += sizeof(topic.air_temperature_celsius);
	static_assert(sizeof(topic.confidence_x) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.confidence_x, sizeof(topic.confidence_x));
	buf.iterator += sizeof(topic.confidence_x);
	buf.offset += sizeof(topic.confidence_x);
	static_assert(sizeof(topic.confidence_y) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.confidence_y, sizeof(topic.confidence_y));
	buf.iterator += sizeof(topic.confidence_y);
	buf.offset += sizeof(topic.confidence_y);
	static_assert(sizeof(topic.confidence_z) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.confidence_z, sizeof(topic.confidence_z));
	buf.iterator += sizeof(topic.confidence_z);
	buf.offset += sizeof(topic.confidence_z);
	static_assert(sizeof(topic.confidence_zs) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.confidence_zs, sizeof(topic.confidence_zs));
	buf.iterator += sizeof(topic.confidence_zs);
	buf.offset += sizeof(topic.confidence_zs);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.orientation, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	return true;
}

bool ucdr_deserialize_windspeed(ucdrBuffer& buf, windspeed_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.id) == 1, "size mismatch");
	memcpy(&topic.id, buf.iterator, sizeof(topic.id));
	buf.iterator += sizeof(topic.id);
	buf.offset += sizeof(topic.id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.measurement_windspeed_x_m_s) == 4, "size mismatch");
	memcpy(&topic.measurement_windspeed_x_m_s, buf.iterator, sizeof(topic.measurement_windspeed_x_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_x_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_x_m_s);
	static_assert(sizeof(topic.measurement_windspeed_y_m_s) == 4, "size mismatch");
	memcpy(&topic.measurement_windspeed_y_m_s, buf.iterator, sizeof(topic.measurement_windspeed_y_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_y_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_y_m_s);
	static_assert(sizeof(topic.measurement_windspeed_z_m_s) == 4, "size mismatch");
	memcpy(&topic.measurement_windspeed_z_m_s, buf.iterator, sizeof(topic.measurement_windspeed_z_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_z_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_z_m_s);
	static_assert(sizeof(topic.measurement_windspeed_zs_m_s) == 4, "size mismatch");
	memcpy(&topic.measurement_windspeed_zs_m_s, buf.iterator, sizeof(topic.measurement_windspeed_zs_m_s));
	buf.iterator += sizeof(topic.measurement_windspeed_zs_m_s);
	buf.offset += sizeof(topic.measurement_windspeed_zs_m_s);
	static_assert(sizeof(topic.filtered_windspeed_x_m_s) == 4, "size mismatch");
	memcpy(&topic.filtered_windspeed_x_m_s, buf.iterator, sizeof(topic.filtered_windspeed_x_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_x_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_x_m_s);
	static_assert(sizeof(topic.filtered_windspeed_y_m_s) == 4, "size mismatch");
	memcpy(&topic.filtered_windspeed_y_m_s, buf.iterator, sizeof(topic.filtered_windspeed_y_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_y_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_y_m_s);
	static_assert(sizeof(topic.filtered_windspeed_z_m_s) == 4, "size mismatch");
	memcpy(&topic.filtered_windspeed_z_m_s, buf.iterator, sizeof(topic.filtered_windspeed_z_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_z_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_z_m_s);
	static_assert(sizeof(topic.filtered_windspeed_zs_m_s) == 4, "size mismatch");
	memcpy(&topic.filtered_windspeed_zs_m_s, buf.iterator, sizeof(topic.filtered_windspeed_zs_m_s));
	buf.iterator += sizeof(topic.filtered_windspeed_zs_m_s);
	buf.offset += sizeof(topic.filtered_windspeed_zs_m_s);
	static_assert(sizeof(topic.air_temperature_celsius) == 4, "size mismatch");
	memcpy(&topic.air_temperature_celsius, buf.iterator, sizeof(topic.air_temperature_celsius));
	buf.iterator += sizeof(topic.air_temperature_celsius);
	buf.offset += sizeof(topic.air_temperature_celsius);
	static_assert(sizeof(topic.confidence_x) == 4, "size mismatch");
	memcpy(&topic.confidence_x, buf.iterator, sizeof(topic.confidence_x));
	buf.iterator += sizeof(topic.confidence_x);
	buf.offset += sizeof(topic.confidence_x);
	static_assert(sizeof(topic.confidence_y) == 4, "size mismatch");
	memcpy(&topic.confidence_y, buf.iterator, sizeof(topic.confidence_y));
	buf.iterator += sizeof(topic.confidence_y);
	buf.offset += sizeof(topic.confidence_y);
	static_assert(sizeof(topic.confidence_z) == 4, "size mismatch");
	memcpy(&topic.confidence_z, buf.iterator, sizeof(topic.confidence_z));
	buf.iterator += sizeof(topic.confidence_z);
	buf.offset += sizeof(topic.confidence_z);
	static_assert(sizeof(topic.confidence_zs) == 4, "size mismatch");
	memcpy(&topic.confidence_zs, buf.iterator, sizeof(topic.confidence_zs));
	buf.iterator += sizeof(topic.confidence_zs);
	buf.offset += sizeof(topic.confidence_zs);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(&topic.orientation, buf.iterator, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	return true;
}
