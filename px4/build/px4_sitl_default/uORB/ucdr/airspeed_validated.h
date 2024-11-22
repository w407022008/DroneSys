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
#include <uORB/topics/airspeed_validated.h>


static inline constexpr int ucdr_topic_size_airspeed_validated()
{
	return 30;
}

bool ucdr_serialize_airspeed_validated(const airspeed_validated_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.indicated_airspeed_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.indicated_airspeed_m_s, sizeof(topic.indicated_airspeed_m_s));
	buf.iterator += sizeof(topic.indicated_airspeed_m_s);
	buf.offset += sizeof(topic.indicated_airspeed_m_s);
	static_assert(sizeof(topic.calibrated_airspeed_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.calibrated_airspeed_m_s, sizeof(topic.calibrated_airspeed_m_s));
	buf.iterator += sizeof(topic.calibrated_airspeed_m_s);
	buf.offset += sizeof(topic.calibrated_airspeed_m_s);
	static_assert(sizeof(topic.true_airspeed_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_m_s, sizeof(topic.true_airspeed_m_s));
	buf.iterator += sizeof(topic.true_airspeed_m_s);
	buf.offset += sizeof(topic.true_airspeed_m_s);
	static_assert(sizeof(topic.calibrated_ground_minus_wind_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.calibrated_ground_minus_wind_m_s, sizeof(topic.calibrated_ground_minus_wind_m_s));
	buf.iterator += sizeof(topic.calibrated_ground_minus_wind_m_s);
	buf.offset += sizeof(topic.calibrated_ground_minus_wind_m_s);
	static_assert(sizeof(topic.true_ground_minus_wind_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_ground_minus_wind_m_s, sizeof(topic.true_ground_minus_wind_m_s));
	buf.iterator += sizeof(topic.true_ground_minus_wind_m_s);
	buf.offset += sizeof(topic.true_ground_minus_wind_m_s);
	static_assert(sizeof(topic.airspeed_sensor_measurement_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.airspeed_sensor_measurement_valid, sizeof(topic.airspeed_sensor_measurement_valid));
	buf.iterator += sizeof(topic.airspeed_sensor_measurement_valid);
	buf.offset += sizeof(topic.airspeed_sensor_measurement_valid);
	static_assert(sizeof(topic.selected_airspeed_index) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.selected_airspeed_index, sizeof(topic.selected_airspeed_index));
	buf.iterator += sizeof(topic.selected_airspeed_index);
	buf.offset += sizeof(topic.selected_airspeed_index);
	return true;
}

bool ucdr_deserialize_airspeed_validated(ucdrBuffer& buf, airspeed_validated_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.indicated_airspeed_m_s) == 4, "size mismatch");
	memcpy(&topic.indicated_airspeed_m_s, buf.iterator, sizeof(topic.indicated_airspeed_m_s));
	buf.iterator += sizeof(topic.indicated_airspeed_m_s);
	buf.offset += sizeof(topic.indicated_airspeed_m_s);
	static_assert(sizeof(topic.calibrated_airspeed_m_s) == 4, "size mismatch");
	memcpy(&topic.calibrated_airspeed_m_s, buf.iterator, sizeof(topic.calibrated_airspeed_m_s));
	buf.iterator += sizeof(topic.calibrated_airspeed_m_s);
	buf.offset += sizeof(topic.calibrated_airspeed_m_s);
	static_assert(sizeof(topic.true_airspeed_m_s) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_m_s, buf.iterator, sizeof(topic.true_airspeed_m_s));
	buf.iterator += sizeof(topic.true_airspeed_m_s);
	buf.offset += sizeof(topic.true_airspeed_m_s);
	static_assert(sizeof(topic.calibrated_ground_minus_wind_m_s) == 4, "size mismatch");
	memcpy(&topic.calibrated_ground_minus_wind_m_s, buf.iterator, sizeof(topic.calibrated_ground_minus_wind_m_s));
	buf.iterator += sizeof(topic.calibrated_ground_minus_wind_m_s);
	buf.offset += sizeof(topic.calibrated_ground_minus_wind_m_s);
	static_assert(sizeof(topic.true_ground_minus_wind_m_s) == 4, "size mismatch");
	memcpy(&topic.true_ground_minus_wind_m_s, buf.iterator, sizeof(topic.true_ground_minus_wind_m_s));
	buf.iterator += sizeof(topic.true_ground_minus_wind_m_s);
	buf.offset += sizeof(topic.true_ground_minus_wind_m_s);
	static_assert(sizeof(topic.airspeed_sensor_measurement_valid) == 1, "size mismatch");
	memcpy(&topic.airspeed_sensor_measurement_valid, buf.iterator, sizeof(topic.airspeed_sensor_measurement_valid));
	buf.iterator += sizeof(topic.airspeed_sensor_measurement_valid);
	buf.offset += sizeof(topic.airspeed_sensor_measurement_valid);
	static_assert(sizeof(topic.selected_airspeed_index) == 1, "size mismatch");
	memcpy(&topic.selected_airspeed_index, buf.iterator, sizeof(topic.selected_airspeed_index));
	buf.iterator += sizeof(topic.selected_airspeed_index);
	buf.offset += sizeof(topic.selected_airspeed_index);
	return true;
}
