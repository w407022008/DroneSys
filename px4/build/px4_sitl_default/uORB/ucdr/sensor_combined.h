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
#include <uORB/topics/sensor_combined.h>


static inline constexpr int ucdr_topic_size_sensor_combined()
{
	return 48;
}

bool ucdr_serialize_sensor_combined(const sensor_combined_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.gyro_rad) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_rad, sizeof(topic.gyro_rad));
	buf.iterator += sizeof(topic.gyro_rad);
	buf.offset += sizeof(topic.gyro_rad);
	static_assert(sizeof(topic.gyro_integral_dt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_integral_dt, sizeof(topic.gyro_integral_dt));
	buf.iterator += sizeof(topic.gyro_integral_dt);
	buf.offset += sizeof(topic.gyro_integral_dt);
	static_assert(sizeof(topic.accelerometer_timestamp_relative) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accelerometer_timestamp_relative, sizeof(topic.accelerometer_timestamp_relative));
	buf.iterator += sizeof(topic.accelerometer_timestamp_relative);
	buf.offset += sizeof(topic.accelerometer_timestamp_relative);
	static_assert(sizeof(topic.accelerometer_m_s2) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accelerometer_m_s2, sizeof(topic.accelerometer_m_s2));
	buf.iterator += sizeof(topic.accelerometer_m_s2);
	buf.offset += sizeof(topic.accelerometer_m_s2);
	static_assert(sizeof(topic.accelerometer_integral_dt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accelerometer_integral_dt, sizeof(topic.accelerometer_integral_dt));
	buf.iterator += sizeof(topic.accelerometer_integral_dt);
	buf.offset += sizeof(topic.accelerometer_integral_dt);
	static_assert(sizeof(topic.accelerometer_clipping) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.accelerometer_clipping, sizeof(topic.accelerometer_clipping));
	buf.iterator += sizeof(topic.accelerometer_clipping);
	buf.offset += sizeof(topic.accelerometer_clipping);
	static_assert(sizeof(topic.gyro_clipping) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_clipping, sizeof(topic.gyro_clipping));
	buf.iterator += sizeof(topic.gyro_clipping);
	buf.offset += sizeof(topic.gyro_clipping);
	static_assert(sizeof(topic.accel_calibration_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.accel_calibration_count, sizeof(topic.accel_calibration_count));
	buf.iterator += sizeof(topic.accel_calibration_count);
	buf.offset += sizeof(topic.accel_calibration_count);
	static_assert(sizeof(topic.gyro_calibration_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_calibration_count, sizeof(topic.gyro_calibration_count));
	buf.iterator += sizeof(topic.gyro_calibration_count);
	buf.offset += sizeof(topic.gyro_calibration_count);
	return true;
}

bool ucdr_deserialize_sensor_combined(ucdrBuffer& buf, sensor_combined_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.gyro_rad) == 12, "size mismatch");
	memcpy(&topic.gyro_rad, buf.iterator, sizeof(topic.gyro_rad));
	buf.iterator += sizeof(topic.gyro_rad);
	buf.offset += sizeof(topic.gyro_rad);
	static_assert(sizeof(topic.gyro_integral_dt) == 4, "size mismatch");
	memcpy(&topic.gyro_integral_dt, buf.iterator, sizeof(topic.gyro_integral_dt));
	buf.iterator += sizeof(topic.gyro_integral_dt);
	buf.offset += sizeof(topic.gyro_integral_dt);
	static_assert(sizeof(topic.accelerometer_timestamp_relative) == 4, "size mismatch");
	memcpy(&topic.accelerometer_timestamp_relative, buf.iterator, sizeof(topic.accelerometer_timestamp_relative));
	buf.iterator += sizeof(topic.accelerometer_timestamp_relative);
	buf.offset += sizeof(topic.accelerometer_timestamp_relative);
	static_assert(sizeof(topic.accelerometer_m_s2) == 12, "size mismatch");
	memcpy(&topic.accelerometer_m_s2, buf.iterator, sizeof(topic.accelerometer_m_s2));
	buf.iterator += sizeof(topic.accelerometer_m_s2);
	buf.offset += sizeof(topic.accelerometer_m_s2);
	static_assert(sizeof(topic.accelerometer_integral_dt) == 4, "size mismatch");
	memcpy(&topic.accelerometer_integral_dt, buf.iterator, sizeof(topic.accelerometer_integral_dt));
	buf.iterator += sizeof(topic.accelerometer_integral_dt);
	buf.offset += sizeof(topic.accelerometer_integral_dt);
	static_assert(sizeof(topic.accelerometer_clipping) == 1, "size mismatch");
	memcpy(&topic.accelerometer_clipping, buf.iterator, sizeof(topic.accelerometer_clipping));
	buf.iterator += sizeof(topic.accelerometer_clipping);
	buf.offset += sizeof(topic.accelerometer_clipping);
	static_assert(sizeof(topic.gyro_clipping) == 1, "size mismatch");
	memcpy(&topic.gyro_clipping, buf.iterator, sizeof(topic.gyro_clipping));
	buf.iterator += sizeof(topic.gyro_clipping);
	buf.offset += sizeof(topic.gyro_clipping);
	static_assert(sizeof(topic.accel_calibration_count) == 1, "size mismatch");
	memcpy(&topic.accel_calibration_count, buf.iterator, sizeof(topic.accel_calibration_count));
	buf.iterator += sizeof(topic.accel_calibration_count);
	buf.offset += sizeof(topic.accel_calibration_count);
	static_assert(sizeof(topic.gyro_calibration_count) == 1, "size mismatch");
	memcpy(&topic.gyro_calibration_count, buf.iterator, sizeof(topic.gyro_calibration_count));
	buf.iterator += sizeof(topic.gyro_calibration_count);
	buf.offset += sizeof(topic.gyro_calibration_count);
	return true;
}
