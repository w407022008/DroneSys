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
#include <uORB/topics/sensor_correction.h>


static inline constexpr int ucdr_topic_size_sensor_correction()
{
	return 216;
}

bool ucdr_serialize_sensor_correction(const sensor_correction_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.gyro_device_ids) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_ids, sizeof(topic.gyro_device_ids));
	buf.iterator += sizeof(topic.gyro_device_ids);
	buf.offset += sizeof(topic.gyro_device_ids);
	static_assert(sizeof(topic.gyro_temperature) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_temperature, sizeof(topic.gyro_temperature));
	buf.iterator += sizeof(topic.gyro_temperature);
	buf.offset += sizeof(topic.gyro_temperature);
	static_assert(sizeof(topic.gyro_offset_0) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_offset_0, sizeof(topic.gyro_offset_0));
	buf.iterator += sizeof(topic.gyro_offset_0);
	buf.offset += sizeof(topic.gyro_offset_0);
	static_assert(sizeof(topic.gyro_offset_1) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_offset_1, sizeof(topic.gyro_offset_1));
	buf.iterator += sizeof(topic.gyro_offset_1);
	buf.offset += sizeof(topic.gyro_offset_1);
	static_assert(sizeof(topic.gyro_offset_2) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_offset_2, sizeof(topic.gyro_offset_2));
	buf.iterator += sizeof(topic.gyro_offset_2);
	buf.offset += sizeof(topic.gyro_offset_2);
	static_assert(sizeof(topic.gyro_offset_3) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_offset_3, sizeof(topic.gyro_offset_3));
	buf.iterator += sizeof(topic.gyro_offset_3);
	buf.offset += sizeof(topic.gyro_offset_3);
	static_assert(sizeof(topic.accel_device_ids) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_ids, sizeof(topic.accel_device_ids));
	buf.iterator += sizeof(topic.accel_device_ids);
	buf.offset += sizeof(topic.accel_device_ids);
	static_assert(sizeof(topic.accel_temperature) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accel_temperature, sizeof(topic.accel_temperature));
	buf.iterator += sizeof(topic.accel_temperature);
	buf.offset += sizeof(topic.accel_temperature);
	static_assert(sizeof(topic.accel_offset_0) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_offset_0, sizeof(topic.accel_offset_0));
	buf.iterator += sizeof(topic.accel_offset_0);
	buf.offset += sizeof(topic.accel_offset_0);
	static_assert(sizeof(topic.accel_offset_1) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_offset_1, sizeof(topic.accel_offset_1));
	buf.iterator += sizeof(topic.accel_offset_1);
	buf.offset += sizeof(topic.accel_offset_1);
	static_assert(sizeof(topic.accel_offset_2) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_offset_2, sizeof(topic.accel_offset_2));
	buf.iterator += sizeof(topic.accel_offset_2);
	buf.offset += sizeof(topic.accel_offset_2);
	static_assert(sizeof(topic.accel_offset_3) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_offset_3, sizeof(topic.accel_offset_3));
	buf.iterator += sizeof(topic.accel_offset_3);
	buf.offset += sizeof(topic.accel_offset_3);
	static_assert(sizeof(topic.baro_device_ids) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.baro_device_ids, sizeof(topic.baro_device_ids));
	buf.iterator += sizeof(topic.baro_device_ids);
	buf.offset += sizeof(topic.baro_device_ids);
	static_assert(sizeof(topic.baro_temperature) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.baro_temperature, sizeof(topic.baro_temperature));
	buf.iterator += sizeof(topic.baro_temperature);
	buf.offset += sizeof(topic.baro_temperature);
	static_assert(sizeof(topic.baro_offset_0) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_offset_0, sizeof(topic.baro_offset_0));
	buf.iterator += sizeof(topic.baro_offset_0);
	buf.offset += sizeof(topic.baro_offset_0);
	static_assert(sizeof(topic.baro_offset_1) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_offset_1, sizeof(topic.baro_offset_1));
	buf.iterator += sizeof(topic.baro_offset_1);
	buf.offset += sizeof(topic.baro_offset_1);
	static_assert(sizeof(topic.baro_offset_2) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_offset_2, sizeof(topic.baro_offset_2));
	buf.iterator += sizeof(topic.baro_offset_2);
	buf.offset += sizeof(topic.baro_offset_2);
	static_assert(sizeof(topic.baro_offset_3) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_offset_3, sizeof(topic.baro_offset_3));
	buf.iterator += sizeof(topic.baro_offset_3);
	buf.offset += sizeof(topic.baro_offset_3);
	return true;
}

bool ucdr_deserialize_sensor_correction(ucdrBuffer& buf, sensor_correction_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.gyro_device_ids) == 16, "size mismatch");
	memcpy(&topic.gyro_device_ids, buf.iterator, sizeof(topic.gyro_device_ids));
	buf.iterator += sizeof(topic.gyro_device_ids);
	buf.offset += sizeof(topic.gyro_device_ids);
	static_assert(sizeof(topic.gyro_temperature) == 16, "size mismatch");
	memcpy(&topic.gyro_temperature, buf.iterator, sizeof(topic.gyro_temperature));
	buf.iterator += sizeof(topic.gyro_temperature);
	buf.offset += sizeof(topic.gyro_temperature);
	static_assert(sizeof(topic.gyro_offset_0) == 12, "size mismatch");
	memcpy(&topic.gyro_offset_0, buf.iterator, sizeof(topic.gyro_offset_0));
	buf.iterator += sizeof(topic.gyro_offset_0);
	buf.offset += sizeof(topic.gyro_offset_0);
	static_assert(sizeof(topic.gyro_offset_1) == 12, "size mismatch");
	memcpy(&topic.gyro_offset_1, buf.iterator, sizeof(topic.gyro_offset_1));
	buf.iterator += sizeof(topic.gyro_offset_1);
	buf.offset += sizeof(topic.gyro_offset_1);
	static_assert(sizeof(topic.gyro_offset_2) == 12, "size mismatch");
	memcpy(&topic.gyro_offset_2, buf.iterator, sizeof(topic.gyro_offset_2));
	buf.iterator += sizeof(topic.gyro_offset_2);
	buf.offset += sizeof(topic.gyro_offset_2);
	static_assert(sizeof(topic.gyro_offset_3) == 12, "size mismatch");
	memcpy(&topic.gyro_offset_3, buf.iterator, sizeof(topic.gyro_offset_3));
	buf.iterator += sizeof(topic.gyro_offset_3);
	buf.offset += sizeof(topic.gyro_offset_3);
	static_assert(sizeof(topic.accel_device_ids) == 16, "size mismatch");
	memcpy(&topic.accel_device_ids, buf.iterator, sizeof(topic.accel_device_ids));
	buf.iterator += sizeof(topic.accel_device_ids);
	buf.offset += sizeof(topic.accel_device_ids);
	static_assert(sizeof(topic.accel_temperature) == 16, "size mismatch");
	memcpy(&topic.accel_temperature, buf.iterator, sizeof(topic.accel_temperature));
	buf.iterator += sizeof(topic.accel_temperature);
	buf.offset += sizeof(topic.accel_temperature);
	static_assert(sizeof(topic.accel_offset_0) == 12, "size mismatch");
	memcpy(&topic.accel_offset_0, buf.iterator, sizeof(topic.accel_offset_0));
	buf.iterator += sizeof(topic.accel_offset_0);
	buf.offset += sizeof(topic.accel_offset_0);
	static_assert(sizeof(topic.accel_offset_1) == 12, "size mismatch");
	memcpy(&topic.accel_offset_1, buf.iterator, sizeof(topic.accel_offset_1));
	buf.iterator += sizeof(topic.accel_offset_1);
	buf.offset += sizeof(topic.accel_offset_1);
	static_assert(sizeof(topic.accel_offset_2) == 12, "size mismatch");
	memcpy(&topic.accel_offset_2, buf.iterator, sizeof(topic.accel_offset_2));
	buf.iterator += sizeof(topic.accel_offset_2);
	buf.offset += sizeof(topic.accel_offset_2);
	static_assert(sizeof(topic.accel_offset_3) == 12, "size mismatch");
	memcpy(&topic.accel_offset_3, buf.iterator, sizeof(topic.accel_offset_3));
	buf.iterator += sizeof(topic.accel_offset_3);
	buf.offset += sizeof(topic.accel_offset_3);
	static_assert(sizeof(topic.baro_device_ids) == 16, "size mismatch");
	memcpy(&topic.baro_device_ids, buf.iterator, sizeof(topic.baro_device_ids));
	buf.iterator += sizeof(topic.baro_device_ids);
	buf.offset += sizeof(topic.baro_device_ids);
	static_assert(sizeof(topic.baro_temperature) == 16, "size mismatch");
	memcpy(&topic.baro_temperature, buf.iterator, sizeof(topic.baro_temperature));
	buf.iterator += sizeof(topic.baro_temperature);
	buf.offset += sizeof(topic.baro_temperature);
	static_assert(sizeof(topic.baro_offset_0) == 4, "size mismatch");
	memcpy(&topic.baro_offset_0, buf.iterator, sizeof(topic.baro_offset_0));
	buf.iterator += sizeof(topic.baro_offset_0);
	buf.offset += sizeof(topic.baro_offset_0);
	static_assert(sizeof(topic.baro_offset_1) == 4, "size mismatch");
	memcpy(&topic.baro_offset_1, buf.iterator, sizeof(topic.baro_offset_1));
	buf.iterator += sizeof(topic.baro_offset_1);
	buf.offset += sizeof(topic.baro_offset_1);
	static_assert(sizeof(topic.baro_offset_2) == 4, "size mismatch");
	memcpy(&topic.baro_offset_2, buf.iterator, sizeof(topic.baro_offset_2));
	buf.iterator += sizeof(topic.baro_offset_2);
	buf.offset += sizeof(topic.baro_offset_2);
	static_assert(sizeof(topic.baro_offset_3) == 4, "size mismatch");
	memcpy(&topic.baro_offset_3, buf.iterator, sizeof(topic.baro_offset_3));
	buf.iterator += sizeof(topic.baro_offset_3);
	buf.offset += sizeof(topic.baro_offset_3);
	return true;
}
