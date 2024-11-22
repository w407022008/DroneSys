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
#include <uORB/topics/vehicle_imu_status.h>


static inline constexpr int ucdr_topic_size_vehicle_imu_status()
{
	return 132;
}

bool ucdr_serialize_vehicle_imu_status(const vehicle_imu_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_id, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_id, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.accel_clipping) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_clipping, sizeof(topic.accel_clipping));
	buf.iterator += sizeof(topic.accel_clipping);
	buf.offset += sizeof(topic.accel_clipping);
	static_assert(sizeof(topic.gyro_clipping) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_clipping, sizeof(topic.gyro_clipping));
	buf.iterator += sizeof(topic.gyro_clipping);
	buf.offset += sizeof(topic.gyro_clipping);
	static_assert(sizeof(topic.accel_error_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_error_count, sizeof(topic.accel_error_count));
	buf.iterator += sizeof(topic.accel_error_count);
	buf.offset += sizeof(topic.accel_error_count);
	static_assert(sizeof(topic.gyro_error_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_error_count, sizeof(topic.gyro_error_count));
	buf.iterator += sizeof(topic.gyro_error_count);
	buf.offset += sizeof(topic.gyro_error_count);
	static_assert(sizeof(topic.accel_rate_hz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_rate_hz, sizeof(topic.accel_rate_hz));
	buf.iterator += sizeof(topic.accel_rate_hz);
	buf.offset += sizeof(topic.accel_rate_hz);
	static_assert(sizeof(topic.gyro_rate_hz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_rate_hz, sizeof(topic.gyro_rate_hz));
	buf.iterator += sizeof(topic.gyro_rate_hz);
	buf.offset += sizeof(topic.gyro_rate_hz);
	static_assert(sizeof(topic.accel_raw_rate_hz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_raw_rate_hz, sizeof(topic.accel_raw_rate_hz));
	buf.iterator += sizeof(topic.accel_raw_rate_hz);
	buf.offset += sizeof(topic.accel_raw_rate_hz);
	static_assert(sizeof(topic.gyro_raw_rate_hz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_raw_rate_hz, sizeof(topic.gyro_raw_rate_hz));
	buf.iterator += sizeof(topic.gyro_raw_rate_hz);
	buf.offset += sizeof(topic.gyro_raw_rate_hz);
	static_assert(sizeof(topic.accel_vibration_metric) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_vibration_metric, sizeof(topic.accel_vibration_metric));
	buf.iterator += sizeof(topic.accel_vibration_metric);
	buf.offset += sizeof(topic.accel_vibration_metric);
	static_assert(sizeof(topic.gyro_vibration_metric) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_vibration_metric, sizeof(topic.gyro_vibration_metric));
	buf.iterator += sizeof(topic.gyro_vibration_metric);
	buf.offset += sizeof(topic.gyro_vibration_metric);
	static_assert(sizeof(topic.delta_angle_coning_metric) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.delta_angle_coning_metric, sizeof(topic.delta_angle_coning_metric));
	buf.iterator += sizeof(topic.delta_angle_coning_metric);
	buf.offset += sizeof(topic.delta_angle_coning_metric);
	static_assert(sizeof(topic.mean_accel) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.mean_accel, sizeof(topic.mean_accel));
	buf.iterator += sizeof(topic.mean_accel);
	buf.offset += sizeof(topic.mean_accel);
	static_assert(sizeof(topic.mean_gyro) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.mean_gyro, sizeof(topic.mean_gyro));
	buf.iterator += sizeof(topic.mean_gyro);
	buf.offset += sizeof(topic.mean_gyro);
	static_assert(sizeof(topic.var_accel) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.var_accel, sizeof(topic.var_accel));
	buf.iterator += sizeof(topic.var_accel);
	buf.offset += sizeof(topic.var_accel);
	static_assert(sizeof(topic.var_gyro) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.var_gyro, sizeof(topic.var_gyro));
	buf.iterator += sizeof(topic.var_gyro);
	buf.offset += sizeof(topic.var_gyro);
	static_assert(sizeof(topic.temperature_accel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_accel, sizeof(topic.temperature_accel));
	buf.iterator += sizeof(topic.temperature_accel);
	buf.offset += sizeof(topic.temperature_accel);
	static_assert(sizeof(topic.temperature_gyro) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_gyro, sizeof(topic.temperature_gyro));
	buf.iterator += sizeof(topic.temperature_gyro);
	buf.offset += sizeof(topic.temperature_gyro);
	return true;
}

bool ucdr_deserialize_vehicle_imu_status(ucdrBuffer& buf, vehicle_imu_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(&topic.accel_device_id, buf.iterator, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(&topic.gyro_device_id, buf.iterator, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.accel_clipping) == 12, "size mismatch");
	memcpy(&topic.accel_clipping, buf.iterator, sizeof(topic.accel_clipping));
	buf.iterator += sizeof(topic.accel_clipping);
	buf.offset += sizeof(topic.accel_clipping);
	static_assert(sizeof(topic.gyro_clipping) == 12, "size mismatch");
	memcpy(&topic.gyro_clipping, buf.iterator, sizeof(topic.gyro_clipping));
	buf.iterator += sizeof(topic.gyro_clipping);
	buf.offset += sizeof(topic.gyro_clipping);
	static_assert(sizeof(topic.accel_error_count) == 4, "size mismatch");
	memcpy(&topic.accel_error_count, buf.iterator, sizeof(topic.accel_error_count));
	buf.iterator += sizeof(topic.accel_error_count);
	buf.offset += sizeof(topic.accel_error_count);
	static_assert(sizeof(topic.gyro_error_count) == 4, "size mismatch");
	memcpy(&topic.gyro_error_count, buf.iterator, sizeof(topic.gyro_error_count));
	buf.iterator += sizeof(topic.gyro_error_count);
	buf.offset += sizeof(topic.gyro_error_count);
	static_assert(sizeof(topic.accel_rate_hz) == 4, "size mismatch");
	memcpy(&topic.accel_rate_hz, buf.iterator, sizeof(topic.accel_rate_hz));
	buf.iterator += sizeof(topic.accel_rate_hz);
	buf.offset += sizeof(topic.accel_rate_hz);
	static_assert(sizeof(topic.gyro_rate_hz) == 4, "size mismatch");
	memcpy(&topic.gyro_rate_hz, buf.iterator, sizeof(topic.gyro_rate_hz));
	buf.iterator += sizeof(topic.gyro_rate_hz);
	buf.offset += sizeof(topic.gyro_rate_hz);
	static_assert(sizeof(topic.accel_raw_rate_hz) == 4, "size mismatch");
	memcpy(&topic.accel_raw_rate_hz, buf.iterator, sizeof(topic.accel_raw_rate_hz));
	buf.iterator += sizeof(topic.accel_raw_rate_hz);
	buf.offset += sizeof(topic.accel_raw_rate_hz);
	static_assert(sizeof(topic.gyro_raw_rate_hz) == 4, "size mismatch");
	memcpy(&topic.gyro_raw_rate_hz, buf.iterator, sizeof(topic.gyro_raw_rate_hz));
	buf.iterator += sizeof(topic.gyro_raw_rate_hz);
	buf.offset += sizeof(topic.gyro_raw_rate_hz);
	static_assert(sizeof(topic.accel_vibration_metric) == 4, "size mismatch");
	memcpy(&topic.accel_vibration_metric, buf.iterator, sizeof(topic.accel_vibration_metric));
	buf.iterator += sizeof(topic.accel_vibration_metric);
	buf.offset += sizeof(topic.accel_vibration_metric);
	static_assert(sizeof(topic.gyro_vibration_metric) == 4, "size mismatch");
	memcpy(&topic.gyro_vibration_metric, buf.iterator, sizeof(topic.gyro_vibration_metric));
	buf.iterator += sizeof(topic.gyro_vibration_metric);
	buf.offset += sizeof(topic.gyro_vibration_metric);
	static_assert(sizeof(topic.delta_angle_coning_metric) == 4, "size mismatch");
	memcpy(&topic.delta_angle_coning_metric, buf.iterator, sizeof(topic.delta_angle_coning_metric));
	buf.iterator += sizeof(topic.delta_angle_coning_metric);
	buf.offset += sizeof(topic.delta_angle_coning_metric);
	static_assert(sizeof(topic.mean_accel) == 12, "size mismatch");
	memcpy(&topic.mean_accel, buf.iterator, sizeof(topic.mean_accel));
	buf.iterator += sizeof(topic.mean_accel);
	buf.offset += sizeof(topic.mean_accel);
	static_assert(sizeof(topic.mean_gyro) == 12, "size mismatch");
	memcpy(&topic.mean_gyro, buf.iterator, sizeof(topic.mean_gyro));
	buf.iterator += sizeof(topic.mean_gyro);
	buf.offset += sizeof(topic.mean_gyro);
	static_assert(sizeof(topic.var_accel) == 12, "size mismatch");
	memcpy(&topic.var_accel, buf.iterator, sizeof(topic.var_accel));
	buf.iterator += sizeof(topic.var_accel);
	buf.offset += sizeof(topic.var_accel);
	static_assert(sizeof(topic.var_gyro) == 12, "size mismatch");
	memcpy(&topic.var_gyro, buf.iterator, sizeof(topic.var_gyro));
	buf.iterator += sizeof(topic.var_gyro);
	buf.offset += sizeof(topic.var_gyro);
	static_assert(sizeof(topic.temperature_accel) == 4, "size mismatch");
	memcpy(&topic.temperature_accel, buf.iterator, sizeof(topic.temperature_accel));
	buf.iterator += sizeof(topic.temperature_accel);
	buf.offset += sizeof(topic.temperature_accel);
	static_assert(sizeof(topic.temperature_gyro) == 4, "size mismatch");
	memcpy(&topic.temperature_gyro, buf.iterator, sizeof(topic.temperature_gyro));
	buf.iterator += sizeof(topic.temperature_gyro);
	buf.offset += sizeof(topic.temperature_gyro);
	return true;
}
