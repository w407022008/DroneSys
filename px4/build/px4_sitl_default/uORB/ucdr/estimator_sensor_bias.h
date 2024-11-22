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
#include <uORB/topics/estimator_sensor_bias.h>


static inline constexpr int ucdr_topic_size_estimator_sensor_bias()
{
	return 122;
}

bool ucdr_serialize_estimator_sensor_bias(const estimator_sensor_bias_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_id, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.gyro_bias) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_bias, sizeof(topic.gyro_bias));
	buf.iterator += sizeof(topic.gyro_bias);
	buf.offset += sizeof(topic.gyro_bias);
	static_assert(sizeof(topic.gyro_bias_limit) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_bias_limit, sizeof(topic.gyro_bias_limit));
	buf.iterator += sizeof(topic.gyro_bias_limit);
	buf.offset += sizeof(topic.gyro_bias_limit);
	static_assert(sizeof(topic.gyro_bias_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_bias_variance, sizeof(topic.gyro_bias_variance));
	buf.iterator += sizeof(topic.gyro_bias_variance);
	buf.offset += sizeof(topic.gyro_bias_variance);
	static_assert(sizeof(topic.gyro_bias_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_bias_valid, sizeof(topic.gyro_bias_valid));
	buf.iterator += sizeof(topic.gyro_bias_valid);
	buf.offset += sizeof(topic.gyro_bias_valid);
	static_assert(sizeof(topic.gyro_bias_stable) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_bias_stable, sizeof(topic.gyro_bias_stable));
	buf.iterator += sizeof(topic.gyro_bias_stable);
	buf.offset += sizeof(topic.gyro_bias_stable);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_id, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.accel_bias) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_bias, sizeof(topic.accel_bias));
	buf.iterator += sizeof(topic.accel_bias);
	buf.offset += sizeof(topic.accel_bias);
	static_assert(sizeof(topic.accel_bias_limit) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_bias_limit, sizeof(topic.accel_bias_limit));
	buf.iterator += sizeof(topic.accel_bias_limit);
	buf.offset += sizeof(topic.accel_bias_limit);
	static_assert(sizeof(topic.accel_bias_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.accel_bias_variance, sizeof(topic.accel_bias_variance));
	buf.iterator += sizeof(topic.accel_bias_variance);
	buf.offset += sizeof(topic.accel_bias_variance);
	static_assert(sizeof(topic.accel_bias_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.accel_bias_valid, sizeof(topic.accel_bias_valid));
	buf.iterator += sizeof(topic.accel_bias_valid);
	buf.offset += sizeof(topic.accel_bias_valid);
	static_assert(sizeof(topic.accel_bias_stable) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.accel_bias_stable, sizeof(topic.accel_bias_stable));
	buf.iterator += sizeof(topic.accel_bias_stable);
	buf.offset += sizeof(topic.accel_bias_stable);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mag_device_id, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.mag_bias) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.mag_bias, sizeof(topic.mag_bias));
	buf.iterator += sizeof(topic.mag_bias);
	buf.offset += sizeof(topic.mag_bias);
	static_assert(sizeof(topic.mag_bias_limit) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mag_bias_limit, sizeof(topic.mag_bias_limit));
	buf.iterator += sizeof(topic.mag_bias_limit);
	buf.offset += sizeof(topic.mag_bias_limit);
	static_assert(sizeof(topic.mag_bias_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.mag_bias_variance, sizeof(topic.mag_bias_variance));
	buf.iterator += sizeof(topic.mag_bias_variance);
	buf.offset += sizeof(topic.mag_bias_variance);
	static_assert(sizeof(topic.mag_bias_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mag_bias_valid, sizeof(topic.mag_bias_valid));
	buf.iterator += sizeof(topic.mag_bias_valid);
	buf.offset += sizeof(topic.mag_bias_valid);
	static_assert(sizeof(topic.mag_bias_stable) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mag_bias_stable, sizeof(topic.mag_bias_stable));
	buf.iterator += sizeof(topic.mag_bias_stable);
	buf.offset += sizeof(topic.mag_bias_stable);
	return true;
}

bool ucdr_deserialize_estimator_sensor_bias(ucdrBuffer& buf, estimator_sensor_bias_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(&topic.gyro_device_id, buf.iterator, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.gyro_bias) == 12, "size mismatch");
	memcpy(&topic.gyro_bias, buf.iterator, sizeof(topic.gyro_bias));
	buf.iterator += sizeof(topic.gyro_bias);
	buf.offset += sizeof(topic.gyro_bias);
	static_assert(sizeof(topic.gyro_bias_limit) == 4, "size mismatch");
	memcpy(&topic.gyro_bias_limit, buf.iterator, sizeof(topic.gyro_bias_limit));
	buf.iterator += sizeof(topic.gyro_bias_limit);
	buf.offset += sizeof(topic.gyro_bias_limit);
	static_assert(sizeof(topic.gyro_bias_variance) == 12, "size mismatch");
	memcpy(&topic.gyro_bias_variance, buf.iterator, sizeof(topic.gyro_bias_variance));
	buf.iterator += sizeof(topic.gyro_bias_variance);
	buf.offset += sizeof(topic.gyro_bias_variance);
	static_assert(sizeof(topic.gyro_bias_valid) == 1, "size mismatch");
	memcpy(&topic.gyro_bias_valid, buf.iterator, sizeof(topic.gyro_bias_valid));
	buf.iterator += sizeof(topic.gyro_bias_valid);
	buf.offset += sizeof(topic.gyro_bias_valid);
	static_assert(sizeof(topic.gyro_bias_stable) == 1, "size mismatch");
	memcpy(&topic.gyro_bias_stable, buf.iterator, sizeof(topic.gyro_bias_stable));
	buf.iterator += sizeof(topic.gyro_bias_stable);
	buf.offset += sizeof(topic.gyro_bias_stable);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(&topic.accel_device_id, buf.iterator, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.accel_bias) == 12, "size mismatch");
	memcpy(&topic.accel_bias, buf.iterator, sizeof(topic.accel_bias));
	buf.iterator += sizeof(topic.accel_bias);
	buf.offset += sizeof(topic.accel_bias);
	static_assert(sizeof(topic.accel_bias_limit) == 4, "size mismatch");
	memcpy(&topic.accel_bias_limit, buf.iterator, sizeof(topic.accel_bias_limit));
	buf.iterator += sizeof(topic.accel_bias_limit);
	buf.offset += sizeof(topic.accel_bias_limit);
	static_assert(sizeof(topic.accel_bias_variance) == 12, "size mismatch");
	memcpy(&topic.accel_bias_variance, buf.iterator, sizeof(topic.accel_bias_variance));
	buf.iterator += sizeof(topic.accel_bias_variance);
	buf.offset += sizeof(topic.accel_bias_variance);
	static_assert(sizeof(topic.accel_bias_valid) == 1, "size mismatch");
	memcpy(&topic.accel_bias_valid, buf.iterator, sizeof(topic.accel_bias_valid));
	buf.iterator += sizeof(topic.accel_bias_valid);
	buf.offset += sizeof(topic.accel_bias_valid);
	static_assert(sizeof(topic.accel_bias_stable) == 1, "size mismatch");
	memcpy(&topic.accel_bias_stable, buf.iterator, sizeof(topic.accel_bias_stable));
	buf.iterator += sizeof(topic.accel_bias_stable);
	buf.offset += sizeof(topic.accel_bias_stable);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(&topic.mag_device_id, buf.iterator, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.mag_bias) == 12, "size mismatch");
	memcpy(&topic.mag_bias, buf.iterator, sizeof(topic.mag_bias));
	buf.iterator += sizeof(topic.mag_bias);
	buf.offset += sizeof(topic.mag_bias);
	static_assert(sizeof(topic.mag_bias_limit) == 4, "size mismatch");
	memcpy(&topic.mag_bias_limit, buf.iterator, sizeof(topic.mag_bias_limit));
	buf.iterator += sizeof(topic.mag_bias_limit);
	buf.offset += sizeof(topic.mag_bias_limit);
	static_assert(sizeof(topic.mag_bias_variance) == 12, "size mismatch");
	memcpy(&topic.mag_bias_variance, buf.iterator, sizeof(topic.mag_bias_variance));
	buf.iterator += sizeof(topic.mag_bias_variance);
	buf.offset += sizeof(topic.mag_bias_variance);
	static_assert(sizeof(topic.mag_bias_valid) == 1, "size mismatch");
	memcpy(&topic.mag_bias_valid, buf.iterator, sizeof(topic.mag_bias_valid));
	buf.iterator += sizeof(topic.mag_bias_valid);
	buf.offset += sizeof(topic.mag_bias_valid);
	static_assert(sizeof(topic.mag_bias_stable) == 1, "size mismatch");
	memcpy(&topic.mag_bias_stable, buf.iterator, sizeof(topic.mag_bias_stable));
	buf.iterator += sizeof(topic.mag_bias_stable);
	buf.offset += sizeof(topic.mag_bias_stable);
	return true;
}
