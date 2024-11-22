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
#include <uORB/topics/estimator_selector_status.h>


static inline constexpr int ucdr_topic_size_estimator_selector_status()
{
	return 158;
}

bool ucdr_serialize_estimator_selector_status(const estimator_selector_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.primary_instance) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.primary_instance, sizeof(topic.primary_instance));
	buf.iterator += sizeof(topic.primary_instance);
	buf.offset += sizeof(topic.primary_instance);
	static_assert(sizeof(topic.instances_available) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.instances_available, sizeof(topic.instances_available));
	buf.iterator += sizeof(topic.instances_available);
	buf.offset += sizeof(topic.instances_available);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.instance_changed_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.instance_changed_count, sizeof(topic.instance_changed_count));
	buf.iterator += sizeof(topic.instance_changed_count);
	buf.offset += sizeof(topic.instance_changed_count);
	static_assert(sizeof(topic.last_instance_change) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.last_instance_change, sizeof(topic.last_instance_change));
	buf.iterator += sizeof(topic.last_instance_change);
	buf.offset += sizeof(topic.last_instance_change);
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_id, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_device_id, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_id, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mag_device_id, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.combined_test_ratio) == 36, "size mismatch");
	memcpy(buf.iterator, &topic.combined_test_ratio, sizeof(topic.combined_test_ratio));
	buf.iterator += sizeof(topic.combined_test_ratio);
	buf.offset += sizeof(topic.combined_test_ratio);
	static_assert(sizeof(topic.relative_test_ratio) == 36, "size mismatch");
	memcpy(buf.iterator, &topic.relative_test_ratio, sizeof(topic.relative_test_ratio));
	buf.iterator += sizeof(topic.relative_test_ratio);
	buf.offset += sizeof(topic.relative_test_ratio);
	static_assert(sizeof(topic.healthy) == 9, "size mismatch");
	memcpy(buf.iterator, &topic.healthy, sizeof(topic.healthy));
	buf.iterator += sizeof(topic.healthy);
	buf.offset += sizeof(topic.healthy);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.accumulated_gyro_error) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accumulated_gyro_error, sizeof(topic.accumulated_gyro_error));
	buf.iterator += sizeof(topic.accumulated_gyro_error);
	buf.offset += sizeof(topic.accumulated_gyro_error);
	static_assert(sizeof(topic.accumulated_accel_error) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accumulated_accel_error, sizeof(topic.accumulated_accel_error));
	buf.iterator += sizeof(topic.accumulated_accel_error);
	buf.offset += sizeof(topic.accumulated_accel_error);
	static_assert(sizeof(topic.gyro_fault_detected) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_fault_detected, sizeof(topic.gyro_fault_detected));
	buf.iterator += sizeof(topic.gyro_fault_detected);
	buf.offset += sizeof(topic.gyro_fault_detected);
	static_assert(sizeof(topic.accel_fault_detected) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.accel_fault_detected, sizeof(topic.accel_fault_detected));
	buf.iterator += sizeof(topic.accel_fault_detected);
	buf.offset += sizeof(topic.accel_fault_detected);
	return true;
}

bool ucdr_deserialize_estimator_selector_status(ucdrBuffer& buf, estimator_selector_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.primary_instance) == 1, "size mismatch");
	memcpy(&topic.primary_instance, buf.iterator, sizeof(topic.primary_instance));
	buf.iterator += sizeof(topic.primary_instance);
	buf.offset += sizeof(topic.primary_instance);
	static_assert(sizeof(topic.instances_available) == 1, "size mismatch");
	memcpy(&topic.instances_available, buf.iterator, sizeof(topic.instances_available));
	buf.iterator += sizeof(topic.instances_available);
	buf.offset += sizeof(topic.instances_available);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.instance_changed_count) == 4, "size mismatch");
	memcpy(&topic.instance_changed_count, buf.iterator, sizeof(topic.instance_changed_count));
	buf.iterator += sizeof(topic.instance_changed_count);
	buf.offset += sizeof(topic.instance_changed_count);
	static_assert(sizeof(topic.last_instance_change) == 8, "size mismatch");
	memcpy(&topic.last_instance_change, buf.iterator, sizeof(topic.last_instance_change));
	buf.iterator += sizeof(topic.last_instance_change);
	buf.offset += sizeof(topic.last_instance_change);
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(&topic.accel_device_id, buf.iterator, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(&topic.baro_device_id, buf.iterator, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(&topic.gyro_device_id, buf.iterator, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(&topic.mag_device_id, buf.iterator, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.combined_test_ratio) == 36, "size mismatch");
	memcpy(&topic.combined_test_ratio, buf.iterator, sizeof(topic.combined_test_ratio));
	buf.iterator += sizeof(topic.combined_test_ratio);
	buf.offset += sizeof(topic.combined_test_ratio);
	static_assert(sizeof(topic.relative_test_ratio) == 36, "size mismatch");
	memcpy(&topic.relative_test_ratio, buf.iterator, sizeof(topic.relative_test_ratio));
	buf.iterator += sizeof(topic.relative_test_ratio);
	buf.offset += sizeof(topic.relative_test_ratio);
	static_assert(sizeof(topic.healthy) == 9, "size mismatch");
	memcpy(&topic.healthy, buf.iterator, sizeof(topic.healthy));
	buf.iterator += sizeof(topic.healthy);
	buf.offset += sizeof(topic.healthy);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.accumulated_gyro_error) == 16, "size mismatch");
	memcpy(&topic.accumulated_gyro_error, buf.iterator, sizeof(topic.accumulated_gyro_error));
	buf.iterator += sizeof(topic.accumulated_gyro_error);
	buf.offset += sizeof(topic.accumulated_gyro_error);
	static_assert(sizeof(topic.accumulated_accel_error) == 16, "size mismatch");
	memcpy(&topic.accumulated_accel_error, buf.iterator, sizeof(topic.accumulated_accel_error));
	buf.iterator += sizeof(topic.accumulated_accel_error);
	buf.offset += sizeof(topic.accumulated_accel_error);
	static_assert(sizeof(topic.gyro_fault_detected) == 1, "size mismatch");
	memcpy(&topic.gyro_fault_detected, buf.iterator, sizeof(topic.gyro_fault_detected));
	buf.iterator += sizeof(topic.gyro_fault_detected);
	buf.offset += sizeof(topic.gyro_fault_detected);
	static_assert(sizeof(topic.accel_fault_detected) == 1, "size mismatch");
	memcpy(&topic.accel_fault_detected, buf.iterator, sizeof(topic.accel_fault_detected));
	buf.iterator += sizeof(topic.accel_fault_detected);
	buf.offset += sizeof(topic.accel_fault_detected);
	return true;
}
