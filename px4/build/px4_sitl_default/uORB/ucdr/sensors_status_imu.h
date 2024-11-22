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
#include <uORB/topics/sensors_status_imu.h>


static inline constexpr int ucdr_topic_size_sensors_status_imu()
{
	return 96;
}

bool ucdr_serialize_sensors_status_imu(const sensors_status_imu_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.accel_device_id_primary) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_id_primary, sizeof(topic.accel_device_id_primary));
	buf.iterator += sizeof(topic.accel_device_id_primary);
	buf.offset += sizeof(topic.accel_device_id_primary);
	static_assert(sizeof(topic.accel_device_ids) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_ids, sizeof(topic.accel_device_ids));
	buf.iterator += sizeof(topic.accel_device_ids);
	buf.offset += sizeof(topic.accel_device_ids);
	static_assert(sizeof(topic.accel_inconsistency_m_s_s) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.accel_inconsistency_m_s_s, sizeof(topic.accel_inconsistency_m_s_s));
	buf.iterator += sizeof(topic.accel_inconsistency_m_s_s);
	buf.offset += sizeof(topic.accel_inconsistency_m_s_s);
	static_assert(sizeof(topic.accel_healthy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_healthy, sizeof(topic.accel_healthy));
	buf.iterator += sizeof(topic.accel_healthy);
	buf.offset += sizeof(topic.accel_healthy);
	static_assert(sizeof(topic.accel_priority) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_priority, sizeof(topic.accel_priority));
	buf.iterator += sizeof(topic.accel_priority);
	buf.offset += sizeof(topic.accel_priority);
	static_assert(sizeof(topic.gyro_device_id_primary) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_id_primary, sizeof(topic.gyro_device_id_primary));
	buf.iterator += sizeof(topic.gyro_device_id_primary);
	buf.offset += sizeof(topic.gyro_device_id_primary);
	static_assert(sizeof(topic.gyro_device_ids) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_ids, sizeof(topic.gyro_device_ids));
	buf.iterator += sizeof(topic.gyro_device_ids);
	buf.offset += sizeof(topic.gyro_device_ids);
	static_assert(sizeof(topic.gyro_inconsistency_rad_s) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_inconsistency_rad_s, sizeof(topic.gyro_inconsistency_rad_s));
	buf.iterator += sizeof(topic.gyro_inconsistency_rad_s);
	buf.offset += sizeof(topic.gyro_inconsistency_rad_s);
	static_assert(sizeof(topic.gyro_healthy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_healthy, sizeof(topic.gyro_healthy));
	buf.iterator += sizeof(topic.gyro_healthy);
	buf.offset += sizeof(topic.gyro_healthy);
	static_assert(sizeof(topic.gyro_priority) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_priority, sizeof(topic.gyro_priority));
	buf.iterator += sizeof(topic.gyro_priority);
	buf.offset += sizeof(topic.gyro_priority);
	return true;
}

bool ucdr_deserialize_sensors_status_imu(ucdrBuffer& buf, sensors_status_imu_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.accel_device_id_primary) == 4, "size mismatch");
	memcpy(&topic.accel_device_id_primary, buf.iterator, sizeof(topic.accel_device_id_primary));
	buf.iterator += sizeof(topic.accel_device_id_primary);
	buf.offset += sizeof(topic.accel_device_id_primary);
	static_assert(sizeof(topic.accel_device_ids) == 16, "size mismatch");
	memcpy(&topic.accel_device_ids, buf.iterator, sizeof(topic.accel_device_ids));
	buf.iterator += sizeof(topic.accel_device_ids);
	buf.offset += sizeof(topic.accel_device_ids);
	static_assert(sizeof(topic.accel_inconsistency_m_s_s) == 16, "size mismatch");
	memcpy(&topic.accel_inconsistency_m_s_s, buf.iterator, sizeof(topic.accel_inconsistency_m_s_s));
	buf.iterator += sizeof(topic.accel_inconsistency_m_s_s);
	buf.offset += sizeof(topic.accel_inconsistency_m_s_s);
	static_assert(sizeof(topic.accel_healthy) == 4, "size mismatch");
	memcpy(&topic.accel_healthy, buf.iterator, sizeof(topic.accel_healthy));
	buf.iterator += sizeof(topic.accel_healthy);
	buf.offset += sizeof(topic.accel_healthy);
	static_assert(sizeof(topic.accel_priority) == 4, "size mismatch");
	memcpy(&topic.accel_priority, buf.iterator, sizeof(topic.accel_priority));
	buf.iterator += sizeof(topic.accel_priority);
	buf.offset += sizeof(topic.accel_priority);
	static_assert(sizeof(topic.gyro_device_id_primary) == 4, "size mismatch");
	memcpy(&topic.gyro_device_id_primary, buf.iterator, sizeof(topic.gyro_device_id_primary));
	buf.iterator += sizeof(topic.gyro_device_id_primary);
	buf.offset += sizeof(topic.gyro_device_id_primary);
	static_assert(sizeof(topic.gyro_device_ids) == 16, "size mismatch");
	memcpy(&topic.gyro_device_ids, buf.iterator, sizeof(topic.gyro_device_ids));
	buf.iterator += sizeof(topic.gyro_device_ids);
	buf.offset += sizeof(topic.gyro_device_ids);
	static_assert(sizeof(topic.gyro_inconsistency_rad_s) == 16, "size mismatch");
	memcpy(&topic.gyro_inconsistency_rad_s, buf.iterator, sizeof(topic.gyro_inconsistency_rad_s));
	buf.iterator += sizeof(topic.gyro_inconsistency_rad_s);
	buf.offset += sizeof(topic.gyro_inconsistency_rad_s);
	static_assert(sizeof(topic.gyro_healthy) == 4, "size mismatch");
	memcpy(&topic.gyro_healthy, buf.iterator, sizeof(topic.gyro_healthy));
	buf.iterator += sizeof(topic.gyro_healthy);
	buf.offset += sizeof(topic.gyro_healthy);
	static_assert(sizeof(topic.gyro_priority) == 4, "size mismatch");
	memcpy(&topic.gyro_priority, buf.iterator, sizeof(topic.gyro_priority));
	buf.iterator += sizeof(topic.gyro_priority);
	buf.offset += sizeof(topic.gyro_priority);
	return true;
}
