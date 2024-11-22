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
#include <uORB/topics/gimbal_device_information.h>


static inline constexpr int ucdr_topic_size_gimbal_device_information()
{
	return 149;
}

bool ucdr_serialize_gimbal_device_information(const gimbal_device_information_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.vendor_name) == 32, "size mismatch");
	memcpy(buf.iterator, &topic.vendor_name, sizeof(topic.vendor_name));
	buf.iterator += sizeof(topic.vendor_name);
	buf.offset += sizeof(topic.vendor_name);
	static_assert(sizeof(topic.model_name) == 32, "size mismatch");
	memcpy(buf.iterator, &topic.model_name, sizeof(topic.model_name));
	buf.iterator += sizeof(topic.model_name);
	buf.offset += sizeof(topic.model_name);
	static_assert(sizeof(topic.custom_name) == 32, "size mismatch");
	memcpy(buf.iterator, &topic.custom_name, sizeof(topic.custom_name));
	buf.iterator += sizeof(topic.custom_name);
	buf.offset += sizeof(topic.custom_name);
	static_assert(sizeof(topic.firmware_version) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.firmware_version, sizeof(topic.firmware_version));
	buf.iterator += sizeof(topic.firmware_version);
	buf.offset += sizeof(topic.firmware_version);
	static_assert(sizeof(topic.hardware_version) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hardware_version, sizeof(topic.hardware_version));
	buf.iterator += sizeof(topic.hardware_version);
	buf.offset += sizeof(topic.hardware_version);
	static_assert(sizeof(topic.uid) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.uid, sizeof(topic.uid));
	buf.iterator += sizeof(topic.uid);
	buf.offset += sizeof(topic.uid);
	static_assert(sizeof(topic.cap_flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.cap_flags, sizeof(topic.cap_flags));
	buf.iterator += sizeof(topic.cap_flags);
	buf.offset += sizeof(topic.cap_flags);
	static_assert(sizeof(topic.custom_cap_flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.custom_cap_flags, sizeof(topic.custom_cap_flags));
	buf.iterator += sizeof(topic.custom_cap_flags);
	buf.offset += sizeof(topic.custom_cap_flags);
	static_assert(sizeof(topic.roll_min) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.roll_min, sizeof(topic.roll_min));
	buf.iterator += sizeof(topic.roll_min);
	buf.offset += sizeof(topic.roll_min);
	static_assert(sizeof(topic.roll_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.roll_max, sizeof(topic.roll_max));
	buf.iterator += sizeof(topic.roll_max);
	buf.offset += sizeof(topic.roll_max);
	static_assert(sizeof(topic.pitch_min) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_min, sizeof(topic.pitch_min));
	buf.iterator += sizeof(topic.pitch_min);
	buf.offset += sizeof(topic.pitch_min);
	static_assert(sizeof(topic.pitch_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_max, sizeof(topic.pitch_max));
	buf.iterator += sizeof(topic.pitch_max);
	buf.offset += sizeof(topic.pitch_max);
	static_assert(sizeof(topic.yaw_min) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_min, sizeof(topic.yaw_min));
	buf.iterator += sizeof(topic.yaw_min);
	buf.offset += sizeof(topic.yaw_min);
	static_assert(sizeof(topic.yaw_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_max, sizeof(topic.yaw_max));
	buf.iterator += sizeof(topic.yaw_max);
	buf.offset += sizeof(topic.yaw_max);
	static_assert(sizeof(topic.gimbal_device_compid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gimbal_device_compid, sizeof(topic.gimbal_device_compid));
	buf.iterator += sizeof(topic.gimbal_device_compid);
	buf.offset += sizeof(topic.gimbal_device_compid);
	return true;
}

bool ucdr_deserialize_gimbal_device_information(ucdrBuffer& buf, gimbal_device_information_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.vendor_name) == 32, "size mismatch");
	memcpy(&topic.vendor_name, buf.iterator, sizeof(topic.vendor_name));
	buf.iterator += sizeof(topic.vendor_name);
	buf.offset += sizeof(topic.vendor_name);
	static_assert(sizeof(topic.model_name) == 32, "size mismatch");
	memcpy(&topic.model_name, buf.iterator, sizeof(topic.model_name));
	buf.iterator += sizeof(topic.model_name);
	buf.offset += sizeof(topic.model_name);
	static_assert(sizeof(topic.custom_name) == 32, "size mismatch");
	memcpy(&topic.custom_name, buf.iterator, sizeof(topic.custom_name));
	buf.iterator += sizeof(topic.custom_name);
	buf.offset += sizeof(topic.custom_name);
	static_assert(sizeof(topic.firmware_version) == 4, "size mismatch");
	memcpy(&topic.firmware_version, buf.iterator, sizeof(topic.firmware_version));
	buf.iterator += sizeof(topic.firmware_version);
	buf.offset += sizeof(topic.firmware_version);
	static_assert(sizeof(topic.hardware_version) == 4, "size mismatch");
	memcpy(&topic.hardware_version, buf.iterator, sizeof(topic.hardware_version));
	buf.iterator += sizeof(topic.hardware_version);
	buf.offset += sizeof(topic.hardware_version);
	static_assert(sizeof(topic.uid) == 8, "size mismatch");
	memcpy(&topic.uid, buf.iterator, sizeof(topic.uid));
	buf.iterator += sizeof(topic.uid);
	buf.offset += sizeof(topic.uid);
	static_assert(sizeof(topic.cap_flags) == 2, "size mismatch");
	memcpy(&topic.cap_flags, buf.iterator, sizeof(topic.cap_flags));
	buf.iterator += sizeof(topic.cap_flags);
	buf.offset += sizeof(topic.cap_flags);
	static_assert(sizeof(topic.custom_cap_flags) == 2, "size mismatch");
	memcpy(&topic.custom_cap_flags, buf.iterator, sizeof(topic.custom_cap_flags));
	buf.iterator += sizeof(topic.custom_cap_flags);
	buf.offset += sizeof(topic.custom_cap_flags);
	static_assert(sizeof(topic.roll_min) == 4, "size mismatch");
	memcpy(&topic.roll_min, buf.iterator, sizeof(topic.roll_min));
	buf.iterator += sizeof(topic.roll_min);
	buf.offset += sizeof(topic.roll_min);
	static_assert(sizeof(topic.roll_max) == 4, "size mismatch");
	memcpy(&topic.roll_max, buf.iterator, sizeof(topic.roll_max));
	buf.iterator += sizeof(topic.roll_max);
	buf.offset += sizeof(topic.roll_max);
	static_assert(sizeof(topic.pitch_min) == 4, "size mismatch");
	memcpy(&topic.pitch_min, buf.iterator, sizeof(topic.pitch_min));
	buf.iterator += sizeof(topic.pitch_min);
	buf.offset += sizeof(topic.pitch_min);
	static_assert(sizeof(topic.pitch_max) == 4, "size mismatch");
	memcpy(&topic.pitch_max, buf.iterator, sizeof(topic.pitch_max));
	buf.iterator += sizeof(topic.pitch_max);
	buf.offset += sizeof(topic.pitch_max);
	static_assert(sizeof(topic.yaw_min) == 4, "size mismatch");
	memcpy(&topic.yaw_min, buf.iterator, sizeof(topic.yaw_min));
	buf.iterator += sizeof(topic.yaw_min);
	buf.offset += sizeof(topic.yaw_min);
	static_assert(sizeof(topic.yaw_max) == 4, "size mismatch");
	memcpy(&topic.yaw_max, buf.iterator, sizeof(topic.yaw_max));
	buf.iterator += sizeof(topic.yaw_max);
	buf.offset += sizeof(topic.yaw_max);
	static_assert(sizeof(topic.gimbal_device_compid) == 1, "size mismatch");
	memcpy(&topic.gimbal_device_compid, buf.iterator, sizeof(topic.gimbal_device_compid));
	buf.iterator += sizeof(topic.gimbal_device_compid);
	buf.offset += sizeof(topic.gimbal_device_compid);
	return true;
}
