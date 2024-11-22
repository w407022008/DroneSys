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
#include <uORB/topics/gimbal_manager_information.h>


static inline constexpr int ucdr_topic_size_gimbal_manager_information()
{
	return 40;
}

bool ucdr_serialize_gimbal_manager_information(const gimbal_manager_information_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.cap_flags) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cap_flags, sizeof(topic.cap_flags));
	buf.iterator += sizeof(topic.cap_flags);
	buf.offset += sizeof(topic.cap_flags);
	static_assert(sizeof(topic.gimbal_device_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gimbal_device_id, sizeof(topic.gimbal_device_id));
	buf.iterator += sizeof(topic.gimbal_device_id);
	buf.offset += sizeof(topic.gimbal_device_id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
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
	return true;
}

bool ucdr_deserialize_gimbal_manager_information(ucdrBuffer& buf, gimbal_manager_information_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.cap_flags) == 4, "size mismatch");
	memcpy(&topic.cap_flags, buf.iterator, sizeof(topic.cap_flags));
	buf.iterator += sizeof(topic.cap_flags);
	buf.offset += sizeof(topic.cap_flags);
	static_assert(sizeof(topic.gimbal_device_id) == 1, "size mismatch");
	memcpy(&topic.gimbal_device_id, buf.iterator, sizeof(topic.gimbal_device_id));
	buf.iterator += sizeof(topic.gimbal_device_id);
	buf.offset += sizeof(topic.gimbal_device_id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
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
	return true;
}
