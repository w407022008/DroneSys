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
#include <uORB/topics/gimbal_manager_set_manual_control.h>


static inline constexpr int ucdr_topic_size_gimbal_manager_set_manual_control()
{
	return 36;
}

bool ucdr_serialize_gimbal_manager_set_manual_control(const gimbal_manager_set_manual_control_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.origin_sysid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.origin_sysid, sizeof(topic.origin_sysid));
	buf.iterator += sizeof(topic.origin_sysid);
	buf.offset += sizeof(topic.origin_sysid);
	static_assert(sizeof(topic.origin_compid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.origin_compid, sizeof(topic.origin_compid));
	buf.iterator += sizeof(topic.origin_compid);
	buf.offset += sizeof(topic.origin_compid);
	static_assert(sizeof(topic.target_system) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.target_system, sizeof(topic.target_system));
	buf.iterator += sizeof(topic.target_system);
	buf.offset += sizeof(topic.target_system);
	static_assert(sizeof(topic.target_component) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.target_component, sizeof(topic.target_component));
	buf.iterator += sizeof(topic.target_component);
	buf.offset += sizeof(topic.target_component);
	static_assert(sizeof(topic.flags) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.flags, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.gimbal_device_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gimbal_device_id, sizeof(topic.gimbal_device_id));
	buf.iterator += sizeof(topic.gimbal_device_id);
	buf.offset += sizeof(topic.gimbal_device_id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.pitch) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch, sizeof(topic.pitch));
	buf.iterator += sizeof(topic.pitch);
	buf.offset += sizeof(topic.pitch);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.pitch_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_rate, sizeof(topic.pitch_rate));
	buf.iterator += sizeof(topic.pitch_rate);
	buf.offset += sizeof(topic.pitch_rate);
	static_assert(sizeof(topic.yaw_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_rate, sizeof(topic.yaw_rate));
	buf.iterator += sizeof(topic.yaw_rate);
	buf.offset += sizeof(topic.yaw_rate);
	return true;
}

bool ucdr_deserialize_gimbal_manager_set_manual_control(ucdrBuffer& buf, gimbal_manager_set_manual_control_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.origin_sysid) == 1, "size mismatch");
	memcpy(&topic.origin_sysid, buf.iterator, sizeof(topic.origin_sysid));
	buf.iterator += sizeof(topic.origin_sysid);
	buf.offset += sizeof(topic.origin_sysid);
	static_assert(sizeof(topic.origin_compid) == 1, "size mismatch");
	memcpy(&topic.origin_compid, buf.iterator, sizeof(topic.origin_compid));
	buf.iterator += sizeof(topic.origin_compid);
	buf.offset += sizeof(topic.origin_compid);
	static_assert(sizeof(topic.target_system) == 1, "size mismatch");
	memcpy(&topic.target_system, buf.iterator, sizeof(topic.target_system));
	buf.iterator += sizeof(topic.target_system);
	buf.offset += sizeof(topic.target_system);
	static_assert(sizeof(topic.target_component) == 1, "size mismatch");
	memcpy(&topic.target_component, buf.iterator, sizeof(topic.target_component));
	buf.iterator += sizeof(topic.target_component);
	buf.offset += sizeof(topic.target_component);
	static_assert(sizeof(topic.flags) == 4, "size mismatch");
	memcpy(&topic.flags, buf.iterator, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.gimbal_device_id) == 1, "size mismatch");
	memcpy(&topic.gimbal_device_id, buf.iterator, sizeof(topic.gimbal_device_id));
	buf.iterator += sizeof(topic.gimbal_device_id);
	buf.offset += sizeof(topic.gimbal_device_id);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.pitch) == 4, "size mismatch");
	memcpy(&topic.pitch, buf.iterator, sizeof(topic.pitch));
	buf.iterator += sizeof(topic.pitch);
	buf.offset += sizeof(topic.pitch);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.pitch_rate) == 4, "size mismatch");
	memcpy(&topic.pitch_rate, buf.iterator, sizeof(topic.pitch_rate));
	buf.iterator += sizeof(topic.pitch_rate);
	buf.offset += sizeof(topic.pitch_rate);
	static_assert(sizeof(topic.yaw_rate) == 4, "size mismatch");
	memcpy(&topic.yaw_rate, buf.iterator, sizeof(topic.yaw_rate));
	buf.iterator += sizeof(topic.yaw_rate);
	buf.offset += sizeof(topic.yaw_rate);
	return true;
}
