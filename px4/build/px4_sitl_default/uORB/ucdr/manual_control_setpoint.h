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
#include <uORB/topics/manual_control_setpoint.h>


static inline constexpr int ucdr_topic_size_manual_control_setpoint()
{
	return 69;
}

bool ucdr_serialize_manual_control_setpoint(const manual_control_setpoint_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.data_source) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.data_source, sizeof(topic.data_source));
	buf.iterator += sizeof(topic.data_source);
	buf.offset += sizeof(topic.data_source);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.roll) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.roll, sizeof(topic.roll));
	buf.iterator += sizeof(topic.roll);
	buf.offset += sizeof(topic.roll);
	static_assert(sizeof(topic.pitch) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch, sizeof(topic.pitch));
	buf.iterator += sizeof(topic.pitch);
	buf.offset += sizeof(topic.pitch);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.throttle) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.throttle, sizeof(topic.throttle));
	buf.iterator += sizeof(topic.throttle);
	buf.offset += sizeof(topic.throttle);
	static_assert(sizeof(topic.flaps) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.flaps, sizeof(topic.flaps));
	buf.iterator += sizeof(topic.flaps);
	buf.offset += sizeof(topic.flaps);
	static_assert(sizeof(topic.buttons) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.buttons, sizeof(topic.buttons));
	buf.iterator += sizeof(topic.buttons);
	buf.offset += sizeof(topic.buttons);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.aux1) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux1, sizeof(topic.aux1));
	buf.iterator += sizeof(topic.aux1);
	buf.offset += sizeof(topic.aux1);
	static_assert(sizeof(topic.aux2) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux2, sizeof(topic.aux2));
	buf.iterator += sizeof(topic.aux2);
	buf.offset += sizeof(topic.aux2);
	static_assert(sizeof(topic.aux3) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux3, sizeof(topic.aux3));
	buf.iterator += sizeof(topic.aux3);
	buf.offset += sizeof(topic.aux3);
	static_assert(sizeof(topic.aux4) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux4, sizeof(topic.aux4));
	buf.iterator += sizeof(topic.aux4);
	buf.offset += sizeof(topic.aux4);
	static_assert(sizeof(topic.aux5) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux5, sizeof(topic.aux5));
	buf.iterator += sizeof(topic.aux5);
	buf.offset += sizeof(topic.aux5);
	static_assert(sizeof(topic.aux6) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux6, sizeof(topic.aux6));
	buf.iterator += sizeof(topic.aux6);
	buf.offset += sizeof(topic.aux6);
	static_assert(sizeof(topic.sticks_moving) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.sticks_moving, sizeof(topic.sticks_moving));
	buf.iterator += sizeof(topic.sticks_moving);
	buf.offset += sizeof(topic.sticks_moving);
	return true;
}

bool ucdr_deserialize_manual_control_setpoint(ucdrBuffer& buf, manual_control_setpoint_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(&topic.valid, buf.iterator, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.data_source) == 1, "size mismatch");
	memcpy(&topic.data_source, buf.iterator, sizeof(topic.data_source));
	buf.iterator += sizeof(topic.data_source);
	buf.offset += sizeof(topic.data_source);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.roll) == 4, "size mismatch");
	memcpy(&topic.roll, buf.iterator, sizeof(topic.roll));
	buf.iterator += sizeof(topic.roll);
	buf.offset += sizeof(topic.roll);
	static_assert(sizeof(topic.pitch) == 4, "size mismatch");
	memcpy(&topic.pitch, buf.iterator, sizeof(topic.pitch));
	buf.iterator += sizeof(topic.pitch);
	buf.offset += sizeof(topic.pitch);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.throttle) == 4, "size mismatch");
	memcpy(&topic.throttle, buf.iterator, sizeof(topic.throttle));
	buf.iterator += sizeof(topic.throttle);
	buf.offset += sizeof(topic.throttle);
	static_assert(sizeof(topic.flaps) == 4, "size mismatch");
	memcpy(&topic.flaps, buf.iterator, sizeof(topic.flaps));
	buf.iterator += sizeof(topic.flaps);
	buf.offset += sizeof(topic.flaps);
	static_assert(sizeof(topic.buttons) == 2, "size mismatch");
	memcpy(&topic.buttons, buf.iterator, sizeof(topic.buttons));
	buf.iterator += sizeof(topic.buttons);
	buf.offset += sizeof(topic.buttons);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.aux1) == 4, "size mismatch");
	memcpy(&topic.aux1, buf.iterator, sizeof(topic.aux1));
	buf.iterator += sizeof(topic.aux1);
	buf.offset += sizeof(topic.aux1);
	static_assert(sizeof(topic.aux2) == 4, "size mismatch");
	memcpy(&topic.aux2, buf.iterator, sizeof(topic.aux2));
	buf.iterator += sizeof(topic.aux2);
	buf.offset += sizeof(topic.aux2);
	static_assert(sizeof(topic.aux3) == 4, "size mismatch");
	memcpy(&topic.aux3, buf.iterator, sizeof(topic.aux3));
	buf.iterator += sizeof(topic.aux3);
	buf.offset += sizeof(topic.aux3);
	static_assert(sizeof(topic.aux4) == 4, "size mismatch");
	memcpy(&topic.aux4, buf.iterator, sizeof(topic.aux4));
	buf.iterator += sizeof(topic.aux4);
	buf.offset += sizeof(topic.aux4);
	static_assert(sizeof(topic.aux5) == 4, "size mismatch");
	memcpy(&topic.aux5, buf.iterator, sizeof(topic.aux5));
	buf.iterator += sizeof(topic.aux5);
	buf.offset += sizeof(topic.aux5);
	static_assert(sizeof(topic.aux6) == 4, "size mismatch");
	memcpy(&topic.aux6, buf.iterator, sizeof(topic.aux6));
	buf.iterator += sizeof(topic.aux6);
	buf.offset += sizeof(topic.aux6);
	static_assert(sizeof(topic.sticks_moving) == 1, "size mismatch");
	memcpy(&topic.sticks_moving, buf.iterator, sizeof(topic.sticks_moving));
	buf.iterator += sizeof(topic.sticks_moving);
	buf.offset += sizeof(topic.sticks_moving);
	return true;
}
