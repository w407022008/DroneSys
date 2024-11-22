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
#include <uORB/topics/failure_detector_status.h>


static inline constexpr int ucdr_topic_size_failure_detector_status()
{
	return 22;
}

bool ucdr_serialize_failure_detector_status(const failure_detector_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.fd_roll) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_roll, sizeof(topic.fd_roll));
	buf.iterator += sizeof(topic.fd_roll);
	buf.offset += sizeof(topic.fd_roll);
	static_assert(sizeof(topic.fd_pitch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_pitch, sizeof(topic.fd_pitch));
	buf.iterator += sizeof(topic.fd_pitch);
	buf.offset += sizeof(topic.fd_pitch);
	static_assert(sizeof(topic.fd_alt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_alt, sizeof(topic.fd_alt));
	buf.iterator += sizeof(topic.fd_alt);
	buf.offset += sizeof(topic.fd_alt);
	static_assert(sizeof(topic.fd_ext) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_ext, sizeof(topic.fd_ext));
	buf.iterator += sizeof(topic.fd_ext);
	buf.offset += sizeof(topic.fd_ext);
	static_assert(sizeof(topic.fd_arm_escs) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_arm_escs, sizeof(topic.fd_arm_escs));
	buf.iterator += sizeof(topic.fd_arm_escs);
	buf.offset += sizeof(topic.fd_arm_escs);
	static_assert(sizeof(topic.fd_battery) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_battery, sizeof(topic.fd_battery));
	buf.iterator += sizeof(topic.fd_battery);
	buf.offset += sizeof(topic.fd_battery);
	static_assert(sizeof(topic.fd_imbalanced_prop) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_imbalanced_prop, sizeof(topic.fd_imbalanced_prop));
	buf.iterator += sizeof(topic.fd_imbalanced_prop);
	buf.offset += sizeof(topic.fd_imbalanced_prop);
	static_assert(sizeof(topic.fd_motor) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fd_motor, sizeof(topic.fd_motor));
	buf.iterator += sizeof(topic.fd_motor);
	buf.offset += sizeof(topic.fd_motor);
	static_assert(sizeof(topic.imbalanced_prop_metric) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.imbalanced_prop_metric, sizeof(topic.imbalanced_prop_metric));
	buf.iterator += sizeof(topic.imbalanced_prop_metric);
	buf.offset += sizeof(topic.imbalanced_prop_metric);
	static_assert(sizeof(topic.motor_failure_mask) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.motor_failure_mask, sizeof(topic.motor_failure_mask));
	buf.iterator += sizeof(topic.motor_failure_mask);
	buf.offset += sizeof(topic.motor_failure_mask);
	return true;
}

bool ucdr_deserialize_failure_detector_status(ucdrBuffer& buf, failure_detector_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.fd_roll) == 1, "size mismatch");
	memcpy(&topic.fd_roll, buf.iterator, sizeof(topic.fd_roll));
	buf.iterator += sizeof(topic.fd_roll);
	buf.offset += sizeof(topic.fd_roll);
	static_assert(sizeof(topic.fd_pitch) == 1, "size mismatch");
	memcpy(&topic.fd_pitch, buf.iterator, sizeof(topic.fd_pitch));
	buf.iterator += sizeof(topic.fd_pitch);
	buf.offset += sizeof(topic.fd_pitch);
	static_assert(sizeof(topic.fd_alt) == 1, "size mismatch");
	memcpy(&topic.fd_alt, buf.iterator, sizeof(topic.fd_alt));
	buf.iterator += sizeof(topic.fd_alt);
	buf.offset += sizeof(topic.fd_alt);
	static_assert(sizeof(topic.fd_ext) == 1, "size mismatch");
	memcpy(&topic.fd_ext, buf.iterator, sizeof(topic.fd_ext));
	buf.iterator += sizeof(topic.fd_ext);
	buf.offset += sizeof(topic.fd_ext);
	static_assert(sizeof(topic.fd_arm_escs) == 1, "size mismatch");
	memcpy(&topic.fd_arm_escs, buf.iterator, sizeof(topic.fd_arm_escs));
	buf.iterator += sizeof(topic.fd_arm_escs);
	buf.offset += sizeof(topic.fd_arm_escs);
	static_assert(sizeof(topic.fd_battery) == 1, "size mismatch");
	memcpy(&topic.fd_battery, buf.iterator, sizeof(topic.fd_battery));
	buf.iterator += sizeof(topic.fd_battery);
	buf.offset += sizeof(topic.fd_battery);
	static_assert(sizeof(topic.fd_imbalanced_prop) == 1, "size mismatch");
	memcpy(&topic.fd_imbalanced_prop, buf.iterator, sizeof(topic.fd_imbalanced_prop));
	buf.iterator += sizeof(topic.fd_imbalanced_prop);
	buf.offset += sizeof(topic.fd_imbalanced_prop);
	static_assert(sizeof(topic.fd_motor) == 1, "size mismatch");
	memcpy(&topic.fd_motor, buf.iterator, sizeof(topic.fd_motor));
	buf.iterator += sizeof(topic.fd_motor);
	buf.offset += sizeof(topic.fd_motor);
	static_assert(sizeof(topic.imbalanced_prop_metric) == 4, "size mismatch");
	memcpy(&topic.imbalanced_prop_metric, buf.iterator, sizeof(topic.imbalanced_prop_metric));
	buf.iterator += sizeof(topic.imbalanced_prop_metric);
	buf.offset += sizeof(topic.imbalanced_prop_metric);
	static_assert(sizeof(topic.motor_failure_mask) == 2, "size mismatch");
	memcpy(&topic.motor_failure_mask, buf.iterator, sizeof(topic.motor_failure_mask));
	buf.iterator += sizeof(topic.motor_failure_mask);
	buf.offset += sizeof(topic.motor_failure_mask);
	return true;
}
