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
#include <uORB/topics/manual_control_switches.h>


static inline constexpr int ucdr_topic_size_manual_control_switches()
{
	return 32;
}

bool ucdr_serialize_manual_control_switches(const manual_control_switches_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.mode_slot) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mode_slot, sizeof(topic.mode_slot));
	buf.iterator += sizeof(topic.mode_slot);
	buf.offset += sizeof(topic.mode_slot);
	static_assert(sizeof(topic.arm_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arm_switch, sizeof(topic.arm_switch));
	buf.iterator += sizeof(topic.arm_switch);
	buf.offset += sizeof(topic.arm_switch);
	static_assert(sizeof(topic.return_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.return_switch, sizeof(topic.return_switch));
	buf.iterator += sizeof(topic.return_switch);
	buf.offset += sizeof(topic.return_switch);
	static_assert(sizeof(topic.loiter_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.loiter_switch, sizeof(topic.loiter_switch));
	buf.iterator += sizeof(topic.loiter_switch);
	buf.offset += sizeof(topic.loiter_switch);
	static_assert(sizeof(topic.offboard_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.offboard_switch, sizeof(topic.offboard_switch));
	buf.iterator += sizeof(topic.offboard_switch);
	buf.offset += sizeof(topic.offboard_switch);
	static_assert(sizeof(topic.kill_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.kill_switch, sizeof(topic.kill_switch));
	buf.iterator += sizeof(topic.kill_switch);
	buf.offset += sizeof(topic.kill_switch);
	static_assert(sizeof(topic.gear_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gear_switch, sizeof(topic.gear_switch));
	buf.iterator += sizeof(topic.gear_switch);
	buf.offset += sizeof(topic.gear_switch);
	static_assert(sizeof(topic.transition_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.transition_switch, sizeof(topic.transition_switch));
	buf.iterator += sizeof(topic.transition_switch);
	buf.offset += sizeof(topic.transition_switch);
	static_assert(sizeof(topic.photo_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.photo_switch, sizeof(topic.photo_switch));
	buf.iterator += sizeof(topic.photo_switch);
	buf.offset += sizeof(topic.photo_switch);
	static_assert(sizeof(topic.video_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.video_switch, sizeof(topic.video_switch));
	buf.iterator += sizeof(topic.video_switch);
	buf.offset += sizeof(topic.video_switch);
	static_assert(sizeof(topic.engage_main_motor_switch) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.engage_main_motor_switch, sizeof(topic.engage_main_motor_switch));
	buf.iterator += sizeof(topic.engage_main_motor_switch);
	buf.offset += sizeof(topic.engage_main_motor_switch);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.switch_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.switch_changes, sizeof(topic.switch_changes));
	buf.iterator += sizeof(topic.switch_changes);
	buf.offset += sizeof(topic.switch_changes);
	return true;
}

bool ucdr_deserialize_manual_control_switches(ucdrBuffer& buf, manual_control_switches_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.mode_slot) == 1, "size mismatch");
	memcpy(&topic.mode_slot, buf.iterator, sizeof(topic.mode_slot));
	buf.iterator += sizeof(topic.mode_slot);
	buf.offset += sizeof(topic.mode_slot);
	static_assert(sizeof(topic.arm_switch) == 1, "size mismatch");
	memcpy(&topic.arm_switch, buf.iterator, sizeof(topic.arm_switch));
	buf.iterator += sizeof(topic.arm_switch);
	buf.offset += sizeof(topic.arm_switch);
	static_assert(sizeof(topic.return_switch) == 1, "size mismatch");
	memcpy(&topic.return_switch, buf.iterator, sizeof(topic.return_switch));
	buf.iterator += sizeof(topic.return_switch);
	buf.offset += sizeof(topic.return_switch);
	static_assert(sizeof(topic.loiter_switch) == 1, "size mismatch");
	memcpy(&topic.loiter_switch, buf.iterator, sizeof(topic.loiter_switch));
	buf.iterator += sizeof(topic.loiter_switch);
	buf.offset += sizeof(topic.loiter_switch);
	static_assert(sizeof(topic.offboard_switch) == 1, "size mismatch");
	memcpy(&topic.offboard_switch, buf.iterator, sizeof(topic.offboard_switch));
	buf.iterator += sizeof(topic.offboard_switch);
	buf.offset += sizeof(topic.offboard_switch);
	static_assert(sizeof(topic.kill_switch) == 1, "size mismatch");
	memcpy(&topic.kill_switch, buf.iterator, sizeof(topic.kill_switch));
	buf.iterator += sizeof(topic.kill_switch);
	buf.offset += sizeof(topic.kill_switch);
	static_assert(sizeof(topic.gear_switch) == 1, "size mismatch");
	memcpy(&topic.gear_switch, buf.iterator, sizeof(topic.gear_switch));
	buf.iterator += sizeof(topic.gear_switch);
	buf.offset += sizeof(topic.gear_switch);
	static_assert(sizeof(topic.transition_switch) == 1, "size mismatch");
	memcpy(&topic.transition_switch, buf.iterator, sizeof(topic.transition_switch));
	buf.iterator += sizeof(topic.transition_switch);
	buf.offset += sizeof(topic.transition_switch);
	static_assert(sizeof(topic.photo_switch) == 1, "size mismatch");
	memcpy(&topic.photo_switch, buf.iterator, sizeof(topic.photo_switch));
	buf.iterator += sizeof(topic.photo_switch);
	buf.offset += sizeof(topic.photo_switch);
	static_assert(sizeof(topic.video_switch) == 1, "size mismatch");
	memcpy(&topic.video_switch, buf.iterator, sizeof(topic.video_switch));
	buf.iterator += sizeof(topic.video_switch);
	buf.offset += sizeof(topic.video_switch);
	static_assert(sizeof(topic.engage_main_motor_switch) == 1, "size mismatch");
	memcpy(&topic.engage_main_motor_switch, buf.iterator, sizeof(topic.engage_main_motor_switch));
	buf.iterator += sizeof(topic.engage_main_motor_switch);
	buf.offset += sizeof(topic.engage_main_motor_switch);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.switch_changes) == 4, "size mismatch");
	memcpy(&topic.switch_changes, buf.iterator, sizeof(topic.switch_changes));
	buf.iterator += sizeof(topic.switch_changes);
	buf.offset += sizeof(topic.switch_changes);
	return true;
}
