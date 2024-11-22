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
#include <uORB/topics/control_allocator_status.h>


static inline constexpr int ucdr_topic_size_control_allocator_status()
{
	return 58;
}

bool ucdr_serialize_control_allocator_status(const control_allocator_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.torque_setpoint_achieved) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.torque_setpoint_achieved, sizeof(topic.torque_setpoint_achieved));
	buf.iterator += sizeof(topic.torque_setpoint_achieved);
	buf.offset += sizeof(topic.torque_setpoint_achieved);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.unallocated_torque) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.unallocated_torque, sizeof(topic.unallocated_torque));
	buf.iterator += sizeof(topic.unallocated_torque);
	buf.offset += sizeof(topic.unallocated_torque);
	static_assert(sizeof(topic.thrust_setpoint_achieved) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.thrust_setpoint_achieved, sizeof(topic.thrust_setpoint_achieved));
	buf.iterator += sizeof(topic.thrust_setpoint_achieved);
	buf.offset += sizeof(topic.thrust_setpoint_achieved);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.unallocated_thrust) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.unallocated_thrust, sizeof(topic.unallocated_thrust));
	buf.iterator += sizeof(topic.unallocated_thrust);
	buf.offset += sizeof(topic.unallocated_thrust);
	static_assert(sizeof(topic.actuator_saturation) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.actuator_saturation, sizeof(topic.actuator_saturation));
	buf.iterator += sizeof(topic.actuator_saturation);
	buf.offset += sizeof(topic.actuator_saturation);
	static_assert(sizeof(topic.handled_motor_failure_mask) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.handled_motor_failure_mask, sizeof(topic.handled_motor_failure_mask));
	buf.iterator += sizeof(topic.handled_motor_failure_mask);
	buf.offset += sizeof(topic.handled_motor_failure_mask);
	return true;
}

bool ucdr_deserialize_control_allocator_status(ucdrBuffer& buf, control_allocator_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.torque_setpoint_achieved) == 1, "size mismatch");
	memcpy(&topic.torque_setpoint_achieved, buf.iterator, sizeof(topic.torque_setpoint_achieved));
	buf.iterator += sizeof(topic.torque_setpoint_achieved);
	buf.offset += sizeof(topic.torque_setpoint_achieved);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.unallocated_torque) == 12, "size mismatch");
	memcpy(&topic.unallocated_torque, buf.iterator, sizeof(topic.unallocated_torque));
	buf.iterator += sizeof(topic.unallocated_torque);
	buf.offset += sizeof(topic.unallocated_torque);
	static_assert(sizeof(topic.thrust_setpoint_achieved) == 1, "size mismatch");
	memcpy(&topic.thrust_setpoint_achieved, buf.iterator, sizeof(topic.thrust_setpoint_achieved));
	buf.iterator += sizeof(topic.thrust_setpoint_achieved);
	buf.offset += sizeof(topic.thrust_setpoint_achieved);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.unallocated_thrust) == 12, "size mismatch");
	memcpy(&topic.unallocated_thrust, buf.iterator, sizeof(topic.unallocated_thrust));
	buf.iterator += sizeof(topic.unallocated_thrust);
	buf.offset += sizeof(topic.unallocated_thrust);
	static_assert(sizeof(topic.actuator_saturation) == 16, "size mismatch");
	memcpy(&topic.actuator_saturation, buf.iterator, sizeof(topic.actuator_saturation));
	buf.iterator += sizeof(topic.actuator_saturation);
	buf.offset += sizeof(topic.actuator_saturation);
	static_assert(sizeof(topic.handled_motor_failure_mask) == 2, "size mismatch");
	memcpy(&topic.handled_motor_failure_mask, buf.iterator, sizeof(topic.handled_motor_failure_mask));
	buf.iterator += sizeof(topic.handled_motor_failure_mask);
	buf.offset += sizeof(topic.handled_motor_failure_mask);
	return true;
}
