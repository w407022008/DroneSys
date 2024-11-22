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
#include <uORB/topics/heater_status.h>


static inline constexpr int ucdr_topic_size_heater_status()
{
	return 45;
}

bool ucdr_serialize_heater_status(const heater_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.device_id, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.heater_on) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heater_on, sizeof(topic.heater_on));
	buf.iterator += sizeof(topic.heater_on);
	buf.offset += sizeof(topic.heater_on);
	static_assert(sizeof(topic.temperature_target_met) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_target_met, sizeof(topic.temperature_target_met));
	buf.iterator += sizeof(topic.temperature_target_met);
	buf.offset += sizeof(topic.temperature_target_met);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.temperature_sensor) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_sensor, sizeof(topic.temperature_sensor));
	buf.iterator += sizeof(topic.temperature_sensor);
	buf.offset += sizeof(topic.temperature_sensor);
	static_assert(sizeof(topic.temperature_target) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_target, sizeof(topic.temperature_target));
	buf.iterator += sizeof(topic.temperature_target);
	buf.offset += sizeof(topic.temperature_target);
	static_assert(sizeof(topic.controller_period_usec) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.controller_period_usec, sizeof(topic.controller_period_usec));
	buf.iterator += sizeof(topic.controller_period_usec);
	buf.offset += sizeof(topic.controller_period_usec);
	static_assert(sizeof(topic.controller_time_on_usec) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.controller_time_on_usec, sizeof(topic.controller_time_on_usec));
	buf.iterator += sizeof(topic.controller_time_on_usec);
	buf.offset += sizeof(topic.controller_time_on_usec);
	static_assert(sizeof(topic.proportional_value) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.proportional_value, sizeof(topic.proportional_value));
	buf.iterator += sizeof(topic.proportional_value);
	buf.offset += sizeof(topic.proportional_value);
	static_assert(sizeof(topic.integrator_value) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.integrator_value, sizeof(topic.integrator_value));
	buf.iterator += sizeof(topic.integrator_value);
	buf.offset += sizeof(topic.integrator_value);
	static_assert(sizeof(topic.feed_forward_value) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.feed_forward_value, sizeof(topic.feed_forward_value));
	buf.iterator += sizeof(topic.feed_forward_value);
	buf.offset += sizeof(topic.feed_forward_value);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mode, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	return true;
}

bool ucdr_deserialize_heater_status(ucdrBuffer& buf, heater_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(&topic.device_id, buf.iterator, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.heater_on) == 1, "size mismatch");
	memcpy(&topic.heater_on, buf.iterator, sizeof(topic.heater_on));
	buf.iterator += sizeof(topic.heater_on);
	buf.offset += sizeof(topic.heater_on);
	static_assert(sizeof(topic.temperature_target_met) == 1, "size mismatch");
	memcpy(&topic.temperature_target_met, buf.iterator, sizeof(topic.temperature_target_met));
	buf.iterator += sizeof(topic.temperature_target_met);
	buf.offset += sizeof(topic.temperature_target_met);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.temperature_sensor) == 4, "size mismatch");
	memcpy(&topic.temperature_sensor, buf.iterator, sizeof(topic.temperature_sensor));
	buf.iterator += sizeof(topic.temperature_sensor);
	buf.offset += sizeof(topic.temperature_sensor);
	static_assert(sizeof(topic.temperature_target) == 4, "size mismatch");
	memcpy(&topic.temperature_target, buf.iterator, sizeof(topic.temperature_target));
	buf.iterator += sizeof(topic.temperature_target);
	buf.offset += sizeof(topic.temperature_target);
	static_assert(sizeof(topic.controller_period_usec) == 4, "size mismatch");
	memcpy(&topic.controller_period_usec, buf.iterator, sizeof(topic.controller_period_usec));
	buf.iterator += sizeof(topic.controller_period_usec);
	buf.offset += sizeof(topic.controller_period_usec);
	static_assert(sizeof(topic.controller_time_on_usec) == 4, "size mismatch");
	memcpy(&topic.controller_time_on_usec, buf.iterator, sizeof(topic.controller_time_on_usec));
	buf.iterator += sizeof(topic.controller_time_on_usec);
	buf.offset += sizeof(topic.controller_time_on_usec);
	static_assert(sizeof(topic.proportional_value) == 4, "size mismatch");
	memcpy(&topic.proportional_value, buf.iterator, sizeof(topic.proportional_value));
	buf.iterator += sizeof(topic.proportional_value);
	buf.offset += sizeof(topic.proportional_value);
	static_assert(sizeof(topic.integrator_value) == 4, "size mismatch");
	memcpy(&topic.integrator_value, buf.iterator, sizeof(topic.integrator_value));
	buf.iterator += sizeof(topic.integrator_value);
	buf.offset += sizeof(topic.integrator_value);
	static_assert(sizeof(topic.feed_forward_value) == 4, "size mismatch");
	memcpy(&topic.feed_forward_value, buf.iterator, sizeof(topic.feed_forward_value));
	buf.iterator += sizeof(topic.feed_forward_value);
	buf.offset += sizeof(topic.feed_forward_value);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(&topic.mode, buf.iterator, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	return true;
}
