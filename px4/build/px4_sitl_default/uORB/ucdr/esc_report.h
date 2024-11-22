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
#include <uORB/topics/esc_report.h>


static inline constexpr int ucdr_topic_size_esc_report()
{
	return 35;
}

bool ucdr_serialize_esc_report(const esc_report_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.esc_errorcount) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.esc_errorcount, sizeof(topic.esc_errorcount));
	buf.iterator += sizeof(topic.esc_errorcount);
	buf.offset += sizeof(topic.esc_errorcount);
	static_assert(sizeof(topic.esc_rpm) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.esc_rpm, sizeof(topic.esc_rpm));
	buf.iterator += sizeof(topic.esc_rpm);
	buf.offset += sizeof(topic.esc_rpm);
	static_assert(sizeof(topic.esc_voltage) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.esc_voltage, sizeof(topic.esc_voltage));
	buf.iterator += sizeof(topic.esc_voltage);
	buf.offset += sizeof(topic.esc_voltage);
	static_assert(sizeof(topic.esc_current) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.esc_current, sizeof(topic.esc_current));
	buf.iterator += sizeof(topic.esc_current);
	buf.offset += sizeof(topic.esc_current);
	static_assert(sizeof(topic.esc_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.esc_temperature, sizeof(topic.esc_temperature));
	buf.iterator += sizeof(topic.esc_temperature);
	buf.offset += sizeof(topic.esc_temperature);
	static_assert(sizeof(topic.esc_address) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.esc_address, sizeof(topic.esc_address));
	buf.iterator += sizeof(topic.esc_address);
	buf.offset += sizeof(topic.esc_address);
	static_assert(sizeof(topic.esc_cmdcount) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.esc_cmdcount, sizeof(topic.esc_cmdcount));
	buf.iterator += sizeof(topic.esc_cmdcount);
	buf.offset += sizeof(topic.esc_cmdcount);
	static_assert(sizeof(topic.esc_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.esc_state, sizeof(topic.esc_state));
	buf.iterator += sizeof(topic.esc_state);
	buf.offset += sizeof(topic.esc_state);
	static_assert(sizeof(topic.actuator_function) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.actuator_function, sizeof(topic.actuator_function));
	buf.iterator += sizeof(topic.actuator_function);
	buf.offset += sizeof(topic.actuator_function);
	static_assert(sizeof(topic.failures) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.failures, sizeof(topic.failures));
	buf.iterator += sizeof(topic.failures);
	buf.offset += sizeof(topic.failures);
	static_assert(sizeof(topic.esc_power) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.esc_power, sizeof(topic.esc_power));
	buf.iterator += sizeof(topic.esc_power);
	buf.offset += sizeof(topic.esc_power);
	return true;
}

bool ucdr_deserialize_esc_report(ucdrBuffer& buf, esc_report_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.esc_errorcount) == 4, "size mismatch");
	memcpy(&topic.esc_errorcount, buf.iterator, sizeof(topic.esc_errorcount));
	buf.iterator += sizeof(topic.esc_errorcount);
	buf.offset += sizeof(topic.esc_errorcount);
	static_assert(sizeof(topic.esc_rpm) == 4, "size mismatch");
	memcpy(&topic.esc_rpm, buf.iterator, sizeof(topic.esc_rpm));
	buf.iterator += sizeof(topic.esc_rpm);
	buf.offset += sizeof(topic.esc_rpm);
	static_assert(sizeof(topic.esc_voltage) == 4, "size mismatch");
	memcpy(&topic.esc_voltage, buf.iterator, sizeof(topic.esc_voltage));
	buf.iterator += sizeof(topic.esc_voltage);
	buf.offset += sizeof(topic.esc_voltage);
	static_assert(sizeof(topic.esc_current) == 4, "size mismatch");
	memcpy(&topic.esc_current, buf.iterator, sizeof(topic.esc_current));
	buf.iterator += sizeof(topic.esc_current);
	buf.offset += sizeof(topic.esc_current);
	static_assert(sizeof(topic.esc_temperature) == 4, "size mismatch");
	memcpy(&topic.esc_temperature, buf.iterator, sizeof(topic.esc_temperature));
	buf.iterator += sizeof(topic.esc_temperature);
	buf.offset += sizeof(topic.esc_temperature);
	static_assert(sizeof(topic.esc_address) == 1, "size mismatch");
	memcpy(&topic.esc_address, buf.iterator, sizeof(topic.esc_address));
	buf.iterator += sizeof(topic.esc_address);
	buf.offset += sizeof(topic.esc_address);
	static_assert(sizeof(topic.esc_cmdcount) == 1, "size mismatch");
	memcpy(&topic.esc_cmdcount, buf.iterator, sizeof(topic.esc_cmdcount));
	buf.iterator += sizeof(topic.esc_cmdcount);
	buf.offset += sizeof(topic.esc_cmdcount);
	static_assert(sizeof(topic.esc_state) == 1, "size mismatch");
	memcpy(&topic.esc_state, buf.iterator, sizeof(topic.esc_state));
	buf.iterator += sizeof(topic.esc_state);
	buf.offset += sizeof(topic.esc_state);
	static_assert(sizeof(topic.actuator_function) == 1, "size mismatch");
	memcpy(&topic.actuator_function, buf.iterator, sizeof(topic.actuator_function));
	buf.iterator += sizeof(topic.actuator_function);
	buf.offset += sizeof(topic.actuator_function);
	static_assert(sizeof(topic.failures) == 2, "size mismatch");
	memcpy(&topic.failures, buf.iterator, sizeof(topic.failures));
	buf.iterator += sizeof(topic.failures);
	buf.offset += sizeof(topic.failures);
	static_assert(sizeof(topic.esc_power) == 1, "size mismatch");
	memcpy(&topic.esc_power, buf.iterator, sizeof(topic.esc_power));
	buf.iterator += sizeof(topic.esc_power);
	buf.offset += sizeof(topic.esc_power);
	return true;
}
