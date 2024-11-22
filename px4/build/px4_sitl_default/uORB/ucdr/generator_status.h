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
#include <uORB/topics/generator_status.h>


static inline constexpr int ucdr_topic_size_generator_status()
{
	return 50;
}

bool ucdr_serialize_generator_status(const generator_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.status) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.status, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.battery_current) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.battery_current, sizeof(topic.battery_current));
	buf.iterator += sizeof(topic.battery_current);
	buf.offset += sizeof(topic.battery_current);
	static_assert(sizeof(topic.load_current) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.load_current, sizeof(topic.load_current));
	buf.iterator += sizeof(topic.load_current);
	buf.offset += sizeof(topic.load_current);
	static_assert(sizeof(topic.power_generated) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.power_generated, sizeof(topic.power_generated));
	buf.iterator += sizeof(topic.power_generated);
	buf.offset += sizeof(topic.power_generated);
	static_assert(sizeof(topic.bus_voltage) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.bus_voltage, sizeof(topic.bus_voltage));
	buf.iterator += sizeof(topic.bus_voltage);
	buf.offset += sizeof(topic.bus_voltage);
	static_assert(sizeof(topic.bat_current_setpoint) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.bat_current_setpoint, sizeof(topic.bat_current_setpoint));
	buf.iterator += sizeof(topic.bat_current_setpoint);
	buf.offset += sizeof(topic.bat_current_setpoint);
	static_assert(sizeof(topic.runtime) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.runtime, sizeof(topic.runtime));
	buf.iterator += sizeof(topic.runtime);
	buf.offset += sizeof(topic.runtime);
	static_assert(sizeof(topic.time_until_maintenance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.time_until_maintenance, sizeof(topic.time_until_maintenance));
	buf.iterator += sizeof(topic.time_until_maintenance);
	buf.offset += sizeof(topic.time_until_maintenance);
	static_assert(sizeof(topic.generator_speed) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.generator_speed, sizeof(topic.generator_speed));
	buf.iterator += sizeof(topic.generator_speed);
	buf.offset += sizeof(topic.generator_speed);
	static_assert(sizeof(topic.rectifier_temperature) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rectifier_temperature, sizeof(topic.rectifier_temperature));
	buf.iterator += sizeof(topic.rectifier_temperature);
	buf.offset += sizeof(topic.rectifier_temperature);
	static_assert(sizeof(topic.generator_temperature) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.generator_temperature, sizeof(topic.generator_temperature));
	buf.iterator += sizeof(topic.generator_temperature);
	buf.offset += sizeof(topic.generator_temperature);
	return true;
}

bool ucdr_deserialize_generator_status(ucdrBuffer& buf, generator_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.status) == 8, "size mismatch");
	memcpy(&topic.status, buf.iterator, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.battery_current) == 4, "size mismatch");
	memcpy(&topic.battery_current, buf.iterator, sizeof(topic.battery_current));
	buf.iterator += sizeof(topic.battery_current);
	buf.offset += sizeof(topic.battery_current);
	static_assert(sizeof(topic.load_current) == 4, "size mismatch");
	memcpy(&topic.load_current, buf.iterator, sizeof(topic.load_current));
	buf.iterator += sizeof(topic.load_current);
	buf.offset += sizeof(topic.load_current);
	static_assert(sizeof(topic.power_generated) == 4, "size mismatch");
	memcpy(&topic.power_generated, buf.iterator, sizeof(topic.power_generated));
	buf.iterator += sizeof(topic.power_generated);
	buf.offset += sizeof(topic.power_generated);
	static_assert(sizeof(topic.bus_voltage) == 4, "size mismatch");
	memcpy(&topic.bus_voltage, buf.iterator, sizeof(topic.bus_voltage));
	buf.iterator += sizeof(topic.bus_voltage);
	buf.offset += sizeof(topic.bus_voltage);
	static_assert(sizeof(topic.bat_current_setpoint) == 4, "size mismatch");
	memcpy(&topic.bat_current_setpoint, buf.iterator, sizeof(topic.bat_current_setpoint));
	buf.iterator += sizeof(topic.bat_current_setpoint);
	buf.offset += sizeof(topic.bat_current_setpoint);
	static_assert(sizeof(topic.runtime) == 4, "size mismatch");
	memcpy(&topic.runtime, buf.iterator, sizeof(topic.runtime));
	buf.iterator += sizeof(topic.runtime);
	buf.offset += sizeof(topic.runtime);
	static_assert(sizeof(topic.time_until_maintenance) == 4, "size mismatch");
	memcpy(&topic.time_until_maintenance, buf.iterator, sizeof(topic.time_until_maintenance));
	buf.iterator += sizeof(topic.time_until_maintenance);
	buf.offset += sizeof(topic.time_until_maintenance);
	static_assert(sizeof(topic.generator_speed) == 2, "size mismatch");
	memcpy(&topic.generator_speed, buf.iterator, sizeof(topic.generator_speed));
	buf.iterator += sizeof(topic.generator_speed);
	buf.offset += sizeof(topic.generator_speed);
	static_assert(sizeof(topic.rectifier_temperature) == 2, "size mismatch");
	memcpy(&topic.rectifier_temperature, buf.iterator, sizeof(topic.rectifier_temperature));
	buf.iterator += sizeof(topic.rectifier_temperature);
	buf.offset += sizeof(topic.rectifier_temperature);
	static_assert(sizeof(topic.generator_temperature) == 2, "size mismatch");
	memcpy(&topic.generator_temperature, buf.iterator, sizeof(topic.generator_temperature));
	buf.iterator += sizeof(topic.generator_temperature);
	buf.offset += sizeof(topic.generator_temperature);
	return true;
}
