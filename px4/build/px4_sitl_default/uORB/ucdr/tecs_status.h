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
#include <uORB/topics/tecs_status.h>


static inline constexpr int ucdr_topic_size_tecs_status()
{
	return 93;
}

bool ucdr_serialize_tecs_status(const tecs_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.altitude_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.altitude_sp, sizeof(topic.altitude_sp));
	buf.iterator += sizeof(topic.altitude_sp);
	buf.offset += sizeof(topic.altitude_sp);
	static_assert(sizeof(topic.altitude_reference) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.altitude_reference, sizeof(topic.altitude_reference));
	buf.iterator += sizeof(topic.altitude_reference);
	buf.offset += sizeof(topic.altitude_reference);
	static_assert(sizeof(topic.height_rate_reference) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.height_rate_reference, sizeof(topic.height_rate_reference));
	buf.iterator += sizeof(topic.height_rate_reference);
	buf.offset += sizeof(topic.height_rate_reference);
	static_assert(sizeof(topic.height_rate_direct) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.height_rate_direct, sizeof(topic.height_rate_direct));
	buf.iterator += sizeof(topic.height_rate_direct);
	buf.offset += sizeof(topic.height_rate_direct);
	static_assert(sizeof(topic.height_rate_setpoint) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.height_rate_setpoint, sizeof(topic.height_rate_setpoint));
	buf.iterator += sizeof(topic.height_rate_setpoint);
	buf.offset += sizeof(topic.height_rate_setpoint);
	static_assert(sizeof(topic.height_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.height_rate, sizeof(topic.height_rate));
	buf.iterator += sizeof(topic.height_rate);
	buf.offset += sizeof(topic.height_rate);
	static_assert(sizeof(topic.equivalent_airspeed_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.equivalent_airspeed_sp, sizeof(topic.equivalent_airspeed_sp));
	buf.iterator += sizeof(topic.equivalent_airspeed_sp);
	buf.offset += sizeof(topic.equivalent_airspeed_sp);
	static_assert(sizeof(topic.true_airspeed_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_sp, sizeof(topic.true_airspeed_sp));
	buf.iterator += sizeof(topic.true_airspeed_sp);
	buf.offset += sizeof(topic.true_airspeed_sp);
	static_assert(sizeof(topic.true_airspeed_filtered) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_filtered, sizeof(topic.true_airspeed_filtered));
	buf.iterator += sizeof(topic.true_airspeed_filtered);
	buf.offset += sizeof(topic.true_airspeed_filtered);
	static_assert(sizeof(topic.true_airspeed_derivative_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_derivative_sp, sizeof(topic.true_airspeed_derivative_sp));
	buf.iterator += sizeof(topic.true_airspeed_derivative_sp);
	buf.offset += sizeof(topic.true_airspeed_derivative_sp);
	static_assert(sizeof(topic.true_airspeed_derivative) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_derivative, sizeof(topic.true_airspeed_derivative));
	buf.iterator += sizeof(topic.true_airspeed_derivative);
	buf.offset += sizeof(topic.true_airspeed_derivative);
	static_assert(sizeof(topic.true_airspeed_derivative_raw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.true_airspeed_derivative_raw, sizeof(topic.true_airspeed_derivative_raw));
	buf.iterator += sizeof(topic.true_airspeed_derivative_raw);
	buf.offset += sizeof(topic.true_airspeed_derivative_raw);
	static_assert(sizeof(topic.total_energy_rate_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.total_energy_rate_sp, sizeof(topic.total_energy_rate_sp));
	buf.iterator += sizeof(topic.total_energy_rate_sp);
	buf.offset += sizeof(topic.total_energy_rate_sp);
	static_assert(sizeof(topic.total_energy_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.total_energy_rate, sizeof(topic.total_energy_rate));
	buf.iterator += sizeof(topic.total_energy_rate);
	buf.offset += sizeof(topic.total_energy_rate);
	static_assert(sizeof(topic.total_energy_balance_rate_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.total_energy_balance_rate_sp, sizeof(topic.total_energy_balance_rate_sp));
	buf.iterator += sizeof(topic.total_energy_balance_rate_sp);
	buf.offset += sizeof(topic.total_energy_balance_rate_sp);
	static_assert(sizeof(topic.total_energy_balance_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.total_energy_balance_rate, sizeof(topic.total_energy_balance_rate));
	buf.iterator += sizeof(topic.total_energy_balance_rate);
	buf.offset += sizeof(topic.total_energy_balance_rate);
	static_assert(sizeof(topic.throttle_integ) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.throttle_integ, sizeof(topic.throttle_integ));
	buf.iterator += sizeof(topic.throttle_integ);
	buf.offset += sizeof(topic.throttle_integ);
	static_assert(sizeof(topic.pitch_integ) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_integ, sizeof(topic.pitch_integ));
	buf.iterator += sizeof(topic.pitch_integ);
	buf.offset += sizeof(topic.pitch_integ);
	static_assert(sizeof(topic.throttle_sp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.throttle_sp, sizeof(topic.throttle_sp));
	buf.iterator += sizeof(topic.throttle_sp);
	buf.offset += sizeof(topic.throttle_sp);
	static_assert(sizeof(topic.pitch_sp_rad) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pitch_sp_rad, sizeof(topic.pitch_sp_rad));
	buf.iterator += sizeof(topic.pitch_sp_rad);
	buf.offset += sizeof(topic.pitch_sp_rad);
	static_assert(sizeof(topic.throttle_trim) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.throttle_trim, sizeof(topic.throttle_trim));
	buf.iterator += sizeof(topic.throttle_trim);
	buf.offset += sizeof(topic.throttle_trim);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mode, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	return true;
}

bool ucdr_deserialize_tecs_status(ucdrBuffer& buf, tecs_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.altitude_sp) == 4, "size mismatch");
	memcpy(&topic.altitude_sp, buf.iterator, sizeof(topic.altitude_sp));
	buf.iterator += sizeof(topic.altitude_sp);
	buf.offset += sizeof(topic.altitude_sp);
	static_assert(sizeof(topic.altitude_reference) == 4, "size mismatch");
	memcpy(&topic.altitude_reference, buf.iterator, sizeof(topic.altitude_reference));
	buf.iterator += sizeof(topic.altitude_reference);
	buf.offset += sizeof(topic.altitude_reference);
	static_assert(sizeof(topic.height_rate_reference) == 4, "size mismatch");
	memcpy(&topic.height_rate_reference, buf.iterator, sizeof(topic.height_rate_reference));
	buf.iterator += sizeof(topic.height_rate_reference);
	buf.offset += sizeof(topic.height_rate_reference);
	static_assert(sizeof(topic.height_rate_direct) == 4, "size mismatch");
	memcpy(&topic.height_rate_direct, buf.iterator, sizeof(topic.height_rate_direct));
	buf.iterator += sizeof(topic.height_rate_direct);
	buf.offset += sizeof(topic.height_rate_direct);
	static_assert(sizeof(topic.height_rate_setpoint) == 4, "size mismatch");
	memcpy(&topic.height_rate_setpoint, buf.iterator, sizeof(topic.height_rate_setpoint));
	buf.iterator += sizeof(topic.height_rate_setpoint);
	buf.offset += sizeof(topic.height_rate_setpoint);
	static_assert(sizeof(topic.height_rate) == 4, "size mismatch");
	memcpy(&topic.height_rate, buf.iterator, sizeof(topic.height_rate));
	buf.iterator += sizeof(topic.height_rate);
	buf.offset += sizeof(topic.height_rate);
	static_assert(sizeof(topic.equivalent_airspeed_sp) == 4, "size mismatch");
	memcpy(&topic.equivalent_airspeed_sp, buf.iterator, sizeof(topic.equivalent_airspeed_sp));
	buf.iterator += sizeof(topic.equivalent_airspeed_sp);
	buf.offset += sizeof(topic.equivalent_airspeed_sp);
	static_assert(sizeof(topic.true_airspeed_sp) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_sp, buf.iterator, sizeof(topic.true_airspeed_sp));
	buf.iterator += sizeof(topic.true_airspeed_sp);
	buf.offset += sizeof(topic.true_airspeed_sp);
	static_assert(sizeof(topic.true_airspeed_filtered) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_filtered, buf.iterator, sizeof(topic.true_airspeed_filtered));
	buf.iterator += sizeof(topic.true_airspeed_filtered);
	buf.offset += sizeof(topic.true_airspeed_filtered);
	static_assert(sizeof(topic.true_airspeed_derivative_sp) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_derivative_sp, buf.iterator, sizeof(topic.true_airspeed_derivative_sp));
	buf.iterator += sizeof(topic.true_airspeed_derivative_sp);
	buf.offset += sizeof(topic.true_airspeed_derivative_sp);
	static_assert(sizeof(topic.true_airspeed_derivative) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_derivative, buf.iterator, sizeof(topic.true_airspeed_derivative));
	buf.iterator += sizeof(topic.true_airspeed_derivative);
	buf.offset += sizeof(topic.true_airspeed_derivative);
	static_assert(sizeof(topic.true_airspeed_derivative_raw) == 4, "size mismatch");
	memcpy(&topic.true_airspeed_derivative_raw, buf.iterator, sizeof(topic.true_airspeed_derivative_raw));
	buf.iterator += sizeof(topic.true_airspeed_derivative_raw);
	buf.offset += sizeof(topic.true_airspeed_derivative_raw);
	static_assert(sizeof(topic.total_energy_rate_sp) == 4, "size mismatch");
	memcpy(&topic.total_energy_rate_sp, buf.iterator, sizeof(topic.total_energy_rate_sp));
	buf.iterator += sizeof(topic.total_energy_rate_sp);
	buf.offset += sizeof(topic.total_energy_rate_sp);
	static_assert(sizeof(topic.total_energy_rate) == 4, "size mismatch");
	memcpy(&topic.total_energy_rate, buf.iterator, sizeof(topic.total_energy_rate));
	buf.iterator += sizeof(topic.total_energy_rate);
	buf.offset += sizeof(topic.total_energy_rate);
	static_assert(sizeof(topic.total_energy_balance_rate_sp) == 4, "size mismatch");
	memcpy(&topic.total_energy_balance_rate_sp, buf.iterator, sizeof(topic.total_energy_balance_rate_sp));
	buf.iterator += sizeof(topic.total_energy_balance_rate_sp);
	buf.offset += sizeof(topic.total_energy_balance_rate_sp);
	static_assert(sizeof(topic.total_energy_balance_rate) == 4, "size mismatch");
	memcpy(&topic.total_energy_balance_rate, buf.iterator, sizeof(topic.total_energy_balance_rate));
	buf.iterator += sizeof(topic.total_energy_balance_rate);
	buf.offset += sizeof(topic.total_energy_balance_rate);
	static_assert(sizeof(topic.throttle_integ) == 4, "size mismatch");
	memcpy(&topic.throttle_integ, buf.iterator, sizeof(topic.throttle_integ));
	buf.iterator += sizeof(topic.throttle_integ);
	buf.offset += sizeof(topic.throttle_integ);
	static_assert(sizeof(topic.pitch_integ) == 4, "size mismatch");
	memcpy(&topic.pitch_integ, buf.iterator, sizeof(topic.pitch_integ));
	buf.iterator += sizeof(topic.pitch_integ);
	buf.offset += sizeof(topic.pitch_integ);
	static_assert(sizeof(topic.throttle_sp) == 4, "size mismatch");
	memcpy(&topic.throttle_sp, buf.iterator, sizeof(topic.throttle_sp));
	buf.iterator += sizeof(topic.throttle_sp);
	buf.offset += sizeof(topic.throttle_sp);
	static_assert(sizeof(topic.pitch_sp_rad) == 4, "size mismatch");
	memcpy(&topic.pitch_sp_rad, buf.iterator, sizeof(topic.pitch_sp_rad));
	buf.iterator += sizeof(topic.pitch_sp_rad);
	buf.offset += sizeof(topic.pitch_sp_rad);
	static_assert(sizeof(topic.throttle_trim) == 4, "size mismatch");
	memcpy(&topic.throttle_trim, buf.iterator, sizeof(topic.throttle_trim));
	buf.iterator += sizeof(topic.throttle_trim);
	buf.offset += sizeof(topic.throttle_trim);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(&topic.mode, buf.iterator, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	return true;
}
