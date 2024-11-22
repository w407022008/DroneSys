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
#include <uORB/topics/px4io_status.h>


static inline constexpr int ucdr_topic_size_px4io_status()
{
	return 144;
}

bool ucdr_serialize_px4io_status(const px4io_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.free_memory_bytes) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.free_memory_bytes, sizeof(topic.free_memory_bytes));
	buf.iterator += sizeof(topic.free_memory_bytes);
	buf.offset += sizeof(topic.free_memory_bytes);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.voltage_v) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.voltage_v, sizeof(topic.voltage_v));
	buf.iterator += sizeof(topic.voltage_v);
	buf.offset += sizeof(topic.voltage_v);
	static_assert(sizeof(topic.rssi_v) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rssi_v, sizeof(topic.rssi_v));
	buf.iterator += sizeof(topic.rssi_v);
	buf.offset += sizeof(topic.rssi_v);
	static_assert(sizeof(topic.status_arm_sync) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_arm_sync, sizeof(topic.status_arm_sync));
	buf.iterator += sizeof(topic.status_arm_sync);
	buf.offset += sizeof(topic.status_arm_sync);
	static_assert(sizeof(topic.status_failsafe) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_failsafe, sizeof(topic.status_failsafe));
	buf.iterator += sizeof(topic.status_failsafe);
	buf.offset += sizeof(topic.status_failsafe);
	static_assert(sizeof(topic.status_fmu_initialized) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_fmu_initialized, sizeof(topic.status_fmu_initialized));
	buf.iterator += sizeof(topic.status_fmu_initialized);
	buf.offset += sizeof(topic.status_fmu_initialized);
	static_assert(sizeof(topic.status_fmu_ok) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_fmu_ok, sizeof(topic.status_fmu_ok));
	buf.iterator += sizeof(topic.status_fmu_ok);
	buf.offset += sizeof(topic.status_fmu_ok);
	static_assert(sizeof(topic.status_init_ok) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_init_ok, sizeof(topic.status_init_ok));
	buf.iterator += sizeof(topic.status_init_ok);
	buf.offset += sizeof(topic.status_init_ok);
	static_assert(sizeof(topic.status_outputs_armed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_outputs_armed, sizeof(topic.status_outputs_armed));
	buf.iterator += sizeof(topic.status_outputs_armed);
	buf.offset += sizeof(topic.status_outputs_armed);
	static_assert(sizeof(topic.status_raw_pwm) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_raw_pwm, sizeof(topic.status_raw_pwm));
	buf.iterator += sizeof(topic.status_raw_pwm);
	buf.offset += sizeof(topic.status_raw_pwm);
	static_assert(sizeof(topic.status_rc_ok) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_ok, sizeof(topic.status_rc_ok));
	buf.iterator += sizeof(topic.status_rc_ok);
	buf.offset += sizeof(topic.status_rc_ok);
	static_assert(sizeof(topic.status_rc_dsm) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_dsm, sizeof(topic.status_rc_dsm));
	buf.iterator += sizeof(topic.status_rc_dsm);
	buf.offset += sizeof(topic.status_rc_dsm);
	static_assert(sizeof(topic.status_rc_ppm) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_ppm, sizeof(topic.status_rc_ppm));
	buf.iterator += sizeof(topic.status_rc_ppm);
	buf.offset += sizeof(topic.status_rc_ppm);
	static_assert(sizeof(topic.status_rc_sbus) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_sbus, sizeof(topic.status_rc_sbus));
	buf.iterator += sizeof(topic.status_rc_sbus);
	buf.offset += sizeof(topic.status_rc_sbus);
	static_assert(sizeof(topic.status_rc_st24) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_st24, sizeof(topic.status_rc_st24));
	buf.iterator += sizeof(topic.status_rc_st24);
	buf.offset += sizeof(topic.status_rc_st24);
	static_assert(sizeof(topic.status_rc_sumd) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_rc_sumd, sizeof(topic.status_rc_sumd));
	buf.iterator += sizeof(topic.status_rc_sumd);
	buf.offset += sizeof(topic.status_rc_sumd);
	static_assert(sizeof(topic.status_safety_button_event) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.status_safety_button_event, sizeof(topic.status_safety_button_event));
	buf.iterator += sizeof(topic.status_safety_button_event);
	buf.offset += sizeof(topic.status_safety_button_event);
	static_assert(sizeof(topic.alarm_pwm_error) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.alarm_pwm_error, sizeof(topic.alarm_pwm_error));
	buf.iterator += sizeof(topic.alarm_pwm_error);
	buf.offset += sizeof(topic.alarm_pwm_error);
	static_assert(sizeof(topic.alarm_rc_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.alarm_rc_lost, sizeof(topic.alarm_rc_lost));
	buf.iterator += sizeof(topic.alarm_rc_lost);
	buf.offset += sizeof(topic.alarm_rc_lost);
	static_assert(sizeof(topic.arming_failsafe_custom) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_failsafe_custom, sizeof(topic.arming_failsafe_custom));
	buf.iterator += sizeof(topic.arming_failsafe_custom);
	buf.offset += sizeof(topic.arming_failsafe_custom);
	static_assert(sizeof(topic.arming_fmu_armed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_fmu_armed, sizeof(topic.arming_fmu_armed));
	buf.iterator += sizeof(topic.arming_fmu_armed);
	buf.offset += sizeof(topic.arming_fmu_armed);
	static_assert(sizeof(topic.arming_fmu_prearmed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_fmu_prearmed, sizeof(topic.arming_fmu_prearmed));
	buf.iterator += sizeof(topic.arming_fmu_prearmed);
	buf.offset += sizeof(topic.arming_fmu_prearmed);
	static_assert(sizeof(topic.arming_force_failsafe) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_force_failsafe, sizeof(topic.arming_force_failsafe));
	buf.iterator += sizeof(topic.arming_force_failsafe);
	buf.offset += sizeof(topic.arming_force_failsafe);
	static_assert(sizeof(topic.arming_io_arm_ok) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_io_arm_ok, sizeof(topic.arming_io_arm_ok));
	buf.iterator += sizeof(topic.arming_io_arm_ok);
	buf.offset += sizeof(topic.arming_io_arm_ok);
	static_assert(sizeof(topic.arming_lockdown) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_lockdown, sizeof(topic.arming_lockdown));
	buf.iterator += sizeof(topic.arming_lockdown);
	buf.offset += sizeof(topic.arming_lockdown);
	static_assert(sizeof(topic.arming_termination_failsafe) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.arming_termination_failsafe, sizeof(topic.arming_termination_failsafe));
	buf.iterator += sizeof(topic.arming_termination_failsafe);
	buf.offset += sizeof(topic.arming_termination_failsafe);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.pwm) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.pwm, sizeof(topic.pwm));
	buf.iterator += sizeof(topic.pwm);
	buf.offset += sizeof(topic.pwm);
	static_assert(sizeof(topic.pwm_disarmed) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.pwm_disarmed, sizeof(topic.pwm_disarmed));
	buf.iterator += sizeof(topic.pwm_disarmed);
	buf.offset += sizeof(topic.pwm_disarmed);
	static_assert(sizeof(topic.pwm_failsafe) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.pwm_failsafe, sizeof(topic.pwm_failsafe));
	buf.iterator += sizeof(topic.pwm_failsafe);
	buf.offset += sizeof(topic.pwm_failsafe);
	static_assert(sizeof(topic.pwm_rate_hz) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.pwm_rate_hz, sizeof(topic.pwm_rate_hz));
	buf.iterator += sizeof(topic.pwm_rate_hz);
	buf.offset += sizeof(topic.pwm_rate_hz);
	static_assert(sizeof(topic.raw_inputs) == 36, "size mismatch");
	memcpy(buf.iterator, &topic.raw_inputs, sizeof(topic.raw_inputs));
	buf.iterator += sizeof(topic.raw_inputs);
	buf.offset += sizeof(topic.raw_inputs);
	return true;
}

bool ucdr_deserialize_px4io_status(ucdrBuffer& buf, px4io_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.free_memory_bytes) == 2, "size mismatch");
	memcpy(&topic.free_memory_bytes, buf.iterator, sizeof(topic.free_memory_bytes));
	buf.iterator += sizeof(topic.free_memory_bytes);
	buf.offset += sizeof(topic.free_memory_bytes);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.voltage_v) == 4, "size mismatch");
	memcpy(&topic.voltage_v, buf.iterator, sizeof(topic.voltage_v));
	buf.iterator += sizeof(topic.voltage_v);
	buf.offset += sizeof(topic.voltage_v);
	static_assert(sizeof(topic.rssi_v) == 4, "size mismatch");
	memcpy(&topic.rssi_v, buf.iterator, sizeof(topic.rssi_v));
	buf.iterator += sizeof(topic.rssi_v);
	buf.offset += sizeof(topic.rssi_v);
	static_assert(sizeof(topic.status_arm_sync) == 1, "size mismatch");
	memcpy(&topic.status_arm_sync, buf.iterator, sizeof(topic.status_arm_sync));
	buf.iterator += sizeof(topic.status_arm_sync);
	buf.offset += sizeof(topic.status_arm_sync);
	static_assert(sizeof(topic.status_failsafe) == 1, "size mismatch");
	memcpy(&topic.status_failsafe, buf.iterator, sizeof(topic.status_failsafe));
	buf.iterator += sizeof(topic.status_failsafe);
	buf.offset += sizeof(topic.status_failsafe);
	static_assert(sizeof(topic.status_fmu_initialized) == 1, "size mismatch");
	memcpy(&topic.status_fmu_initialized, buf.iterator, sizeof(topic.status_fmu_initialized));
	buf.iterator += sizeof(topic.status_fmu_initialized);
	buf.offset += sizeof(topic.status_fmu_initialized);
	static_assert(sizeof(topic.status_fmu_ok) == 1, "size mismatch");
	memcpy(&topic.status_fmu_ok, buf.iterator, sizeof(topic.status_fmu_ok));
	buf.iterator += sizeof(topic.status_fmu_ok);
	buf.offset += sizeof(topic.status_fmu_ok);
	static_assert(sizeof(topic.status_init_ok) == 1, "size mismatch");
	memcpy(&topic.status_init_ok, buf.iterator, sizeof(topic.status_init_ok));
	buf.iterator += sizeof(topic.status_init_ok);
	buf.offset += sizeof(topic.status_init_ok);
	static_assert(sizeof(topic.status_outputs_armed) == 1, "size mismatch");
	memcpy(&topic.status_outputs_armed, buf.iterator, sizeof(topic.status_outputs_armed));
	buf.iterator += sizeof(topic.status_outputs_armed);
	buf.offset += sizeof(topic.status_outputs_armed);
	static_assert(sizeof(topic.status_raw_pwm) == 1, "size mismatch");
	memcpy(&topic.status_raw_pwm, buf.iterator, sizeof(topic.status_raw_pwm));
	buf.iterator += sizeof(topic.status_raw_pwm);
	buf.offset += sizeof(topic.status_raw_pwm);
	static_assert(sizeof(topic.status_rc_ok) == 1, "size mismatch");
	memcpy(&topic.status_rc_ok, buf.iterator, sizeof(topic.status_rc_ok));
	buf.iterator += sizeof(topic.status_rc_ok);
	buf.offset += sizeof(topic.status_rc_ok);
	static_assert(sizeof(topic.status_rc_dsm) == 1, "size mismatch");
	memcpy(&topic.status_rc_dsm, buf.iterator, sizeof(topic.status_rc_dsm));
	buf.iterator += sizeof(topic.status_rc_dsm);
	buf.offset += sizeof(topic.status_rc_dsm);
	static_assert(sizeof(topic.status_rc_ppm) == 1, "size mismatch");
	memcpy(&topic.status_rc_ppm, buf.iterator, sizeof(topic.status_rc_ppm));
	buf.iterator += sizeof(topic.status_rc_ppm);
	buf.offset += sizeof(topic.status_rc_ppm);
	static_assert(sizeof(topic.status_rc_sbus) == 1, "size mismatch");
	memcpy(&topic.status_rc_sbus, buf.iterator, sizeof(topic.status_rc_sbus));
	buf.iterator += sizeof(topic.status_rc_sbus);
	buf.offset += sizeof(topic.status_rc_sbus);
	static_assert(sizeof(topic.status_rc_st24) == 1, "size mismatch");
	memcpy(&topic.status_rc_st24, buf.iterator, sizeof(topic.status_rc_st24));
	buf.iterator += sizeof(topic.status_rc_st24);
	buf.offset += sizeof(topic.status_rc_st24);
	static_assert(sizeof(topic.status_rc_sumd) == 1, "size mismatch");
	memcpy(&topic.status_rc_sumd, buf.iterator, sizeof(topic.status_rc_sumd));
	buf.iterator += sizeof(topic.status_rc_sumd);
	buf.offset += sizeof(topic.status_rc_sumd);
	static_assert(sizeof(topic.status_safety_button_event) == 1, "size mismatch");
	memcpy(&topic.status_safety_button_event, buf.iterator, sizeof(topic.status_safety_button_event));
	buf.iterator += sizeof(topic.status_safety_button_event);
	buf.offset += sizeof(topic.status_safety_button_event);
	static_assert(sizeof(topic.alarm_pwm_error) == 1, "size mismatch");
	memcpy(&topic.alarm_pwm_error, buf.iterator, sizeof(topic.alarm_pwm_error));
	buf.iterator += sizeof(topic.alarm_pwm_error);
	buf.offset += sizeof(topic.alarm_pwm_error);
	static_assert(sizeof(topic.alarm_rc_lost) == 1, "size mismatch");
	memcpy(&topic.alarm_rc_lost, buf.iterator, sizeof(topic.alarm_rc_lost));
	buf.iterator += sizeof(topic.alarm_rc_lost);
	buf.offset += sizeof(topic.alarm_rc_lost);
	static_assert(sizeof(topic.arming_failsafe_custom) == 1, "size mismatch");
	memcpy(&topic.arming_failsafe_custom, buf.iterator, sizeof(topic.arming_failsafe_custom));
	buf.iterator += sizeof(topic.arming_failsafe_custom);
	buf.offset += sizeof(topic.arming_failsafe_custom);
	static_assert(sizeof(topic.arming_fmu_armed) == 1, "size mismatch");
	memcpy(&topic.arming_fmu_armed, buf.iterator, sizeof(topic.arming_fmu_armed));
	buf.iterator += sizeof(topic.arming_fmu_armed);
	buf.offset += sizeof(topic.arming_fmu_armed);
	static_assert(sizeof(topic.arming_fmu_prearmed) == 1, "size mismatch");
	memcpy(&topic.arming_fmu_prearmed, buf.iterator, sizeof(topic.arming_fmu_prearmed));
	buf.iterator += sizeof(topic.arming_fmu_prearmed);
	buf.offset += sizeof(topic.arming_fmu_prearmed);
	static_assert(sizeof(topic.arming_force_failsafe) == 1, "size mismatch");
	memcpy(&topic.arming_force_failsafe, buf.iterator, sizeof(topic.arming_force_failsafe));
	buf.iterator += sizeof(topic.arming_force_failsafe);
	buf.offset += sizeof(topic.arming_force_failsafe);
	static_assert(sizeof(topic.arming_io_arm_ok) == 1, "size mismatch");
	memcpy(&topic.arming_io_arm_ok, buf.iterator, sizeof(topic.arming_io_arm_ok));
	buf.iterator += sizeof(topic.arming_io_arm_ok);
	buf.offset += sizeof(topic.arming_io_arm_ok);
	static_assert(sizeof(topic.arming_lockdown) == 1, "size mismatch");
	memcpy(&topic.arming_lockdown, buf.iterator, sizeof(topic.arming_lockdown));
	buf.iterator += sizeof(topic.arming_lockdown);
	buf.offset += sizeof(topic.arming_lockdown);
	static_assert(sizeof(topic.arming_termination_failsafe) == 1, "size mismatch");
	memcpy(&topic.arming_termination_failsafe, buf.iterator, sizeof(topic.arming_termination_failsafe));
	buf.iterator += sizeof(topic.arming_termination_failsafe);
	buf.offset += sizeof(topic.arming_termination_failsafe);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.pwm) == 16, "size mismatch");
	memcpy(&topic.pwm, buf.iterator, sizeof(topic.pwm));
	buf.iterator += sizeof(topic.pwm);
	buf.offset += sizeof(topic.pwm);
	static_assert(sizeof(topic.pwm_disarmed) == 16, "size mismatch");
	memcpy(&topic.pwm_disarmed, buf.iterator, sizeof(topic.pwm_disarmed));
	buf.iterator += sizeof(topic.pwm_disarmed);
	buf.offset += sizeof(topic.pwm_disarmed);
	static_assert(sizeof(topic.pwm_failsafe) == 16, "size mismatch");
	memcpy(&topic.pwm_failsafe, buf.iterator, sizeof(topic.pwm_failsafe));
	buf.iterator += sizeof(topic.pwm_failsafe);
	buf.offset += sizeof(topic.pwm_failsafe);
	static_assert(sizeof(topic.pwm_rate_hz) == 16, "size mismatch");
	memcpy(&topic.pwm_rate_hz, buf.iterator, sizeof(topic.pwm_rate_hz));
	buf.iterator += sizeof(topic.pwm_rate_hz);
	buf.offset += sizeof(topic.pwm_rate_hz);
	static_assert(sizeof(topic.raw_inputs) == 36, "size mismatch");
	memcpy(&topic.raw_inputs, buf.iterator, sizeof(topic.raw_inputs));
	buf.iterator += sizeof(topic.raw_inputs);
	buf.offset += sizeof(topic.raw_inputs);
	return true;
}
