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
#include <uORB/topics/internal_combustion_engine_status.h>


static inline constexpr int ucdr_topic_size_internal_combustion_engine_status()
{
	return 88;
}

bool ucdr_serialize_internal_combustion_engine_status(const internal_combustion_engine_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.state, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.flags) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.flags, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.engine_load_percent) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.engine_load_percent, sizeof(topic.engine_load_percent));
	buf.iterator += sizeof(topic.engine_load_percent);
	buf.offset += sizeof(topic.engine_load_percent);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.engine_speed_rpm) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.engine_speed_rpm, sizeof(topic.engine_speed_rpm));
	buf.iterator += sizeof(topic.engine_speed_rpm);
	buf.offset += sizeof(topic.engine_speed_rpm);
	static_assert(sizeof(topic.spark_dwell_time_ms) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.spark_dwell_time_ms, sizeof(topic.spark_dwell_time_ms));
	buf.iterator += sizeof(topic.spark_dwell_time_ms);
	buf.offset += sizeof(topic.spark_dwell_time_ms);
	static_assert(sizeof(topic.atmospheric_pressure_kpa) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.atmospheric_pressure_kpa, sizeof(topic.atmospheric_pressure_kpa));
	buf.iterator += sizeof(topic.atmospheric_pressure_kpa);
	buf.offset += sizeof(topic.atmospheric_pressure_kpa);
	static_assert(sizeof(topic.intake_manifold_pressure_kpa) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.intake_manifold_pressure_kpa, sizeof(topic.intake_manifold_pressure_kpa));
	buf.iterator += sizeof(topic.intake_manifold_pressure_kpa);
	buf.offset += sizeof(topic.intake_manifold_pressure_kpa);
	static_assert(sizeof(topic.intake_manifold_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.intake_manifold_temperature, sizeof(topic.intake_manifold_temperature));
	buf.iterator += sizeof(topic.intake_manifold_temperature);
	buf.offset += sizeof(topic.intake_manifold_temperature);
	static_assert(sizeof(topic.coolant_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.coolant_temperature, sizeof(topic.coolant_temperature));
	buf.iterator += sizeof(topic.coolant_temperature);
	buf.offset += sizeof(topic.coolant_temperature);
	static_assert(sizeof(topic.oil_pressure) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.oil_pressure, sizeof(topic.oil_pressure));
	buf.iterator += sizeof(topic.oil_pressure);
	buf.offset += sizeof(topic.oil_pressure);
	static_assert(sizeof(topic.oil_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.oil_temperature, sizeof(topic.oil_temperature));
	buf.iterator += sizeof(topic.oil_temperature);
	buf.offset += sizeof(topic.oil_temperature);
	static_assert(sizeof(topic.fuel_pressure) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.fuel_pressure, sizeof(topic.fuel_pressure));
	buf.iterator += sizeof(topic.fuel_pressure);
	buf.offset += sizeof(topic.fuel_pressure);
	static_assert(sizeof(topic.fuel_consumption_rate_cm3pm) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.fuel_consumption_rate_cm3pm, sizeof(topic.fuel_consumption_rate_cm3pm));
	buf.iterator += sizeof(topic.fuel_consumption_rate_cm3pm);
	buf.offset += sizeof(topic.fuel_consumption_rate_cm3pm);
	static_assert(sizeof(topic.estimated_consumed_fuel_volume_cm3) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.estimated_consumed_fuel_volume_cm3, sizeof(topic.estimated_consumed_fuel_volume_cm3));
	buf.iterator += sizeof(topic.estimated_consumed_fuel_volume_cm3);
	buf.offset += sizeof(topic.estimated_consumed_fuel_volume_cm3);
	static_assert(sizeof(topic.throttle_position_percent) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.throttle_position_percent, sizeof(topic.throttle_position_percent));
	buf.iterator += sizeof(topic.throttle_position_percent);
	buf.offset += sizeof(topic.throttle_position_percent);
	static_assert(sizeof(topic.ecu_index) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.ecu_index, sizeof(topic.ecu_index));
	buf.iterator += sizeof(topic.ecu_index);
	buf.offset += sizeof(topic.ecu_index);
	static_assert(sizeof(topic.spark_plug_usage) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.spark_plug_usage, sizeof(topic.spark_plug_usage));
	buf.iterator += sizeof(topic.spark_plug_usage);
	buf.offset += sizeof(topic.spark_plug_usage);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.ignition_timing_deg) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ignition_timing_deg, sizeof(topic.ignition_timing_deg));
	buf.iterator += sizeof(topic.ignition_timing_deg);
	buf.offset += sizeof(topic.ignition_timing_deg);
	static_assert(sizeof(topic.injection_time_ms) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.injection_time_ms, sizeof(topic.injection_time_ms));
	buf.iterator += sizeof(topic.injection_time_ms);
	buf.offset += sizeof(topic.injection_time_ms);
	static_assert(sizeof(topic.cylinder_head_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cylinder_head_temperature, sizeof(topic.cylinder_head_temperature));
	buf.iterator += sizeof(topic.cylinder_head_temperature);
	buf.offset += sizeof(topic.cylinder_head_temperature);
	static_assert(sizeof(topic.exhaust_gas_temperature) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.exhaust_gas_temperature, sizeof(topic.exhaust_gas_temperature));
	buf.iterator += sizeof(topic.exhaust_gas_temperature);
	buf.offset += sizeof(topic.exhaust_gas_temperature);
	static_assert(sizeof(topic.lambda_coefficient) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.lambda_coefficient, sizeof(topic.lambda_coefficient));
	buf.iterator += sizeof(topic.lambda_coefficient);
	buf.offset += sizeof(topic.lambda_coefficient);
	return true;
}

bool ucdr_deserialize_internal_combustion_engine_status(ucdrBuffer& buf, internal_combustion_engine_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(&topic.state, buf.iterator, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.flags) == 4, "size mismatch");
	memcpy(&topic.flags, buf.iterator, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.engine_load_percent) == 1, "size mismatch");
	memcpy(&topic.engine_load_percent, buf.iterator, sizeof(topic.engine_load_percent));
	buf.iterator += sizeof(topic.engine_load_percent);
	buf.offset += sizeof(topic.engine_load_percent);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.engine_speed_rpm) == 4, "size mismatch");
	memcpy(&topic.engine_speed_rpm, buf.iterator, sizeof(topic.engine_speed_rpm));
	buf.iterator += sizeof(topic.engine_speed_rpm);
	buf.offset += sizeof(topic.engine_speed_rpm);
	static_assert(sizeof(topic.spark_dwell_time_ms) == 4, "size mismatch");
	memcpy(&topic.spark_dwell_time_ms, buf.iterator, sizeof(topic.spark_dwell_time_ms));
	buf.iterator += sizeof(topic.spark_dwell_time_ms);
	buf.offset += sizeof(topic.spark_dwell_time_ms);
	static_assert(sizeof(topic.atmospheric_pressure_kpa) == 4, "size mismatch");
	memcpy(&topic.atmospheric_pressure_kpa, buf.iterator, sizeof(topic.atmospheric_pressure_kpa));
	buf.iterator += sizeof(topic.atmospheric_pressure_kpa);
	buf.offset += sizeof(topic.atmospheric_pressure_kpa);
	static_assert(sizeof(topic.intake_manifold_pressure_kpa) == 4, "size mismatch");
	memcpy(&topic.intake_manifold_pressure_kpa, buf.iterator, sizeof(topic.intake_manifold_pressure_kpa));
	buf.iterator += sizeof(topic.intake_manifold_pressure_kpa);
	buf.offset += sizeof(topic.intake_manifold_pressure_kpa);
	static_assert(sizeof(topic.intake_manifold_temperature) == 4, "size mismatch");
	memcpy(&topic.intake_manifold_temperature, buf.iterator, sizeof(topic.intake_manifold_temperature));
	buf.iterator += sizeof(topic.intake_manifold_temperature);
	buf.offset += sizeof(topic.intake_manifold_temperature);
	static_assert(sizeof(topic.coolant_temperature) == 4, "size mismatch");
	memcpy(&topic.coolant_temperature, buf.iterator, sizeof(topic.coolant_temperature));
	buf.iterator += sizeof(topic.coolant_temperature);
	buf.offset += sizeof(topic.coolant_temperature);
	static_assert(sizeof(topic.oil_pressure) == 4, "size mismatch");
	memcpy(&topic.oil_pressure, buf.iterator, sizeof(topic.oil_pressure));
	buf.iterator += sizeof(topic.oil_pressure);
	buf.offset += sizeof(topic.oil_pressure);
	static_assert(sizeof(topic.oil_temperature) == 4, "size mismatch");
	memcpy(&topic.oil_temperature, buf.iterator, sizeof(topic.oil_temperature));
	buf.iterator += sizeof(topic.oil_temperature);
	buf.offset += sizeof(topic.oil_temperature);
	static_assert(sizeof(topic.fuel_pressure) == 4, "size mismatch");
	memcpy(&topic.fuel_pressure, buf.iterator, sizeof(topic.fuel_pressure));
	buf.iterator += sizeof(topic.fuel_pressure);
	buf.offset += sizeof(topic.fuel_pressure);
	static_assert(sizeof(topic.fuel_consumption_rate_cm3pm) == 4, "size mismatch");
	memcpy(&topic.fuel_consumption_rate_cm3pm, buf.iterator, sizeof(topic.fuel_consumption_rate_cm3pm));
	buf.iterator += sizeof(topic.fuel_consumption_rate_cm3pm);
	buf.offset += sizeof(topic.fuel_consumption_rate_cm3pm);
	static_assert(sizeof(topic.estimated_consumed_fuel_volume_cm3) == 4, "size mismatch");
	memcpy(&topic.estimated_consumed_fuel_volume_cm3, buf.iterator, sizeof(topic.estimated_consumed_fuel_volume_cm3));
	buf.iterator += sizeof(topic.estimated_consumed_fuel_volume_cm3);
	buf.offset += sizeof(topic.estimated_consumed_fuel_volume_cm3);
	static_assert(sizeof(topic.throttle_position_percent) == 1, "size mismatch");
	memcpy(&topic.throttle_position_percent, buf.iterator, sizeof(topic.throttle_position_percent));
	buf.iterator += sizeof(topic.throttle_position_percent);
	buf.offset += sizeof(topic.throttle_position_percent);
	static_assert(sizeof(topic.ecu_index) == 1, "size mismatch");
	memcpy(&topic.ecu_index, buf.iterator, sizeof(topic.ecu_index));
	buf.iterator += sizeof(topic.ecu_index);
	buf.offset += sizeof(topic.ecu_index);
	static_assert(sizeof(topic.spark_plug_usage) == 1, "size mismatch");
	memcpy(&topic.spark_plug_usage, buf.iterator, sizeof(topic.spark_plug_usage));
	buf.iterator += sizeof(topic.spark_plug_usage);
	buf.offset += sizeof(topic.spark_plug_usage);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.ignition_timing_deg) == 4, "size mismatch");
	memcpy(&topic.ignition_timing_deg, buf.iterator, sizeof(topic.ignition_timing_deg));
	buf.iterator += sizeof(topic.ignition_timing_deg);
	buf.offset += sizeof(topic.ignition_timing_deg);
	static_assert(sizeof(topic.injection_time_ms) == 4, "size mismatch");
	memcpy(&topic.injection_time_ms, buf.iterator, sizeof(topic.injection_time_ms));
	buf.iterator += sizeof(topic.injection_time_ms);
	buf.offset += sizeof(topic.injection_time_ms);
	static_assert(sizeof(topic.cylinder_head_temperature) == 4, "size mismatch");
	memcpy(&topic.cylinder_head_temperature, buf.iterator, sizeof(topic.cylinder_head_temperature));
	buf.iterator += sizeof(topic.cylinder_head_temperature);
	buf.offset += sizeof(topic.cylinder_head_temperature);
	static_assert(sizeof(topic.exhaust_gas_temperature) == 4, "size mismatch");
	memcpy(&topic.exhaust_gas_temperature, buf.iterator, sizeof(topic.exhaust_gas_temperature));
	buf.iterator += sizeof(topic.exhaust_gas_temperature);
	buf.offset += sizeof(topic.exhaust_gas_temperature);
	static_assert(sizeof(topic.lambda_coefficient) == 4, "size mismatch");
	memcpy(&topic.lambda_coefficient, buf.iterator, sizeof(topic.lambda_coefficient));
	buf.iterator += sizeof(topic.lambda_coefficient);
	buf.offset += sizeof(topic.lambda_coefficient);
	return true;
}
