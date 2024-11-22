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
#include <uORB/topics/vehicle_air_data.h>


static inline constexpr int ucdr_topic_size_vehicle_air_data()
{
	return 37;
}

bool ucdr_serialize_vehicle_air_data(const vehicle_air_data_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_device_id, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.baro_alt_meter) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_alt_meter, sizeof(topic.baro_alt_meter));
	buf.iterator += sizeof(topic.baro_alt_meter);
	buf.offset += sizeof(topic.baro_alt_meter);
	static_assert(sizeof(topic.baro_temp_celcius) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_temp_celcius, sizeof(topic.baro_temp_celcius));
	buf.iterator += sizeof(topic.baro_temp_celcius);
	buf.offset += sizeof(topic.baro_temp_celcius);
	static_assert(sizeof(topic.baro_pressure_pa) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_pressure_pa, sizeof(topic.baro_pressure_pa));
	buf.iterator += sizeof(topic.baro_pressure_pa);
	buf.offset += sizeof(topic.baro_pressure_pa);
	static_assert(sizeof(topic.rho) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rho, sizeof(topic.rho));
	buf.iterator += sizeof(topic.rho);
	buf.offset += sizeof(topic.rho);
	static_assert(sizeof(topic.calibration_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.calibration_count, sizeof(topic.calibration_count));
	buf.iterator += sizeof(topic.calibration_count);
	buf.offset += sizeof(topic.calibration_count);
	return true;
}

bool ucdr_deserialize_vehicle_air_data(ucdrBuffer& buf, vehicle_air_data_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(&topic.baro_device_id, buf.iterator, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.baro_alt_meter) == 4, "size mismatch");
	memcpy(&topic.baro_alt_meter, buf.iterator, sizeof(topic.baro_alt_meter));
	buf.iterator += sizeof(topic.baro_alt_meter);
	buf.offset += sizeof(topic.baro_alt_meter);
	static_assert(sizeof(topic.baro_temp_celcius) == 4, "size mismatch");
	memcpy(&topic.baro_temp_celcius, buf.iterator, sizeof(topic.baro_temp_celcius));
	buf.iterator += sizeof(topic.baro_temp_celcius);
	buf.offset += sizeof(topic.baro_temp_celcius);
	static_assert(sizeof(topic.baro_pressure_pa) == 4, "size mismatch");
	memcpy(&topic.baro_pressure_pa, buf.iterator, sizeof(topic.baro_pressure_pa));
	buf.iterator += sizeof(topic.baro_pressure_pa);
	buf.offset += sizeof(topic.baro_pressure_pa);
	static_assert(sizeof(topic.rho) == 4, "size mismatch");
	memcpy(&topic.rho, buf.iterator, sizeof(topic.rho));
	buf.iterator += sizeof(topic.rho);
	buf.offset += sizeof(topic.rho);
	static_assert(sizeof(topic.calibration_count) == 1, "size mismatch");
	memcpy(&topic.calibration_count, buf.iterator, sizeof(topic.calibration_count));
	buf.iterator += sizeof(topic.calibration_count);
	buf.offset += sizeof(topic.calibration_count);
	return true;
}
