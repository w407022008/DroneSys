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
#include <uORB/topics/airspeed_wind.h>


static inline constexpr int ucdr_topic_size_airspeed_wind()
{
	return 61;
}

bool ucdr_serialize_airspeed_wind(const airspeed_wind_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.windspeed_north) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.windspeed_north, sizeof(topic.windspeed_north));
	buf.iterator += sizeof(topic.windspeed_north);
	buf.offset += sizeof(topic.windspeed_north);
	static_assert(sizeof(topic.windspeed_east) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.windspeed_east, sizeof(topic.windspeed_east));
	buf.iterator += sizeof(topic.windspeed_east);
	buf.offset += sizeof(topic.windspeed_east);
	static_assert(sizeof(topic.variance_north) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.variance_north, sizeof(topic.variance_north));
	buf.iterator += sizeof(topic.variance_north);
	buf.offset += sizeof(topic.variance_north);
	static_assert(sizeof(topic.variance_east) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.variance_east, sizeof(topic.variance_east));
	buf.iterator += sizeof(topic.variance_east);
	buf.offset += sizeof(topic.variance_east);
	static_assert(sizeof(topic.tas_innov) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_innov, sizeof(topic.tas_innov));
	buf.iterator += sizeof(topic.tas_innov);
	buf.offset += sizeof(topic.tas_innov);
	static_assert(sizeof(topic.tas_innov_var) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_innov_var, sizeof(topic.tas_innov_var));
	buf.iterator += sizeof(topic.tas_innov_var);
	buf.offset += sizeof(topic.tas_innov_var);
	static_assert(sizeof(topic.tas_scale_raw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_scale_raw, sizeof(topic.tas_scale_raw));
	buf.iterator += sizeof(topic.tas_scale_raw);
	buf.offset += sizeof(topic.tas_scale_raw);
	static_assert(sizeof(topic.tas_scale_raw_var) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_scale_raw_var, sizeof(topic.tas_scale_raw_var));
	buf.iterator += sizeof(topic.tas_scale_raw_var);
	buf.offset += sizeof(topic.tas_scale_raw_var);
	static_assert(sizeof(topic.tas_scale_validated) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_scale_validated, sizeof(topic.tas_scale_validated));
	buf.iterator += sizeof(topic.tas_scale_validated);
	buf.offset += sizeof(topic.tas_scale_validated);
	static_assert(sizeof(topic.beta_innov) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.beta_innov, sizeof(topic.beta_innov));
	buf.iterator += sizeof(topic.beta_innov);
	buf.offset += sizeof(topic.beta_innov);
	static_assert(sizeof(topic.beta_innov_var) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.beta_innov_var, sizeof(topic.beta_innov_var));
	buf.iterator += sizeof(topic.beta_innov_var);
	buf.offset += sizeof(topic.beta_innov_var);
	static_assert(sizeof(topic.source) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.source, sizeof(topic.source));
	buf.iterator += sizeof(topic.source);
	buf.offset += sizeof(topic.source);
	return true;
}

bool ucdr_deserialize_airspeed_wind(ucdrBuffer& buf, airspeed_wind_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.windspeed_north) == 4, "size mismatch");
	memcpy(&topic.windspeed_north, buf.iterator, sizeof(topic.windspeed_north));
	buf.iterator += sizeof(topic.windspeed_north);
	buf.offset += sizeof(topic.windspeed_north);
	static_assert(sizeof(topic.windspeed_east) == 4, "size mismatch");
	memcpy(&topic.windspeed_east, buf.iterator, sizeof(topic.windspeed_east));
	buf.iterator += sizeof(topic.windspeed_east);
	buf.offset += sizeof(topic.windspeed_east);
	static_assert(sizeof(topic.variance_north) == 4, "size mismatch");
	memcpy(&topic.variance_north, buf.iterator, sizeof(topic.variance_north));
	buf.iterator += sizeof(topic.variance_north);
	buf.offset += sizeof(topic.variance_north);
	static_assert(sizeof(topic.variance_east) == 4, "size mismatch");
	memcpy(&topic.variance_east, buf.iterator, sizeof(topic.variance_east));
	buf.iterator += sizeof(topic.variance_east);
	buf.offset += sizeof(topic.variance_east);
	static_assert(sizeof(topic.tas_innov) == 4, "size mismatch");
	memcpy(&topic.tas_innov, buf.iterator, sizeof(topic.tas_innov));
	buf.iterator += sizeof(topic.tas_innov);
	buf.offset += sizeof(topic.tas_innov);
	static_assert(sizeof(topic.tas_innov_var) == 4, "size mismatch");
	memcpy(&topic.tas_innov_var, buf.iterator, sizeof(topic.tas_innov_var));
	buf.iterator += sizeof(topic.tas_innov_var);
	buf.offset += sizeof(topic.tas_innov_var);
	static_assert(sizeof(topic.tas_scale_raw) == 4, "size mismatch");
	memcpy(&topic.tas_scale_raw, buf.iterator, sizeof(topic.tas_scale_raw));
	buf.iterator += sizeof(topic.tas_scale_raw);
	buf.offset += sizeof(topic.tas_scale_raw);
	static_assert(sizeof(topic.tas_scale_raw_var) == 4, "size mismatch");
	memcpy(&topic.tas_scale_raw_var, buf.iterator, sizeof(topic.tas_scale_raw_var));
	buf.iterator += sizeof(topic.tas_scale_raw_var);
	buf.offset += sizeof(topic.tas_scale_raw_var);
	static_assert(sizeof(topic.tas_scale_validated) == 4, "size mismatch");
	memcpy(&topic.tas_scale_validated, buf.iterator, sizeof(topic.tas_scale_validated));
	buf.iterator += sizeof(topic.tas_scale_validated);
	buf.offset += sizeof(topic.tas_scale_validated);
	static_assert(sizeof(topic.beta_innov) == 4, "size mismatch");
	memcpy(&topic.beta_innov, buf.iterator, sizeof(topic.beta_innov));
	buf.iterator += sizeof(topic.beta_innov);
	buf.offset += sizeof(topic.beta_innov);
	static_assert(sizeof(topic.beta_innov_var) == 4, "size mismatch");
	memcpy(&topic.beta_innov_var, buf.iterator, sizeof(topic.beta_innov_var));
	buf.iterator += sizeof(topic.beta_innov_var);
	buf.offset += sizeof(topic.beta_innov_var);
	static_assert(sizeof(topic.source) == 1, "size mismatch");
	memcpy(&topic.source, buf.iterator, sizeof(topic.source));
	buf.iterator += sizeof(topic.source);
	buf.offset += sizeof(topic.source);
	return true;
}
