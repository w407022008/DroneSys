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
#include <uORB/topics/npfg_status.h>


static inline constexpr int ucdr_topic_size_npfg_status()
{
	return 64;
}

bool ucdr_serialize_npfg_status(const npfg_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.wind_est_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.wind_est_valid, sizeof(topic.wind_est_valid));
	buf.iterator += sizeof(topic.wind_est_valid);
	buf.offset += sizeof(topic.wind_est_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.lat_accel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.lat_accel, sizeof(topic.lat_accel));
	buf.iterator += sizeof(topic.lat_accel);
	buf.offset += sizeof(topic.lat_accel);
	static_assert(sizeof(topic.lat_accel_ff) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.lat_accel_ff, sizeof(topic.lat_accel_ff));
	buf.iterator += sizeof(topic.lat_accel_ff);
	buf.offset += sizeof(topic.lat_accel_ff);
	static_assert(sizeof(topic.bearing_feas) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.bearing_feas, sizeof(topic.bearing_feas));
	buf.iterator += sizeof(topic.bearing_feas);
	buf.offset += sizeof(topic.bearing_feas);
	static_assert(sizeof(topic.bearing_feas_on_track) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.bearing_feas_on_track, sizeof(topic.bearing_feas_on_track));
	buf.iterator += sizeof(topic.bearing_feas_on_track);
	buf.offset += sizeof(topic.bearing_feas_on_track);
	static_assert(sizeof(topic.signed_track_error) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.signed_track_error, sizeof(topic.signed_track_error));
	buf.iterator += sizeof(topic.signed_track_error);
	buf.offset += sizeof(topic.signed_track_error);
	static_assert(sizeof(topic.track_error_bound) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.track_error_bound, sizeof(topic.track_error_bound));
	buf.iterator += sizeof(topic.track_error_bound);
	buf.offset += sizeof(topic.track_error_bound);
	static_assert(sizeof(topic.airspeed_ref) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.airspeed_ref, sizeof(topic.airspeed_ref));
	buf.iterator += sizeof(topic.airspeed_ref);
	buf.offset += sizeof(topic.airspeed_ref);
	static_assert(sizeof(topic.bearing) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.bearing, sizeof(topic.bearing));
	buf.iterator += sizeof(topic.bearing);
	buf.offset += sizeof(topic.bearing);
	static_assert(sizeof(topic.heading_ref) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading_ref, sizeof(topic.heading_ref));
	buf.iterator += sizeof(topic.heading_ref);
	buf.offset += sizeof(topic.heading_ref);
	static_assert(sizeof(topic.min_ground_speed_ref) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.min_ground_speed_ref, sizeof(topic.min_ground_speed_ref));
	buf.iterator += sizeof(topic.min_ground_speed_ref);
	buf.offset += sizeof(topic.min_ground_speed_ref);
	static_assert(sizeof(topic.adapted_period) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.adapted_period, sizeof(topic.adapted_period));
	buf.iterator += sizeof(topic.adapted_period);
	buf.offset += sizeof(topic.adapted_period);
	static_assert(sizeof(topic.p_gain) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.p_gain, sizeof(topic.p_gain));
	buf.iterator += sizeof(topic.p_gain);
	buf.offset += sizeof(topic.p_gain);
	static_assert(sizeof(topic.time_const) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.time_const, sizeof(topic.time_const));
	buf.iterator += sizeof(topic.time_const);
	buf.offset += sizeof(topic.time_const);
	return true;
}

bool ucdr_deserialize_npfg_status(ucdrBuffer& buf, npfg_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.wind_est_valid) == 1, "size mismatch");
	memcpy(&topic.wind_est_valid, buf.iterator, sizeof(topic.wind_est_valid));
	buf.iterator += sizeof(topic.wind_est_valid);
	buf.offset += sizeof(topic.wind_est_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.lat_accel) == 4, "size mismatch");
	memcpy(&topic.lat_accel, buf.iterator, sizeof(topic.lat_accel));
	buf.iterator += sizeof(topic.lat_accel);
	buf.offset += sizeof(topic.lat_accel);
	static_assert(sizeof(topic.lat_accel_ff) == 4, "size mismatch");
	memcpy(&topic.lat_accel_ff, buf.iterator, sizeof(topic.lat_accel_ff));
	buf.iterator += sizeof(topic.lat_accel_ff);
	buf.offset += sizeof(topic.lat_accel_ff);
	static_assert(sizeof(topic.bearing_feas) == 4, "size mismatch");
	memcpy(&topic.bearing_feas, buf.iterator, sizeof(topic.bearing_feas));
	buf.iterator += sizeof(topic.bearing_feas);
	buf.offset += sizeof(topic.bearing_feas);
	static_assert(sizeof(topic.bearing_feas_on_track) == 4, "size mismatch");
	memcpy(&topic.bearing_feas_on_track, buf.iterator, sizeof(topic.bearing_feas_on_track));
	buf.iterator += sizeof(topic.bearing_feas_on_track);
	buf.offset += sizeof(topic.bearing_feas_on_track);
	static_assert(sizeof(topic.signed_track_error) == 4, "size mismatch");
	memcpy(&topic.signed_track_error, buf.iterator, sizeof(topic.signed_track_error));
	buf.iterator += sizeof(topic.signed_track_error);
	buf.offset += sizeof(topic.signed_track_error);
	static_assert(sizeof(topic.track_error_bound) == 4, "size mismatch");
	memcpy(&topic.track_error_bound, buf.iterator, sizeof(topic.track_error_bound));
	buf.iterator += sizeof(topic.track_error_bound);
	buf.offset += sizeof(topic.track_error_bound);
	static_assert(sizeof(topic.airspeed_ref) == 4, "size mismatch");
	memcpy(&topic.airspeed_ref, buf.iterator, sizeof(topic.airspeed_ref));
	buf.iterator += sizeof(topic.airspeed_ref);
	buf.offset += sizeof(topic.airspeed_ref);
	static_assert(sizeof(topic.bearing) == 4, "size mismatch");
	memcpy(&topic.bearing, buf.iterator, sizeof(topic.bearing));
	buf.iterator += sizeof(topic.bearing);
	buf.offset += sizeof(topic.bearing);
	static_assert(sizeof(topic.heading_ref) == 4, "size mismatch");
	memcpy(&topic.heading_ref, buf.iterator, sizeof(topic.heading_ref));
	buf.iterator += sizeof(topic.heading_ref);
	buf.offset += sizeof(topic.heading_ref);
	static_assert(sizeof(topic.min_ground_speed_ref) == 4, "size mismatch");
	memcpy(&topic.min_ground_speed_ref, buf.iterator, sizeof(topic.min_ground_speed_ref));
	buf.iterator += sizeof(topic.min_ground_speed_ref);
	buf.offset += sizeof(topic.min_ground_speed_ref);
	static_assert(sizeof(topic.adapted_period) == 4, "size mismatch");
	memcpy(&topic.adapted_period, buf.iterator, sizeof(topic.adapted_period));
	buf.iterator += sizeof(topic.adapted_period);
	buf.offset += sizeof(topic.adapted_period);
	static_assert(sizeof(topic.p_gain) == 4, "size mismatch");
	memcpy(&topic.p_gain, buf.iterator, sizeof(topic.p_gain));
	buf.iterator += sizeof(topic.p_gain);
	buf.offset += sizeof(topic.p_gain);
	static_assert(sizeof(topic.time_const) == 4, "size mismatch");
	memcpy(&topic.time_const, buf.iterator, sizeof(topic.time_const));
	buf.iterator += sizeof(topic.time_const);
	buf.offset += sizeof(topic.time_const);
	return true;
}
