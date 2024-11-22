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
#include <uORB/topics/vehicle_global_position.h>


static inline constexpr int ucdr_topic_size_vehicle_global_position()
{
	return 62;
}

bool ucdr_serialize_vehicle_global_position(const vehicle_global_position_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lat, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lon, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.alt, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.alt_ellipsoid) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.alt_ellipsoid, sizeof(topic.alt_ellipsoid));
	buf.iterator += sizeof(topic.alt_ellipsoid);
	buf.offset += sizeof(topic.alt_ellipsoid);
	static_assert(sizeof(topic.delta_alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.delta_alt, sizeof(topic.delta_alt));
	buf.iterator += sizeof(topic.delta_alt);
	buf.offset += sizeof(topic.delta_alt);
	static_assert(sizeof(topic.lat_lon_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.lat_lon_reset_counter, sizeof(topic.lat_lon_reset_counter));
	buf.iterator += sizeof(topic.lat_lon_reset_counter);
	buf.offset += sizeof(topic.lat_lon_reset_counter);
	static_assert(sizeof(topic.alt_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.alt_reset_counter, sizeof(topic.alt_reset_counter));
	buf.iterator += sizeof(topic.alt_reset_counter);
	buf.offset += sizeof(topic.alt_reset_counter);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.eph) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.eph, sizeof(topic.eph));
	buf.iterator += sizeof(topic.eph);
	buf.offset += sizeof(topic.eph);
	static_assert(sizeof(topic.epv) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.epv, sizeof(topic.epv));
	buf.iterator += sizeof(topic.epv);
	buf.offset += sizeof(topic.epv);
	static_assert(sizeof(topic.terrain_alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.terrain_alt, sizeof(topic.terrain_alt));
	buf.iterator += sizeof(topic.terrain_alt);
	buf.offset += sizeof(topic.terrain_alt);
	static_assert(sizeof(topic.terrain_alt_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.terrain_alt_valid, sizeof(topic.terrain_alt_valid));
	buf.iterator += sizeof(topic.terrain_alt_valid);
	buf.offset += sizeof(topic.terrain_alt_valid);
	static_assert(sizeof(topic.dead_reckoning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.dead_reckoning, sizeof(topic.dead_reckoning));
	buf.iterator += sizeof(topic.dead_reckoning);
	buf.offset += sizeof(topic.dead_reckoning);
	return true;
}

bool ucdr_deserialize_vehicle_global_position(ucdrBuffer& buf, vehicle_global_position_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(&topic.lat, buf.iterator, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(&topic.lon, buf.iterator, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(&topic.alt, buf.iterator, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.alt_ellipsoid) == 4, "size mismatch");
	memcpy(&topic.alt_ellipsoid, buf.iterator, sizeof(topic.alt_ellipsoid));
	buf.iterator += sizeof(topic.alt_ellipsoid);
	buf.offset += sizeof(topic.alt_ellipsoid);
	static_assert(sizeof(topic.delta_alt) == 4, "size mismatch");
	memcpy(&topic.delta_alt, buf.iterator, sizeof(topic.delta_alt));
	buf.iterator += sizeof(topic.delta_alt);
	buf.offset += sizeof(topic.delta_alt);
	static_assert(sizeof(topic.lat_lon_reset_counter) == 1, "size mismatch");
	memcpy(&topic.lat_lon_reset_counter, buf.iterator, sizeof(topic.lat_lon_reset_counter));
	buf.iterator += sizeof(topic.lat_lon_reset_counter);
	buf.offset += sizeof(topic.lat_lon_reset_counter);
	static_assert(sizeof(topic.alt_reset_counter) == 1, "size mismatch");
	memcpy(&topic.alt_reset_counter, buf.iterator, sizeof(topic.alt_reset_counter));
	buf.iterator += sizeof(topic.alt_reset_counter);
	buf.offset += sizeof(topic.alt_reset_counter);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.eph) == 4, "size mismatch");
	memcpy(&topic.eph, buf.iterator, sizeof(topic.eph));
	buf.iterator += sizeof(topic.eph);
	buf.offset += sizeof(topic.eph);
	static_assert(sizeof(topic.epv) == 4, "size mismatch");
	memcpy(&topic.epv, buf.iterator, sizeof(topic.epv));
	buf.iterator += sizeof(topic.epv);
	buf.offset += sizeof(topic.epv);
	static_assert(sizeof(topic.terrain_alt) == 4, "size mismatch");
	memcpy(&topic.terrain_alt, buf.iterator, sizeof(topic.terrain_alt));
	buf.iterator += sizeof(topic.terrain_alt);
	buf.offset += sizeof(topic.terrain_alt);
	static_assert(sizeof(topic.terrain_alt_valid) == 1, "size mismatch");
	memcpy(&topic.terrain_alt_valid, buf.iterator, sizeof(topic.terrain_alt_valid));
	buf.iterator += sizeof(topic.terrain_alt_valid);
	buf.offset += sizeof(topic.terrain_alt_valid);
	static_assert(sizeof(topic.dead_reckoning) == 1, "size mismatch");
	memcpy(&topic.dead_reckoning, buf.iterator, sizeof(topic.dead_reckoning));
	buf.iterator += sizeof(topic.dead_reckoning);
	buf.offset += sizeof(topic.dead_reckoning);
	return true;
}
