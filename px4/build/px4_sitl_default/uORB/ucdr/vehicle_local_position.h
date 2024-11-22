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
#include <uORB/topics/vehicle_local_position.h>


static inline constexpr int ucdr_topic_size_vehicle_local_position()
{
	return 184;
}

bool ucdr_serialize_vehicle_local_position(const vehicle_local_position_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.xy_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.xy_valid, sizeof(topic.xy_valid));
	buf.iterator += sizeof(topic.xy_valid);
	buf.offset += sizeof(topic.xy_valid);
	static_assert(sizeof(topic.z_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.z_valid, sizeof(topic.z_valid));
	buf.iterator += sizeof(topic.z_valid);
	buf.offset += sizeof(topic.z_valid);
	static_assert(sizeof(topic.v_xy_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.v_xy_valid, sizeof(topic.v_xy_valid));
	buf.iterator += sizeof(topic.v_xy_valid);
	buf.offset += sizeof(topic.v_xy_valid);
	static_assert(sizeof(topic.v_z_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.v_z_valid, sizeof(topic.v_z_valid));
	buf.iterator += sizeof(topic.v_z_valid);
	buf.offset += sizeof(topic.v_z_valid);
	static_assert(sizeof(topic.x) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.x, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.y, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.z, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	static_assert(sizeof(topic.delta_xy) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.delta_xy, sizeof(topic.delta_xy));
	buf.iterator += sizeof(topic.delta_xy);
	buf.offset += sizeof(topic.delta_xy);
	static_assert(sizeof(topic.xy_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.xy_reset_counter, sizeof(topic.xy_reset_counter));
	buf.iterator += sizeof(topic.xy_reset_counter);
	buf.offset += sizeof(topic.xy_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.delta_z) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.delta_z, sizeof(topic.delta_z));
	buf.iterator += sizeof(topic.delta_z);
	buf.offset += sizeof(topic.delta_z);
	static_assert(sizeof(topic.z_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.z_reset_counter, sizeof(topic.z_reset_counter));
	buf.iterator += sizeof(topic.z_reset_counter);
	buf.offset += sizeof(topic.z_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vx) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vx, sizeof(topic.vx));
	buf.iterator += sizeof(topic.vx);
	buf.offset += sizeof(topic.vx);
	static_assert(sizeof(topic.vy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vy, sizeof(topic.vy));
	buf.iterator += sizeof(topic.vy);
	buf.offset += sizeof(topic.vy);
	static_assert(sizeof(topic.vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vz, sizeof(topic.vz));
	buf.iterator += sizeof(topic.vz);
	buf.offset += sizeof(topic.vz);
	static_assert(sizeof(topic.z_deriv) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.z_deriv, sizeof(topic.z_deriv));
	buf.iterator += sizeof(topic.z_deriv);
	buf.offset += sizeof(topic.z_deriv);
	static_assert(sizeof(topic.delta_vxy) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.delta_vxy, sizeof(topic.delta_vxy));
	buf.iterator += sizeof(topic.delta_vxy);
	buf.offset += sizeof(topic.delta_vxy);
	static_assert(sizeof(topic.vxy_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vxy_reset_counter, sizeof(topic.vxy_reset_counter));
	buf.iterator += sizeof(topic.vxy_reset_counter);
	buf.offset += sizeof(topic.vxy_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.delta_vz) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.delta_vz, sizeof(topic.delta_vz));
	buf.iterator += sizeof(topic.delta_vz);
	buf.offset += sizeof(topic.delta_vz);
	static_assert(sizeof(topic.vz_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vz_reset_counter, sizeof(topic.vz_reset_counter));
	buf.iterator += sizeof(topic.vz_reset_counter);
	buf.offset += sizeof(topic.vz_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.ax) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ax, sizeof(topic.ax));
	buf.iterator += sizeof(topic.ax);
	buf.offset += sizeof(topic.ax);
	static_assert(sizeof(topic.ay) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ay, sizeof(topic.ay));
	buf.iterator += sizeof(topic.ay);
	buf.offset += sizeof(topic.ay);
	static_assert(sizeof(topic.az) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.az, sizeof(topic.az));
	buf.iterator += sizeof(topic.az);
	buf.offset += sizeof(topic.az);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.delta_heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.delta_heading, sizeof(topic.delta_heading));
	buf.iterator += sizeof(topic.delta_heading);
	buf.offset += sizeof(topic.delta_heading);
	static_assert(sizeof(topic.heading_reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heading_reset_counter, sizeof(topic.heading_reset_counter));
	buf.iterator += sizeof(topic.heading_reset_counter);
	buf.offset += sizeof(topic.heading_reset_counter);
	static_assert(sizeof(topic.heading_good_for_control) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heading_good_for_control, sizeof(topic.heading_good_for_control));
	buf.iterator += sizeof(topic.heading_good_for_control);
	buf.offset += sizeof(topic.heading_good_for_control);
	static_assert(sizeof(topic.xy_global) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.xy_global, sizeof(topic.xy_global));
	buf.iterator += sizeof(topic.xy_global);
	buf.offset += sizeof(topic.xy_global);
	static_assert(sizeof(topic.z_global) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.z_global, sizeof(topic.z_global));
	buf.iterator += sizeof(topic.z_global);
	buf.offset += sizeof(topic.z_global);
	static_assert(sizeof(topic.ref_timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ref_timestamp, sizeof(topic.ref_timestamp));
	buf.iterator += sizeof(topic.ref_timestamp);
	buf.offset += sizeof(topic.ref_timestamp);
	static_assert(sizeof(topic.ref_lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ref_lat, sizeof(topic.ref_lat));
	buf.iterator += sizeof(topic.ref_lat);
	buf.offset += sizeof(topic.ref_lat);
	static_assert(sizeof(topic.ref_lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ref_lon, sizeof(topic.ref_lon));
	buf.iterator += sizeof(topic.ref_lon);
	buf.offset += sizeof(topic.ref_lon);
	static_assert(sizeof(topic.ref_alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ref_alt, sizeof(topic.ref_alt));
	buf.iterator += sizeof(topic.ref_alt);
	buf.offset += sizeof(topic.ref_alt);
	static_assert(sizeof(topic.dist_bottom) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.dist_bottom, sizeof(topic.dist_bottom));
	buf.iterator += sizeof(topic.dist_bottom);
	buf.offset += sizeof(topic.dist_bottom);
	static_assert(sizeof(topic.dist_bottom_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.dist_bottom_valid, sizeof(topic.dist_bottom_valid));
	buf.iterator += sizeof(topic.dist_bottom_valid);
	buf.offset += sizeof(topic.dist_bottom_valid);
	static_assert(sizeof(topic.dist_bottom_sensor_bitfield) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.dist_bottom_sensor_bitfield, sizeof(topic.dist_bottom_sensor_bitfield));
	buf.iterator += sizeof(topic.dist_bottom_sensor_bitfield);
	buf.offset += sizeof(topic.dist_bottom_sensor_bitfield);
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
	static_assert(sizeof(topic.evh) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.evh, sizeof(topic.evh));
	buf.iterator += sizeof(topic.evh);
	buf.offset += sizeof(topic.evh);
	static_assert(sizeof(topic.evv) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.evv, sizeof(topic.evv));
	buf.iterator += sizeof(topic.evv);
	buf.offset += sizeof(topic.evv);
	static_assert(sizeof(topic.dead_reckoning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.dead_reckoning, sizeof(topic.dead_reckoning));
	buf.iterator += sizeof(topic.dead_reckoning);
	buf.offset += sizeof(topic.dead_reckoning);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vxy_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vxy_max, sizeof(topic.vxy_max));
	buf.iterator += sizeof(topic.vxy_max);
	buf.offset += sizeof(topic.vxy_max);
	static_assert(sizeof(topic.vz_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vz_max, sizeof(topic.vz_max));
	buf.iterator += sizeof(topic.vz_max);
	buf.offset += sizeof(topic.vz_max);
	static_assert(sizeof(topic.hagl_min) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hagl_min, sizeof(topic.hagl_min));
	buf.iterator += sizeof(topic.hagl_min);
	buf.offset += sizeof(topic.hagl_min);
	static_assert(sizeof(topic.hagl_max) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hagl_max, sizeof(topic.hagl_max));
	buf.iterator += sizeof(topic.hagl_max);
	buf.offset += sizeof(topic.hagl_max);
	return true;
}

bool ucdr_deserialize_vehicle_local_position(ucdrBuffer& buf, vehicle_local_position_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.xy_valid) == 1, "size mismatch");
	memcpy(&topic.xy_valid, buf.iterator, sizeof(topic.xy_valid));
	buf.iterator += sizeof(topic.xy_valid);
	buf.offset += sizeof(topic.xy_valid);
	static_assert(sizeof(topic.z_valid) == 1, "size mismatch");
	memcpy(&topic.z_valid, buf.iterator, sizeof(topic.z_valid));
	buf.iterator += sizeof(topic.z_valid);
	buf.offset += sizeof(topic.z_valid);
	static_assert(sizeof(topic.v_xy_valid) == 1, "size mismatch");
	memcpy(&topic.v_xy_valid, buf.iterator, sizeof(topic.v_xy_valid));
	buf.iterator += sizeof(topic.v_xy_valid);
	buf.offset += sizeof(topic.v_xy_valid);
	static_assert(sizeof(topic.v_z_valid) == 1, "size mismatch");
	memcpy(&topic.v_z_valid, buf.iterator, sizeof(topic.v_z_valid));
	buf.iterator += sizeof(topic.v_z_valid);
	buf.offset += sizeof(topic.v_z_valid);
	static_assert(sizeof(topic.x) == 4, "size mismatch");
	memcpy(&topic.x, buf.iterator, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 4, "size mismatch");
	memcpy(&topic.y, buf.iterator, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 4, "size mismatch");
	memcpy(&topic.z, buf.iterator, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	static_assert(sizeof(topic.delta_xy) == 8, "size mismatch");
	memcpy(&topic.delta_xy, buf.iterator, sizeof(topic.delta_xy));
	buf.iterator += sizeof(topic.delta_xy);
	buf.offset += sizeof(topic.delta_xy);
	static_assert(sizeof(topic.xy_reset_counter) == 1, "size mismatch");
	memcpy(&topic.xy_reset_counter, buf.iterator, sizeof(topic.xy_reset_counter));
	buf.iterator += sizeof(topic.xy_reset_counter);
	buf.offset += sizeof(topic.xy_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.delta_z) == 4, "size mismatch");
	memcpy(&topic.delta_z, buf.iterator, sizeof(topic.delta_z));
	buf.iterator += sizeof(topic.delta_z);
	buf.offset += sizeof(topic.delta_z);
	static_assert(sizeof(topic.z_reset_counter) == 1, "size mismatch");
	memcpy(&topic.z_reset_counter, buf.iterator, sizeof(topic.z_reset_counter));
	buf.iterator += sizeof(topic.z_reset_counter);
	buf.offset += sizeof(topic.z_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vx) == 4, "size mismatch");
	memcpy(&topic.vx, buf.iterator, sizeof(topic.vx));
	buf.iterator += sizeof(topic.vx);
	buf.offset += sizeof(topic.vx);
	static_assert(sizeof(topic.vy) == 4, "size mismatch");
	memcpy(&topic.vy, buf.iterator, sizeof(topic.vy));
	buf.iterator += sizeof(topic.vy);
	buf.offset += sizeof(topic.vy);
	static_assert(sizeof(topic.vz) == 4, "size mismatch");
	memcpy(&topic.vz, buf.iterator, sizeof(topic.vz));
	buf.iterator += sizeof(topic.vz);
	buf.offset += sizeof(topic.vz);
	static_assert(sizeof(topic.z_deriv) == 4, "size mismatch");
	memcpy(&topic.z_deriv, buf.iterator, sizeof(topic.z_deriv));
	buf.iterator += sizeof(topic.z_deriv);
	buf.offset += sizeof(topic.z_deriv);
	static_assert(sizeof(topic.delta_vxy) == 8, "size mismatch");
	memcpy(&topic.delta_vxy, buf.iterator, sizeof(topic.delta_vxy));
	buf.iterator += sizeof(topic.delta_vxy);
	buf.offset += sizeof(topic.delta_vxy);
	static_assert(sizeof(topic.vxy_reset_counter) == 1, "size mismatch");
	memcpy(&topic.vxy_reset_counter, buf.iterator, sizeof(topic.vxy_reset_counter));
	buf.iterator += sizeof(topic.vxy_reset_counter);
	buf.offset += sizeof(topic.vxy_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.delta_vz) == 4, "size mismatch");
	memcpy(&topic.delta_vz, buf.iterator, sizeof(topic.delta_vz));
	buf.iterator += sizeof(topic.delta_vz);
	buf.offset += sizeof(topic.delta_vz);
	static_assert(sizeof(topic.vz_reset_counter) == 1, "size mismatch");
	memcpy(&topic.vz_reset_counter, buf.iterator, sizeof(topic.vz_reset_counter));
	buf.iterator += sizeof(topic.vz_reset_counter);
	buf.offset += sizeof(topic.vz_reset_counter);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.ax) == 4, "size mismatch");
	memcpy(&topic.ax, buf.iterator, sizeof(topic.ax));
	buf.iterator += sizeof(topic.ax);
	buf.offset += sizeof(topic.ax);
	static_assert(sizeof(topic.ay) == 4, "size mismatch");
	memcpy(&topic.ay, buf.iterator, sizeof(topic.ay));
	buf.iterator += sizeof(topic.ay);
	buf.offset += sizeof(topic.ay);
	static_assert(sizeof(topic.az) == 4, "size mismatch");
	memcpy(&topic.az, buf.iterator, sizeof(topic.az));
	buf.iterator += sizeof(topic.az);
	buf.offset += sizeof(topic.az);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(&topic.heading, buf.iterator, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.delta_heading) == 4, "size mismatch");
	memcpy(&topic.delta_heading, buf.iterator, sizeof(topic.delta_heading));
	buf.iterator += sizeof(topic.delta_heading);
	buf.offset += sizeof(topic.delta_heading);
	static_assert(sizeof(topic.heading_reset_counter) == 1, "size mismatch");
	memcpy(&topic.heading_reset_counter, buf.iterator, sizeof(topic.heading_reset_counter));
	buf.iterator += sizeof(topic.heading_reset_counter);
	buf.offset += sizeof(topic.heading_reset_counter);
	static_assert(sizeof(topic.heading_good_for_control) == 1, "size mismatch");
	memcpy(&topic.heading_good_for_control, buf.iterator, sizeof(topic.heading_good_for_control));
	buf.iterator += sizeof(topic.heading_good_for_control);
	buf.offset += sizeof(topic.heading_good_for_control);
	static_assert(sizeof(topic.xy_global) == 1, "size mismatch");
	memcpy(&topic.xy_global, buf.iterator, sizeof(topic.xy_global));
	buf.iterator += sizeof(topic.xy_global);
	buf.offset += sizeof(topic.xy_global);
	static_assert(sizeof(topic.z_global) == 1, "size mismatch");
	memcpy(&topic.z_global, buf.iterator, sizeof(topic.z_global));
	buf.iterator += sizeof(topic.z_global);
	buf.offset += sizeof(topic.z_global);
	static_assert(sizeof(topic.ref_timestamp) == 8, "size mismatch");
	memcpy(&topic.ref_timestamp, buf.iterator, sizeof(topic.ref_timestamp));
	buf.iterator += sizeof(topic.ref_timestamp);
	buf.offset += sizeof(topic.ref_timestamp);
	static_assert(sizeof(topic.ref_lat) == 8, "size mismatch");
	memcpy(&topic.ref_lat, buf.iterator, sizeof(topic.ref_lat));
	buf.iterator += sizeof(topic.ref_lat);
	buf.offset += sizeof(topic.ref_lat);
	static_assert(sizeof(topic.ref_lon) == 8, "size mismatch");
	memcpy(&topic.ref_lon, buf.iterator, sizeof(topic.ref_lon));
	buf.iterator += sizeof(topic.ref_lon);
	buf.offset += sizeof(topic.ref_lon);
	static_assert(sizeof(topic.ref_alt) == 4, "size mismatch");
	memcpy(&topic.ref_alt, buf.iterator, sizeof(topic.ref_alt));
	buf.iterator += sizeof(topic.ref_alt);
	buf.offset += sizeof(topic.ref_alt);
	static_assert(sizeof(topic.dist_bottom) == 4, "size mismatch");
	memcpy(&topic.dist_bottom, buf.iterator, sizeof(topic.dist_bottom));
	buf.iterator += sizeof(topic.dist_bottom);
	buf.offset += sizeof(topic.dist_bottom);
	static_assert(sizeof(topic.dist_bottom_valid) == 1, "size mismatch");
	memcpy(&topic.dist_bottom_valid, buf.iterator, sizeof(topic.dist_bottom_valid));
	buf.iterator += sizeof(topic.dist_bottom_valid);
	buf.offset += sizeof(topic.dist_bottom_valid);
	static_assert(sizeof(topic.dist_bottom_sensor_bitfield) == 1, "size mismatch");
	memcpy(&topic.dist_bottom_sensor_bitfield, buf.iterator, sizeof(topic.dist_bottom_sensor_bitfield));
	buf.iterator += sizeof(topic.dist_bottom_sensor_bitfield);
	buf.offset += sizeof(topic.dist_bottom_sensor_bitfield);
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
	static_assert(sizeof(topic.evh) == 4, "size mismatch");
	memcpy(&topic.evh, buf.iterator, sizeof(topic.evh));
	buf.iterator += sizeof(topic.evh);
	buf.offset += sizeof(topic.evh);
	static_assert(sizeof(topic.evv) == 4, "size mismatch");
	memcpy(&topic.evv, buf.iterator, sizeof(topic.evv));
	buf.iterator += sizeof(topic.evv);
	buf.offset += sizeof(topic.evv);
	static_assert(sizeof(topic.dead_reckoning) == 1, "size mismatch");
	memcpy(&topic.dead_reckoning, buf.iterator, sizeof(topic.dead_reckoning));
	buf.iterator += sizeof(topic.dead_reckoning);
	buf.offset += sizeof(topic.dead_reckoning);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vxy_max) == 4, "size mismatch");
	memcpy(&topic.vxy_max, buf.iterator, sizeof(topic.vxy_max));
	buf.iterator += sizeof(topic.vxy_max);
	buf.offset += sizeof(topic.vxy_max);
	static_assert(sizeof(topic.vz_max) == 4, "size mismatch");
	memcpy(&topic.vz_max, buf.iterator, sizeof(topic.vz_max));
	buf.iterator += sizeof(topic.vz_max);
	buf.offset += sizeof(topic.vz_max);
	static_assert(sizeof(topic.hagl_min) == 4, "size mismatch");
	memcpy(&topic.hagl_min, buf.iterator, sizeof(topic.hagl_min));
	buf.iterator += sizeof(topic.hagl_min);
	buf.offset += sizeof(topic.hagl_min);
	static_assert(sizeof(topic.hagl_max) == 4, "size mismatch");
	memcpy(&topic.hagl_max, buf.iterator, sizeof(topic.hagl_max));
	buf.iterator += sizeof(topic.hagl_max);
	buf.offset += sizeof(topic.hagl_max);
	return true;
}
