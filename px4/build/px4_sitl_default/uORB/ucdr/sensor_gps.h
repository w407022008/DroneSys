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
#include <uORB/topics/sensor_gps.h>


static inline constexpr int ucdr_topic_size_sensor_gps()
{
	return 141;
}

bool ucdr_serialize_sensor_gps(const sensor_gps_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.device_id, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.lat) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.lat, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 4, "size mismatch");
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
	static_assert(sizeof(topic.s_variance_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.s_variance_m_s, sizeof(topic.s_variance_m_s));
	buf.iterator += sizeof(topic.s_variance_m_s);
	buf.offset += sizeof(topic.s_variance_m_s);
	static_assert(sizeof(topic.c_variance_rad) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.c_variance_rad, sizeof(topic.c_variance_rad));
	buf.iterator += sizeof(topic.c_variance_rad);
	buf.offset += sizeof(topic.c_variance_rad);
	static_assert(sizeof(topic.fix_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fix_type, sizeof(topic.fix_type));
	buf.iterator += sizeof(topic.fix_type);
	buf.offset += sizeof(topic.fix_type);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.eph) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.eph, sizeof(topic.eph));
	buf.iterator += sizeof(topic.eph);
	buf.offset += sizeof(topic.eph);
	static_assert(sizeof(topic.epv) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.epv, sizeof(topic.epv));
	buf.iterator += sizeof(topic.epv);
	buf.offset += sizeof(topic.epv);
	static_assert(sizeof(topic.hdop) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hdop, sizeof(topic.hdop));
	buf.iterator += sizeof(topic.hdop);
	buf.offset += sizeof(topic.hdop);
	static_assert(sizeof(topic.vdop) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vdop, sizeof(topic.vdop));
	buf.iterator += sizeof(topic.vdop);
	buf.offset += sizeof(topic.vdop);
	static_assert(sizeof(topic.noise_per_ms) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.noise_per_ms, sizeof(topic.noise_per_ms));
	buf.iterator += sizeof(topic.noise_per_ms);
	buf.offset += sizeof(topic.noise_per_ms);
	static_assert(sizeof(topic.automatic_gain_control) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.automatic_gain_control, sizeof(topic.automatic_gain_control));
	buf.iterator += sizeof(topic.automatic_gain_control);
	buf.offset += sizeof(topic.automatic_gain_control);
	static_assert(sizeof(topic.jamming_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.jamming_state, sizeof(topic.jamming_state));
	buf.iterator += sizeof(topic.jamming_state);
	buf.offset += sizeof(topic.jamming_state);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.jamming_indicator) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.jamming_indicator, sizeof(topic.jamming_indicator));
	buf.iterator += sizeof(topic.jamming_indicator);
	buf.offset += sizeof(topic.jamming_indicator);
	static_assert(sizeof(topic.spoofing_state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.spoofing_state, sizeof(topic.spoofing_state));
	buf.iterator += sizeof(topic.spoofing_state);
	buf.offset += sizeof(topic.spoofing_state);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vel_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vel_m_s, sizeof(topic.vel_m_s));
	buf.iterator += sizeof(topic.vel_m_s);
	buf.offset += sizeof(topic.vel_m_s);
	static_assert(sizeof(topic.vel_n_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vel_n_m_s, sizeof(topic.vel_n_m_s));
	buf.iterator += sizeof(topic.vel_n_m_s);
	buf.offset += sizeof(topic.vel_n_m_s);
	static_assert(sizeof(topic.vel_e_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vel_e_m_s, sizeof(topic.vel_e_m_s));
	buf.iterator += sizeof(topic.vel_e_m_s);
	buf.offset += sizeof(topic.vel_e_m_s);
	static_assert(sizeof(topic.vel_d_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vel_d_m_s, sizeof(topic.vel_d_m_s));
	buf.iterator += sizeof(topic.vel_d_m_s);
	buf.offset += sizeof(topic.vel_d_m_s);
	static_assert(sizeof(topic.cog_rad) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cog_rad, sizeof(topic.cog_rad));
	buf.iterator += sizeof(topic.cog_rad);
	buf.offset += sizeof(topic.cog_rad);
	static_assert(sizeof(topic.vel_ned_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vel_ned_valid, sizeof(topic.vel_ned_valid));
	buf.iterator += sizeof(topic.vel_ned_valid);
	buf.offset += sizeof(topic.vel_ned_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.timestamp_time_relative) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.timestamp_time_relative, sizeof(topic.timestamp_time_relative));
	buf.iterator += sizeof(topic.timestamp_time_relative);
	buf.offset += sizeof(topic.timestamp_time_relative);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.time_utc_usec) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.time_utc_usec, sizeof(topic.time_utc_usec));
	buf.iterator += sizeof(topic.time_utc_usec);
	buf.offset += sizeof(topic.time_utc_usec);
	static_assert(sizeof(topic.satellites_used) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.satellites_used, sizeof(topic.satellites_used));
	buf.iterator += sizeof(topic.satellites_used);
	buf.offset += sizeof(topic.satellites_used);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.heading_offset) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading_offset, sizeof(topic.heading_offset));
	buf.iterator += sizeof(topic.heading_offset);
	buf.offset += sizeof(topic.heading_offset);
	static_assert(sizeof(topic.heading_accuracy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading_accuracy, sizeof(topic.heading_accuracy));
	buf.iterator += sizeof(topic.heading_accuracy);
	buf.offset += sizeof(topic.heading_accuracy);
	static_assert(sizeof(topic.rtcm_injection_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rtcm_injection_rate, sizeof(topic.rtcm_injection_rate));
	buf.iterator += sizeof(topic.rtcm_injection_rate);
	buf.offset += sizeof(topic.rtcm_injection_rate);
	static_assert(sizeof(topic.selected_rtcm_instance) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.selected_rtcm_instance, sizeof(topic.selected_rtcm_instance));
	buf.iterator += sizeof(topic.selected_rtcm_instance);
	buf.offset += sizeof(topic.selected_rtcm_instance);
	return true;
}

bool ucdr_deserialize_sensor_gps(ucdrBuffer& buf, sensor_gps_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(&topic.device_id, buf.iterator, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.lat) == 4, "size mismatch");
	memcpy(&topic.lat, buf.iterator, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 4, "size mismatch");
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
	static_assert(sizeof(topic.s_variance_m_s) == 4, "size mismatch");
	memcpy(&topic.s_variance_m_s, buf.iterator, sizeof(topic.s_variance_m_s));
	buf.iterator += sizeof(topic.s_variance_m_s);
	buf.offset += sizeof(topic.s_variance_m_s);
	static_assert(sizeof(topic.c_variance_rad) == 4, "size mismatch");
	memcpy(&topic.c_variance_rad, buf.iterator, sizeof(topic.c_variance_rad));
	buf.iterator += sizeof(topic.c_variance_rad);
	buf.offset += sizeof(topic.c_variance_rad);
	static_assert(sizeof(topic.fix_type) == 1, "size mismatch");
	memcpy(&topic.fix_type, buf.iterator, sizeof(topic.fix_type));
	buf.iterator += sizeof(topic.fix_type);
	buf.offset += sizeof(topic.fix_type);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.eph) == 4, "size mismatch");
	memcpy(&topic.eph, buf.iterator, sizeof(topic.eph));
	buf.iterator += sizeof(topic.eph);
	buf.offset += sizeof(topic.eph);
	static_assert(sizeof(topic.epv) == 4, "size mismatch");
	memcpy(&topic.epv, buf.iterator, sizeof(topic.epv));
	buf.iterator += sizeof(topic.epv);
	buf.offset += sizeof(topic.epv);
	static_assert(sizeof(topic.hdop) == 4, "size mismatch");
	memcpy(&topic.hdop, buf.iterator, sizeof(topic.hdop));
	buf.iterator += sizeof(topic.hdop);
	buf.offset += sizeof(topic.hdop);
	static_assert(sizeof(topic.vdop) == 4, "size mismatch");
	memcpy(&topic.vdop, buf.iterator, sizeof(topic.vdop));
	buf.iterator += sizeof(topic.vdop);
	buf.offset += sizeof(topic.vdop);
	static_assert(sizeof(topic.noise_per_ms) == 4, "size mismatch");
	memcpy(&topic.noise_per_ms, buf.iterator, sizeof(topic.noise_per_ms));
	buf.iterator += sizeof(topic.noise_per_ms);
	buf.offset += sizeof(topic.noise_per_ms);
	static_assert(sizeof(topic.automatic_gain_control) == 2, "size mismatch");
	memcpy(&topic.automatic_gain_control, buf.iterator, sizeof(topic.automatic_gain_control));
	buf.iterator += sizeof(topic.automatic_gain_control);
	buf.offset += sizeof(topic.automatic_gain_control);
	static_assert(sizeof(topic.jamming_state) == 1, "size mismatch");
	memcpy(&topic.jamming_state, buf.iterator, sizeof(topic.jamming_state));
	buf.iterator += sizeof(topic.jamming_state);
	buf.offset += sizeof(topic.jamming_state);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.jamming_indicator) == 4, "size mismatch");
	memcpy(&topic.jamming_indicator, buf.iterator, sizeof(topic.jamming_indicator));
	buf.iterator += sizeof(topic.jamming_indicator);
	buf.offset += sizeof(topic.jamming_indicator);
	static_assert(sizeof(topic.spoofing_state) == 1, "size mismatch");
	memcpy(&topic.spoofing_state, buf.iterator, sizeof(topic.spoofing_state));
	buf.iterator += sizeof(topic.spoofing_state);
	buf.offset += sizeof(topic.spoofing_state);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.vel_m_s) == 4, "size mismatch");
	memcpy(&topic.vel_m_s, buf.iterator, sizeof(topic.vel_m_s));
	buf.iterator += sizeof(topic.vel_m_s);
	buf.offset += sizeof(topic.vel_m_s);
	static_assert(sizeof(topic.vel_n_m_s) == 4, "size mismatch");
	memcpy(&topic.vel_n_m_s, buf.iterator, sizeof(topic.vel_n_m_s));
	buf.iterator += sizeof(topic.vel_n_m_s);
	buf.offset += sizeof(topic.vel_n_m_s);
	static_assert(sizeof(topic.vel_e_m_s) == 4, "size mismatch");
	memcpy(&topic.vel_e_m_s, buf.iterator, sizeof(topic.vel_e_m_s));
	buf.iterator += sizeof(topic.vel_e_m_s);
	buf.offset += sizeof(topic.vel_e_m_s);
	static_assert(sizeof(topic.vel_d_m_s) == 4, "size mismatch");
	memcpy(&topic.vel_d_m_s, buf.iterator, sizeof(topic.vel_d_m_s));
	buf.iterator += sizeof(topic.vel_d_m_s);
	buf.offset += sizeof(topic.vel_d_m_s);
	static_assert(sizeof(topic.cog_rad) == 4, "size mismatch");
	memcpy(&topic.cog_rad, buf.iterator, sizeof(topic.cog_rad));
	buf.iterator += sizeof(topic.cog_rad);
	buf.offset += sizeof(topic.cog_rad);
	static_assert(sizeof(topic.vel_ned_valid) == 1, "size mismatch");
	memcpy(&topic.vel_ned_valid, buf.iterator, sizeof(topic.vel_ned_valid));
	buf.iterator += sizeof(topic.vel_ned_valid);
	buf.offset += sizeof(topic.vel_ned_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.timestamp_time_relative) == 4, "size mismatch");
	memcpy(&topic.timestamp_time_relative, buf.iterator, sizeof(topic.timestamp_time_relative));
	buf.iterator += sizeof(topic.timestamp_time_relative);
	buf.offset += sizeof(topic.timestamp_time_relative);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.time_utc_usec) == 8, "size mismatch");
	memcpy(&topic.time_utc_usec, buf.iterator, sizeof(topic.time_utc_usec));
	buf.iterator += sizeof(topic.time_utc_usec);
	buf.offset += sizeof(topic.time_utc_usec);
	static_assert(sizeof(topic.satellites_used) == 1, "size mismatch");
	memcpy(&topic.satellites_used, buf.iterator, sizeof(topic.satellites_used));
	buf.iterator += sizeof(topic.satellites_used);
	buf.offset += sizeof(topic.satellites_used);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(&topic.heading, buf.iterator, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.heading_offset) == 4, "size mismatch");
	memcpy(&topic.heading_offset, buf.iterator, sizeof(topic.heading_offset));
	buf.iterator += sizeof(topic.heading_offset);
	buf.offset += sizeof(topic.heading_offset);
	static_assert(sizeof(topic.heading_accuracy) == 4, "size mismatch");
	memcpy(&topic.heading_accuracy, buf.iterator, sizeof(topic.heading_accuracy));
	buf.iterator += sizeof(topic.heading_accuracy);
	buf.offset += sizeof(topic.heading_accuracy);
	static_assert(sizeof(topic.rtcm_injection_rate) == 4, "size mismatch");
	memcpy(&topic.rtcm_injection_rate, buf.iterator, sizeof(topic.rtcm_injection_rate));
	buf.iterator += sizeof(topic.rtcm_injection_rate);
	buf.offset += sizeof(topic.rtcm_injection_rate);
	static_assert(sizeof(topic.selected_rtcm_instance) == 1, "size mismatch");
	memcpy(&topic.selected_rtcm_instance, buf.iterator, sizeof(topic.selected_rtcm_instance));
	buf.iterator += sizeof(topic.selected_rtcm_instance);
	buf.offset += sizeof(topic.selected_rtcm_instance);
	return true;
}
