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
#include <uORB/topics/estimator_gps_status.h>


static inline constexpr int ucdr_topic_size_estimator_gps_status()
{
	return 40;
}

bool ucdr_serialize_estimator_gps_status(const estimator_gps_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.checks_passed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.checks_passed, sizeof(topic.checks_passed));
	buf.iterator += sizeof(topic.checks_passed);
	buf.offset += sizeof(topic.checks_passed);
	static_assert(sizeof(topic.check_fail_gps_fix) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_gps_fix, sizeof(topic.check_fail_gps_fix));
	buf.iterator += sizeof(topic.check_fail_gps_fix);
	buf.offset += sizeof(topic.check_fail_gps_fix);
	static_assert(sizeof(topic.check_fail_min_sat_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_min_sat_count, sizeof(topic.check_fail_min_sat_count));
	buf.iterator += sizeof(topic.check_fail_min_sat_count);
	buf.offset += sizeof(topic.check_fail_min_sat_count);
	static_assert(sizeof(topic.check_fail_max_pdop) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_pdop, sizeof(topic.check_fail_max_pdop));
	buf.iterator += sizeof(topic.check_fail_max_pdop);
	buf.offset += sizeof(topic.check_fail_max_pdop);
	static_assert(sizeof(topic.check_fail_max_horz_err) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_horz_err, sizeof(topic.check_fail_max_horz_err));
	buf.iterator += sizeof(topic.check_fail_max_horz_err);
	buf.offset += sizeof(topic.check_fail_max_horz_err);
	static_assert(sizeof(topic.check_fail_max_vert_err) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_vert_err, sizeof(topic.check_fail_max_vert_err));
	buf.iterator += sizeof(topic.check_fail_max_vert_err);
	buf.offset += sizeof(topic.check_fail_max_vert_err);
	static_assert(sizeof(topic.check_fail_max_spd_err) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_spd_err, sizeof(topic.check_fail_max_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_spd_err);
	buf.offset += sizeof(topic.check_fail_max_spd_err);
	static_assert(sizeof(topic.check_fail_max_horz_drift) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_horz_drift, sizeof(topic.check_fail_max_horz_drift));
	buf.iterator += sizeof(topic.check_fail_max_horz_drift);
	buf.offset += sizeof(topic.check_fail_max_horz_drift);
	static_assert(sizeof(topic.check_fail_max_vert_drift) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_vert_drift, sizeof(topic.check_fail_max_vert_drift));
	buf.iterator += sizeof(topic.check_fail_max_vert_drift);
	buf.offset += sizeof(topic.check_fail_max_vert_drift);
	static_assert(sizeof(topic.check_fail_max_horz_spd_err) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_horz_spd_err, sizeof(topic.check_fail_max_horz_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_horz_spd_err);
	buf.offset += sizeof(topic.check_fail_max_horz_spd_err);
	static_assert(sizeof(topic.check_fail_max_vert_spd_err) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.check_fail_max_vert_spd_err, sizeof(topic.check_fail_max_vert_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_vert_spd_err);
	buf.offset += sizeof(topic.check_fail_max_vert_spd_err);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.position_drift_rate_horizontal_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.position_drift_rate_horizontal_m_s, sizeof(topic.position_drift_rate_horizontal_m_s));
	buf.iterator += sizeof(topic.position_drift_rate_horizontal_m_s);
	buf.offset += sizeof(topic.position_drift_rate_horizontal_m_s);
	static_assert(sizeof(topic.position_drift_rate_vertical_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.position_drift_rate_vertical_m_s, sizeof(topic.position_drift_rate_vertical_m_s));
	buf.iterator += sizeof(topic.position_drift_rate_vertical_m_s);
	buf.offset += sizeof(topic.position_drift_rate_vertical_m_s);
	static_assert(sizeof(topic.filtered_horizontal_speed_m_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filtered_horizontal_speed_m_s, sizeof(topic.filtered_horizontal_speed_m_s));
	buf.iterator += sizeof(topic.filtered_horizontal_speed_m_s);
	buf.offset += sizeof(topic.filtered_horizontal_speed_m_s);
	return true;
}

bool ucdr_deserialize_estimator_gps_status(ucdrBuffer& buf, estimator_gps_status_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.checks_passed) == 1, "size mismatch");
	memcpy(&topic.checks_passed, buf.iterator, sizeof(topic.checks_passed));
	buf.iterator += sizeof(topic.checks_passed);
	buf.offset += sizeof(topic.checks_passed);
	static_assert(sizeof(topic.check_fail_gps_fix) == 1, "size mismatch");
	memcpy(&topic.check_fail_gps_fix, buf.iterator, sizeof(topic.check_fail_gps_fix));
	buf.iterator += sizeof(topic.check_fail_gps_fix);
	buf.offset += sizeof(topic.check_fail_gps_fix);
	static_assert(sizeof(topic.check_fail_min_sat_count) == 1, "size mismatch");
	memcpy(&topic.check_fail_min_sat_count, buf.iterator, sizeof(topic.check_fail_min_sat_count));
	buf.iterator += sizeof(topic.check_fail_min_sat_count);
	buf.offset += sizeof(topic.check_fail_min_sat_count);
	static_assert(sizeof(topic.check_fail_max_pdop) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_pdop, buf.iterator, sizeof(topic.check_fail_max_pdop));
	buf.iterator += sizeof(topic.check_fail_max_pdop);
	buf.offset += sizeof(topic.check_fail_max_pdop);
	static_assert(sizeof(topic.check_fail_max_horz_err) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_horz_err, buf.iterator, sizeof(topic.check_fail_max_horz_err));
	buf.iterator += sizeof(topic.check_fail_max_horz_err);
	buf.offset += sizeof(topic.check_fail_max_horz_err);
	static_assert(sizeof(topic.check_fail_max_vert_err) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_vert_err, buf.iterator, sizeof(topic.check_fail_max_vert_err));
	buf.iterator += sizeof(topic.check_fail_max_vert_err);
	buf.offset += sizeof(topic.check_fail_max_vert_err);
	static_assert(sizeof(topic.check_fail_max_spd_err) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_spd_err, buf.iterator, sizeof(topic.check_fail_max_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_spd_err);
	buf.offset += sizeof(topic.check_fail_max_spd_err);
	static_assert(sizeof(topic.check_fail_max_horz_drift) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_horz_drift, buf.iterator, sizeof(topic.check_fail_max_horz_drift));
	buf.iterator += sizeof(topic.check_fail_max_horz_drift);
	buf.offset += sizeof(topic.check_fail_max_horz_drift);
	static_assert(sizeof(topic.check_fail_max_vert_drift) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_vert_drift, buf.iterator, sizeof(topic.check_fail_max_vert_drift));
	buf.iterator += sizeof(topic.check_fail_max_vert_drift);
	buf.offset += sizeof(topic.check_fail_max_vert_drift);
	static_assert(sizeof(topic.check_fail_max_horz_spd_err) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_horz_spd_err, buf.iterator, sizeof(topic.check_fail_max_horz_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_horz_spd_err);
	buf.offset += sizeof(topic.check_fail_max_horz_spd_err);
	static_assert(sizeof(topic.check_fail_max_vert_spd_err) == 1, "size mismatch");
	memcpy(&topic.check_fail_max_vert_spd_err, buf.iterator, sizeof(topic.check_fail_max_vert_spd_err));
	buf.iterator += sizeof(topic.check_fail_max_vert_spd_err);
	buf.offset += sizeof(topic.check_fail_max_vert_spd_err);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.position_drift_rate_horizontal_m_s) == 4, "size mismatch");
	memcpy(&topic.position_drift_rate_horizontal_m_s, buf.iterator, sizeof(topic.position_drift_rate_horizontal_m_s));
	buf.iterator += sizeof(topic.position_drift_rate_horizontal_m_s);
	buf.offset += sizeof(topic.position_drift_rate_horizontal_m_s);
	static_assert(sizeof(topic.position_drift_rate_vertical_m_s) == 4, "size mismatch");
	memcpy(&topic.position_drift_rate_vertical_m_s, buf.iterator, sizeof(topic.position_drift_rate_vertical_m_s));
	buf.iterator += sizeof(topic.position_drift_rate_vertical_m_s);
	buf.offset += sizeof(topic.position_drift_rate_vertical_m_s);
	static_assert(sizeof(topic.filtered_horizontal_speed_m_s) == 4, "size mismatch");
	memcpy(&topic.filtered_horizontal_speed_m_s, buf.iterator, sizeof(topic.filtered_horizontal_speed_m_s));
	buf.iterator += sizeof(topic.filtered_horizontal_speed_m_s);
	buf.offset += sizeof(topic.filtered_horizontal_speed_m_s);
	return true;
}
