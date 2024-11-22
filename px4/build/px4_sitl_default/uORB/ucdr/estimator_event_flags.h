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
#include <uORB/topics/estimator_event_flags.h>


static inline constexpr int ucdr_topic_size_estimator_event_flags()
{
	return 56;
}

bool ucdr_serialize_estimator_event_flags(const estimator_event_flags_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.information_event_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.information_event_changes, sizeof(topic.information_event_changes));
	buf.iterator += sizeof(topic.information_event_changes);
	buf.offset += sizeof(topic.information_event_changes);
	static_assert(sizeof(topic.gps_checks_passed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gps_checks_passed, sizeof(topic.gps_checks_passed));
	buf.iterator += sizeof(topic.gps_checks_passed);
	buf.offset += sizeof(topic.gps_checks_passed);
	static_assert(sizeof(topic.reset_vel_to_gps) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_vel_to_gps, sizeof(topic.reset_vel_to_gps));
	buf.iterator += sizeof(topic.reset_vel_to_gps);
	buf.offset += sizeof(topic.reset_vel_to_gps);
	static_assert(sizeof(topic.reset_vel_to_flow) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_vel_to_flow, sizeof(topic.reset_vel_to_flow));
	buf.iterator += sizeof(topic.reset_vel_to_flow);
	buf.offset += sizeof(topic.reset_vel_to_flow);
	static_assert(sizeof(topic.reset_vel_to_vision) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_vel_to_vision, sizeof(topic.reset_vel_to_vision));
	buf.iterator += sizeof(topic.reset_vel_to_vision);
	buf.offset += sizeof(topic.reset_vel_to_vision);
	static_assert(sizeof(topic.reset_vel_to_zero) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_vel_to_zero, sizeof(topic.reset_vel_to_zero));
	buf.iterator += sizeof(topic.reset_vel_to_zero);
	buf.offset += sizeof(topic.reset_vel_to_zero);
	static_assert(sizeof(topic.reset_pos_to_last_known) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_pos_to_last_known, sizeof(topic.reset_pos_to_last_known));
	buf.iterator += sizeof(topic.reset_pos_to_last_known);
	buf.offset += sizeof(topic.reset_pos_to_last_known);
	static_assert(sizeof(topic.reset_pos_to_gps) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_pos_to_gps, sizeof(topic.reset_pos_to_gps));
	buf.iterator += sizeof(topic.reset_pos_to_gps);
	buf.offset += sizeof(topic.reset_pos_to_gps);
	static_assert(sizeof(topic.reset_pos_to_vision) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_pos_to_vision, sizeof(topic.reset_pos_to_vision));
	buf.iterator += sizeof(topic.reset_pos_to_vision);
	buf.offset += sizeof(topic.reset_pos_to_vision);
	static_assert(sizeof(topic.starting_gps_fusion) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.starting_gps_fusion, sizeof(topic.starting_gps_fusion));
	buf.iterator += sizeof(topic.starting_gps_fusion);
	buf.offset += sizeof(topic.starting_gps_fusion);
	static_assert(sizeof(topic.starting_vision_pos_fusion) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.starting_vision_pos_fusion, sizeof(topic.starting_vision_pos_fusion));
	buf.iterator += sizeof(topic.starting_vision_pos_fusion);
	buf.offset += sizeof(topic.starting_vision_pos_fusion);
	static_assert(sizeof(topic.starting_vision_vel_fusion) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.starting_vision_vel_fusion, sizeof(topic.starting_vision_vel_fusion));
	buf.iterator += sizeof(topic.starting_vision_vel_fusion);
	buf.offset += sizeof(topic.starting_vision_vel_fusion);
	static_assert(sizeof(topic.starting_vision_yaw_fusion) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.starting_vision_yaw_fusion, sizeof(topic.starting_vision_yaw_fusion));
	buf.iterator += sizeof(topic.starting_vision_yaw_fusion);
	buf.offset += sizeof(topic.starting_vision_yaw_fusion);
	static_assert(sizeof(topic.yaw_aligned_to_imu_gps) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_aligned_to_imu_gps, sizeof(topic.yaw_aligned_to_imu_gps));
	buf.iterator += sizeof(topic.yaw_aligned_to_imu_gps);
	buf.offset += sizeof(topic.yaw_aligned_to_imu_gps);
	static_assert(sizeof(topic.reset_hgt_to_baro) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_hgt_to_baro, sizeof(topic.reset_hgt_to_baro));
	buf.iterator += sizeof(topic.reset_hgt_to_baro);
	buf.offset += sizeof(topic.reset_hgt_to_baro);
	static_assert(sizeof(topic.reset_hgt_to_gps) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_hgt_to_gps, sizeof(topic.reset_hgt_to_gps));
	buf.iterator += sizeof(topic.reset_hgt_to_gps);
	buf.offset += sizeof(topic.reset_hgt_to_gps);
	static_assert(sizeof(topic.reset_hgt_to_rng) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_hgt_to_rng, sizeof(topic.reset_hgt_to_rng));
	buf.iterator += sizeof(topic.reset_hgt_to_rng);
	buf.offset += sizeof(topic.reset_hgt_to_rng);
	static_assert(sizeof(topic.reset_hgt_to_ev) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_hgt_to_ev, sizeof(topic.reset_hgt_to_ev));
	buf.iterator += sizeof(topic.reset_hgt_to_ev);
	buf.offset += sizeof(topic.reset_hgt_to_ev);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.warning_event_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.warning_event_changes, sizeof(topic.warning_event_changes));
	buf.iterator += sizeof(topic.warning_event_changes);
	buf.offset += sizeof(topic.warning_event_changes);
	static_assert(sizeof(topic.gps_quality_poor) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gps_quality_poor, sizeof(topic.gps_quality_poor));
	buf.iterator += sizeof(topic.gps_quality_poor);
	buf.offset += sizeof(topic.gps_quality_poor);
	static_assert(sizeof(topic.gps_fusion_timout) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gps_fusion_timout, sizeof(topic.gps_fusion_timout));
	buf.iterator += sizeof(topic.gps_fusion_timout);
	buf.offset += sizeof(topic.gps_fusion_timout);
	static_assert(sizeof(topic.gps_data_stopped) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gps_data_stopped, sizeof(topic.gps_data_stopped));
	buf.iterator += sizeof(topic.gps_data_stopped);
	buf.offset += sizeof(topic.gps_data_stopped);
	static_assert(sizeof(topic.gps_data_stopped_using_alternate) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.gps_data_stopped_using_alternate, sizeof(topic.gps_data_stopped_using_alternate));
	buf.iterator += sizeof(topic.gps_data_stopped_using_alternate);
	buf.offset += sizeof(topic.gps_data_stopped_using_alternate);
	static_assert(sizeof(topic.height_sensor_timeout) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.height_sensor_timeout, sizeof(topic.height_sensor_timeout));
	buf.iterator += sizeof(topic.height_sensor_timeout);
	buf.offset += sizeof(topic.height_sensor_timeout);
	static_assert(sizeof(topic.stopping_navigation) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.stopping_navigation, sizeof(topic.stopping_navigation));
	buf.iterator += sizeof(topic.stopping_navigation);
	buf.offset += sizeof(topic.stopping_navigation);
	static_assert(sizeof(topic.invalid_accel_bias_cov_reset) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.invalid_accel_bias_cov_reset, sizeof(topic.invalid_accel_bias_cov_reset));
	buf.iterator += sizeof(topic.invalid_accel_bias_cov_reset);
	buf.offset += sizeof(topic.invalid_accel_bias_cov_reset);
	static_assert(sizeof(topic.bad_yaw_using_gps_course) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.bad_yaw_using_gps_course, sizeof(topic.bad_yaw_using_gps_course));
	buf.iterator += sizeof(topic.bad_yaw_using_gps_course);
	buf.offset += sizeof(topic.bad_yaw_using_gps_course);
	static_assert(sizeof(topic.stopping_mag_use) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.stopping_mag_use, sizeof(topic.stopping_mag_use));
	buf.iterator += sizeof(topic.stopping_mag_use);
	buf.offset += sizeof(topic.stopping_mag_use);
	static_assert(sizeof(topic.vision_data_stopped) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vision_data_stopped, sizeof(topic.vision_data_stopped));
	buf.iterator += sizeof(topic.vision_data_stopped);
	buf.offset += sizeof(topic.vision_data_stopped);
	static_assert(sizeof(topic.emergency_yaw_reset_mag_stopped) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.emergency_yaw_reset_mag_stopped, sizeof(topic.emergency_yaw_reset_mag_stopped));
	buf.iterator += sizeof(topic.emergency_yaw_reset_mag_stopped);
	buf.offset += sizeof(topic.emergency_yaw_reset_mag_stopped);
	static_assert(sizeof(topic.emergency_yaw_reset_gps_yaw_stopped) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.emergency_yaw_reset_gps_yaw_stopped, sizeof(topic.emergency_yaw_reset_gps_yaw_stopped));
	buf.iterator += sizeof(topic.emergency_yaw_reset_gps_yaw_stopped);
	buf.offset += sizeof(topic.emergency_yaw_reset_gps_yaw_stopped);
	return true;
}

bool ucdr_deserialize_estimator_event_flags(ucdrBuffer& buf, estimator_event_flags_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.information_event_changes) == 4, "size mismatch");
	memcpy(&topic.information_event_changes, buf.iterator, sizeof(topic.information_event_changes));
	buf.iterator += sizeof(topic.information_event_changes);
	buf.offset += sizeof(topic.information_event_changes);
	static_assert(sizeof(topic.gps_checks_passed) == 1, "size mismatch");
	memcpy(&topic.gps_checks_passed, buf.iterator, sizeof(topic.gps_checks_passed));
	buf.iterator += sizeof(topic.gps_checks_passed);
	buf.offset += sizeof(topic.gps_checks_passed);
	static_assert(sizeof(topic.reset_vel_to_gps) == 1, "size mismatch");
	memcpy(&topic.reset_vel_to_gps, buf.iterator, sizeof(topic.reset_vel_to_gps));
	buf.iterator += sizeof(topic.reset_vel_to_gps);
	buf.offset += sizeof(topic.reset_vel_to_gps);
	static_assert(sizeof(topic.reset_vel_to_flow) == 1, "size mismatch");
	memcpy(&topic.reset_vel_to_flow, buf.iterator, sizeof(topic.reset_vel_to_flow));
	buf.iterator += sizeof(topic.reset_vel_to_flow);
	buf.offset += sizeof(topic.reset_vel_to_flow);
	static_assert(sizeof(topic.reset_vel_to_vision) == 1, "size mismatch");
	memcpy(&topic.reset_vel_to_vision, buf.iterator, sizeof(topic.reset_vel_to_vision));
	buf.iterator += sizeof(topic.reset_vel_to_vision);
	buf.offset += sizeof(topic.reset_vel_to_vision);
	static_assert(sizeof(topic.reset_vel_to_zero) == 1, "size mismatch");
	memcpy(&topic.reset_vel_to_zero, buf.iterator, sizeof(topic.reset_vel_to_zero));
	buf.iterator += sizeof(topic.reset_vel_to_zero);
	buf.offset += sizeof(topic.reset_vel_to_zero);
	static_assert(sizeof(topic.reset_pos_to_last_known) == 1, "size mismatch");
	memcpy(&topic.reset_pos_to_last_known, buf.iterator, sizeof(topic.reset_pos_to_last_known));
	buf.iterator += sizeof(topic.reset_pos_to_last_known);
	buf.offset += sizeof(topic.reset_pos_to_last_known);
	static_assert(sizeof(topic.reset_pos_to_gps) == 1, "size mismatch");
	memcpy(&topic.reset_pos_to_gps, buf.iterator, sizeof(topic.reset_pos_to_gps));
	buf.iterator += sizeof(topic.reset_pos_to_gps);
	buf.offset += sizeof(topic.reset_pos_to_gps);
	static_assert(sizeof(topic.reset_pos_to_vision) == 1, "size mismatch");
	memcpy(&topic.reset_pos_to_vision, buf.iterator, sizeof(topic.reset_pos_to_vision));
	buf.iterator += sizeof(topic.reset_pos_to_vision);
	buf.offset += sizeof(topic.reset_pos_to_vision);
	static_assert(sizeof(topic.starting_gps_fusion) == 1, "size mismatch");
	memcpy(&topic.starting_gps_fusion, buf.iterator, sizeof(topic.starting_gps_fusion));
	buf.iterator += sizeof(topic.starting_gps_fusion);
	buf.offset += sizeof(topic.starting_gps_fusion);
	static_assert(sizeof(topic.starting_vision_pos_fusion) == 1, "size mismatch");
	memcpy(&topic.starting_vision_pos_fusion, buf.iterator, sizeof(topic.starting_vision_pos_fusion));
	buf.iterator += sizeof(topic.starting_vision_pos_fusion);
	buf.offset += sizeof(topic.starting_vision_pos_fusion);
	static_assert(sizeof(topic.starting_vision_vel_fusion) == 1, "size mismatch");
	memcpy(&topic.starting_vision_vel_fusion, buf.iterator, sizeof(topic.starting_vision_vel_fusion));
	buf.iterator += sizeof(topic.starting_vision_vel_fusion);
	buf.offset += sizeof(topic.starting_vision_vel_fusion);
	static_assert(sizeof(topic.starting_vision_yaw_fusion) == 1, "size mismatch");
	memcpy(&topic.starting_vision_yaw_fusion, buf.iterator, sizeof(topic.starting_vision_yaw_fusion));
	buf.iterator += sizeof(topic.starting_vision_yaw_fusion);
	buf.offset += sizeof(topic.starting_vision_yaw_fusion);
	static_assert(sizeof(topic.yaw_aligned_to_imu_gps) == 1, "size mismatch");
	memcpy(&topic.yaw_aligned_to_imu_gps, buf.iterator, sizeof(topic.yaw_aligned_to_imu_gps));
	buf.iterator += sizeof(topic.yaw_aligned_to_imu_gps);
	buf.offset += sizeof(topic.yaw_aligned_to_imu_gps);
	static_assert(sizeof(topic.reset_hgt_to_baro) == 1, "size mismatch");
	memcpy(&topic.reset_hgt_to_baro, buf.iterator, sizeof(topic.reset_hgt_to_baro));
	buf.iterator += sizeof(topic.reset_hgt_to_baro);
	buf.offset += sizeof(topic.reset_hgt_to_baro);
	static_assert(sizeof(topic.reset_hgt_to_gps) == 1, "size mismatch");
	memcpy(&topic.reset_hgt_to_gps, buf.iterator, sizeof(topic.reset_hgt_to_gps));
	buf.iterator += sizeof(topic.reset_hgt_to_gps);
	buf.offset += sizeof(topic.reset_hgt_to_gps);
	static_assert(sizeof(topic.reset_hgt_to_rng) == 1, "size mismatch");
	memcpy(&topic.reset_hgt_to_rng, buf.iterator, sizeof(topic.reset_hgt_to_rng));
	buf.iterator += sizeof(topic.reset_hgt_to_rng);
	buf.offset += sizeof(topic.reset_hgt_to_rng);
	static_assert(sizeof(topic.reset_hgt_to_ev) == 1, "size mismatch");
	memcpy(&topic.reset_hgt_to_ev, buf.iterator, sizeof(topic.reset_hgt_to_ev));
	buf.iterator += sizeof(topic.reset_hgt_to_ev);
	buf.offset += sizeof(topic.reset_hgt_to_ev);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.warning_event_changes) == 4, "size mismatch");
	memcpy(&topic.warning_event_changes, buf.iterator, sizeof(topic.warning_event_changes));
	buf.iterator += sizeof(topic.warning_event_changes);
	buf.offset += sizeof(topic.warning_event_changes);
	static_assert(sizeof(topic.gps_quality_poor) == 1, "size mismatch");
	memcpy(&topic.gps_quality_poor, buf.iterator, sizeof(topic.gps_quality_poor));
	buf.iterator += sizeof(topic.gps_quality_poor);
	buf.offset += sizeof(topic.gps_quality_poor);
	static_assert(sizeof(topic.gps_fusion_timout) == 1, "size mismatch");
	memcpy(&topic.gps_fusion_timout, buf.iterator, sizeof(topic.gps_fusion_timout));
	buf.iterator += sizeof(topic.gps_fusion_timout);
	buf.offset += sizeof(topic.gps_fusion_timout);
	static_assert(sizeof(topic.gps_data_stopped) == 1, "size mismatch");
	memcpy(&topic.gps_data_stopped, buf.iterator, sizeof(topic.gps_data_stopped));
	buf.iterator += sizeof(topic.gps_data_stopped);
	buf.offset += sizeof(topic.gps_data_stopped);
	static_assert(sizeof(topic.gps_data_stopped_using_alternate) == 1, "size mismatch");
	memcpy(&topic.gps_data_stopped_using_alternate, buf.iterator, sizeof(topic.gps_data_stopped_using_alternate));
	buf.iterator += sizeof(topic.gps_data_stopped_using_alternate);
	buf.offset += sizeof(topic.gps_data_stopped_using_alternate);
	static_assert(sizeof(topic.height_sensor_timeout) == 1, "size mismatch");
	memcpy(&topic.height_sensor_timeout, buf.iterator, sizeof(topic.height_sensor_timeout));
	buf.iterator += sizeof(topic.height_sensor_timeout);
	buf.offset += sizeof(topic.height_sensor_timeout);
	static_assert(sizeof(topic.stopping_navigation) == 1, "size mismatch");
	memcpy(&topic.stopping_navigation, buf.iterator, sizeof(topic.stopping_navigation));
	buf.iterator += sizeof(topic.stopping_navigation);
	buf.offset += sizeof(topic.stopping_navigation);
	static_assert(sizeof(topic.invalid_accel_bias_cov_reset) == 1, "size mismatch");
	memcpy(&topic.invalid_accel_bias_cov_reset, buf.iterator, sizeof(topic.invalid_accel_bias_cov_reset));
	buf.iterator += sizeof(topic.invalid_accel_bias_cov_reset);
	buf.offset += sizeof(topic.invalid_accel_bias_cov_reset);
	static_assert(sizeof(topic.bad_yaw_using_gps_course) == 1, "size mismatch");
	memcpy(&topic.bad_yaw_using_gps_course, buf.iterator, sizeof(topic.bad_yaw_using_gps_course));
	buf.iterator += sizeof(topic.bad_yaw_using_gps_course);
	buf.offset += sizeof(topic.bad_yaw_using_gps_course);
	static_assert(sizeof(topic.stopping_mag_use) == 1, "size mismatch");
	memcpy(&topic.stopping_mag_use, buf.iterator, sizeof(topic.stopping_mag_use));
	buf.iterator += sizeof(topic.stopping_mag_use);
	buf.offset += sizeof(topic.stopping_mag_use);
	static_assert(sizeof(topic.vision_data_stopped) == 1, "size mismatch");
	memcpy(&topic.vision_data_stopped, buf.iterator, sizeof(topic.vision_data_stopped));
	buf.iterator += sizeof(topic.vision_data_stopped);
	buf.offset += sizeof(topic.vision_data_stopped);
	static_assert(sizeof(topic.emergency_yaw_reset_mag_stopped) == 1, "size mismatch");
	memcpy(&topic.emergency_yaw_reset_mag_stopped, buf.iterator, sizeof(topic.emergency_yaw_reset_mag_stopped));
	buf.iterator += sizeof(topic.emergency_yaw_reset_mag_stopped);
	buf.offset += sizeof(topic.emergency_yaw_reset_mag_stopped);
	static_assert(sizeof(topic.emergency_yaw_reset_gps_yaw_stopped) == 1, "size mismatch");
	memcpy(&topic.emergency_yaw_reset_gps_yaw_stopped, buf.iterator, sizeof(topic.emergency_yaw_reset_gps_yaw_stopped));
	buf.iterator += sizeof(topic.emergency_yaw_reset_gps_yaw_stopped);
	buf.offset += sizeof(topic.emergency_yaw_reset_gps_yaw_stopped);
	return true;
}
