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
#include <uORB/topics/estimator_status.h>


static inline constexpr int ucdr_topic_size_estimator_status()
{
	return 122;
}

bool ucdr_serialize_estimator_status(const estimator_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.output_tracking_error) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.output_tracking_error, sizeof(topic.output_tracking_error));
	buf.iterator += sizeof(topic.output_tracking_error);
	buf.offset += sizeof(topic.output_tracking_error);
	static_assert(sizeof(topic.gps_check_fail_flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.gps_check_fail_flags, sizeof(topic.gps_check_fail_flags));
	buf.iterator += sizeof(topic.gps_check_fail_flags);
	buf.offset += sizeof(topic.gps_check_fail_flags);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.control_mode_flags) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.control_mode_flags, sizeof(topic.control_mode_flags));
	buf.iterator += sizeof(topic.control_mode_flags);
	buf.offset += sizeof(topic.control_mode_flags);
	static_assert(sizeof(topic.filter_fault_flags) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.filter_fault_flags, sizeof(topic.filter_fault_flags));
	buf.iterator += sizeof(topic.filter_fault_flags);
	buf.offset += sizeof(topic.filter_fault_flags);
	static_assert(sizeof(topic.pos_horiz_accuracy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pos_horiz_accuracy, sizeof(topic.pos_horiz_accuracy));
	buf.iterator += sizeof(topic.pos_horiz_accuracy);
	buf.offset += sizeof(topic.pos_horiz_accuracy);
	static_assert(sizeof(topic.pos_vert_accuracy) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pos_vert_accuracy, sizeof(topic.pos_vert_accuracy));
	buf.iterator += sizeof(topic.pos_vert_accuracy);
	buf.offset += sizeof(topic.pos_vert_accuracy);
	static_assert(sizeof(topic.innovation_check_flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.innovation_check_flags, sizeof(topic.innovation_check_flags));
	buf.iterator += sizeof(topic.innovation_check_flags);
	buf.offset += sizeof(topic.innovation_check_flags);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.mag_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mag_test_ratio, sizeof(topic.mag_test_ratio));
	buf.iterator += sizeof(topic.mag_test_ratio);
	buf.offset += sizeof(topic.mag_test_ratio);
	static_assert(sizeof(topic.vel_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vel_test_ratio, sizeof(topic.vel_test_ratio));
	buf.iterator += sizeof(topic.vel_test_ratio);
	buf.offset += sizeof(topic.vel_test_ratio);
	static_assert(sizeof(topic.pos_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pos_test_ratio, sizeof(topic.pos_test_ratio));
	buf.iterator += sizeof(topic.pos_test_ratio);
	buf.offset += sizeof(topic.pos_test_ratio);
	static_assert(sizeof(topic.hgt_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hgt_test_ratio, sizeof(topic.hgt_test_ratio));
	buf.iterator += sizeof(topic.hgt_test_ratio);
	buf.offset += sizeof(topic.hgt_test_ratio);
	static_assert(sizeof(topic.tas_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tas_test_ratio, sizeof(topic.tas_test_ratio));
	buf.iterator += sizeof(topic.tas_test_ratio);
	buf.offset += sizeof(topic.tas_test_ratio);
	static_assert(sizeof(topic.hagl_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hagl_test_ratio, sizeof(topic.hagl_test_ratio));
	buf.iterator += sizeof(topic.hagl_test_ratio);
	buf.offset += sizeof(topic.hagl_test_ratio);
	static_assert(sizeof(topic.beta_test_ratio) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.beta_test_ratio, sizeof(topic.beta_test_ratio));
	buf.iterator += sizeof(topic.beta_test_ratio);
	buf.offset += sizeof(topic.beta_test_ratio);
	static_assert(sizeof(topic.solution_status_flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.solution_status_flags, sizeof(topic.solution_status_flags));
	buf.iterator += sizeof(topic.solution_status_flags);
	buf.offset += sizeof(topic.solution_status_flags);
	static_assert(sizeof(topic.reset_count_vel_ne) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_count_vel_ne, sizeof(topic.reset_count_vel_ne));
	buf.iterator += sizeof(topic.reset_count_vel_ne);
	buf.offset += sizeof(topic.reset_count_vel_ne);
	static_assert(sizeof(topic.reset_count_vel_d) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_count_vel_d, sizeof(topic.reset_count_vel_d));
	buf.iterator += sizeof(topic.reset_count_vel_d);
	buf.offset += sizeof(topic.reset_count_vel_d);
	static_assert(sizeof(topic.reset_count_pos_ne) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_count_pos_ne, sizeof(topic.reset_count_pos_ne));
	buf.iterator += sizeof(topic.reset_count_pos_ne);
	buf.offset += sizeof(topic.reset_count_pos_ne);
	static_assert(sizeof(topic.reset_count_pod_d) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_count_pod_d, sizeof(topic.reset_count_pod_d));
	buf.iterator += sizeof(topic.reset_count_pod_d);
	buf.offset += sizeof(topic.reset_count_pod_d);
	static_assert(sizeof(topic.reset_count_quat) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_count_quat, sizeof(topic.reset_count_quat));
	buf.iterator += sizeof(topic.reset_count_quat);
	buf.offset += sizeof(topic.reset_count_quat);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.time_slip) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.time_slip, sizeof(topic.time_slip));
	buf.iterator += sizeof(topic.time_slip);
	buf.offset += sizeof(topic.time_slip);
	static_assert(sizeof(topic.pre_flt_fail_innov_heading) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flt_fail_innov_heading, sizeof(topic.pre_flt_fail_innov_heading));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_heading);
	buf.offset += sizeof(topic.pre_flt_fail_innov_heading);
	static_assert(sizeof(topic.pre_flt_fail_innov_vel_horiz) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flt_fail_innov_vel_horiz, sizeof(topic.pre_flt_fail_innov_vel_horiz));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_vel_horiz);
	buf.offset += sizeof(topic.pre_flt_fail_innov_vel_horiz);
	static_assert(sizeof(topic.pre_flt_fail_innov_vel_vert) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flt_fail_innov_vel_vert, sizeof(topic.pre_flt_fail_innov_vel_vert));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_vel_vert);
	buf.offset += sizeof(topic.pre_flt_fail_innov_vel_vert);
	static_assert(sizeof(topic.pre_flt_fail_innov_height) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flt_fail_innov_height, sizeof(topic.pre_flt_fail_innov_height));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_height);
	buf.offset += sizeof(topic.pre_flt_fail_innov_height);
	static_assert(sizeof(topic.pre_flt_fail_mag_field_disturbed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pre_flt_fail_mag_field_disturbed, sizeof(topic.pre_flt_fail_mag_field_disturbed));
	buf.iterator += sizeof(topic.pre_flt_fail_mag_field_disturbed);
	buf.offset += sizeof(topic.pre_flt_fail_mag_field_disturbed);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.accel_device_id, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gyro_device_id, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_device_id, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.mag_device_id, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.health_flags) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.health_flags, sizeof(topic.health_flags));
	buf.iterator += sizeof(topic.health_flags);
	buf.offset += sizeof(topic.health_flags);
	static_assert(sizeof(topic.timeout_flags) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.timeout_flags, sizeof(topic.timeout_flags));
	buf.iterator += sizeof(topic.timeout_flags);
	buf.offset += sizeof(topic.timeout_flags);
	return true;
}

bool ucdr_deserialize_estimator_status(ucdrBuffer& buf, estimator_status_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.output_tracking_error) == 12, "size mismatch");
	memcpy(&topic.output_tracking_error, buf.iterator, sizeof(topic.output_tracking_error));
	buf.iterator += sizeof(topic.output_tracking_error);
	buf.offset += sizeof(topic.output_tracking_error);
	static_assert(sizeof(topic.gps_check_fail_flags) == 2, "size mismatch");
	memcpy(&topic.gps_check_fail_flags, buf.iterator, sizeof(topic.gps_check_fail_flags));
	buf.iterator += sizeof(topic.gps_check_fail_flags);
	buf.offset += sizeof(topic.gps_check_fail_flags);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.control_mode_flags) == 8, "size mismatch");
	memcpy(&topic.control_mode_flags, buf.iterator, sizeof(topic.control_mode_flags));
	buf.iterator += sizeof(topic.control_mode_flags);
	buf.offset += sizeof(topic.control_mode_flags);
	static_assert(sizeof(topic.filter_fault_flags) == 4, "size mismatch");
	memcpy(&topic.filter_fault_flags, buf.iterator, sizeof(topic.filter_fault_flags));
	buf.iterator += sizeof(topic.filter_fault_flags);
	buf.offset += sizeof(topic.filter_fault_flags);
	static_assert(sizeof(topic.pos_horiz_accuracy) == 4, "size mismatch");
	memcpy(&topic.pos_horiz_accuracy, buf.iterator, sizeof(topic.pos_horiz_accuracy));
	buf.iterator += sizeof(topic.pos_horiz_accuracy);
	buf.offset += sizeof(topic.pos_horiz_accuracy);
	static_assert(sizeof(topic.pos_vert_accuracy) == 4, "size mismatch");
	memcpy(&topic.pos_vert_accuracy, buf.iterator, sizeof(topic.pos_vert_accuracy));
	buf.iterator += sizeof(topic.pos_vert_accuracy);
	buf.offset += sizeof(topic.pos_vert_accuracy);
	static_assert(sizeof(topic.innovation_check_flags) == 2, "size mismatch");
	memcpy(&topic.innovation_check_flags, buf.iterator, sizeof(topic.innovation_check_flags));
	buf.iterator += sizeof(topic.innovation_check_flags);
	buf.offset += sizeof(topic.innovation_check_flags);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.mag_test_ratio) == 4, "size mismatch");
	memcpy(&topic.mag_test_ratio, buf.iterator, sizeof(topic.mag_test_ratio));
	buf.iterator += sizeof(topic.mag_test_ratio);
	buf.offset += sizeof(topic.mag_test_ratio);
	static_assert(sizeof(topic.vel_test_ratio) == 4, "size mismatch");
	memcpy(&topic.vel_test_ratio, buf.iterator, sizeof(topic.vel_test_ratio));
	buf.iterator += sizeof(topic.vel_test_ratio);
	buf.offset += sizeof(topic.vel_test_ratio);
	static_assert(sizeof(topic.pos_test_ratio) == 4, "size mismatch");
	memcpy(&topic.pos_test_ratio, buf.iterator, sizeof(topic.pos_test_ratio));
	buf.iterator += sizeof(topic.pos_test_ratio);
	buf.offset += sizeof(topic.pos_test_ratio);
	static_assert(sizeof(topic.hgt_test_ratio) == 4, "size mismatch");
	memcpy(&topic.hgt_test_ratio, buf.iterator, sizeof(topic.hgt_test_ratio));
	buf.iterator += sizeof(topic.hgt_test_ratio);
	buf.offset += sizeof(topic.hgt_test_ratio);
	static_assert(sizeof(topic.tas_test_ratio) == 4, "size mismatch");
	memcpy(&topic.tas_test_ratio, buf.iterator, sizeof(topic.tas_test_ratio));
	buf.iterator += sizeof(topic.tas_test_ratio);
	buf.offset += sizeof(topic.tas_test_ratio);
	static_assert(sizeof(topic.hagl_test_ratio) == 4, "size mismatch");
	memcpy(&topic.hagl_test_ratio, buf.iterator, sizeof(topic.hagl_test_ratio));
	buf.iterator += sizeof(topic.hagl_test_ratio);
	buf.offset += sizeof(topic.hagl_test_ratio);
	static_assert(sizeof(topic.beta_test_ratio) == 4, "size mismatch");
	memcpy(&topic.beta_test_ratio, buf.iterator, sizeof(topic.beta_test_ratio));
	buf.iterator += sizeof(topic.beta_test_ratio);
	buf.offset += sizeof(topic.beta_test_ratio);
	static_assert(sizeof(topic.solution_status_flags) == 2, "size mismatch");
	memcpy(&topic.solution_status_flags, buf.iterator, sizeof(topic.solution_status_flags));
	buf.iterator += sizeof(topic.solution_status_flags);
	buf.offset += sizeof(topic.solution_status_flags);
	static_assert(sizeof(topic.reset_count_vel_ne) == 1, "size mismatch");
	memcpy(&topic.reset_count_vel_ne, buf.iterator, sizeof(topic.reset_count_vel_ne));
	buf.iterator += sizeof(topic.reset_count_vel_ne);
	buf.offset += sizeof(topic.reset_count_vel_ne);
	static_assert(sizeof(topic.reset_count_vel_d) == 1, "size mismatch");
	memcpy(&topic.reset_count_vel_d, buf.iterator, sizeof(topic.reset_count_vel_d));
	buf.iterator += sizeof(topic.reset_count_vel_d);
	buf.offset += sizeof(topic.reset_count_vel_d);
	static_assert(sizeof(topic.reset_count_pos_ne) == 1, "size mismatch");
	memcpy(&topic.reset_count_pos_ne, buf.iterator, sizeof(topic.reset_count_pos_ne));
	buf.iterator += sizeof(topic.reset_count_pos_ne);
	buf.offset += sizeof(topic.reset_count_pos_ne);
	static_assert(sizeof(topic.reset_count_pod_d) == 1, "size mismatch");
	memcpy(&topic.reset_count_pod_d, buf.iterator, sizeof(topic.reset_count_pod_d));
	buf.iterator += sizeof(topic.reset_count_pod_d);
	buf.offset += sizeof(topic.reset_count_pod_d);
	static_assert(sizeof(topic.reset_count_quat) == 1, "size mismatch");
	memcpy(&topic.reset_count_quat, buf.iterator, sizeof(topic.reset_count_quat));
	buf.iterator += sizeof(topic.reset_count_quat);
	buf.offset += sizeof(topic.reset_count_quat);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.time_slip) == 4, "size mismatch");
	memcpy(&topic.time_slip, buf.iterator, sizeof(topic.time_slip));
	buf.iterator += sizeof(topic.time_slip);
	buf.offset += sizeof(topic.time_slip);
	static_assert(sizeof(topic.pre_flt_fail_innov_heading) == 1, "size mismatch");
	memcpy(&topic.pre_flt_fail_innov_heading, buf.iterator, sizeof(topic.pre_flt_fail_innov_heading));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_heading);
	buf.offset += sizeof(topic.pre_flt_fail_innov_heading);
	static_assert(sizeof(topic.pre_flt_fail_innov_vel_horiz) == 1, "size mismatch");
	memcpy(&topic.pre_flt_fail_innov_vel_horiz, buf.iterator, sizeof(topic.pre_flt_fail_innov_vel_horiz));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_vel_horiz);
	buf.offset += sizeof(topic.pre_flt_fail_innov_vel_horiz);
	static_assert(sizeof(topic.pre_flt_fail_innov_vel_vert) == 1, "size mismatch");
	memcpy(&topic.pre_flt_fail_innov_vel_vert, buf.iterator, sizeof(topic.pre_flt_fail_innov_vel_vert));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_vel_vert);
	buf.offset += sizeof(topic.pre_flt_fail_innov_vel_vert);
	static_assert(sizeof(topic.pre_flt_fail_innov_height) == 1, "size mismatch");
	memcpy(&topic.pre_flt_fail_innov_height, buf.iterator, sizeof(topic.pre_flt_fail_innov_height));
	buf.iterator += sizeof(topic.pre_flt_fail_innov_height);
	buf.offset += sizeof(topic.pre_flt_fail_innov_height);
	static_assert(sizeof(topic.pre_flt_fail_mag_field_disturbed) == 1, "size mismatch");
	memcpy(&topic.pre_flt_fail_mag_field_disturbed, buf.iterator, sizeof(topic.pre_flt_fail_mag_field_disturbed));
	buf.iterator += sizeof(topic.pre_flt_fail_mag_field_disturbed);
	buf.offset += sizeof(topic.pre_flt_fail_mag_field_disturbed);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.accel_device_id) == 4, "size mismatch");
	memcpy(&topic.accel_device_id, buf.iterator, sizeof(topic.accel_device_id));
	buf.iterator += sizeof(topic.accel_device_id);
	buf.offset += sizeof(topic.accel_device_id);
	static_assert(sizeof(topic.gyro_device_id) == 4, "size mismatch");
	memcpy(&topic.gyro_device_id, buf.iterator, sizeof(topic.gyro_device_id));
	buf.iterator += sizeof(topic.gyro_device_id);
	buf.offset += sizeof(topic.gyro_device_id);
	static_assert(sizeof(topic.baro_device_id) == 4, "size mismatch");
	memcpy(&topic.baro_device_id, buf.iterator, sizeof(topic.baro_device_id));
	buf.iterator += sizeof(topic.baro_device_id);
	buf.offset += sizeof(topic.baro_device_id);
	static_assert(sizeof(topic.mag_device_id) == 4, "size mismatch");
	memcpy(&topic.mag_device_id, buf.iterator, sizeof(topic.mag_device_id));
	buf.iterator += sizeof(topic.mag_device_id);
	buf.offset += sizeof(topic.mag_device_id);
	static_assert(sizeof(topic.health_flags) == 1, "size mismatch");
	memcpy(&topic.health_flags, buf.iterator, sizeof(topic.health_flags));
	buf.iterator += sizeof(topic.health_flags);
	buf.offset += sizeof(topic.health_flags);
	static_assert(sizeof(topic.timeout_flags) == 1, "size mismatch");
	memcpy(&topic.timeout_flags, buf.iterator, sizeof(topic.timeout_flags));
	buf.iterator += sizeof(topic.timeout_flags);
	buf.offset += sizeof(topic.timeout_flags);
	return true;
}
