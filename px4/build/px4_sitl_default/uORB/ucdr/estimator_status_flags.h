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
#include <uORB/topics/estimator_status_flags.h>


static inline constexpr int ucdr_topic_size_estimator_status_flags()
{
	return 94;
}

bool ucdr_serialize_estimator_status_flags(const estimator_status_flags_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.control_status_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.control_status_changes, sizeof(topic.control_status_changes));
	buf.iterator += sizeof(topic.control_status_changes);
	buf.offset += sizeof(topic.control_status_changes);
	static_assert(sizeof(topic.cs_tilt_align) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_tilt_align, sizeof(topic.cs_tilt_align));
	buf.iterator += sizeof(topic.cs_tilt_align);
	buf.offset += sizeof(topic.cs_tilt_align);
	static_assert(sizeof(topic.cs_yaw_align) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_yaw_align, sizeof(topic.cs_yaw_align));
	buf.iterator += sizeof(topic.cs_yaw_align);
	buf.offset += sizeof(topic.cs_yaw_align);
	static_assert(sizeof(topic.cs_gps) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gps, sizeof(topic.cs_gps));
	buf.iterator += sizeof(topic.cs_gps);
	buf.offset += sizeof(topic.cs_gps);
	static_assert(sizeof(topic.cs_opt_flow) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_opt_flow, sizeof(topic.cs_opt_flow));
	buf.iterator += sizeof(topic.cs_opt_flow);
	buf.offset += sizeof(topic.cs_opt_flow);
	static_assert(sizeof(topic.cs_mag_hdg) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_hdg, sizeof(topic.cs_mag_hdg));
	buf.iterator += sizeof(topic.cs_mag_hdg);
	buf.offset += sizeof(topic.cs_mag_hdg);
	static_assert(sizeof(topic.cs_mag_3d) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_3d, sizeof(topic.cs_mag_3d));
	buf.iterator += sizeof(topic.cs_mag_3d);
	buf.offset += sizeof(topic.cs_mag_3d);
	static_assert(sizeof(topic.cs_mag_dec) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_dec, sizeof(topic.cs_mag_dec));
	buf.iterator += sizeof(topic.cs_mag_dec);
	buf.offset += sizeof(topic.cs_mag_dec);
	static_assert(sizeof(topic.cs_in_air) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_in_air, sizeof(topic.cs_in_air));
	buf.iterator += sizeof(topic.cs_in_air);
	buf.offset += sizeof(topic.cs_in_air);
	static_assert(sizeof(topic.cs_wind) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_wind, sizeof(topic.cs_wind));
	buf.iterator += sizeof(topic.cs_wind);
	buf.offset += sizeof(topic.cs_wind);
	static_assert(sizeof(topic.cs_baro_hgt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_baro_hgt, sizeof(topic.cs_baro_hgt));
	buf.iterator += sizeof(topic.cs_baro_hgt);
	buf.offset += sizeof(topic.cs_baro_hgt);
	static_assert(sizeof(topic.cs_rng_hgt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_rng_hgt, sizeof(topic.cs_rng_hgt));
	buf.iterator += sizeof(topic.cs_rng_hgt);
	buf.offset += sizeof(topic.cs_rng_hgt);
	static_assert(sizeof(topic.cs_gps_hgt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gps_hgt, sizeof(topic.cs_gps_hgt));
	buf.iterator += sizeof(topic.cs_gps_hgt);
	buf.offset += sizeof(topic.cs_gps_hgt);
	static_assert(sizeof(topic.cs_ev_pos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_ev_pos, sizeof(topic.cs_ev_pos));
	buf.iterator += sizeof(topic.cs_ev_pos);
	buf.offset += sizeof(topic.cs_ev_pos);
	static_assert(sizeof(topic.cs_ev_yaw) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_ev_yaw, sizeof(topic.cs_ev_yaw));
	buf.iterator += sizeof(topic.cs_ev_yaw);
	buf.offset += sizeof(topic.cs_ev_yaw);
	static_assert(sizeof(topic.cs_ev_hgt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_ev_hgt, sizeof(topic.cs_ev_hgt));
	buf.iterator += sizeof(topic.cs_ev_hgt);
	buf.offset += sizeof(topic.cs_ev_hgt);
	static_assert(sizeof(topic.cs_fuse_beta) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_fuse_beta, sizeof(topic.cs_fuse_beta));
	buf.iterator += sizeof(topic.cs_fuse_beta);
	buf.offset += sizeof(topic.cs_fuse_beta);
	static_assert(sizeof(topic.cs_mag_field_disturbed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_field_disturbed, sizeof(topic.cs_mag_field_disturbed));
	buf.iterator += sizeof(topic.cs_mag_field_disturbed);
	buf.offset += sizeof(topic.cs_mag_field_disturbed);
	static_assert(sizeof(topic.cs_fixed_wing) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_fixed_wing, sizeof(topic.cs_fixed_wing));
	buf.iterator += sizeof(topic.cs_fixed_wing);
	buf.offset += sizeof(topic.cs_fixed_wing);
	static_assert(sizeof(topic.cs_mag_fault) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_fault, sizeof(topic.cs_mag_fault));
	buf.iterator += sizeof(topic.cs_mag_fault);
	buf.offset += sizeof(topic.cs_mag_fault);
	static_assert(sizeof(topic.cs_fuse_aspd) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_fuse_aspd, sizeof(topic.cs_fuse_aspd));
	buf.iterator += sizeof(topic.cs_fuse_aspd);
	buf.offset += sizeof(topic.cs_fuse_aspd);
	static_assert(sizeof(topic.cs_gnd_effect) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gnd_effect, sizeof(topic.cs_gnd_effect));
	buf.iterator += sizeof(topic.cs_gnd_effect);
	buf.offset += sizeof(topic.cs_gnd_effect);
	static_assert(sizeof(topic.cs_rng_stuck) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_rng_stuck, sizeof(topic.cs_rng_stuck));
	buf.iterator += sizeof(topic.cs_rng_stuck);
	buf.offset += sizeof(topic.cs_rng_stuck);
	static_assert(sizeof(topic.cs_gps_yaw) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gps_yaw, sizeof(topic.cs_gps_yaw));
	buf.iterator += sizeof(topic.cs_gps_yaw);
	buf.offset += sizeof(topic.cs_gps_yaw);
	static_assert(sizeof(topic.cs_mag_aligned_in_flight) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_mag_aligned_in_flight, sizeof(topic.cs_mag_aligned_in_flight));
	buf.iterator += sizeof(topic.cs_mag_aligned_in_flight);
	buf.offset += sizeof(topic.cs_mag_aligned_in_flight);
	static_assert(sizeof(topic.cs_ev_vel) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_ev_vel, sizeof(topic.cs_ev_vel));
	buf.iterator += sizeof(topic.cs_ev_vel);
	buf.offset += sizeof(topic.cs_ev_vel);
	static_assert(sizeof(topic.cs_synthetic_mag_z) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_synthetic_mag_z, sizeof(topic.cs_synthetic_mag_z));
	buf.iterator += sizeof(topic.cs_synthetic_mag_z);
	buf.offset += sizeof(topic.cs_synthetic_mag_z);
	static_assert(sizeof(topic.cs_vehicle_at_rest) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_vehicle_at_rest, sizeof(topic.cs_vehicle_at_rest));
	buf.iterator += sizeof(topic.cs_vehicle_at_rest);
	buf.offset += sizeof(topic.cs_vehicle_at_rest);
	static_assert(sizeof(topic.cs_gps_yaw_fault) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gps_yaw_fault, sizeof(topic.cs_gps_yaw_fault));
	buf.iterator += sizeof(topic.cs_gps_yaw_fault);
	buf.offset += sizeof(topic.cs_gps_yaw_fault);
	static_assert(sizeof(topic.cs_rng_fault) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_rng_fault, sizeof(topic.cs_rng_fault));
	buf.iterator += sizeof(topic.cs_rng_fault);
	buf.offset += sizeof(topic.cs_rng_fault);
	static_assert(sizeof(topic.cs_inertial_dead_reckoning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_inertial_dead_reckoning, sizeof(topic.cs_inertial_dead_reckoning));
	buf.iterator += sizeof(topic.cs_inertial_dead_reckoning);
	buf.offset += sizeof(topic.cs_inertial_dead_reckoning);
	static_assert(sizeof(topic.cs_wind_dead_reckoning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_wind_dead_reckoning, sizeof(topic.cs_wind_dead_reckoning));
	buf.iterator += sizeof(topic.cs_wind_dead_reckoning);
	buf.offset += sizeof(topic.cs_wind_dead_reckoning);
	static_assert(sizeof(topic.cs_rng_kin_consistent) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_rng_kin_consistent, sizeof(topic.cs_rng_kin_consistent));
	buf.iterator += sizeof(topic.cs_rng_kin_consistent);
	buf.offset += sizeof(topic.cs_rng_kin_consistent);
	static_assert(sizeof(topic.cs_fake_pos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_fake_pos, sizeof(topic.cs_fake_pos));
	buf.iterator += sizeof(topic.cs_fake_pos);
	buf.offset += sizeof(topic.cs_fake_pos);
	static_assert(sizeof(topic.cs_fake_hgt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_fake_hgt, sizeof(topic.cs_fake_hgt));
	buf.iterator += sizeof(topic.cs_fake_hgt);
	buf.offset += sizeof(topic.cs_fake_hgt);
	static_assert(sizeof(topic.cs_gravity_vector) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.cs_gravity_vector, sizeof(topic.cs_gravity_vector));
	buf.iterator += sizeof(topic.cs_gravity_vector);
	buf.offset += sizeof(topic.cs_gravity_vector);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.fault_status_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.fault_status_changes, sizeof(topic.fault_status_changes));
	buf.iterator += sizeof(topic.fault_status_changes);
	buf.offset += sizeof(topic.fault_status_changes);
	static_assert(sizeof(topic.fs_bad_mag_x) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_mag_x, sizeof(topic.fs_bad_mag_x));
	buf.iterator += sizeof(topic.fs_bad_mag_x);
	buf.offset += sizeof(topic.fs_bad_mag_x);
	static_assert(sizeof(topic.fs_bad_mag_y) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_mag_y, sizeof(topic.fs_bad_mag_y));
	buf.iterator += sizeof(topic.fs_bad_mag_y);
	buf.offset += sizeof(topic.fs_bad_mag_y);
	static_assert(sizeof(topic.fs_bad_mag_z) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_mag_z, sizeof(topic.fs_bad_mag_z));
	buf.iterator += sizeof(topic.fs_bad_mag_z);
	buf.offset += sizeof(topic.fs_bad_mag_z);
	static_assert(sizeof(topic.fs_bad_hdg) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_hdg, sizeof(topic.fs_bad_hdg));
	buf.iterator += sizeof(topic.fs_bad_hdg);
	buf.offset += sizeof(topic.fs_bad_hdg);
	static_assert(sizeof(topic.fs_bad_mag_decl) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_mag_decl, sizeof(topic.fs_bad_mag_decl));
	buf.iterator += sizeof(topic.fs_bad_mag_decl);
	buf.offset += sizeof(topic.fs_bad_mag_decl);
	static_assert(sizeof(topic.fs_bad_airspeed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_airspeed, sizeof(topic.fs_bad_airspeed));
	buf.iterator += sizeof(topic.fs_bad_airspeed);
	buf.offset += sizeof(topic.fs_bad_airspeed);
	static_assert(sizeof(topic.fs_bad_sideslip) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_sideslip, sizeof(topic.fs_bad_sideslip));
	buf.iterator += sizeof(topic.fs_bad_sideslip);
	buf.offset += sizeof(topic.fs_bad_sideslip);
	static_assert(sizeof(topic.fs_bad_optflow_x) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_optflow_x, sizeof(topic.fs_bad_optflow_x));
	buf.iterator += sizeof(topic.fs_bad_optflow_x);
	buf.offset += sizeof(topic.fs_bad_optflow_x);
	static_assert(sizeof(topic.fs_bad_optflow_y) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_optflow_y, sizeof(topic.fs_bad_optflow_y));
	buf.iterator += sizeof(topic.fs_bad_optflow_y);
	buf.offset += sizeof(topic.fs_bad_optflow_y);
	static_assert(sizeof(topic.fs_bad_vel_n) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_vel_n, sizeof(topic.fs_bad_vel_n));
	buf.iterator += sizeof(topic.fs_bad_vel_n);
	buf.offset += sizeof(topic.fs_bad_vel_n);
	static_assert(sizeof(topic.fs_bad_vel_e) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_vel_e, sizeof(topic.fs_bad_vel_e));
	buf.iterator += sizeof(topic.fs_bad_vel_e);
	buf.offset += sizeof(topic.fs_bad_vel_e);
	static_assert(sizeof(topic.fs_bad_vel_d) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_vel_d, sizeof(topic.fs_bad_vel_d));
	buf.iterator += sizeof(topic.fs_bad_vel_d);
	buf.offset += sizeof(topic.fs_bad_vel_d);
	static_assert(sizeof(topic.fs_bad_pos_n) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_pos_n, sizeof(topic.fs_bad_pos_n));
	buf.iterator += sizeof(topic.fs_bad_pos_n);
	buf.offset += sizeof(topic.fs_bad_pos_n);
	static_assert(sizeof(topic.fs_bad_pos_e) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_pos_e, sizeof(topic.fs_bad_pos_e));
	buf.iterator += sizeof(topic.fs_bad_pos_e);
	buf.offset += sizeof(topic.fs_bad_pos_e);
	static_assert(sizeof(topic.fs_bad_pos_d) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_pos_d, sizeof(topic.fs_bad_pos_d));
	buf.iterator += sizeof(topic.fs_bad_pos_d);
	buf.offset += sizeof(topic.fs_bad_pos_d);
	static_assert(sizeof(topic.fs_bad_acc_bias) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_acc_bias, sizeof(topic.fs_bad_acc_bias));
	buf.iterator += sizeof(topic.fs_bad_acc_bias);
	buf.offset += sizeof(topic.fs_bad_acc_bias);
	static_assert(sizeof(topic.fs_bad_acc_vertical) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_acc_vertical, sizeof(topic.fs_bad_acc_vertical));
	buf.iterator += sizeof(topic.fs_bad_acc_vertical);
	buf.offset += sizeof(topic.fs_bad_acc_vertical);
	static_assert(sizeof(topic.fs_bad_acc_clipping) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fs_bad_acc_clipping, sizeof(topic.fs_bad_acc_clipping));
	buf.iterator += sizeof(topic.fs_bad_acc_clipping);
	buf.offset += sizeof(topic.fs_bad_acc_clipping);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.innovation_fault_status_changes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.innovation_fault_status_changes, sizeof(topic.innovation_fault_status_changes));
	buf.iterator += sizeof(topic.innovation_fault_status_changes);
	buf.offset += sizeof(topic.innovation_fault_status_changes);
	static_assert(sizeof(topic.reject_hor_vel) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_hor_vel, sizeof(topic.reject_hor_vel));
	buf.iterator += sizeof(topic.reject_hor_vel);
	buf.offset += sizeof(topic.reject_hor_vel);
	static_assert(sizeof(topic.reject_ver_vel) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_ver_vel, sizeof(topic.reject_ver_vel));
	buf.iterator += sizeof(topic.reject_ver_vel);
	buf.offset += sizeof(topic.reject_ver_vel);
	static_assert(sizeof(topic.reject_hor_pos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_hor_pos, sizeof(topic.reject_hor_pos));
	buf.iterator += sizeof(topic.reject_hor_pos);
	buf.offset += sizeof(topic.reject_hor_pos);
	static_assert(sizeof(topic.reject_ver_pos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_ver_pos, sizeof(topic.reject_ver_pos));
	buf.iterator += sizeof(topic.reject_ver_pos);
	buf.offset += sizeof(topic.reject_ver_pos);
	static_assert(sizeof(topic.reject_yaw) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_yaw, sizeof(topic.reject_yaw));
	buf.iterator += sizeof(topic.reject_yaw);
	buf.offset += sizeof(topic.reject_yaw);
	static_assert(sizeof(topic.reject_airspeed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_airspeed, sizeof(topic.reject_airspeed));
	buf.iterator += sizeof(topic.reject_airspeed);
	buf.offset += sizeof(topic.reject_airspeed);
	static_assert(sizeof(topic.reject_sideslip) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_sideslip, sizeof(topic.reject_sideslip));
	buf.iterator += sizeof(topic.reject_sideslip);
	buf.offset += sizeof(topic.reject_sideslip);
	static_assert(sizeof(topic.reject_hagl) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_hagl, sizeof(topic.reject_hagl));
	buf.iterator += sizeof(topic.reject_hagl);
	buf.offset += sizeof(topic.reject_hagl);
	static_assert(sizeof(topic.reject_optflow_x) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_optflow_x, sizeof(topic.reject_optflow_x));
	buf.iterator += sizeof(topic.reject_optflow_x);
	buf.offset += sizeof(topic.reject_optflow_x);
	static_assert(sizeof(topic.reject_optflow_y) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reject_optflow_y, sizeof(topic.reject_optflow_y));
	buf.iterator += sizeof(topic.reject_optflow_y);
	buf.offset += sizeof(topic.reject_optflow_y);
	return true;
}

bool ucdr_deserialize_estimator_status_flags(ucdrBuffer& buf, estimator_status_flags_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.control_status_changes) == 4, "size mismatch");
	memcpy(&topic.control_status_changes, buf.iterator, sizeof(topic.control_status_changes));
	buf.iterator += sizeof(topic.control_status_changes);
	buf.offset += sizeof(topic.control_status_changes);
	static_assert(sizeof(topic.cs_tilt_align) == 1, "size mismatch");
	memcpy(&topic.cs_tilt_align, buf.iterator, sizeof(topic.cs_tilt_align));
	buf.iterator += sizeof(topic.cs_tilt_align);
	buf.offset += sizeof(topic.cs_tilt_align);
	static_assert(sizeof(topic.cs_yaw_align) == 1, "size mismatch");
	memcpy(&topic.cs_yaw_align, buf.iterator, sizeof(topic.cs_yaw_align));
	buf.iterator += sizeof(topic.cs_yaw_align);
	buf.offset += sizeof(topic.cs_yaw_align);
	static_assert(sizeof(topic.cs_gps) == 1, "size mismatch");
	memcpy(&topic.cs_gps, buf.iterator, sizeof(topic.cs_gps));
	buf.iterator += sizeof(topic.cs_gps);
	buf.offset += sizeof(topic.cs_gps);
	static_assert(sizeof(topic.cs_opt_flow) == 1, "size mismatch");
	memcpy(&topic.cs_opt_flow, buf.iterator, sizeof(topic.cs_opt_flow));
	buf.iterator += sizeof(topic.cs_opt_flow);
	buf.offset += sizeof(topic.cs_opt_flow);
	static_assert(sizeof(topic.cs_mag_hdg) == 1, "size mismatch");
	memcpy(&topic.cs_mag_hdg, buf.iterator, sizeof(topic.cs_mag_hdg));
	buf.iterator += sizeof(topic.cs_mag_hdg);
	buf.offset += sizeof(topic.cs_mag_hdg);
	static_assert(sizeof(topic.cs_mag_3d) == 1, "size mismatch");
	memcpy(&topic.cs_mag_3d, buf.iterator, sizeof(topic.cs_mag_3d));
	buf.iterator += sizeof(topic.cs_mag_3d);
	buf.offset += sizeof(topic.cs_mag_3d);
	static_assert(sizeof(topic.cs_mag_dec) == 1, "size mismatch");
	memcpy(&topic.cs_mag_dec, buf.iterator, sizeof(topic.cs_mag_dec));
	buf.iterator += sizeof(topic.cs_mag_dec);
	buf.offset += sizeof(topic.cs_mag_dec);
	static_assert(sizeof(topic.cs_in_air) == 1, "size mismatch");
	memcpy(&topic.cs_in_air, buf.iterator, sizeof(topic.cs_in_air));
	buf.iterator += sizeof(topic.cs_in_air);
	buf.offset += sizeof(topic.cs_in_air);
	static_assert(sizeof(topic.cs_wind) == 1, "size mismatch");
	memcpy(&topic.cs_wind, buf.iterator, sizeof(topic.cs_wind));
	buf.iterator += sizeof(topic.cs_wind);
	buf.offset += sizeof(topic.cs_wind);
	static_assert(sizeof(topic.cs_baro_hgt) == 1, "size mismatch");
	memcpy(&topic.cs_baro_hgt, buf.iterator, sizeof(topic.cs_baro_hgt));
	buf.iterator += sizeof(topic.cs_baro_hgt);
	buf.offset += sizeof(topic.cs_baro_hgt);
	static_assert(sizeof(topic.cs_rng_hgt) == 1, "size mismatch");
	memcpy(&topic.cs_rng_hgt, buf.iterator, sizeof(topic.cs_rng_hgt));
	buf.iterator += sizeof(topic.cs_rng_hgt);
	buf.offset += sizeof(topic.cs_rng_hgt);
	static_assert(sizeof(topic.cs_gps_hgt) == 1, "size mismatch");
	memcpy(&topic.cs_gps_hgt, buf.iterator, sizeof(topic.cs_gps_hgt));
	buf.iterator += sizeof(topic.cs_gps_hgt);
	buf.offset += sizeof(topic.cs_gps_hgt);
	static_assert(sizeof(topic.cs_ev_pos) == 1, "size mismatch");
	memcpy(&topic.cs_ev_pos, buf.iterator, sizeof(topic.cs_ev_pos));
	buf.iterator += sizeof(topic.cs_ev_pos);
	buf.offset += sizeof(topic.cs_ev_pos);
	static_assert(sizeof(topic.cs_ev_yaw) == 1, "size mismatch");
	memcpy(&topic.cs_ev_yaw, buf.iterator, sizeof(topic.cs_ev_yaw));
	buf.iterator += sizeof(topic.cs_ev_yaw);
	buf.offset += sizeof(topic.cs_ev_yaw);
	static_assert(sizeof(topic.cs_ev_hgt) == 1, "size mismatch");
	memcpy(&topic.cs_ev_hgt, buf.iterator, sizeof(topic.cs_ev_hgt));
	buf.iterator += sizeof(topic.cs_ev_hgt);
	buf.offset += sizeof(topic.cs_ev_hgt);
	static_assert(sizeof(topic.cs_fuse_beta) == 1, "size mismatch");
	memcpy(&topic.cs_fuse_beta, buf.iterator, sizeof(topic.cs_fuse_beta));
	buf.iterator += sizeof(topic.cs_fuse_beta);
	buf.offset += sizeof(topic.cs_fuse_beta);
	static_assert(sizeof(topic.cs_mag_field_disturbed) == 1, "size mismatch");
	memcpy(&topic.cs_mag_field_disturbed, buf.iterator, sizeof(topic.cs_mag_field_disturbed));
	buf.iterator += sizeof(topic.cs_mag_field_disturbed);
	buf.offset += sizeof(topic.cs_mag_field_disturbed);
	static_assert(sizeof(topic.cs_fixed_wing) == 1, "size mismatch");
	memcpy(&topic.cs_fixed_wing, buf.iterator, sizeof(topic.cs_fixed_wing));
	buf.iterator += sizeof(topic.cs_fixed_wing);
	buf.offset += sizeof(topic.cs_fixed_wing);
	static_assert(sizeof(topic.cs_mag_fault) == 1, "size mismatch");
	memcpy(&topic.cs_mag_fault, buf.iterator, sizeof(topic.cs_mag_fault));
	buf.iterator += sizeof(topic.cs_mag_fault);
	buf.offset += sizeof(topic.cs_mag_fault);
	static_assert(sizeof(topic.cs_fuse_aspd) == 1, "size mismatch");
	memcpy(&topic.cs_fuse_aspd, buf.iterator, sizeof(topic.cs_fuse_aspd));
	buf.iterator += sizeof(topic.cs_fuse_aspd);
	buf.offset += sizeof(topic.cs_fuse_aspd);
	static_assert(sizeof(topic.cs_gnd_effect) == 1, "size mismatch");
	memcpy(&topic.cs_gnd_effect, buf.iterator, sizeof(topic.cs_gnd_effect));
	buf.iterator += sizeof(topic.cs_gnd_effect);
	buf.offset += sizeof(topic.cs_gnd_effect);
	static_assert(sizeof(topic.cs_rng_stuck) == 1, "size mismatch");
	memcpy(&topic.cs_rng_stuck, buf.iterator, sizeof(topic.cs_rng_stuck));
	buf.iterator += sizeof(topic.cs_rng_stuck);
	buf.offset += sizeof(topic.cs_rng_stuck);
	static_assert(sizeof(topic.cs_gps_yaw) == 1, "size mismatch");
	memcpy(&topic.cs_gps_yaw, buf.iterator, sizeof(topic.cs_gps_yaw));
	buf.iterator += sizeof(topic.cs_gps_yaw);
	buf.offset += sizeof(topic.cs_gps_yaw);
	static_assert(sizeof(topic.cs_mag_aligned_in_flight) == 1, "size mismatch");
	memcpy(&topic.cs_mag_aligned_in_flight, buf.iterator, sizeof(topic.cs_mag_aligned_in_flight));
	buf.iterator += sizeof(topic.cs_mag_aligned_in_flight);
	buf.offset += sizeof(topic.cs_mag_aligned_in_flight);
	static_assert(sizeof(topic.cs_ev_vel) == 1, "size mismatch");
	memcpy(&topic.cs_ev_vel, buf.iterator, sizeof(topic.cs_ev_vel));
	buf.iterator += sizeof(topic.cs_ev_vel);
	buf.offset += sizeof(topic.cs_ev_vel);
	static_assert(sizeof(topic.cs_synthetic_mag_z) == 1, "size mismatch");
	memcpy(&topic.cs_synthetic_mag_z, buf.iterator, sizeof(topic.cs_synthetic_mag_z));
	buf.iterator += sizeof(topic.cs_synthetic_mag_z);
	buf.offset += sizeof(topic.cs_synthetic_mag_z);
	static_assert(sizeof(topic.cs_vehicle_at_rest) == 1, "size mismatch");
	memcpy(&topic.cs_vehicle_at_rest, buf.iterator, sizeof(topic.cs_vehicle_at_rest));
	buf.iterator += sizeof(topic.cs_vehicle_at_rest);
	buf.offset += sizeof(topic.cs_vehicle_at_rest);
	static_assert(sizeof(topic.cs_gps_yaw_fault) == 1, "size mismatch");
	memcpy(&topic.cs_gps_yaw_fault, buf.iterator, sizeof(topic.cs_gps_yaw_fault));
	buf.iterator += sizeof(topic.cs_gps_yaw_fault);
	buf.offset += sizeof(topic.cs_gps_yaw_fault);
	static_assert(sizeof(topic.cs_rng_fault) == 1, "size mismatch");
	memcpy(&topic.cs_rng_fault, buf.iterator, sizeof(topic.cs_rng_fault));
	buf.iterator += sizeof(topic.cs_rng_fault);
	buf.offset += sizeof(topic.cs_rng_fault);
	static_assert(sizeof(topic.cs_inertial_dead_reckoning) == 1, "size mismatch");
	memcpy(&topic.cs_inertial_dead_reckoning, buf.iterator, sizeof(topic.cs_inertial_dead_reckoning));
	buf.iterator += sizeof(topic.cs_inertial_dead_reckoning);
	buf.offset += sizeof(topic.cs_inertial_dead_reckoning);
	static_assert(sizeof(topic.cs_wind_dead_reckoning) == 1, "size mismatch");
	memcpy(&topic.cs_wind_dead_reckoning, buf.iterator, sizeof(topic.cs_wind_dead_reckoning));
	buf.iterator += sizeof(topic.cs_wind_dead_reckoning);
	buf.offset += sizeof(topic.cs_wind_dead_reckoning);
	static_assert(sizeof(topic.cs_rng_kin_consistent) == 1, "size mismatch");
	memcpy(&topic.cs_rng_kin_consistent, buf.iterator, sizeof(topic.cs_rng_kin_consistent));
	buf.iterator += sizeof(topic.cs_rng_kin_consistent);
	buf.offset += sizeof(topic.cs_rng_kin_consistent);
	static_assert(sizeof(topic.cs_fake_pos) == 1, "size mismatch");
	memcpy(&topic.cs_fake_pos, buf.iterator, sizeof(topic.cs_fake_pos));
	buf.iterator += sizeof(topic.cs_fake_pos);
	buf.offset += sizeof(topic.cs_fake_pos);
	static_assert(sizeof(topic.cs_fake_hgt) == 1, "size mismatch");
	memcpy(&topic.cs_fake_hgt, buf.iterator, sizeof(topic.cs_fake_hgt));
	buf.iterator += sizeof(topic.cs_fake_hgt);
	buf.offset += sizeof(topic.cs_fake_hgt);
	static_assert(sizeof(topic.cs_gravity_vector) == 1, "size mismatch");
	memcpy(&topic.cs_gravity_vector, buf.iterator, sizeof(topic.cs_gravity_vector));
	buf.iterator += sizeof(topic.cs_gravity_vector);
	buf.offset += sizeof(topic.cs_gravity_vector);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.fault_status_changes) == 4, "size mismatch");
	memcpy(&topic.fault_status_changes, buf.iterator, sizeof(topic.fault_status_changes));
	buf.iterator += sizeof(topic.fault_status_changes);
	buf.offset += sizeof(topic.fault_status_changes);
	static_assert(sizeof(topic.fs_bad_mag_x) == 1, "size mismatch");
	memcpy(&topic.fs_bad_mag_x, buf.iterator, sizeof(topic.fs_bad_mag_x));
	buf.iterator += sizeof(topic.fs_bad_mag_x);
	buf.offset += sizeof(topic.fs_bad_mag_x);
	static_assert(sizeof(topic.fs_bad_mag_y) == 1, "size mismatch");
	memcpy(&topic.fs_bad_mag_y, buf.iterator, sizeof(topic.fs_bad_mag_y));
	buf.iterator += sizeof(topic.fs_bad_mag_y);
	buf.offset += sizeof(topic.fs_bad_mag_y);
	static_assert(sizeof(topic.fs_bad_mag_z) == 1, "size mismatch");
	memcpy(&topic.fs_bad_mag_z, buf.iterator, sizeof(topic.fs_bad_mag_z));
	buf.iterator += sizeof(topic.fs_bad_mag_z);
	buf.offset += sizeof(topic.fs_bad_mag_z);
	static_assert(sizeof(topic.fs_bad_hdg) == 1, "size mismatch");
	memcpy(&topic.fs_bad_hdg, buf.iterator, sizeof(topic.fs_bad_hdg));
	buf.iterator += sizeof(topic.fs_bad_hdg);
	buf.offset += sizeof(topic.fs_bad_hdg);
	static_assert(sizeof(topic.fs_bad_mag_decl) == 1, "size mismatch");
	memcpy(&topic.fs_bad_mag_decl, buf.iterator, sizeof(topic.fs_bad_mag_decl));
	buf.iterator += sizeof(topic.fs_bad_mag_decl);
	buf.offset += sizeof(topic.fs_bad_mag_decl);
	static_assert(sizeof(topic.fs_bad_airspeed) == 1, "size mismatch");
	memcpy(&topic.fs_bad_airspeed, buf.iterator, sizeof(topic.fs_bad_airspeed));
	buf.iterator += sizeof(topic.fs_bad_airspeed);
	buf.offset += sizeof(topic.fs_bad_airspeed);
	static_assert(sizeof(topic.fs_bad_sideslip) == 1, "size mismatch");
	memcpy(&topic.fs_bad_sideslip, buf.iterator, sizeof(topic.fs_bad_sideslip));
	buf.iterator += sizeof(topic.fs_bad_sideslip);
	buf.offset += sizeof(topic.fs_bad_sideslip);
	static_assert(sizeof(topic.fs_bad_optflow_x) == 1, "size mismatch");
	memcpy(&topic.fs_bad_optflow_x, buf.iterator, sizeof(topic.fs_bad_optflow_x));
	buf.iterator += sizeof(topic.fs_bad_optflow_x);
	buf.offset += sizeof(topic.fs_bad_optflow_x);
	static_assert(sizeof(topic.fs_bad_optflow_y) == 1, "size mismatch");
	memcpy(&topic.fs_bad_optflow_y, buf.iterator, sizeof(topic.fs_bad_optflow_y));
	buf.iterator += sizeof(topic.fs_bad_optflow_y);
	buf.offset += sizeof(topic.fs_bad_optflow_y);
	static_assert(sizeof(topic.fs_bad_vel_n) == 1, "size mismatch");
	memcpy(&topic.fs_bad_vel_n, buf.iterator, sizeof(topic.fs_bad_vel_n));
	buf.iterator += sizeof(topic.fs_bad_vel_n);
	buf.offset += sizeof(topic.fs_bad_vel_n);
	static_assert(sizeof(topic.fs_bad_vel_e) == 1, "size mismatch");
	memcpy(&topic.fs_bad_vel_e, buf.iterator, sizeof(topic.fs_bad_vel_e));
	buf.iterator += sizeof(topic.fs_bad_vel_e);
	buf.offset += sizeof(topic.fs_bad_vel_e);
	static_assert(sizeof(topic.fs_bad_vel_d) == 1, "size mismatch");
	memcpy(&topic.fs_bad_vel_d, buf.iterator, sizeof(topic.fs_bad_vel_d));
	buf.iterator += sizeof(topic.fs_bad_vel_d);
	buf.offset += sizeof(topic.fs_bad_vel_d);
	static_assert(sizeof(topic.fs_bad_pos_n) == 1, "size mismatch");
	memcpy(&topic.fs_bad_pos_n, buf.iterator, sizeof(topic.fs_bad_pos_n));
	buf.iterator += sizeof(topic.fs_bad_pos_n);
	buf.offset += sizeof(topic.fs_bad_pos_n);
	static_assert(sizeof(topic.fs_bad_pos_e) == 1, "size mismatch");
	memcpy(&topic.fs_bad_pos_e, buf.iterator, sizeof(topic.fs_bad_pos_e));
	buf.iterator += sizeof(topic.fs_bad_pos_e);
	buf.offset += sizeof(topic.fs_bad_pos_e);
	static_assert(sizeof(topic.fs_bad_pos_d) == 1, "size mismatch");
	memcpy(&topic.fs_bad_pos_d, buf.iterator, sizeof(topic.fs_bad_pos_d));
	buf.iterator += sizeof(topic.fs_bad_pos_d);
	buf.offset += sizeof(topic.fs_bad_pos_d);
	static_assert(sizeof(topic.fs_bad_acc_bias) == 1, "size mismatch");
	memcpy(&topic.fs_bad_acc_bias, buf.iterator, sizeof(topic.fs_bad_acc_bias));
	buf.iterator += sizeof(topic.fs_bad_acc_bias);
	buf.offset += sizeof(topic.fs_bad_acc_bias);
	static_assert(sizeof(topic.fs_bad_acc_vertical) == 1, "size mismatch");
	memcpy(&topic.fs_bad_acc_vertical, buf.iterator, sizeof(topic.fs_bad_acc_vertical));
	buf.iterator += sizeof(topic.fs_bad_acc_vertical);
	buf.offset += sizeof(topic.fs_bad_acc_vertical);
	static_assert(sizeof(topic.fs_bad_acc_clipping) == 1, "size mismatch");
	memcpy(&topic.fs_bad_acc_clipping, buf.iterator, sizeof(topic.fs_bad_acc_clipping));
	buf.iterator += sizeof(topic.fs_bad_acc_clipping);
	buf.offset += sizeof(topic.fs_bad_acc_clipping);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.innovation_fault_status_changes) == 4, "size mismatch");
	memcpy(&topic.innovation_fault_status_changes, buf.iterator, sizeof(topic.innovation_fault_status_changes));
	buf.iterator += sizeof(topic.innovation_fault_status_changes);
	buf.offset += sizeof(topic.innovation_fault_status_changes);
	static_assert(sizeof(topic.reject_hor_vel) == 1, "size mismatch");
	memcpy(&topic.reject_hor_vel, buf.iterator, sizeof(topic.reject_hor_vel));
	buf.iterator += sizeof(topic.reject_hor_vel);
	buf.offset += sizeof(topic.reject_hor_vel);
	static_assert(sizeof(topic.reject_ver_vel) == 1, "size mismatch");
	memcpy(&topic.reject_ver_vel, buf.iterator, sizeof(topic.reject_ver_vel));
	buf.iterator += sizeof(topic.reject_ver_vel);
	buf.offset += sizeof(topic.reject_ver_vel);
	static_assert(sizeof(topic.reject_hor_pos) == 1, "size mismatch");
	memcpy(&topic.reject_hor_pos, buf.iterator, sizeof(topic.reject_hor_pos));
	buf.iterator += sizeof(topic.reject_hor_pos);
	buf.offset += sizeof(topic.reject_hor_pos);
	static_assert(sizeof(topic.reject_ver_pos) == 1, "size mismatch");
	memcpy(&topic.reject_ver_pos, buf.iterator, sizeof(topic.reject_ver_pos));
	buf.iterator += sizeof(topic.reject_ver_pos);
	buf.offset += sizeof(topic.reject_ver_pos);
	static_assert(sizeof(topic.reject_yaw) == 1, "size mismatch");
	memcpy(&topic.reject_yaw, buf.iterator, sizeof(topic.reject_yaw));
	buf.iterator += sizeof(topic.reject_yaw);
	buf.offset += sizeof(topic.reject_yaw);
	static_assert(sizeof(topic.reject_airspeed) == 1, "size mismatch");
	memcpy(&topic.reject_airspeed, buf.iterator, sizeof(topic.reject_airspeed));
	buf.iterator += sizeof(topic.reject_airspeed);
	buf.offset += sizeof(topic.reject_airspeed);
	static_assert(sizeof(topic.reject_sideslip) == 1, "size mismatch");
	memcpy(&topic.reject_sideslip, buf.iterator, sizeof(topic.reject_sideslip));
	buf.iterator += sizeof(topic.reject_sideslip);
	buf.offset += sizeof(topic.reject_sideslip);
	static_assert(sizeof(topic.reject_hagl) == 1, "size mismatch");
	memcpy(&topic.reject_hagl, buf.iterator, sizeof(topic.reject_hagl));
	buf.iterator += sizeof(topic.reject_hagl);
	buf.offset += sizeof(topic.reject_hagl);
	static_assert(sizeof(topic.reject_optflow_x) == 1, "size mismatch");
	memcpy(&topic.reject_optflow_x, buf.iterator, sizeof(topic.reject_optflow_x));
	buf.iterator += sizeof(topic.reject_optflow_x);
	buf.offset += sizeof(topic.reject_optflow_x);
	static_assert(sizeof(topic.reject_optflow_y) == 1, "size mismatch");
	memcpy(&topic.reject_optflow_y, buf.iterator, sizeof(topic.reject_optflow_y));
	buf.iterator += sizeof(topic.reject_optflow_y);
	buf.offset += sizeof(topic.reject_optflow_y);
	return true;
}
