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
#include <uORB/topics/landing_target_pose.h>


static inline constexpr int ucdr_topic_size_landing_target_pose()
{
	return 64;
}

bool ucdr_serialize_landing_target_pose(const landing_target_pose_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.is_static) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.is_static, sizeof(topic.is_static));
	buf.iterator += sizeof(topic.is_static);
	buf.offset += sizeof(topic.is_static);
	static_assert(sizeof(topic.rel_pos_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rel_pos_valid, sizeof(topic.rel_pos_valid));
	buf.iterator += sizeof(topic.rel_pos_valid);
	buf.offset += sizeof(topic.rel_pos_valid);
	static_assert(sizeof(topic.rel_vel_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rel_vel_valid, sizeof(topic.rel_vel_valid));
	buf.iterator += sizeof(topic.rel_vel_valid);
	buf.offset += sizeof(topic.rel_vel_valid);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.x_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.x_rel, sizeof(topic.x_rel));
	buf.iterator += sizeof(topic.x_rel);
	buf.offset += sizeof(topic.x_rel);
	static_assert(sizeof(topic.y_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.y_rel, sizeof(topic.y_rel));
	buf.iterator += sizeof(topic.y_rel);
	buf.offset += sizeof(topic.y_rel);
	static_assert(sizeof(topic.z_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.z_rel, sizeof(topic.z_rel));
	buf.iterator += sizeof(topic.z_rel);
	buf.offset += sizeof(topic.z_rel);
	static_assert(sizeof(topic.vx_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vx_rel, sizeof(topic.vx_rel));
	buf.iterator += sizeof(topic.vx_rel);
	buf.offset += sizeof(topic.vx_rel);
	static_assert(sizeof(topic.vy_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.vy_rel, sizeof(topic.vy_rel));
	buf.iterator += sizeof(topic.vy_rel);
	buf.offset += sizeof(topic.vy_rel);
	static_assert(sizeof(topic.cov_x_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cov_x_rel, sizeof(topic.cov_x_rel));
	buf.iterator += sizeof(topic.cov_x_rel);
	buf.offset += sizeof(topic.cov_x_rel);
	static_assert(sizeof(topic.cov_y_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cov_y_rel, sizeof(topic.cov_y_rel));
	buf.iterator += sizeof(topic.cov_y_rel);
	buf.offset += sizeof(topic.cov_y_rel);
	static_assert(sizeof(topic.cov_vx_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cov_vx_rel, sizeof(topic.cov_vx_rel));
	buf.iterator += sizeof(topic.cov_vx_rel);
	buf.offset += sizeof(topic.cov_vx_rel);
	static_assert(sizeof(topic.cov_vy_rel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.cov_vy_rel, sizeof(topic.cov_vy_rel));
	buf.iterator += sizeof(topic.cov_vy_rel);
	buf.offset += sizeof(topic.cov_vy_rel);
	static_assert(sizeof(topic.abs_pos_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.abs_pos_valid, sizeof(topic.abs_pos_valid));
	buf.iterator += sizeof(topic.abs_pos_valid);
	buf.offset += sizeof(topic.abs_pos_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.x_abs) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.x_abs, sizeof(topic.x_abs));
	buf.iterator += sizeof(topic.x_abs);
	buf.offset += sizeof(topic.x_abs);
	static_assert(sizeof(topic.y_abs) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.y_abs, sizeof(topic.y_abs));
	buf.iterator += sizeof(topic.y_abs);
	buf.offset += sizeof(topic.y_abs);
	static_assert(sizeof(topic.z_abs) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.z_abs, sizeof(topic.z_abs));
	buf.iterator += sizeof(topic.z_abs);
	buf.offset += sizeof(topic.z_abs);
	return true;
}

bool ucdr_deserialize_landing_target_pose(ucdrBuffer& buf, landing_target_pose_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.is_static) == 1, "size mismatch");
	memcpy(&topic.is_static, buf.iterator, sizeof(topic.is_static));
	buf.iterator += sizeof(topic.is_static);
	buf.offset += sizeof(topic.is_static);
	static_assert(sizeof(topic.rel_pos_valid) == 1, "size mismatch");
	memcpy(&topic.rel_pos_valid, buf.iterator, sizeof(topic.rel_pos_valid));
	buf.iterator += sizeof(topic.rel_pos_valid);
	buf.offset += sizeof(topic.rel_pos_valid);
	static_assert(sizeof(topic.rel_vel_valid) == 1, "size mismatch");
	memcpy(&topic.rel_vel_valid, buf.iterator, sizeof(topic.rel_vel_valid));
	buf.iterator += sizeof(topic.rel_vel_valid);
	buf.offset += sizeof(topic.rel_vel_valid);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.x_rel) == 4, "size mismatch");
	memcpy(&topic.x_rel, buf.iterator, sizeof(topic.x_rel));
	buf.iterator += sizeof(topic.x_rel);
	buf.offset += sizeof(topic.x_rel);
	static_assert(sizeof(topic.y_rel) == 4, "size mismatch");
	memcpy(&topic.y_rel, buf.iterator, sizeof(topic.y_rel));
	buf.iterator += sizeof(topic.y_rel);
	buf.offset += sizeof(topic.y_rel);
	static_assert(sizeof(topic.z_rel) == 4, "size mismatch");
	memcpy(&topic.z_rel, buf.iterator, sizeof(topic.z_rel));
	buf.iterator += sizeof(topic.z_rel);
	buf.offset += sizeof(topic.z_rel);
	static_assert(sizeof(topic.vx_rel) == 4, "size mismatch");
	memcpy(&topic.vx_rel, buf.iterator, sizeof(topic.vx_rel));
	buf.iterator += sizeof(topic.vx_rel);
	buf.offset += sizeof(topic.vx_rel);
	static_assert(sizeof(topic.vy_rel) == 4, "size mismatch");
	memcpy(&topic.vy_rel, buf.iterator, sizeof(topic.vy_rel));
	buf.iterator += sizeof(topic.vy_rel);
	buf.offset += sizeof(topic.vy_rel);
	static_assert(sizeof(topic.cov_x_rel) == 4, "size mismatch");
	memcpy(&topic.cov_x_rel, buf.iterator, sizeof(topic.cov_x_rel));
	buf.iterator += sizeof(topic.cov_x_rel);
	buf.offset += sizeof(topic.cov_x_rel);
	static_assert(sizeof(topic.cov_y_rel) == 4, "size mismatch");
	memcpy(&topic.cov_y_rel, buf.iterator, sizeof(topic.cov_y_rel));
	buf.iterator += sizeof(topic.cov_y_rel);
	buf.offset += sizeof(topic.cov_y_rel);
	static_assert(sizeof(topic.cov_vx_rel) == 4, "size mismatch");
	memcpy(&topic.cov_vx_rel, buf.iterator, sizeof(topic.cov_vx_rel));
	buf.iterator += sizeof(topic.cov_vx_rel);
	buf.offset += sizeof(topic.cov_vx_rel);
	static_assert(sizeof(topic.cov_vy_rel) == 4, "size mismatch");
	memcpy(&topic.cov_vy_rel, buf.iterator, sizeof(topic.cov_vy_rel));
	buf.iterator += sizeof(topic.cov_vy_rel);
	buf.offset += sizeof(topic.cov_vy_rel);
	static_assert(sizeof(topic.abs_pos_valid) == 1, "size mismatch");
	memcpy(&topic.abs_pos_valid, buf.iterator, sizeof(topic.abs_pos_valid));
	buf.iterator += sizeof(topic.abs_pos_valid);
	buf.offset += sizeof(topic.abs_pos_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.x_abs) == 4, "size mismatch");
	memcpy(&topic.x_abs, buf.iterator, sizeof(topic.x_abs));
	buf.iterator += sizeof(topic.x_abs);
	buf.offset += sizeof(topic.x_abs);
	static_assert(sizeof(topic.y_abs) == 4, "size mismatch");
	memcpy(&topic.y_abs, buf.iterator, sizeof(topic.y_abs));
	buf.iterator += sizeof(topic.y_abs);
	buf.offset += sizeof(topic.y_abs);
	static_assert(sizeof(topic.z_abs) == 4, "size mismatch");
	memcpy(&topic.z_abs, buf.iterator, sizeof(topic.z_abs));
	buf.iterator += sizeof(topic.z_abs);
	buf.offset += sizeof(topic.z_abs);
	return true;
}
