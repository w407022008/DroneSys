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
#include <uORB/topics/vehicle_odometry.h>


static inline constexpr int ucdr_topic_size_vehicle_odometry()
{
	return 114;
}

bool ucdr_serialize_vehicle_odometry(const vehicle_odometry_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.pose_frame) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.pose_frame, sizeof(topic.pose_frame));
	buf.iterator += sizeof(topic.pose_frame);
	buf.offset += sizeof(topic.pose_frame);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.position, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.q, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.velocity_frame) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.velocity_frame, sizeof(topic.velocity_frame));
	buf.iterator += sizeof(topic.velocity_frame);
	buf.offset += sizeof(topic.velocity_frame);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.velocity, sizeof(topic.velocity));
	buf.iterator += sizeof(topic.velocity);
	buf.offset += sizeof(topic.velocity);
	static_assert(sizeof(topic.angular_velocity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.angular_velocity, sizeof(topic.angular_velocity));
	buf.iterator += sizeof(topic.angular_velocity);
	buf.offset += sizeof(topic.angular_velocity);
	static_assert(sizeof(topic.position_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.position_variance, sizeof(topic.position_variance));
	buf.iterator += sizeof(topic.position_variance);
	buf.offset += sizeof(topic.position_variance);
	static_assert(sizeof(topic.orientation_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.orientation_variance, sizeof(topic.orientation_variance));
	buf.iterator += sizeof(topic.orientation_variance);
	buf.offset += sizeof(topic.orientation_variance);
	static_assert(sizeof(topic.velocity_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.velocity_variance, sizeof(topic.velocity_variance));
	buf.iterator += sizeof(topic.velocity_variance);
	buf.offset += sizeof(topic.velocity_variance);
	static_assert(sizeof(topic.reset_counter) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.reset_counter, sizeof(topic.reset_counter));
	buf.iterator += sizeof(topic.reset_counter);
	buf.offset += sizeof(topic.reset_counter);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.quality, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	return true;
}

bool ucdr_deserialize_vehicle_odometry(ucdrBuffer& buf, vehicle_odometry_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.pose_frame) == 1, "size mismatch");
	memcpy(&topic.pose_frame, buf.iterator, sizeof(topic.pose_frame));
	buf.iterator += sizeof(topic.pose_frame);
	buf.offset += sizeof(topic.pose_frame);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.position) == 12, "size mismatch");
	memcpy(&topic.position, buf.iterator, sizeof(topic.position));
	buf.iterator += sizeof(topic.position);
	buf.offset += sizeof(topic.position);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(&topic.q, buf.iterator, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.velocity_frame) == 1, "size mismatch");
	memcpy(&topic.velocity_frame, buf.iterator, sizeof(topic.velocity_frame));
	buf.iterator += sizeof(topic.velocity_frame);
	buf.offset += sizeof(topic.velocity_frame);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.velocity) == 12, "size mismatch");
	memcpy(&topic.velocity, buf.iterator, sizeof(topic.velocity));
	buf.iterator += sizeof(topic.velocity);
	buf.offset += sizeof(topic.velocity);
	static_assert(sizeof(topic.angular_velocity) == 12, "size mismatch");
	memcpy(&topic.angular_velocity, buf.iterator, sizeof(topic.angular_velocity));
	buf.iterator += sizeof(topic.angular_velocity);
	buf.offset += sizeof(topic.angular_velocity);
	static_assert(sizeof(topic.position_variance) == 12, "size mismatch");
	memcpy(&topic.position_variance, buf.iterator, sizeof(topic.position_variance));
	buf.iterator += sizeof(topic.position_variance);
	buf.offset += sizeof(topic.position_variance);
	static_assert(sizeof(topic.orientation_variance) == 12, "size mismatch");
	memcpy(&topic.orientation_variance, buf.iterator, sizeof(topic.orientation_variance));
	buf.iterator += sizeof(topic.orientation_variance);
	buf.offset += sizeof(topic.orientation_variance);
	static_assert(sizeof(topic.velocity_variance) == 12, "size mismatch");
	memcpy(&topic.velocity_variance, buf.iterator, sizeof(topic.velocity_variance));
	buf.iterator += sizeof(topic.velocity_variance);
	buf.offset += sizeof(topic.velocity_variance);
	static_assert(sizeof(topic.reset_counter) == 1, "size mismatch");
	memcpy(&topic.reset_counter, buf.iterator, sizeof(topic.reset_counter));
	buf.iterator += sizeof(topic.reset_counter);
	buf.offset += sizeof(topic.reset_counter);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(&topic.quality, buf.iterator, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	return true;
}
