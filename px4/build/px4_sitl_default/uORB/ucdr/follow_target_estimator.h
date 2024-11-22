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
#include <uORB/topics/follow_target_estimator.h>


static inline constexpr int ucdr_topic_size_follow_target_estimator()
{
	return 96;
}

bool ucdr_serialize_follow_target_estimator(const follow_target_estimator_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.last_filter_reset_timestamp) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.last_filter_reset_timestamp, sizeof(topic.last_filter_reset_timestamp));
	buf.iterator += sizeof(topic.last_filter_reset_timestamp);
	buf.offset += sizeof(topic.last_filter_reset_timestamp);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.stale) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.stale, sizeof(topic.stale));
	buf.iterator += sizeof(topic.stale);
	buf.offset += sizeof(topic.stale);
	buf.iterator += 6; // padding
	buf.offset += 6; // padding
	static_assert(sizeof(topic.lat_est) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lat_est, sizeof(topic.lat_est));
	buf.iterator += sizeof(topic.lat_est);
	buf.offset += sizeof(topic.lat_est);
	static_assert(sizeof(topic.lon_est) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lon_est, sizeof(topic.lon_est));
	buf.iterator += sizeof(topic.lon_est);
	buf.offset += sizeof(topic.lon_est);
	static_assert(sizeof(topic.alt_est) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.alt_est, sizeof(topic.alt_est));
	buf.iterator += sizeof(topic.alt_est);
	buf.offset += sizeof(topic.alt_est);
	static_assert(sizeof(topic.pos_est) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.pos_est, sizeof(topic.pos_est));
	buf.iterator += sizeof(topic.pos_est);
	buf.offset += sizeof(topic.pos_est);
	static_assert(sizeof(topic.vel_est) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.vel_est, sizeof(topic.vel_est));
	buf.iterator += sizeof(topic.vel_est);
	buf.offset += sizeof(topic.vel_est);
	static_assert(sizeof(topic.acc_est) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.acc_est, sizeof(topic.acc_est));
	buf.iterator += sizeof(topic.acc_est);
	buf.offset += sizeof(topic.acc_est);
	static_assert(sizeof(topic.prediction_count) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.prediction_count, sizeof(topic.prediction_count));
	buf.iterator += sizeof(topic.prediction_count);
	buf.offset += sizeof(topic.prediction_count);
	static_assert(sizeof(topic.fusion_count) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.fusion_count, sizeof(topic.fusion_count));
	buf.iterator += sizeof(topic.fusion_count);
	buf.offset += sizeof(topic.fusion_count);
	return true;
}

bool ucdr_deserialize_follow_target_estimator(ucdrBuffer& buf, follow_target_estimator_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.last_filter_reset_timestamp) == 8, "size mismatch");
	memcpy(&topic.last_filter_reset_timestamp, buf.iterator, sizeof(topic.last_filter_reset_timestamp));
	buf.iterator += sizeof(topic.last_filter_reset_timestamp);
	buf.offset += sizeof(topic.last_filter_reset_timestamp);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(&topic.valid, buf.iterator, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.stale) == 1, "size mismatch");
	memcpy(&topic.stale, buf.iterator, sizeof(topic.stale));
	buf.iterator += sizeof(topic.stale);
	buf.offset += sizeof(topic.stale);
	buf.iterator += 6; // padding
	buf.offset += 6; // padding
	static_assert(sizeof(topic.lat_est) == 8, "size mismatch");
	memcpy(&topic.lat_est, buf.iterator, sizeof(topic.lat_est));
	buf.iterator += sizeof(topic.lat_est);
	buf.offset += sizeof(topic.lat_est);
	static_assert(sizeof(topic.lon_est) == 8, "size mismatch");
	memcpy(&topic.lon_est, buf.iterator, sizeof(topic.lon_est));
	buf.iterator += sizeof(topic.lon_est);
	buf.offset += sizeof(topic.lon_est);
	static_assert(sizeof(topic.alt_est) == 4, "size mismatch");
	memcpy(&topic.alt_est, buf.iterator, sizeof(topic.alt_est));
	buf.iterator += sizeof(topic.alt_est);
	buf.offset += sizeof(topic.alt_est);
	static_assert(sizeof(topic.pos_est) == 12, "size mismatch");
	memcpy(&topic.pos_est, buf.iterator, sizeof(topic.pos_est));
	buf.iterator += sizeof(topic.pos_est);
	buf.offset += sizeof(topic.pos_est);
	static_assert(sizeof(topic.vel_est) == 12, "size mismatch");
	memcpy(&topic.vel_est, buf.iterator, sizeof(topic.vel_est));
	buf.iterator += sizeof(topic.vel_est);
	buf.offset += sizeof(topic.vel_est);
	static_assert(sizeof(topic.acc_est) == 12, "size mismatch");
	memcpy(&topic.acc_est, buf.iterator, sizeof(topic.acc_est));
	buf.iterator += sizeof(topic.acc_est);
	buf.offset += sizeof(topic.acc_est);
	static_assert(sizeof(topic.prediction_count) == 8, "size mismatch");
	memcpy(&topic.prediction_count, buf.iterator, sizeof(topic.prediction_count));
	buf.iterator += sizeof(topic.prediction_count);
	buf.offset += sizeof(topic.prediction_count);
	static_assert(sizeof(topic.fusion_count) == 8, "size mismatch");
	memcpy(&topic.fusion_count, buf.iterator, sizeof(topic.fusion_count));
	buf.iterator += sizeof(topic.fusion_count);
	buf.offset += sizeof(topic.fusion_count);
	return true;
}
