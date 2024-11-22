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
#include <uORB/topics/estimator_aid_source3d.h>


static inline constexpr int ucdr_topic_size_estimator_aid_source3d()
{
	return 95;
}

bool ucdr_serialize_estimator_aid_source3d(const estimator_aid_source3d_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.estimator_instance) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.estimator_instance, sizeof(topic.estimator_instance));
	buf.iterator += sizeof(topic.estimator_instance);
	buf.offset += sizeof(topic.estimator_instance);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.device_id, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.time_last_fuse) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.time_last_fuse, sizeof(topic.time_last_fuse));
	buf.iterator += sizeof(topic.time_last_fuse);
	buf.offset += sizeof(topic.time_last_fuse);
	static_assert(sizeof(topic.observation) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.observation, sizeof(topic.observation));
	buf.iterator += sizeof(topic.observation);
	buf.offset += sizeof(topic.observation);
	static_assert(sizeof(topic.observation_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.observation_variance, sizeof(topic.observation_variance));
	buf.iterator += sizeof(topic.observation_variance);
	buf.offset += sizeof(topic.observation_variance);
	static_assert(sizeof(topic.innovation) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.innovation, sizeof(topic.innovation));
	buf.iterator += sizeof(topic.innovation);
	buf.offset += sizeof(topic.innovation);
	static_assert(sizeof(topic.innovation_variance) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.innovation_variance, sizeof(topic.innovation_variance));
	buf.iterator += sizeof(topic.innovation_variance);
	buf.offset += sizeof(topic.innovation_variance);
	static_assert(sizeof(topic.test_ratio) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.test_ratio, sizeof(topic.test_ratio));
	buf.iterator += sizeof(topic.test_ratio);
	buf.offset += sizeof(topic.test_ratio);
	static_assert(sizeof(topic.fusion_enabled) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fusion_enabled, sizeof(topic.fusion_enabled));
	buf.iterator += sizeof(topic.fusion_enabled);
	buf.offset += sizeof(topic.fusion_enabled);
	static_assert(sizeof(topic.innovation_rejected) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.innovation_rejected, sizeof(topic.innovation_rejected));
	buf.iterator += sizeof(topic.innovation_rejected);
	buf.offset += sizeof(topic.innovation_rejected);
	static_assert(sizeof(topic.fused) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.fused, sizeof(topic.fused));
	buf.iterator += sizeof(topic.fused);
	buf.offset += sizeof(topic.fused);
	return true;
}

bool ucdr_deserialize_estimator_aid_source3d(ucdrBuffer& buf, estimator_aid_source3d_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.estimator_instance) == 1, "size mismatch");
	memcpy(&topic.estimator_instance, buf.iterator, sizeof(topic.estimator_instance));
	buf.iterator += sizeof(topic.estimator_instance);
	buf.offset += sizeof(topic.estimator_instance);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.device_id) == 4, "size mismatch");
	memcpy(&topic.device_id, buf.iterator, sizeof(topic.device_id));
	buf.iterator += sizeof(topic.device_id);
	buf.offset += sizeof(topic.device_id);
	static_assert(sizeof(topic.time_last_fuse) == 8, "size mismatch");
	memcpy(&topic.time_last_fuse, buf.iterator, sizeof(topic.time_last_fuse));
	buf.iterator += sizeof(topic.time_last_fuse);
	buf.offset += sizeof(topic.time_last_fuse);
	static_assert(sizeof(topic.observation) == 12, "size mismatch");
	memcpy(&topic.observation, buf.iterator, sizeof(topic.observation));
	buf.iterator += sizeof(topic.observation);
	buf.offset += sizeof(topic.observation);
	static_assert(sizeof(topic.observation_variance) == 12, "size mismatch");
	memcpy(&topic.observation_variance, buf.iterator, sizeof(topic.observation_variance));
	buf.iterator += sizeof(topic.observation_variance);
	buf.offset += sizeof(topic.observation_variance);
	static_assert(sizeof(topic.innovation) == 12, "size mismatch");
	memcpy(&topic.innovation, buf.iterator, sizeof(topic.innovation));
	buf.iterator += sizeof(topic.innovation);
	buf.offset += sizeof(topic.innovation);
	static_assert(sizeof(topic.innovation_variance) == 12, "size mismatch");
	memcpy(&topic.innovation_variance, buf.iterator, sizeof(topic.innovation_variance));
	buf.iterator += sizeof(topic.innovation_variance);
	buf.offset += sizeof(topic.innovation_variance);
	static_assert(sizeof(topic.test_ratio) == 12, "size mismatch");
	memcpy(&topic.test_ratio, buf.iterator, sizeof(topic.test_ratio));
	buf.iterator += sizeof(topic.test_ratio);
	buf.offset += sizeof(topic.test_ratio);
	static_assert(sizeof(topic.fusion_enabled) == 1, "size mismatch");
	memcpy(&topic.fusion_enabled, buf.iterator, sizeof(topic.fusion_enabled));
	buf.iterator += sizeof(topic.fusion_enabled);
	buf.offset += sizeof(topic.fusion_enabled);
	static_assert(sizeof(topic.innovation_rejected) == 1, "size mismatch");
	memcpy(&topic.innovation_rejected, buf.iterator, sizeof(topic.innovation_rejected));
	buf.iterator += sizeof(topic.innovation_rejected);
	buf.offset += sizeof(topic.innovation_rejected);
	static_assert(sizeof(topic.fused) == 1, "size mismatch");
	memcpy(&topic.fused, buf.iterator, sizeof(topic.fused));
	buf.iterator += sizeof(topic.fused);
	buf.offset += sizeof(topic.fused);
	return true;
}
