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
#include <uORB/topics/yaw_estimator_status.h>


static inline constexpr int ucdr_topic_size_yaw_estimator_status()
{
	return 108;
}

bool ucdr_serialize_yaw_estimator_status(const yaw_estimator_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.yaw_composite) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_composite, sizeof(topic.yaw_composite));
	buf.iterator += sizeof(topic.yaw_composite);
	buf.offset += sizeof(topic.yaw_composite);
	static_assert(sizeof(topic.yaw_variance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_variance, sizeof(topic.yaw_variance));
	buf.iterator += sizeof(topic.yaw_variance);
	buf.offset += sizeof(topic.yaw_variance);
	static_assert(sizeof(topic.yaw_composite_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.yaw_composite_valid, sizeof(topic.yaw_composite_valid));
	buf.iterator += sizeof(topic.yaw_composite_valid);
	buf.offset += sizeof(topic.yaw_composite_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.yaw) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.innov_vn) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.innov_vn, sizeof(topic.innov_vn));
	buf.iterator += sizeof(topic.innov_vn);
	buf.offset += sizeof(topic.innov_vn);
	static_assert(sizeof(topic.innov_ve) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.innov_ve, sizeof(topic.innov_ve));
	buf.iterator += sizeof(topic.innov_ve);
	buf.offset += sizeof(topic.innov_ve);
	static_assert(sizeof(topic.weight) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.weight, sizeof(topic.weight));
	buf.iterator += sizeof(topic.weight);
	buf.offset += sizeof(topic.weight);
	return true;
}

bool ucdr_deserialize_yaw_estimator_status(ucdrBuffer& buf, yaw_estimator_status_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.yaw_composite) == 4, "size mismatch");
	memcpy(&topic.yaw_composite, buf.iterator, sizeof(topic.yaw_composite));
	buf.iterator += sizeof(topic.yaw_composite);
	buf.offset += sizeof(topic.yaw_composite);
	static_assert(sizeof(topic.yaw_variance) == 4, "size mismatch");
	memcpy(&topic.yaw_variance, buf.iterator, sizeof(topic.yaw_variance));
	buf.iterator += sizeof(topic.yaw_variance);
	buf.offset += sizeof(topic.yaw_variance);
	static_assert(sizeof(topic.yaw_composite_valid) == 1, "size mismatch");
	memcpy(&topic.yaw_composite_valid, buf.iterator, sizeof(topic.yaw_composite_valid));
	buf.iterator += sizeof(topic.yaw_composite_valid);
	buf.offset += sizeof(topic.yaw_composite_valid);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.yaw) == 20, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.innov_vn) == 20, "size mismatch");
	memcpy(&topic.innov_vn, buf.iterator, sizeof(topic.innov_vn));
	buf.iterator += sizeof(topic.innov_vn);
	buf.offset += sizeof(topic.innov_vn);
	static_assert(sizeof(topic.innov_ve) == 20, "size mismatch");
	memcpy(&topic.innov_ve, buf.iterator, sizeof(topic.innov_ve));
	buf.iterator += sizeof(topic.innov_ve);
	buf.offset += sizeof(topic.innov_ve);
	static_assert(sizeof(topic.weight) == 20, "size mismatch");
	memcpy(&topic.weight, buf.iterator, sizeof(topic.weight));
	buf.iterator += sizeof(topic.weight);
	buf.offset += sizeof(topic.weight);
	return true;
}
