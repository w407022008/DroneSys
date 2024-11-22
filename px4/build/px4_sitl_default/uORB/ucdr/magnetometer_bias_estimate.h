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
#include <uORB/topics/magnetometer_bias_estimate.h>


static inline constexpr int ucdr_topic_size_magnetometer_bias_estimate()
{
	return 64;
}

bool ucdr_serialize_magnetometer_bias_estimate(const magnetometer_bias_estimate_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.bias_x) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.bias_x, sizeof(topic.bias_x));
	buf.iterator += sizeof(topic.bias_x);
	buf.offset += sizeof(topic.bias_x);
	static_assert(sizeof(topic.bias_y) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.bias_y, sizeof(topic.bias_y));
	buf.iterator += sizeof(topic.bias_y);
	buf.offset += sizeof(topic.bias_y);
	static_assert(sizeof(topic.bias_z) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.bias_z, sizeof(topic.bias_z));
	buf.iterator += sizeof(topic.bias_z);
	buf.offset += sizeof(topic.bias_z);
	static_assert(sizeof(topic.valid) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.valid, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.stable) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.stable, sizeof(topic.stable));
	buf.iterator += sizeof(topic.stable);
	buf.offset += sizeof(topic.stable);
	return true;
}

bool ucdr_deserialize_magnetometer_bias_estimate(ucdrBuffer& buf, magnetometer_bias_estimate_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.bias_x) == 16, "size mismatch");
	memcpy(&topic.bias_x, buf.iterator, sizeof(topic.bias_x));
	buf.iterator += sizeof(topic.bias_x);
	buf.offset += sizeof(topic.bias_x);
	static_assert(sizeof(topic.bias_y) == 16, "size mismatch");
	memcpy(&topic.bias_y, buf.iterator, sizeof(topic.bias_y));
	buf.iterator += sizeof(topic.bias_y);
	buf.offset += sizeof(topic.bias_y);
	static_assert(sizeof(topic.bias_z) == 16, "size mismatch");
	memcpy(&topic.bias_z, buf.iterator, sizeof(topic.bias_z));
	buf.iterator += sizeof(topic.bias_z);
	buf.offset += sizeof(topic.bias_z);
	static_assert(sizeof(topic.valid) == 4, "size mismatch");
	memcpy(&topic.valid, buf.iterator, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.stable) == 4, "size mismatch");
	memcpy(&topic.stable, buf.iterator, sizeof(topic.stable));
	buf.iterator += sizeof(topic.stable);
	buf.offset += sizeof(topic.stable);
	return true;
}
