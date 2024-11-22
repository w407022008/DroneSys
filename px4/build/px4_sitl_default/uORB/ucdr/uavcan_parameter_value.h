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
#include <uORB/topics/uavcan_parameter_value.h>


static inline constexpr int ucdr_topic_size_uavcan_parameter_value()
{
	return 44;
}

bool ucdr_serialize_uavcan_parameter_value(const uavcan_parameter_value_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.node_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.node_id, sizeof(topic.node_id));
	buf.iterator += sizeof(topic.node_id);
	buf.offset += sizeof(topic.node_id);
	static_assert(sizeof(topic.param_id) == 17, "size mismatch");
	memcpy(buf.iterator, &topic.param_id, sizeof(topic.param_id));
	buf.iterator += sizeof(topic.param_id);
	buf.offset += sizeof(topic.param_id);
	static_assert(sizeof(topic.param_index) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.param_index, sizeof(topic.param_index));
	buf.iterator += sizeof(topic.param_index);
	buf.offset += sizeof(topic.param_index);
	static_assert(sizeof(topic.param_count) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.param_count, sizeof(topic.param_count));
	buf.iterator += sizeof(topic.param_count);
	buf.offset += sizeof(topic.param_count);
	static_assert(sizeof(topic.param_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.param_type, sizeof(topic.param_type));
	buf.iterator += sizeof(topic.param_type);
	buf.offset += sizeof(topic.param_type);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.int_value) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.int_value, sizeof(topic.int_value));
	buf.iterator += sizeof(topic.int_value);
	buf.offset += sizeof(topic.int_value);
	static_assert(sizeof(topic.real_value) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.real_value, sizeof(topic.real_value));
	buf.iterator += sizeof(topic.real_value);
	buf.offset += sizeof(topic.real_value);
	return true;
}

bool ucdr_deserialize_uavcan_parameter_value(ucdrBuffer& buf, uavcan_parameter_value_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.node_id) == 1, "size mismatch");
	memcpy(&topic.node_id, buf.iterator, sizeof(topic.node_id));
	buf.iterator += sizeof(topic.node_id);
	buf.offset += sizeof(topic.node_id);
	static_assert(sizeof(topic.param_id) == 17, "size mismatch");
	memcpy(&topic.param_id, buf.iterator, sizeof(topic.param_id));
	buf.iterator += sizeof(topic.param_id);
	buf.offset += sizeof(topic.param_id);
	static_assert(sizeof(topic.param_index) == 2, "size mismatch");
	memcpy(&topic.param_index, buf.iterator, sizeof(topic.param_index));
	buf.iterator += sizeof(topic.param_index);
	buf.offset += sizeof(topic.param_index);
	static_assert(sizeof(topic.param_count) == 2, "size mismatch");
	memcpy(&topic.param_count, buf.iterator, sizeof(topic.param_count));
	buf.iterator += sizeof(topic.param_count);
	buf.offset += sizeof(topic.param_count);
	static_assert(sizeof(topic.param_type) == 1, "size mismatch");
	memcpy(&topic.param_type, buf.iterator, sizeof(topic.param_type));
	buf.iterator += sizeof(topic.param_type);
	buf.offset += sizeof(topic.param_type);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.int_value) == 8, "size mismatch");
	memcpy(&topic.int_value, buf.iterator, sizeof(topic.int_value));
	buf.iterator += sizeof(topic.int_value);
	buf.offset += sizeof(topic.int_value);
	static_assert(sizeof(topic.real_value) == 4, "size mismatch");
	memcpy(&topic.real_value, buf.iterator, sizeof(topic.real_value));
	buf.iterator += sizeof(topic.real_value);
	buf.offset += sizeof(topic.real_value);
	return true;
}
