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
#include <uORB/topics/logger_status.h>


static inline constexpr int ucdr_topic_size_logger_status()
{
	return 37;
}

bool ucdr_serialize_logger_status(const logger_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.backend) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.backend, sizeof(topic.backend));
	buf.iterator += sizeof(topic.backend);
	buf.offset += sizeof(topic.backend);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.total_written_kb) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.total_written_kb, sizeof(topic.total_written_kb));
	buf.iterator += sizeof(topic.total_written_kb);
	buf.offset += sizeof(topic.total_written_kb);
	static_assert(sizeof(topic.write_rate_kb_s) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.write_rate_kb_s, sizeof(topic.write_rate_kb_s));
	buf.iterator += sizeof(topic.write_rate_kb_s);
	buf.offset += sizeof(topic.write_rate_kb_s);
	static_assert(sizeof(topic.dropouts) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.dropouts, sizeof(topic.dropouts));
	buf.iterator += sizeof(topic.dropouts);
	buf.offset += sizeof(topic.dropouts);
	static_assert(sizeof(topic.message_gaps) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.message_gaps, sizeof(topic.message_gaps));
	buf.iterator += sizeof(topic.message_gaps);
	buf.offset += sizeof(topic.message_gaps);
	static_assert(sizeof(topic.buffer_used_bytes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.buffer_used_bytes, sizeof(topic.buffer_used_bytes));
	buf.iterator += sizeof(topic.buffer_used_bytes);
	buf.offset += sizeof(topic.buffer_used_bytes);
	static_assert(sizeof(topic.buffer_size_bytes) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.buffer_size_bytes, sizeof(topic.buffer_size_bytes));
	buf.iterator += sizeof(topic.buffer_size_bytes);
	buf.offset += sizeof(topic.buffer_size_bytes);
	static_assert(sizeof(topic.num_messages) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.num_messages, sizeof(topic.num_messages));
	buf.iterator += sizeof(topic.num_messages);
	buf.offset += sizeof(topic.num_messages);
	return true;
}

bool ucdr_deserialize_logger_status(ucdrBuffer& buf, logger_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.backend) == 1, "size mismatch");
	memcpy(&topic.backend, buf.iterator, sizeof(topic.backend));
	buf.iterator += sizeof(topic.backend);
	buf.offset += sizeof(topic.backend);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.total_written_kb) == 4, "size mismatch");
	memcpy(&topic.total_written_kb, buf.iterator, sizeof(topic.total_written_kb));
	buf.iterator += sizeof(topic.total_written_kb);
	buf.offset += sizeof(topic.total_written_kb);
	static_assert(sizeof(topic.write_rate_kb_s) == 4, "size mismatch");
	memcpy(&topic.write_rate_kb_s, buf.iterator, sizeof(topic.write_rate_kb_s));
	buf.iterator += sizeof(topic.write_rate_kb_s);
	buf.offset += sizeof(topic.write_rate_kb_s);
	static_assert(sizeof(topic.dropouts) == 4, "size mismatch");
	memcpy(&topic.dropouts, buf.iterator, sizeof(topic.dropouts));
	buf.iterator += sizeof(topic.dropouts);
	buf.offset += sizeof(topic.dropouts);
	static_assert(sizeof(topic.message_gaps) == 4, "size mismatch");
	memcpy(&topic.message_gaps, buf.iterator, sizeof(topic.message_gaps));
	buf.iterator += sizeof(topic.message_gaps);
	buf.offset += sizeof(topic.message_gaps);
	static_assert(sizeof(topic.buffer_used_bytes) == 4, "size mismatch");
	memcpy(&topic.buffer_used_bytes, buf.iterator, sizeof(topic.buffer_used_bytes));
	buf.iterator += sizeof(topic.buffer_used_bytes);
	buf.offset += sizeof(topic.buffer_used_bytes);
	static_assert(sizeof(topic.buffer_size_bytes) == 4, "size mismatch");
	memcpy(&topic.buffer_size_bytes, buf.iterator, sizeof(topic.buffer_size_bytes));
	buf.iterator += sizeof(topic.buffer_size_bytes);
	buf.offset += sizeof(topic.buffer_size_bytes);
	static_assert(sizeof(topic.num_messages) == 1, "size mismatch");
	memcpy(&topic.num_messages, buf.iterator, sizeof(topic.num_messages));
	buf.iterator += sizeof(topic.num_messages);
	buf.offset += sizeof(topic.num_messages);
	return true;
}
