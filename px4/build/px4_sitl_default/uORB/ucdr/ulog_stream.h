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
#include <uORB/topics/ulog_stream.h>


static inline constexpr int ucdr_topic_size_ulog_stream()
{
	return 262;
}

bool ucdr_serialize_ulog_stream(const ulog_stream_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.length) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.length, sizeof(topic.length));
	buf.iterator += sizeof(topic.length);
	buf.offset += sizeof(topic.length);
	static_assert(sizeof(topic.first_message_offset) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.first_message_offset, sizeof(topic.first_message_offset));
	buf.iterator += sizeof(topic.first_message_offset);
	buf.offset += sizeof(topic.first_message_offset);
	static_assert(sizeof(topic.msg_sequence) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.msg_sequence, sizeof(topic.msg_sequence));
	buf.iterator += sizeof(topic.msg_sequence);
	buf.offset += sizeof(topic.msg_sequence);
	static_assert(sizeof(topic.flags) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flags, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.data) == 249, "size mismatch");
	memcpy(buf.iterator, &topic.data, sizeof(topic.data));
	buf.iterator += sizeof(topic.data);
	buf.offset += sizeof(topic.data);
	return true;
}

bool ucdr_deserialize_ulog_stream(ucdrBuffer& buf, ulog_stream_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.length) == 1, "size mismatch");
	memcpy(&topic.length, buf.iterator, sizeof(topic.length));
	buf.iterator += sizeof(topic.length);
	buf.offset += sizeof(topic.length);
	static_assert(sizeof(topic.first_message_offset) == 1, "size mismatch");
	memcpy(&topic.first_message_offset, buf.iterator, sizeof(topic.first_message_offset));
	buf.iterator += sizeof(topic.first_message_offset);
	buf.offset += sizeof(topic.first_message_offset);
	static_assert(sizeof(topic.msg_sequence) == 2, "size mismatch");
	memcpy(&topic.msg_sequence, buf.iterator, sizeof(topic.msg_sequence));
	buf.iterator += sizeof(topic.msg_sequence);
	buf.offset += sizeof(topic.msg_sequence);
	static_assert(sizeof(topic.flags) == 1, "size mismatch");
	memcpy(&topic.flags, buf.iterator, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.data) == 249, "size mismatch");
	memcpy(&topic.data, buf.iterator, sizeof(topic.data));
	buf.iterator += sizeof(topic.data);
	buf.offset += sizeof(topic.data);
	return true;
}
