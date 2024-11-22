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
#include <uORB/topics/ping.h>


static inline constexpr int ucdr_topic_size_ping()
{
	return 30;
}

bool ucdr_serialize_ping(const ping_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.ping_time) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ping_time, sizeof(topic.ping_time));
	buf.iterator += sizeof(topic.ping_time);
	buf.offset += sizeof(topic.ping_time);
	static_assert(sizeof(topic.ping_sequence) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ping_sequence, sizeof(topic.ping_sequence));
	buf.iterator += sizeof(topic.ping_sequence);
	buf.offset += sizeof(topic.ping_sequence);
	static_assert(sizeof(topic.dropped_packets) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.dropped_packets, sizeof(topic.dropped_packets));
	buf.iterator += sizeof(topic.dropped_packets);
	buf.offset += sizeof(topic.dropped_packets);
	static_assert(sizeof(topic.rtt_ms) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rtt_ms, sizeof(topic.rtt_ms));
	buf.iterator += sizeof(topic.rtt_ms);
	buf.offset += sizeof(topic.rtt_ms);
	static_assert(sizeof(topic.system_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.system_id, sizeof(topic.system_id));
	buf.iterator += sizeof(topic.system_id);
	buf.offset += sizeof(topic.system_id);
	static_assert(sizeof(topic.component_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.component_id, sizeof(topic.component_id));
	buf.iterator += sizeof(topic.component_id);
	buf.offset += sizeof(topic.component_id);
	return true;
}

bool ucdr_deserialize_ping(ucdrBuffer& buf, ping_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.ping_time) == 8, "size mismatch");
	memcpy(&topic.ping_time, buf.iterator, sizeof(topic.ping_time));
	buf.iterator += sizeof(topic.ping_time);
	buf.offset += sizeof(topic.ping_time);
	static_assert(sizeof(topic.ping_sequence) == 4, "size mismatch");
	memcpy(&topic.ping_sequence, buf.iterator, sizeof(topic.ping_sequence));
	buf.iterator += sizeof(topic.ping_sequence);
	buf.offset += sizeof(topic.ping_sequence);
	static_assert(sizeof(topic.dropped_packets) == 4, "size mismatch");
	memcpy(&topic.dropped_packets, buf.iterator, sizeof(topic.dropped_packets));
	buf.iterator += sizeof(topic.dropped_packets);
	buf.offset += sizeof(topic.dropped_packets);
	static_assert(sizeof(topic.rtt_ms) == 4, "size mismatch");
	memcpy(&topic.rtt_ms, buf.iterator, sizeof(topic.rtt_ms));
	buf.iterator += sizeof(topic.rtt_ms);
	buf.offset += sizeof(topic.rtt_ms);
	static_assert(sizeof(topic.system_id) == 1, "size mismatch");
	memcpy(&topic.system_id, buf.iterator, sizeof(topic.system_id));
	buf.iterator += sizeof(topic.system_id);
	buf.offset += sizeof(topic.system_id);
	static_assert(sizeof(topic.component_id) == 1, "size mismatch");
	memcpy(&topic.component_id, buf.iterator, sizeof(topic.component_id));
	buf.iterator += sizeof(topic.component_id);
	buf.offset += sizeof(topic.component_id);
	return true;
}
