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
#include <uORB/topics/rc_channels.h>


static inline constexpr int ucdr_topic_size_rc_channels()
{
	return 124;
}

bool ucdr_serialize_rc_channels(const rc_channels_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_last_valid) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.timestamp_last_valid, sizeof(topic.timestamp_last_valid));
	buf.iterator += sizeof(topic.timestamp_last_valid);
	buf.offset += sizeof(topic.timestamp_last_valid);
	static_assert(sizeof(topic.channels) == 72, "size mismatch");
	memcpy(buf.iterator, &topic.channels, sizeof(topic.channels));
	buf.iterator += sizeof(topic.channels);
	buf.offset += sizeof(topic.channels);
	static_assert(sizeof(topic.channel_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.channel_count, sizeof(topic.channel_count));
	buf.iterator += sizeof(topic.channel_count);
	buf.offset += sizeof(topic.channel_count);
	static_assert(sizeof(topic.function) == 28, "size mismatch");
	memcpy(buf.iterator, &topic.function, sizeof(topic.function));
	buf.iterator += sizeof(topic.function);
	buf.offset += sizeof(topic.function);
	static_assert(sizeof(topic.rssi) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rssi, sizeof(topic.rssi));
	buf.iterator += sizeof(topic.rssi);
	buf.offset += sizeof(topic.rssi);
	static_assert(sizeof(topic.signal_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.signal_lost, sizeof(topic.signal_lost));
	buf.iterator += sizeof(topic.signal_lost);
	buf.offset += sizeof(topic.signal_lost);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.frame_drop_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.frame_drop_count, sizeof(topic.frame_drop_count));
	buf.iterator += sizeof(topic.frame_drop_count);
	buf.offset += sizeof(topic.frame_drop_count);
	return true;
}

bool ucdr_deserialize_rc_channels(ucdrBuffer& buf, rc_channels_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_last_valid) == 8, "size mismatch");
	memcpy(&topic.timestamp_last_valid, buf.iterator, sizeof(topic.timestamp_last_valid));
	buf.iterator += sizeof(topic.timestamp_last_valid);
	buf.offset += sizeof(topic.timestamp_last_valid);
	static_assert(sizeof(topic.channels) == 72, "size mismatch");
	memcpy(&topic.channels, buf.iterator, sizeof(topic.channels));
	buf.iterator += sizeof(topic.channels);
	buf.offset += sizeof(topic.channels);
	static_assert(sizeof(topic.channel_count) == 1, "size mismatch");
	memcpy(&topic.channel_count, buf.iterator, sizeof(topic.channel_count));
	buf.iterator += sizeof(topic.channel_count);
	buf.offset += sizeof(topic.channel_count);
	static_assert(sizeof(topic.function) == 28, "size mismatch");
	memcpy(&topic.function, buf.iterator, sizeof(topic.function));
	buf.iterator += sizeof(topic.function);
	buf.offset += sizeof(topic.function);
	static_assert(sizeof(topic.rssi) == 1, "size mismatch");
	memcpy(&topic.rssi, buf.iterator, sizeof(topic.rssi));
	buf.iterator += sizeof(topic.rssi);
	buf.offset += sizeof(topic.rssi);
	static_assert(sizeof(topic.signal_lost) == 1, "size mismatch");
	memcpy(&topic.signal_lost, buf.iterator, sizeof(topic.signal_lost));
	buf.iterator += sizeof(topic.signal_lost);
	buf.offset += sizeof(topic.signal_lost);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.frame_drop_count) == 4, "size mismatch");
	memcpy(&topic.frame_drop_count, buf.iterator, sizeof(topic.frame_drop_count));
	buf.iterator += sizeof(topic.frame_drop_count);
	buf.offset += sizeof(topic.frame_drop_count);
	return true;
}
