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
#include <uORB/topics/input_rc.h>


static inline constexpr int ucdr_topic_size_input_rc()
{
	return 76;
}

bool ucdr_serialize_input_rc(const input_rc_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_last_signal) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.timestamp_last_signal, sizeof(topic.timestamp_last_signal));
	buf.iterator += sizeof(topic.timestamp_last_signal);
	buf.offset += sizeof(topic.timestamp_last_signal);
	static_assert(sizeof(topic.channel_count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.channel_count, sizeof(topic.channel_count));
	buf.iterator += sizeof(topic.channel_count);
	buf.offset += sizeof(topic.channel_count);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.rssi) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rssi, sizeof(topic.rssi));
	buf.iterator += sizeof(topic.rssi);
	buf.offset += sizeof(topic.rssi);
	static_assert(sizeof(topic.rc_failsafe) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rc_failsafe, sizeof(topic.rc_failsafe));
	buf.iterator += sizeof(topic.rc_failsafe);
	buf.offset += sizeof(topic.rc_failsafe);
	static_assert(sizeof(topic.rc_lost) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rc_lost, sizeof(topic.rc_lost));
	buf.iterator += sizeof(topic.rc_lost);
	buf.offset += sizeof(topic.rc_lost);
	static_assert(sizeof(topic.rc_lost_frame_count) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rc_lost_frame_count, sizeof(topic.rc_lost_frame_count));
	buf.iterator += sizeof(topic.rc_lost_frame_count);
	buf.offset += sizeof(topic.rc_lost_frame_count);
	static_assert(sizeof(topic.rc_total_frame_count) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rc_total_frame_count, sizeof(topic.rc_total_frame_count));
	buf.iterator += sizeof(topic.rc_total_frame_count);
	buf.offset += sizeof(topic.rc_total_frame_count);
	static_assert(sizeof(topic.rc_ppm_frame_length) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rc_ppm_frame_length, sizeof(topic.rc_ppm_frame_length));
	buf.iterator += sizeof(topic.rc_ppm_frame_length);
	buf.offset += sizeof(topic.rc_ppm_frame_length);
	static_assert(sizeof(topic.input_source) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.input_source, sizeof(topic.input_source));
	buf.iterator += sizeof(topic.input_source);
	buf.offset += sizeof(topic.input_source);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.values) == 36, "size mismatch");
	memcpy(buf.iterator, &topic.values, sizeof(topic.values));
	buf.iterator += sizeof(topic.values);
	buf.offset += sizeof(topic.values);
	static_assert(sizeof(topic.link_quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.link_quality, sizeof(topic.link_quality));
	buf.iterator += sizeof(topic.link_quality);
	buf.offset += sizeof(topic.link_quality);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.rssi_dbm) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rssi_dbm, sizeof(topic.rssi_dbm));
	buf.iterator += sizeof(topic.rssi_dbm);
	buf.offset += sizeof(topic.rssi_dbm);
	return true;
}

bool ucdr_deserialize_input_rc(ucdrBuffer& buf, input_rc_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_last_signal) == 8, "size mismatch");
	memcpy(&topic.timestamp_last_signal, buf.iterator, sizeof(topic.timestamp_last_signal));
	buf.iterator += sizeof(topic.timestamp_last_signal);
	buf.offset += sizeof(topic.timestamp_last_signal);
	static_assert(sizeof(topic.channel_count) == 1, "size mismatch");
	memcpy(&topic.channel_count, buf.iterator, sizeof(topic.channel_count));
	buf.iterator += sizeof(topic.channel_count);
	buf.offset += sizeof(topic.channel_count);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.rssi) == 4, "size mismatch");
	memcpy(&topic.rssi, buf.iterator, sizeof(topic.rssi));
	buf.iterator += sizeof(topic.rssi);
	buf.offset += sizeof(topic.rssi);
	static_assert(sizeof(topic.rc_failsafe) == 1, "size mismatch");
	memcpy(&topic.rc_failsafe, buf.iterator, sizeof(topic.rc_failsafe));
	buf.iterator += sizeof(topic.rc_failsafe);
	buf.offset += sizeof(topic.rc_failsafe);
	static_assert(sizeof(topic.rc_lost) == 1, "size mismatch");
	memcpy(&topic.rc_lost, buf.iterator, sizeof(topic.rc_lost));
	buf.iterator += sizeof(topic.rc_lost);
	buf.offset += sizeof(topic.rc_lost);
	static_assert(sizeof(topic.rc_lost_frame_count) == 2, "size mismatch");
	memcpy(&topic.rc_lost_frame_count, buf.iterator, sizeof(topic.rc_lost_frame_count));
	buf.iterator += sizeof(topic.rc_lost_frame_count);
	buf.offset += sizeof(topic.rc_lost_frame_count);
	static_assert(sizeof(topic.rc_total_frame_count) == 2, "size mismatch");
	memcpy(&topic.rc_total_frame_count, buf.iterator, sizeof(topic.rc_total_frame_count));
	buf.iterator += sizeof(topic.rc_total_frame_count);
	buf.offset += sizeof(topic.rc_total_frame_count);
	static_assert(sizeof(topic.rc_ppm_frame_length) == 2, "size mismatch");
	memcpy(&topic.rc_ppm_frame_length, buf.iterator, sizeof(topic.rc_ppm_frame_length));
	buf.iterator += sizeof(topic.rc_ppm_frame_length);
	buf.offset += sizeof(topic.rc_ppm_frame_length);
	static_assert(sizeof(topic.input_source) == 1, "size mismatch");
	memcpy(&topic.input_source, buf.iterator, sizeof(topic.input_source));
	buf.iterator += sizeof(topic.input_source);
	buf.offset += sizeof(topic.input_source);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.values) == 36, "size mismatch");
	memcpy(&topic.values, buf.iterator, sizeof(topic.values));
	buf.iterator += sizeof(topic.values);
	buf.offset += sizeof(topic.values);
	static_assert(sizeof(topic.link_quality) == 1, "size mismatch");
	memcpy(&topic.link_quality, buf.iterator, sizeof(topic.link_quality));
	buf.iterator += sizeof(topic.link_quality);
	buf.offset += sizeof(topic.link_quality);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.rssi_dbm) == 4, "size mismatch");
	memcpy(&topic.rssi_dbm, buf.iterator, sizeof(topic.rssi_dbm));
	buf.iterator += sizeof(topic.rssi_dbm);
	buf.offset += sizeof(topic.rssi_dbm);
	return true;
}
