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
#include <uORB/topics/camera_capture.h>


static inline constexpr int ucdr_topic_size_camera_capture()
{
	return 65;
}

bool ucdr_serialize_camera_capture(const camera_capture_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_utc) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.timestamp_utc, sizeof(topic.timestamp_utc));
	buf.iterator += sizeof(topic.timestamp_utc);
	buf.offset += sizeof(topic.timestamp_utc);
	static_assert(sizeof(topic.seq) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.seq, sizeof(topic.seq));
	buf.iterator += sizeof(topic.seq);
	buf.offset += sizeof(topic.seq);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lat, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.lon, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.alt, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.ground_distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ground_distance, sizeof(topic.ground_distance));
	buf.iterator += sizeof(topic.ground_distance);
	buf.offset += sizeof(topic.ground_distance);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.q, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.result) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.result, sizeof(topic.result));
	buf.iterator += sizeof(topic.result);
	buf.offset += sizeof(topic.result);
	return true;
}

bool ucdr_deserialize_camera_capture(ucdrBuffer& buf, camera_capture_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_utc) == 8, "size mismatch");
	memcpy(&topic.timestamp_utc, buf.iterator, sizeof(topic.timestamp_utc));
	buf.iterator += sizeof(topic.timestamp_utc);
	buf.offset += sizeof(topic.timestamp_utc);
	static_assert(sizeof(topic.seq) == 4, "size mismatch");
	memcpy(&topic.seq, buf.iterator, sizeof(topic.seq));
	buf.iterator += sizeof(topic.seq);
	buf.offset += sizeof(topic.seq);
	buf.iterator += 4; // padding
	buf.offset += 4; // padding
	static_assert(sizeof(topic.lat) == 8, "size mismatch");
	memcpy(&topic.lat, buf.iterator, sizeof(topic.lat));
	buf.iterator += sizeof(topic.lat);
	buf.offset += sizeof(topic.lat);
	static_assert(sizeof(topic.lon) == 8, "size mismatch");
	memcpy(&topic.lon, buf.iterator, sizeof(topic.lon));
	buf.iterator += sizeof(topic.lon);
	buf.offset += sizeof(topic.lon);
	static_assert(sizeof(topic.alt) == 4, "size mismatch");
	memcpy(&topic.alt, buf.iterator, sizeof(topic.alt));
	buf.iterator += sizeof(topic.alt);
	buf.offset += sizeof(topic.alt);
	static_assert(sizeof(topic.ground_distance) == 4, "size mismatch");
	memcpy(&topic.ground_distance, buf.iterator, sizeof(topic.ground_distance));
	buf.iterator += sizeof(topic.ground_distance);
	buf.offset += sizeof(topic.ground_distance);
	static_assert(sizeof(topic.q) == 16, "size mismatch");
	memcpy(&topic.q, buf.iterator, sizeof(topic.q));
	buf.iterator += sizeof(topic.q);
	buf.offset += sizeof(topic.q);
	static_assert(sizeof(topic.result) == 1, "size mismatch");
	memcpy(&topic.result, buf.iterator, sizeof(topic.result));
	buf.iterator += sizeof(topic.result);
	buf.offset += sizeof(topic.result);
	return true;
}
