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
#include <uORB/topics/home_position.h>


static inline constexpr int ucdr_topic_size_home_position()
{
	return 48;
}

bool ucdr_serialize_home_position(const home_position_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
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
	static_assert(sizeof(topic.x) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.x, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.y, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.z, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.yaw, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.valid_alt) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid_alt, sizeof(topic.valid_alt));
	buf.iterator += sizeof(topic.valid_alt);
	buf.offset += sizeof(topic.valid_alt);
	static_assert(sizeof(topic.valid_hpos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid_hpos, sizeof(topic.valid_hpos));
	buf.iterator += sizeof(topic.valid_hpos);
	buf.offset += sizeof(topic.valid_hpos);
	static_assert(sizeof(topic.valid_lpos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid_lpos, sizeof(topic.valid_lpos));
	buf.iterator += sizeof(topic.valid_lpos);
	buf.offset += sizeof(topic.valid_lpos);
	static_assert(sizeof(topic.manual_home) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.manual_home, sizeof(topic.manual_home));
	buf.iterator += sizeof(topic.manual_home);
	buf.offset += sizeof(topic.manual_home);
	return true;
}

bool ucdr_deserialize_home_position(ucdrBuffer& buf, home_position_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
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
	static_assert(sizeof(topic.x) == 4, "size mismatch");
	memcpy(&topic.x, buf.iterator, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 4, "size mismatch");
	memcpy(&topic.y, buf.iterator, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 4, "size mismatch");
	memcpy(&topic.z, buf.iterator, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	static_assert(sizeof(topic.yaw) == 4, "size mismatch");
	memcpy(&topic.yaw, buf.iterator, sizeof(topic.yaw));
	buf.iterator += sizeof(topic.yaw);
	buf.offset += sizeof(topic.yaw);
	static_assert(sizeof(topic.valid_alt) == 1, "size mismatch");
	memcpy(&topic.valid_alt, buf.iterator, sizeof(topic.valid_alt));
	buf.iterator += sizeof(topic.valid_alt);
	buf.offset += sizeof(topic.valid_alt);
	static_assert(sizeof(topic.valid_hpos) == 1, "size mismatch");
	memcpy(&topic.valid_hpos, buf.iterator, sizeof(topic.valid_hpos));
	buf.iterator += sizeof(topic.valid_hpos);
	buf.offset += sizeof(topic.valid_hpos);
	static_assert(sizeof(topic.valid_lpos) == 1, "size mismatch");
	memcpy(&topic.valid_lpos, buf.iterator, sizeof(topic.valid_lpos));
	buf.iterator += sizeof(topic.valid_lpos);
	buf.offset += sizeof(topic.valid_lpos);
	static_assert(sizeof(topic.manual_home) == 1, "size mismatch");
	memcpy(&topic.manual_home, buf.iterator, sizeof(topic.manual_home));
	buf.iterator += sizeof(topic.manual_home);
	buf.offset += sizeof(topic.manual_home);
	return true;
}
