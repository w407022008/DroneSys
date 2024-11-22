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
#include <uORB/topics/satellite_info.h>


static inline constexpr int ucdr_topic_size_satellite_info()
{
	return 129;
}

bool ucdr_serialize_satellite_info(const satellite_info_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.count) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.count, sizeof(topic.count));
	buf.iterator += sizeof(topic.count);
	buf.offset += sizeof(topic.count);
	static_assert(sizeof(topic.svid) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.svid, sizeof(topic.svid));
	buf.iterator += sizeof(topic.svid);
	buf.offset += sizeof(topic.svid);
	static_assert(sizeof(topic.used) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.used, sizeof(topic.used));
	buf.iterator += sizeof(topic.used);
	buf.offset += sizeof(topic.used);
	static_assert(sizeof(topic.elevation) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.elevation, sizeof(topic.elevation));
	buf.iterator += sizeof(topic.elevation);
	buf.offset += sizeof(topic.elevation);
	static_assert(sizeof(topic.azimuth) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.azimuth, sizeof(topic.azimuth));
	buf.iterator += sizeof(topic.azimuth);
	buf.offset += sizeof(topic.azimuth);
	static_assert(sizeof(topic.snr) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.snr, sizeof(topic.snr));
	buf.iterator += sizeof(topic.snr);
	buf.offset += sizeof(topic.snr);
	static_assert(sizeof(topic.prn) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.prn, sizeof(topic.prn));
	buf.iterator += sizeof(topic.prn);
	buf.offset += sizeof(topic.prn);
	return true;
}

bool ucdr_deserialize_satellite_info(ucdrBuffer& buf, satellite_info_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.count) == 1, "size mismatch");
	memcpy(&topic.count, buf.iterator, sizeof(topic.count));
	buf.iterator += sizeof(topic.count);
	buf.offset += sizeof(topic.count);
	static_assert(sizeof(topic.svid) == 20, "size mismatch");
	memcpy(&topic.svid, buf.iterator, sizeof(topic.svid));
	buf.iterator += sizeof(topic.svid);
	buf.offset += sizeof(topic.svid);
	static_assert(sizeof(topic.used) == 20, "size mismatch");
	memcpy(&topic.used, buf.iterator, sizeof(topic.used));
	buf.iterator += sizeof(topic.used);
	buf.offset += sizeof(topic.used);
	static_assert(sizeof(topic.elevation) == 20, "size mismatch");
	memcpy(&topic.elevation, buf.iterator, sizeof(topic.elevation));
	buf.iterator += sizeof(topic.elevation);
	buf.offset += sizeof(topic.elevation);
	static_assert(sizeof(topic.azimuth) == 20, "size mismatch");
	memcpy(&topic.azimuth, buf.iterator, sizeof(topic.azimuth));
	buf.iterator += sizeof(topic.azimuth);
	buf.offset += sizeof(topic.azimuth);
	static_assert(sizeof(topic.snr) == 20, "size mismatch");
	memcpy(&topic.snr, buf.iterator, sizeof(topic.snr));
	buf.iterator += sizeof(topic.snr);
	buf.offset += sizeof(topic.snr);
	static_assert(sizeof(topic.prn) == 20, "size mismatch");
	memcpy(&topic.prn, buf.iterator, sizeof(topic.prn));
	buf.iterator += sizeof(topic.prn);
	buf.offset += sizeof(topic.prn);
	return true;
}
