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
#include <uORB/topics/cellular_status.h>


static inline constexpr int ucdr_topic_size_cellular_status()
{
	return 20;
}

bool ucdr_serialize_cellular_status(const cellular_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.status) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.status, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.failure_reason) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.failure_reason, sizeof(topic.failure_reason));
	buf.iterator += sizeof(topic.failure_reason);
	buf.offset += sizeof(topic.failure_reason);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.quality, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.mcc) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.mcc, sizeof(topic.mcc));
	buf.iterator += sizeof(topic.mcc);
	buf.offset += sizeof(topic.mcc);
	static_assert(sizeof(topic.mnc) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.mnc, sizeof(topic.mnc));
	buf.iterator += sizeof(topic.mnc);
	buf.offset += sizeof(topic.mnc);
	static_assert(sizeof(topic.lac) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.lac, sizeof(topic.lac));
	buf.iterator += sizeof(topic.lac);
	buf.offset += sizeof(topic.lac);
	return true;
}

bool ucdr_deserialize_cellular_status(ucdrBuffer& buf, cellular_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.status) == 2, "size mismatch");
	memcpy(&topic.status, buf.iterator, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.failure_reason) == 1, "size mismatch");
	memcpy(&topic.failure_reason, buf.iterator, sizeof(topic.failure_reason));
	buf.iterator += sizeof(topic.failure_reason);
	buf.offset += sizeof(topic.failure_reason);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.quality) == 1, "size mismatch");
	memcpy(&topic.quality, buf.iterator, sizeof(topic.quality));
	buf.iterator += sizeof(topic.quality);
	buf.offset += sizeof(topic.quality);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.mcc) == 2, "size mismatch");
	memcpy(&topic.mcc, buf.iterator, sizeof(topic.mcc));
	buf.iterator += sizeof(topic.mcc);
	buf.offset += sizeof(topic.mcc);
	static_assert(sizeof(topic.mnc) == 2, "size mismatch");
	memcpy(&topic.mnc, buf.iterator, sizeof(topic.mnc));
	buf.iterator += sizeof(topic.mnc);
	buf.offset += sizeof(topic.mnc);
	static_assert(sizeof(topic.lac) == 2, "size mismatch");
	memcpy(&topic.lac, buf.iterator, sizeof(topic.lac));
	buf.iterator += sizeof(topic.lac);
	buf.offset += sizeof(topic.lac);
	return true;
}
