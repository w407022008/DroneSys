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
#include <uORB/topics/geofence_result.h>


static inline constexpr int ucdr_topic_size_geofence_result()
{
	return 12;
}

bool ucdr_serialize_geofence_result(const geofence_result_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.geofence_violation_reason) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.geofence_violation_reason, sizeof(topic.geofence_violation_reason));
	buf.iterator += sizeof(topic.geofence_violation_reason);
	buf.offset += sizeof(topic.geofence_violation_reason);
	static_assert(sizeof(topic.primary_geofence_breached) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.primary_geofence_breached, sizeof(topic.primary_geofence_breached));
	buf.iterator += sizeof(topic.primary_geofence_breached);
	buf.offset += sizeof(topic.primary_geofence_breached);
	static_assert(sizeof(topic.primary_geofence_action) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.primary_geofence_action, sizeof(topic.primary_geofence_action));
	buf.iterator += sizeof(topic.primary_geofence_action);
	buf.offset += sizeof(topic.primary_geofence_action);
	static_assert(sizeof(topic.home_required) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.home_required, sizeof(topic.home_required));
	buf.iterator += sizeof(topic.home_required);
	buf.offset += sizeof(topic.home_required);
	return true;
}

bool ucdr_deserialize_geofence_result(ucdrBuffer& buf, geofence_result_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.geofence_violation_reason) == 1, "size mismatch");
	memcpy(&topic.geofence_violation_reason, buf.iterator, sizeof(topic.geofence_violation_reason));
	buf.iterator += sizeof(topic.geofence_violation_reason);
	buf.offset += sizeof(topic.geofence_violation_reason);
	static_assert(sizeof(topic.primary_geofence_breached) == 1, "size mismatch");
	memcpy(&topic.primary_geofence_breached, buf.iterator, sizeof(topic.primary_geofence_breached));
	buf.iterator += sizeof(topic.primary_geofence_breached);
	buf.offset += sizeof(topic.primary_geofence_breached);
	static_assert(sizeof(topic.primary_geofence_action) == 1, "size mismatch");
	memcpy(&topic.primary_geofence_action, buf.iterator, sizeof(topic.primary_geofence_action));
	buf.iterator += sizeof(topic.primary_geofence_action);
	buf.offset += sizeof(topic.primary_geofence_action);
	static_assert(sizeof(topic.home_required) == 1, "size mismatch");
	memcpy(&topic.home_required, buf.iterator, sizeof(topic.home_required));
	buf.iterator += sizeof(topic.home_required);
	buf.offset += sizeof(topic.home_required);
	return true;
}
