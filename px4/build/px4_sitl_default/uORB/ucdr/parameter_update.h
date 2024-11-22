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
#include <uORB/topics/parameter_update.h>


static inline constexpr int ucdr_topic_size_parameter_update()
{
	return 34;
}

bool ucdr_serialize_parameter_update(const parameter_update_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.instance, sizeof(topic.instance));
	buf.iterator += sizeof(topic.instance);
	buf.offset += sizeof(topic.instance);
	static_assert(sizeof(topic.get_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.get_count, sizeof(topic.get_count));
	buf.iterator += sizeof(topic.get_count);
	buf.offset += sizeof(topic.get_count);
	static_assert(sizeof(topic.set_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.set_count, sizeof(topic.set_count));
	buf.iterator += sizeof(topic.set_count);
	buf.offset += sizeof(topic.set_count);
	static_assert(sizeof(topic.find_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.find_count, sizeof(topic.find_count));
	buf.iterator += sizeof(topic.find_count);
	buf.offset += sizeof(topic.find_count);
	static_assert(sizeof(topic.export_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.export_count, sizeof(topic.export_count));
	buf.iterator += sizeof(topic.export_count);
	buf.offset += sizeof(topic.export_count);
	static_assert(sizeof(topic.active) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.active, sizeof(topic.active));
	buf.iterator += sizeof(topic.active);
	buf.offset += sizeof(topic.active);
	static_assert(sizeof(topic.changed) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.changed, sizeof(topic.changed));
	buf.iterator += sizeof(topic.changed);
	buf.offset += sizeof(topic.changed);
	static_assert(sizeof(topic.custom_default) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.custom_default, sizeof(topic.custom_default));
	buf.iterator += sizeof(topic.custom_default);
	buf.offset += sizeof(topic.custom_default);
	return true;
}

bool ucdr_deserialize_parameter_update(ucdrBuffer& buf, parameter_update_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance) == 4, "size mismatch");
	memcpy(&topic.instance, buf.iterator, sizeof(topic.instance));
	buf.iterator += sizeof(topic.instance);
	buf.offset += sizeof(topic.instance);
	static_assert(sizeof(topic.get_count) == 4, "size mismatch");
	memcpy(&topic.get_count, buf.iterator, sizeof(topic.get_count));
	buf.iterator += sizeof(topic.get_count);
	buf.offset += sizeof(topic.get_count);
	static_assert(sizeof(topic.set_count) == 4, "size mismatch");
	memcpy(&topic.set_count, buf.iterator, sizeof(topic.set_count));
	buf.iterator += sizeof(topic.set_count);
	buf.offset += sizeof(topic.set_count);
	static_assert(sizeof(topic.find_count) == 4, "size mismatch");
	memcpy(&topic.find_count, buf.iterator, sizeof(topic.find_count));
	buf.iterator += sizeof(topic.find_count);
	buf.offset += sizeof(topic.find_count);
	static_assert(sizeof(topic.export_count) == 4, "size mismatch");
	memcpy(&topic.export_count, buf.iterator, sizeof(topic.export_count));
	buf.iterator += sizeof(topic.export_count);
	buf.offset += sizeof(topic.export_count);
	static_assert(sizeof(topic.active) == 2, "size mismatch");
	memcpy(&topic.active, buf.iterator, sizeof(topic.active));
	buf.iterator += sizeof(topic.active);
	buf.offset += sizeof(topic.active);
	static_assert(sizeof(topic.changed) == 2, "size mismatch");
	memcpy(&topic.changed, buf.iterator, sizeof(topic.changed));
	buf.iterator += sizeof(topic.changed);
	buf.offset += sizeof(topic.changed);
	static_assert(sizeof(topic.custom_default) == 2, "size mismatch");
	memcpy(&topic.custom_default, buf.iterator, sizeof(topic.custom_default));
	buf.iterator += sizeof(topic.custom_default);
	buf.offset += sizeof(topic.custom_default);
	return true;
}
