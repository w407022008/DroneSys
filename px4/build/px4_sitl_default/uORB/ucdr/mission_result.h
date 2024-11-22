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
#include <uORB/topics/mission_result.h>


static inline constexpr int ucdr_topic_size_mission_result()
{
	return 31;
}

bool ucdr_serialize_mission_result(const mission_result_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.instance_count, sizeof(topic.instance_count));
	buf.iterator += sizeof(topic.instance_count);
	buf.offset += sizeof(topic.instance_count);
	static_assert(sizeof(topic.seq_reached) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.seq_reached, sizeof(topic.seq_reached));
	buf.iterator += sizeof(topic.seq_reached);
	buf.offset += sizeof(topic.seq_reached);
	static_assert(sizeof(topic.seq_current) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.seq_current, sizeof(topic.seq_current));
	buf.iterator += sizeof(topic.seq_current);
	buf.offset += sizeof(topic.seq_current);
	static_assert(sizeof(topic.seq_total) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.seq_total, sizeof(topic.seq_total));
	buf.iterator += sizeof(topic.seq_total);
	buf.offset += sizeof(topic.seq_total);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.valid, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.warning) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.warning, sizeof(topic.warning));
	buf.iterator += sizeof(topic.warning);
	buf.offset += sizeof(topic.warning);
	static_assert(sizeof(topic.finished) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.finished, sizeof(topic.finished));
	buf.iterator += sizeof(topic.finished);
	buf.offset += sizeof(topic.finished);
	static_assert(sizeof(topic.failure) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.failure, sizeof(topic.failure));
	buf.iterator += sizeof(topic.failure);
	buf.offset += sizeof(topic.failure);
	static_assert(sizeof(topic.item_do_jump_changed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.item_do_jump_changed, sizeof(topic.item_do_jump_changed));
	buf.iterator += sizeof(topic.item_do_jump_changed);
	buf.offset += sizeof(topic.item_do_jump_changed);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.item_changed_index) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.item_changed_index, sizeof(topic.item_changed_index));
	buf.iterator += sizeof(topic.item_changed_index);
	buf.offset += sizeof(topic.item_changed_index);
	static_assert(sizeof(topic.item_do_jump_remaining) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.item_do_jump_remaining, sizeof(topic.item_do_jump_remaining));
	buf.iterator += sizeof(topic.item_do_jump_remaining);
	buf.offset += sizeof(topic.item_do_jump_remaining);
	static_assert(sizeof(topic.execution_mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.execution_mode, sizeof(topic.execution_mode));
	buf.iterator += sizeof(topic.execution_mode);
	buf.offset += sizeof(topic.execution_mode);
	return true;
}

bool ucdr_deserialize_mission_result(ucdrBuffer& buf, mission_result_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.instance_count) == 4, "size mismatch");
	memcpy(&topic.instance_count, buf.iterator, sizeof(topic.instance_count));
	buf.iterator += sizeof(topic.instance_count);
	buf.offset += sizeof(topic.instance_count);
	static_assert(sizeof(topic.seq_reached) == 4, "size mismatch");
	memcpy(&topic.seq_reached, buf.iterator, sizeof(topic.seq_reached));
	buf.iterator += sizeof(topic.seq_reached);
	buf.offset += sizeof(topic.seq_reached);
	static_assert(sizeof(topic.seq_current) == 2, "size mismatch");
	memcpy(&topic.seq_current, buf.iterator, sizeof(topic.seq_current));
	buf.iterator += sizeof(topic.seq_current);
	buf.offset += sizeof(topic.seq_current);
	static_assert(sizeof(topic.seq_total) == 2, "size mismatch");
	memcpy(&topic.seq_total, buf.iterator, sizeof(topic.seq_total));
	buf.iterator += sizeof(topic.seq_total);
	buf.offset += sizeof(topic.seq_total);
	static_assert(sizeof(topic.valid) == 1, "size mismatch");
	memcpy(&topic.valid, buf.iterator, sizeof(topic.valid));
	buf.iterator += sizeof(topic.valid);
	buf.offset += sizeof(topic.valid);
	static_assert(sizeof(topic.warning) == 1, "size mismatch");
	memcpy(&topic.warning, buf.iterator, sizeof(topic.warning));
	buf.iterator += sizeof(topic.warning);
	buf.offset += sizeof(topic.warning);
	static_assert(sizeof(topic.finished) == 1, "size mismatch");
	memcpy(&topic.finished, buf.iterator, sizeof(topic.finished));
	buf.iterator += sizeof(topic.finished);
	buf.offset += sizeof(topic.finished);
	static_assert(sizeof(topic.failure) == 1, "size mismatch");
	memcpy(&topic.failure, buf.iterator, sizeof(topic.failure));
	buf.iterator += sizeof(topic.failure);
	buf.offset += sizeof(topic.failure);
	static_assert(sizeof(topic.item_do_jump_changed) == 1, "size mismatch");
	memcpy(&topic.item_do_jump_changed, buf.iterator, sizeof(topic.item_do_jump_changed));
	buf.iterator += sizeof(topic.item_do_jump_changed);
	buf.offset += sizeof(topic.item_do_jump_changed);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.item_changed_index) == 2, "size mismatch");
	memcpy(&topic.item_changed_index, buf.iterator, sizeof(topic.item_changed_index));
	buf.iterator += sizeof(topic.item_changed_index);
	buf.offset += sizeof(topic.item_changed_index);
	static_assert(sizeof(topic.item_do_jump_remaining) == 2, "size mismatch");
	memcpy(&topic.item_do_jump_remaining, buf.iterator, sizeof(topic.item_do_jump_remaining));
	buf.iterator += sizeof(topic.item_do_jump_remaining);
	buf.offset += sizeof(topic.item_do_jump_remaining);
	static_assert(sizeof(topic.execution_mode) == 1, "size mismatch");
	memcpy(&topic.execution_mode, buf.iterator, sizeof(topic.execution_mode));
	buf.iterator += sizeof(topic.execution_mode);
	buf.offset += sizeof(topic.execution_mode);
	return true;
}
