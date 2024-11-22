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
#include <uORB/topics/pwm_input.h>


static inline constexpr int ucdr_topic_size_pwm_input()
{
	return 24;
}

bool ucdr_serialize_pwm_input(const pwm_input_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.error_count) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.error_count, sizeof(topic.error_count));
	buf.iterator += sizeof(topic.error_count);
	buf.offset += sizeof(topic.error_count);
	static_assert(sizeof(topic.pulse_width) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.pulse_width, sizeof(topic.pulse_width));
	buf.iterator += sizeof(topic.pulse_width);
	buf.offset += sizeof(topic.pulse_width);
	static_assert(sizeof(topic.period) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.period, sizeof(topic.period));
	buf.iterator += sizeof(topic.period);
	buf.offset += sizeof(topic.period);
	return true;
}

bool ucdr_deserialize_pwm_input(ucdrBuffer& buf, pwm_input_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.error_count) == 8, "size mismatch");
	memcpy(&topic.error_count, buf.iterator, sizeof(topic.error_count));
	buf.iterator += sizeof(topic.error_count);
	buf.offset += sizeof(topic.error_count);
	static_assert(sizeof(topic.pulse_width) == 4, "size mismatch");
	memcpy(&topic.pulse_width, buf.iterator, sizeof(topic.pulse_width));
	buf.iterator += sizeof(topic.pulse_width);
	buf.offset += sizeof(topic.pulse_width);
	static_assert(sizeof(topic.period) == 4, "size mismatch");
	memcpy(&topic.period, buf.iterator, sizeof(topic.period));
	buf.iterator += sizeof(topic.period);
	buf.offset += sizeof(topic.period);
	return true;
}
