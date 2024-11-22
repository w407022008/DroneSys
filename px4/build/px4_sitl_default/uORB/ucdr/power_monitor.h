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
#include <uORB/topics/power_monitor.h>


static inline constexpr int ucdr_topic_size_power_monitor()
{
	return 36;
}

bool ucdr_serialize_power_monitor(const power_monitor_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.voltage_v) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.voltage_v, sizeof(topic.voltage_v));
	buf.iterator += sizeof(topic.voltage_v);
	buf.offset += sizeof(topic.voltage_v);
	static_assert(sizeof(topic.current_a) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.current_a, sizeof(topic.current_a));
	buf.iterator += sizeof(topic.current_a);
	buf.offset += sizeof(topic.current_a);
	static_assert(sizeof(topic.power_w) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.power_w, sizeof(topic.power_w));
	buf.iterator += sizeof(topic.power_w);
	buf.offset += sizeof(topic.power_w);
	static_assert(sizeof(topic.rconf) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rconf, sizeof(topic.rconf));
	buf.iterator += sizeof(topic.rconf);
	buf.offset += sizeof(topic.rconf);
	static_assert(sizeof(topic.rsv) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rsv, sizeof(topic.rsv));
	buf.iterator += sizeof(topic.rsv);
	buf.offset += sizeof(topic.rsv);
	static_assert(sizeof(topic.rbv) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rbv, sizeof(topic.rbv));
	buf.iterator += sizeof(topic.rbv);
	buf.offset += sizeof(topic.rbv);
	static_assert(sizeof(topic.rp) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rp, sizeof(topic.rp));
	buf.iterator += sizeof(topic.rp);
	buf.offset += sizeof(topic.rp);
	static_assert(sizeof(topic.rc) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rc, sizeof(topic.rc));
	buf.iterator += sizeof(topic.rc);
	buf.offset += sizeof(topic.rc);
	static_assert(sizeof(topic.rcal) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rcal, sizeof(topic.rcal));
	buf.iterator += sizeof(topic.rcal);
	buf.offset += sizeof(topic.rcal);
	static_assert(sizeof(topic.me) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.me, sizeof(topic.me));
	buf.iterator += sizeof(topic.me);
	buf.offset += sizeof(topic.me);
	static_assert(sizeof(topic.al) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.al, sizeof(topic.al));
	buf.iterator += sizeof(topic.al);
	buf.offset += sizeof(topic.al);
	return true;
}

bool ucdr_deserialize_power_monitor(ucdrBuffer& buf, power_monitor_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.voltage_v) == 4, "size mismatch");
	memcpy(&topic.voltage_v, buf.iterator, sizeof(topic.voltage_v));
	buf.iterator += sizeof(topic.voltage_v);
	buf.offset += sizeof(topic.voltage_v);
	static_assert(sizeof(topic.current_a) == 4, "size mismatch");
	memcpy(&topic.current_a, buf.iterator, sizeof(topic.current_a));
	buf.iterator += sizeof(topic.current_a);
	buf.offset += sizeof(topic.current_a);
	static_assert(sizeof(topic.power_w) == 4, "size mismatch");
	memcpy(&topic.power_w, buf.iterator, sizeof(topic.power_w));
	buf.iterator += sizeof(topic.power_w);
	buf.offset += sizeof(topic.power_w);
	static_assert(sizeof(topic.rconf) == 2, "size mismatch");
	memcpy(&topic.rconf, buf.iterator, sizeof(topic.rconf));
	buf.iterator += sizeof(topic.rconf);
	buf.offset += sizeof(topic.rconf);
	static_assert(sizeof(topic.rsv) == 2, "size mismatch");
	memcpy(&topic.rsv, buf.iterator, sizeof(topic.rsv));
	buf.iterator += sizeof(topic.rsv);
	buf.offset += sizeof(topic.rsv);
	static_assert(sizeof(topic.rbv) == 2, "size mismatch");
	memcpy(&topic.rbv, buf.iterator, sizeof(topic.rbv));
	buf.iterator += sizeof(topic.rbv);
	buf.offset += sizeof(topic.rbv);
	static_assert(sizeof(topic.rp) == 2, "size mismatch");
	memcpy(&topic.rp, buf.iterator, sizeof(topic.rp));
	buf.iterator += sizeof(topic.rp);
	buf.offset += sizeof(topic.rp);
	static_assert(sizeof(topic.rc) == 2, "size mismatch");
	memcpy(&topic.rc, buf.iterator, sizeof(topic.rc));
	buf.iterator += sizeof(topic.rc);
	buf.offset += sizeof(topic.rc);
	static_assert(sizeof(topic.rcal) == 2, "size mismatch");
	memcpy(&topic.rcal, buf.iterator, sizeof(topic.rcal));
	buf.iterator += sizeof(topic.rcal);
	buf.offset += sizeof(topic.rcal);
	static_assert(sizeof(topic.me) == 2, "size mismatch");
	memcpy(&topic.me, buf.iterator, sizeof(topic.me));
	buf.iterator += sizeof(topic.me);
	buf.offset += sizeof(topic.me);
	static_assert(sizeof(topic.al) == 2, "size mismatch");
	memcpy(&topic.al, buf.iterator, sizeof(topic.al));
	buf.iterator += sizeof(topic.al);
	buf.offset += sizeof(topic.al);
	return true;
}
