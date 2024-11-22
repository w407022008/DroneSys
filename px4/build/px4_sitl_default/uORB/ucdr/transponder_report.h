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
#include <uORB/topics/transponder_report.h>


static inline constexpr int ucdr_topic_size_transponder_report()
{
	return 86;
}

bool ucdr_serialize_transponder_report(const transponder_report_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.icao_address) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.icao_address, sizeof(topic.icao_address));
	buf.iterator += sizeof(topic.icao_address);
	buf.offset += sizeof(topic.icao_address);
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
	static_assert(sizeof(topic.altitude_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.altitude_type, sizeof(topic.altitude_type));
	buf.iterator += sizeof(topic.altitude_type);
	buf.offset += sizeof(topic.altitude_type);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.altitude) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.altitude, sizeof(topic.altitude));
	buf.iterator += sizeof(topic.altitude);
	buf.offset += sizeof(topic.altitude);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.hor_velocity) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hor_velocity, sizeof(topic.hor_velocity));
	buf.iterator += sizeof(topic.hor_velocity);
	buf.offset += sizeof(topic.hor_velocity);
	static_assert(sizeof(topic.ver_velocity) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ver_velocity, sizeof(topic.ver_velocity));
	buf.iterator += sizeof(topic.ver_velocity);
	buf.offset += sizeof(topic.ver_velocity);
	static_assert(sizeof(topic.callsign) == 9, "size mismatch");
	memcpy(buf.iterator, &topic.callsign, sizeof(topic.callsign));
	buf.iterator += sizeof(topic.callsign);
	buf.offset += sizeof(topic.callsign);
	static_assert(sizeof(topic.emitter_type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.emitter_type, sizeof(topic.emitter_type));
	buf.iterator += sizeof(topic.emitter_type);
	buf.offset += sizeof(topic.emitter_type);
	static_assert(sizeof(topic.tslc) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.tslc, sizeof(topic.tslc));
	buf.iterator += sizeof(topic.tslc);
	buf.offset += sizeof(topic.tslc);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.flags) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.flags, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.squawk) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.squawk, sizeof(topic.squawk));
	buf.iterator += sizeof(topic.squawk);
	buf.offset += sizeof(topic.squawk);
	static_assert(sizeof(topic.uas_id) == 18, "size mismatch");
	memcpy(buf.iterator, &topic.uas_id, sizeof(topic.uas_id));
	buf.iterator += sizeof(topic.uas_id);
	buf.offset += sizeof(topic.uas_id);
	return true;
}

bool ucdr_deserialize_transponder_report(ucdrBuffer& buf, transponder_report_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.icao_address) == 4, "size mismatch");
	memcpy(&topic.icao_address, buf.iterator, sizeof(topic.icao_address));
	buf.iterator += sizeof(topic.icao_address);
	buf.offset += sizeof(topic.icao_address);
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
	static_assert(sizeof(topic.altitude_type) == 1, "size mismatch");
	memcpy(&topic.altitude_type, buf.iterator, sizeof(topic.altitude_type));
	buf.iterator += sizeof(topic.altitude_type);
	buf.offset += sizeof(topic.altitude_type);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.altitude) == 4, "size mismatch");
	memcpy(&topic.altitude, buf.iterator, sizeof(topic.altitude));
	buf.iterator += sizeof(topic.altitude);
	buf.offset += sizeof(topic.altitude);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(&topic.heading, buf.iterator, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.hor_velocity) == 4, "size mismatch");
	memcpy(&topic.hor_velocity, buf.iterator, sizeof(topic.hor_velocity));
	buf.iterator += sizeof(topic.hor_velocity);
	buf.offset += sizeof(topic.hor_velocity);
	static_assert(sizeof(topic.ver_velocity) == 4, "size mismatch");
	memcpy(&topic.ver_velocity, buf.iterator, sizeof(topic.ver_velocity));
	buf.iterator += sizeof(topic.ver_velocity);
	buf.offset += sizeof(topic.ver_velocity);
	static_assert(sizeof(topic.callsign) == 9, "size mismatch");
	memcpy(&topic.callsign, buf.iterator, sizeof(topic.callsign));
	buf.iterator += sizeof(topic.callsign);
	buf.offset += sizeof(topic.callsign);
	static_assert(sizeof(topic.emitter_type) == 1, "size mismatch");
	memcpy(&topic.emitter_type, buf.iterator, sizeof(topic.emitter_type));
	buf.iterator += sizeof(topic.emitter_type);
	buf.offset += sizeof(topic.emitter_type);
	static_assert(sizeof(topic.tslc) == 1, "size mismatch");
	memcpy(&topic.tslc, buf.iterator, sizeof(topic.tslc));
	buf.iterator += sizeof(topic.tslc);
	buf.offset += sizeof(topic.tslc);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.flags) == 2, "size mismatch");
	memcpy(&topic.flags, buf.iterator, sizeof(topic.flags));
	buf.iterator += sizeof(topic.flags);
	buf.offset += sizeof(topic.flags);
	static_assert(sizeof(topic.squawk) == 2, "size mismatch");
	memcpy(&topic.squawk, buf.iterator, sizeof(topic.squawk));
	buf.iterator += sizeof(topic.squawk);
	buf.offset += sizeof(topic.squawk);
	static_assert(sizeof(topic.uas_id) == 18, "size mismatch");
	memcpy(&topic.uas_id, buf.iterator, sizeof(topic.uas_id));
	buf.iterator += sizeof(topic.uas_id);
	buf.offset += sizeof(topic.uas_id);
	return true;
}
