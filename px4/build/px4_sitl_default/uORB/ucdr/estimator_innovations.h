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
#include <uORB/topics/estimator_innovations.h>


static inline constexpr int ucdr_topic_size_estimator_innovations()
{
	return 152;
}

bool ucdr_serialize_estimator_innovations(const estimator_innovations_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_sample) == 8, "size mismatch");
	const uint64_t timestamp_sample_adjusted = topic.timestamp_sample + time_offset;
	memcpy(buf.iterator, &timestamp_sample_adjusted, sizeof(topic.timestamp_sample));
	buf.iterator += sizeof(topic.timestamp_sample);
	buf.offset += sizeof(topic.timestamp_sample);
	static_assert(sizeof(topic.gps_hvel) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.gps_hvel, sizeof(topic.gps_hvel));
	buf.iterator += sizeof(topic.gps_hvel);
	buf.offset += sizeof(topic.gps_hvel);
	static_assert(sizeof(topic.gps_vvel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gps_vvel, sizeof(topic.gps_vvel));
	buf.iterator += sizeof(topic.gps_vvel);
	buf.offset += sizeof(topic.gps_vvel);
	static_assert(sizeof(topic.gps_hpos) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.gps_hpos, sizeof(topic.gps_hpos));
	buf.iterator += sizeof(topic.gps_hpos);
	buf.offset += sizeof(topic.gps_hpos);
	static_assert(sizeof(topic.gps_vpos) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gps_vpos, sizeof(topic.gps_vpos));
	buf.iterator += sizeof(topic.gps_vpos);
	buf.offset += sizeof(topic.gps_vpos);
	static_assert(sizeof(topic.ev_hvel) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ev_hvel, sizeof(topic.ev_hvel));
	buf.iterator += sizeof(topic.ev_hvel);
	buf.offset += sizeof(topic.ev_hvel);
	static_assert(sizeof(topic.ev_vvel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ev_vvel, sizeof(topic.ev_vvel));
	buf.iterator += sizeof(topic.ev_vvel);
	buf.offset += sizeof(topic.ev_vvel);
	static_assert(sizeof(topic.ev_hpos) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.ev_hpos, sizeof(topic.ev_hpos));
	buf.iterator += sizeof(topic.ev_hpos);
	buf.offset += sizeof(topic.ev_hpos);
	static_assert(sizeof(topic.ev_vpos) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ev_vpos, sizeof(topic.ev_vpos));
	buf.iterator += sizeof(topic.ev_vpos);
	buf.offset += sizeof(topic.ev_vpos);
	static_assert(sizeof(topic.rng_vpos) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rng_vpos, sizeof(topic.rng_vpos));
	buf.iterator += sizeof(topic.rng_vpos);
	buf.offset += sizeof(topic.rng_vpos);
	static_assert(sizeof(topic.baro_vpos) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.baro_vpos, sizeof(topic.baro_vpos));
	buf.iterator += sizeof(topic.baro_vpos);
	buf.offset += sizeof(topic.baro_vpos);
	static_assert(sizeof(topic.aux_hvel) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.aux_hvel, sizeof(topic.aux_hvel));
	buf.iterator += sizeof(topic.aux_hvel);
	buf.offset += sizeof(topic.aux_hvel);
	static_assert(sizeof(topic.aux_vvel) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aux_vvel, sizeof(topic.aux_vvel));
	buf.iterator += sizeof(topic.aux_vvel);
	buf.offset += sizeof(topic.aux_vvel);
	static_assert(sizeof(topic.flow) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.flow, sizeof(topic.flow));
	buf.iterator += sizeof(topic.flow);
	buf.offset += sizeof(topic.flow);
	static_assert(sizeof(topic.terr_flow) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.terr_flow, sizeof(topic.terr_flow));
	buf.iterator += sizeof(topic.terr_flow);
	buf.offset += sizeof(topic.terr_flow);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.heading, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.mag_field) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.mag_field, sizeof(topic.mag_field));
	buf.iterator += sizeof(topic.mag_field);
	buf.offset += sizeof(topic.mag_field);
	static_assert(sizeof(topic.gravity) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.gravity, sizeof(topic.gravity));
	buf.iterator += sizeof(topic.gravity);
	buf.offset += sizeof(topic.gravity);
	static_assert(sizeof(topic.drag) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.drag, sizeof(topic.drag));
	buf.iterator += sizeof(topic.drag);
	buf.offset += sizeof(topic.drag);
	static_assert(sizeof(topic.airspeed) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.airspeed, sizeof(topic.airspeed));
	buf.iterator += sizeof(topic.airspeed);
	buf.offset += sizeof(topic.airspeed);
	static_assert(sizeof(topic.beta) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.beta, sizeof(topic.beta));
	buf.iterator += sizeof(topic.beta);
	buf.offset += sizeof(topic.beta);
	static_assert(sizeof(topic.hagl) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hagl, sizeof(topic.hagl));
	buf.iterator += sizeof(topic.hagl);
	buf.offset += sizeof(topic.hagl);
	static_assert(sizeof(topic.hagl_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.hagl_rate, sizeof(topic.hagl_rate));
	buf.iterator += sizeof(topic.hagl_rate);
	buf.offset += sizeof(topic.hagl_rate);
	return true;
}

bool ucdr_deserialize_estimator_innovations(ucdrBuffer& buf, estimator_innovations_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.timestamp_sample) == 8, "size mismatch");
	memcpy(&topic.timestamp_sample, buf.iterator, sizeof(topic.timestamp_sample));
	if (topic.timestamp_sample == 0) topic.timestamp_sample = hrt_absolute_time();
	else topic.timestamp_sample = math::min(topic.timestamp_sample - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp_sample);
	buf.offset += sizeof(topic.timestamp_sample);
	static_assert(sizeof(topic.gps_hvel) == 8, "size mismatch");
	memcpy(&topic.gps_hvel, buf.iterator, sizeof(topic.gps_hvel));
	buf.iterator += sizeof(topic.gps_hvel);
	buf.offset += sizeof(topic.gps_hvel);
	static_assert(sizeof(topic.gps_vvel) == 4, "size mismatch");
	memcpy(&topic.gps_vvel, buf.iterator, sizeof(topic.gps_vvel));
	buf.iterator += sizeof(topic.gps_vvel);
	buf.offset += sizeof(topic.gps_vvel);
	static_assert(sizeof(topic.gps_hpos) == 8, "size mismatch");
	memcpy(&topic.gps_hpos, buf.iterator, sizeof(topic.gps_hpos));
	buf.iterator += sizeof(topic.gps_hpos);
	buf.offset += sizeof(topic.gps_hpos);
	static_assert(sizeof(topic.gps_vpos) == 4, "size mismatch");
	memcpy(&topic.gps_vpos, buf.iterator, sizeof(topic.gps_vpos));
	buf.iterator += sizeof(topic.gps_vpos);
	buf.offset += sizeof(topic.gps_vpos);
	static_assert(sizeof(topic.ev_hvel) == 8, "size mismatch");
	memcpy(&topic.ev_hvel, buf.iterator, sizeof(topic.ev_hvel));
	buf.iterator += sizeof(topic.ev_hvel);
	buf.offset += sizeof(topic.ev_hvel);
	static_assert(sizeof(topic.ev_vvel) == 4, "size mismatch");
	memcpy(&topic.ev_vvel, buf.iterator, sizeof(topic.ev_vvel));
	buf.iterator += sizeof(topic.ev_vvel);
	buf.offset += sizeof(topic.ev_vvel);
	static_assert(sizeof(topic.ev_hpos) == 8, "size mismatch");
	memcpy(&topic.ev_hpos, buf.iterator, sizeof(topic.ev_hpos));
	buf.iterator += sizeof(topic.ev_hpos);
	buf.offset += sizeof(topic.ev_hpos);
	static_assert(sizeof(topic.ev_vpos) == 4, "size mismatch");
	memcpy(&topic.ev_vpos, buf.iterator, sizeof(topic.ev_vpos));
	buf.iterator += sizeof(topic.ev_vpos);
	buf.offset += sizeof(topic.ev_vpos);
	static_assert(sizeof(topic.rng_vpos) == 4, "size mismatch");
	memcpy(&topic.rng_vpos, buf.iterator, sizeof(topic.rng_vpos));
	buf.iterator += sizeof(topic.rng_vpos);
	buf.offset += sizeof(topic.rng_vpos);
	static_assert(sizeof(topic.baro_vpos) == 4, "size mismatch");
	memcpy(&topic.baro_vpos, buf.iterator, sizeof(topic.baro_vpos));
	buf.iterator += sizeof(topic.baro_vpos);
	buf.offset += sizeof(topic.baro_vpos);
	static_assert(sizeof(topic.aux_hvel) == 8, "size mismatch");
	memcpy(&topic.aux_hvel, buf.iterator, sizeof(topic.aux_hvel));
	buf.iterator += sizeof(topic.aux_hvel);
	buf.offset += sizeof(topic.aux_hvel);
	static_assert(sizeof(topic.aux_vvel) == 4, "size mismatch");
	memcpy(&topic.aux_vvel, buf.iterator, sizeof(topic.aux_vvel));
	buf.iterator += sizeof(topic.aux_vvel);
	buf.offset += sizeof(topic.aux_vvel);
	static_assert(sizeof(topic.flow) == 8, "size mismatch");
	memcpy(&topic.flow, buf.iterator, sizeof(topic.flow));
	buf.iterator += sizeof(topic.flow);
	buf.offset += sizeof(topic.flow);
	static_assert(sizeof(topic.terr_flow) == 8, "size mismatch");
	memcpy(&topic.terr_flow, buf.iterator, sizeof(topic.terr_flow));
	buf.iterator += sizeof(topic.terr_flow);
	buf.offset += sizeof(topic.terr_flow);
	static_assert(sizeof(topic.heading) == 4, "size mismatch");
	memcpy(&topic.heading, buf.iterator, sizeof(topic.heading));
	buf.iterator += sizeof(topic.heading);
	buf.offset += sizeof(topic.heading);
	static_assert(sizeof(topic.mag_field) == 12, "size mismatch");
	memcpy(&topic.mag_field, buf.iterator, sizeof(topic.mag_field));
	buf.iterator += sizeof(topic.mag_field);
	buf.offset += sizeof(topic.mag_field);
	static_assert(sizeof(topic.gravity) == 12, "size mismatch");
	memcpy(&topic.gravity, buf.iterator, sizeof(topic.gravity));
	buf.iterator += sizeof(topic.gravity);
	buf.offset += sizeof(topic.gravity);
	static_assert(sizeof(topic.drag) == 8, "size mismatch");
	memcpy(&topic.drag, buf.iterator, sizeof(topic.drag));
	buf.iterator += sizeof(topic.drag);
	buf.offset += sizeof(topic.drag);
	static_assert(sizeof(topic.airspeed) == 4, "size mismatch");
	memcpy(&topic.airspeed, buf.iterator, sizeof(topic.airspeed));
	buf.iterator += sizeof(topic.airspeed);
	buf.offset += sizeof(topic.airspeed);
	static_assert(sizeof(topic.beta) == 4, "size mismatch");
	memcpy(&topic.beta, buf.iterator, sizeof(topic.beta));
	buf.iterator += sizeof(topic.beta);
	buf.offset += sizeof(topic.beta);
	static_assert(sizeof(topic.hagl) == 4, "size mismatch");
	memcpy(&topic.hagl, buf.iterator, sizeof(topic.hagl));
	buf.iterator += sizeof(topic.hagl);
	buf.offset += sizeof(topic.hagl);
	static_assert(sizeof(topic.hagl_rate) == 4, "size mismatch");
	memcpy(&topic.hagl_rate, buf.iterator, sizeof(topic.hagl_rate));
	buf.iterator += sizeof(topic.hagl_rate);
	buf.offset += sizeof(topic.hagl_rate);
	return true;
}
