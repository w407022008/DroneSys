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
#include <uORB/topics/mag_worker_data.h>


static inline constexpr int ucdr_topic_size_mag_worker_data()
{
	return 100;
}

bool ucdr_serialize_mag_worker_data(const mag_worker_data_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.done_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.done_count, sizeof(topic.done_count));
	buf.iterator += sizeof(topic.done_count);
	buf.offset += sizeof(topic.done_count);
	static_assert(sizeof(topic.calibration_points_perside) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.calibration_points_perside, sizeof(topic.calibration_points_perside));
	buf.iterator += sizeof(topic.calibration_points_perside);
	buf.offset += sizeof(topic.calibration_points_perside);
	static_assert(sizeof(topic.calibration_interval_perside_us) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.calibration_interval_perside_us, sizeof(topic.calibration_interval_perside_us));
	buf.iterator += sizeof(topic.calibration_interval_perside_us);
	buf.offset += sizeof(topic.calibration_interval_perside_us);
	static_assert(sizeof(topic.calibration_counter_total) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.calibration_counter_total, sizeof(topic.calibration_counter_total));
	buf.iterator += sizeof(topic.calibration_counter_total);
	buf.offset += sizeof(topic.calibration_counter_total);
	static_assert(sizeof(topic.side_data_collected) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.side_data_collected, sizeof(topic.side_data_collected));
	buf.iterator += sizeof(topic.side_data_collected);
	buf.offset += sizeof(topic.side_data_collected);
	static_assert(sizeof(topic.x) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.x, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.y, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.z, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	return true;
}

bool ucdr_deserialize_mag_worker_data(ucdrBuffer& buf, mag_worker_data_s& topic, int64_t time_offset = 0)
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
	static_assert(sizeof(topic.done_count) == 4, "size mismatch");
	memcpy(&topic.done_count, buf.iterator, sizeof(topic.done_count));
	buf.iterator += sizeof(topic.done_count);
	buf.offset += sizeof(topic.done_count);
	static_assert(sizeof(topic.calibration_points_perside) == 4, "size mismatch");
	memcpy(&topic.calibration_points_perside, buf.iterator, sizeof(topic.calibration_points_perside));
	buf.iterator += sizeof(topic.calibration_points_perside);
	buf.offset += sizeof(topic.calibration_points_perside);
	static_assert(sizeof(topic.calibration_interval_perside_us) == 8, "size mismatch");
	memcpy(&topic.calibration_interval_perside_us, buf.iterator, sizeof(topic.calibration_interval_perside_us));
	buf.iterator += sizeof(topic.calibration_interval_perside_us);
	buf.offset += sizeof(topic.calibration_interval_perside_us);
	static_assert(sizeof(topic.calibration_counter_total) == 16, "size mismatch");
	memcpy(&topic.calibration_counter_total, buf.iterator, sizeof(topic.calibration_counter_total));
	buf.iterator += sizeof(topic.calibration_counter_total);
	buf.offset += sizeof(topic.calibration_counter_total);
	static_assert(sizeof(topic.side_data_collected) == 4, "size mismatch");
	memcpy(&topic.side_data_collected, buf.iterator, sizeof(topic.side_data_collected));
	buf.iterator += sizeof(topic.side_data_collected);
	buf.offset += sizeof(topic.side_data_collected);
	static_assert(sizeof(topic.x) == 16, "size mismatch");
	memcpy(&topic.x, buf.iterator, sizeof(topic.x));
	buf.iterator += sizeof(topic.x);
	buf.offset += sizeof(topic.x);
	static_assert(sizeof(topic.y) == 16, "size mismatch");
	memcpy(&topic.y, buf.iterator, sizeof(topic.y));
	buf.iterator += sizeof(topic.y);
	buf.offset += sizeof(topic.y);
	static_assert(sizeof(topic.z) == 16, "size mismatch");
	memcpy(&topic.z, buf.iterator, sizeof(topic.z));
	buf.iterator += sizeof(topic.z);
	buf.offset += sizeof(topic.z);
	return true;
}
