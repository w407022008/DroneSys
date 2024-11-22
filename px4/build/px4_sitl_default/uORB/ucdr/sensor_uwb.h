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
#include <uORB/topics/sensor_uwb.h>


static inline constexpr int ucdr_topic_size_sensor_uwb()
{
	return 68;
}

bool ucdr_serialize_sensor_uwb(const sensor_uwb_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.sessionid) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.sessionid, sizeof(topic.sessionid));
	buf.iterator += sizeof(topic.sessionid);
	buf.offset += sizeof(topic.sessionid);
	static_assert(sizeof(topic.time_offset) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.time_offset, sizeof(topic.time_offset));
	buf.iterator += sizeof(topic.time_offset);
	buf.offset += sizeof(topic.time_offset);
	static_assert(sizeof(topic.counter) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.counter, sizeof(topic.counter));
	buf.iterator += sizeof(topic.counter);
	buf.offset += sizeof(topic.counter);
	static_assert(sizeof(topic.mac) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.mac, sizeof(topic.mac));
	buf.iterator += sizeof(topic.mac);
	buf.offset += sizeof(topic.mac);
	static_assert(sizeof(topic.mac_dest) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.mac_dest, sizeof(topic.mac_dest));
	buf.iterator += sizeof(topic.mac_dest);
	buf.offset += sizeof(topic.mac_dest);
	static_assert(sizeof(topic.status) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.status, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.nlos) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.nlos, sizeof(topic.nlos));
	buf.iterator += sizeof(topic.nlos);
	buf.offset += sizeof(topic.nlos);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.distance) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.distance, sizeof(topic.distance));
	buf.iterator += sizeof(topic.distance);
	buf.offset += sizeof(topic.distance);
	static_assert(sizeof(topic.aoa_azimuth_dev) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_azimuth_dev, sizeof(topic.aoa_azimuth_dev));
	buf.iterator += sizeof(topic.aoa_azimuth_dev);
	buf.offset += sizeof(topic.aoa_azimuth_dev);
	static_assert(sizeof(topic.aoa_elevation_dev) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_elevation_dev, sizeof(topic.aoa_elevation_dev));
	buf.iterator += sizeof(topic.aoa_elevation_dev);
	buf.offset += sizeof(topic.aoa_elevation_dev);
	static_assert(sizeof(topic.aoa_azimuth_resp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_azimuth_resp, sizeof(topic.aoa_azimuth_resp));
	buf.iterator += sizeof(topic.aoa_azimuth_resp);
	buf.offset += sizeof(topic.aoa_azimuth_resp);
	static_assert(sizeof(topic.aoa_elevation_resp) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_elevation_resp, sizeof(topic.aoa_elevation_resp));
	buf.iterator += sizeof(topic.aoa_elevation_resp);
	buf.offset += sizeof(topic.aoa_elevation_resp);
	static_assert(sizeof(topic.aoa_azimuth_fom) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_azimuth_fom, sizeof(topic.aoa_azimuth_fom));
	buf.iterator += sizeof(topic.aoa_azimuth_fom);
	buf.offset += sizeof(topic.aoa_azimuth_fom);
	static_assert(sizeof(topic.aoa_elevation_fom) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_elevation_fom, sizeof(topic.aoa_elevation_fom));
	buf.iterator += sizeof(topic.aoa_elevation_fom);
	buf.offset += sizeof(topic.aoa_elevation_fom);
	static_assert(sizeof(topic.aoa_dest_azimuth_fom) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_dest_azimuth_fom, sizeof(topic.aoa_dest_azimuth_fom));
	buf.iterator += sizeof(topic.aoa_dest_azimuth_fom);
	buf.offset += sizeof(topic.aoa_dest_azimuth_fom);
	static_assert(sizeof(topic.aoa_dest_elevation_fom) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.aoa_dest_elevation_fom, sizeof(topic.aoa_dest_elevation_fom));
	buf.iterator += sizeof(topic.aoa_dest_elevation_fom);
	buf.offset += sizeof(topic.aoa_dest_elevation_fom);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.orientation, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.offset_x) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.offset_x, sizeof(topic.offset_x));
	buf.iterator += sizeof(topic.offset_x);
	buf.offset += sizeof(topic.offset_x);
	static_assert(sizeof(topic.offset_y) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.offset_y, sizeof(topic.offset_y));
	buf.iterator += sizeof(topic.offset_y);
	buf.offset += sizeof(topic.offset_y);
	static_assert(sizeof(topic.offset_z) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.offset_z, sizeof(topic.offset_z));
	buf.iterator += sizeof(topic.offset_z);
	buf.offset += sizeof(topic.offset_z);
	return true;
}

bool ucdr_deserialize_sensor_uwb(ucdrBuffer& buf, sensor_uwb_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.sessionid) == 4, "size mismatch");
	memcpy(&topic.sessionid, buf.iterator, sizeof(topic.sessionid));
	buf.iterator += sizeof(topic.sessionid);
	buf.offset += sizeof(topic.sessionid);
	static_assert(sizeof(topic.time_offset) == 4, "size mismatch");
	memcpy(&topic.time_offset, buf.iterator, sizeof(topic.time_offset));
	buf.iterator += sizeof(topic.time_offset);
	buf.offset += sizeof(topic.time_offset);
	static_assert(sizeof(topic.counter) == 4, "size mismatch");
	memcpy(&topic.counter, buf.iterator, sizeof(topic.counter));
	buf.iterator += sizeof(topic.counter);
	buf.offset += sizeof(topic.counter);
	static_assert(sizeof(topic.mac) == 2, "size mismatch");
	memcpy(&topic.mac, buf.iterator, sizeof(topic.mac));
	buf.iterator += sizeof(topic.mac);
	buf.offset += sizeof(topic.mac);
	static_assert(sizeof(topic.mac_dest) == 2, "size mismatch");
	memcpy(&topic.mac_dest, buf.iterator, sizeof(topic.mac_dest));
	buf.iterator += sizeof(topic.mac_dest);
	buf.offset += sizeof(topic.mac_dest);
	static_assert(sizeof(topic.status) == 2, "size mismatch");
	memcpy(&topic.status, buf.iterator, sizeof(topic.status));
	buf.iterator += sizeof(topic.status);
	buf.offset += sizeof(topic.status);
	static_assert(sizeof(topic.nlos) == 1, "size mismatch");
	memcpy(&topic.nlos, buf.iterator, sizeof(topic.nlos));
	buf.iterator += sizeof(topic.nlos);
	buf.offset += sizeof(topic.nlos);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.distance) == 4, "size mismatch");
	memcpy(&topic.distance, buf.iterator, sizeof(topic.distance));
	buf.iterator += sizeof(topic.distance);
	buf.offset += sizeof(topic.distance);
	static_assert(sizeof(topic.aoa_azimuth_dev) == 4, "size mismatch");
	memcpy(&topic.aoa_azimuth_dev, buf.iterator, sizeof(topic.aoa_azimuth_dev));
	buf.iterator += sizeof(topic.aoa_azimuth_dev);
	buf.offset += sizeof(topic.aoa_azimuth_dev);
	static_assert(sizeof(topic.aoa_elevation_dev) == 4, "size mismatch");
	memcpy(&topic.aoa_elevation_dev, buf.iterator, sizeof(topic.aoa_elevation_dev));
	buf.iterator += sizeof(topic.aoa_elevation_dev);
	buf.offset += sizeof(topic.aoa_elevation_dev);
	static_assert(sizeof(topic.aoa_azimuth_resp) == 4, "size mismatch");
	memcpy(&topic.aoa_azimuth_resp, buf.iterator, sizeof(topic.aoa_azimuth_resp));
	buf.iterator += sizeof(topic.aoa_azimuth_resp);
	buf.offset += sizeof(topic.aoa_azimuth_resp);
	static_assert(sizeof(topic.aoa_elevation_resp) == 4, "size mismatch");
	memcpy(&topic.aoa_elevation_resp, buf.iterator, sizeof(topic.aoa_elevation_resp));
	buf.iterator += sizeof(topic.aoa_elevation_resp);
	buf.offset += sizeof(topic.aoa_elevation_resp);
	static_assert(sizeof(topic.aoa_azimuth_fom) == 1, "size mismatch");
	memcpy(&topic.aoa_azimuth_fom, buf.iterator, sizeof(topic.aoa_azimuth_fom));
	buf.iterator += sizeof(topic.aoa_azimuth_fom);
	buf.offset += sizeof(topic.aoa_azimuth_fom);
	static_assert(sizeof(topic.aoa_elevation_fom) == 1, "size mismatch");
	memcpy(&topic.aoa_elevation_fom, buf.iterator, sizeof(topic.aoa_elevation_fom));
	buf.iterator += sizeof(topic.aoa_elevation_fom);
	buf.offset += sizeof(topic.aoa_elevation_fom);
	static_assert(sizeof(topic.aoa_dest_azimuth_fom) == 1, "size mismatch");
	memcpy(&topic.aoa_dest_azimuth_fom, buf.iterator, sizeof(topic.aoa_dest_azimuth_fom));
	buf.iterator += sizeof(topic.aoa_dest_azimuth_fom);
	buf.offset += sizeof(topic.aoa_dest_azimuth_fom);
	static_assert(sizeof(topic.aoa_dest_elevation_fom) == 1, "size mismatch");
	memcpy(&topic.aoa_dest_elevation_fom, buf.iterator, sizeof(topic.aoa_dest_elevation_fom));
	buf.iterator += sizeof(topic.aoa_dest_elevation_fom);
	buf.offset += sizeof(topic.aoa_dest_elevation_fom);
	static_assert(sizeof(topic.orientation) == 1, "size mismatch");
	memcpy(&topic.orientation, buf.iterator, sizeof(topic.orientation));
	buf.iterator += sizeof(topic.orientation);
	buf.offset += sizeof(topic.orientation);
	buf.iterator += 3; // padding
	buf.offset += 3; // padding
	static_assert(sizeof(topic.offset_x) == 4, "size mismatch");
	memcpy(&topic.offset_x, buf.iterator, sizeof(topic.offset_x));
	buf.iterator += sizeof(topic.offset_x);
	buf.offset += sizeof(topic.offset_x);
	static_assert(sizeof(topic.offset_y) == 4, "size mismatch");
	memcpy(&topic.offset_y, buf.iterator, sizeof(topic.offset_y));
	buf.iterator += sizeof(topic.offset_y);
	buf.offset += sizeof(topic.offset_y);
	static_assert(sizeof(topic.offset_z) == 4, "size mismatch");
	memcpy(&topic.offset_z, buf.iterator, sizeof(topic.offset_z));
	buf.iterator += sizeof(topic.offset_z);
	buf.offset += sizeof(topic.offset_z);
	return true;
}
