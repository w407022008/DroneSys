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
#include <uORB/topics/vehicle_land_detected.h>


static inline constexpr int ucdr_topic_size_vehicle_land_detected()
{
	return 20;
}

bool ucdr_serialize_vehicle_land_detected(const vehicle_land_detected_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.freefall) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.freefall, sizeof(topic.freefall));
	buf.iterator += sizeof(topic.freefall);
	buf.offset += sizeof(topic.freefall);
	static_assert(sizeof(topic.ground_contact) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.ground_contact, sizeof(topic.ground_contact));
	buf.iterator += sizeof(topic.ground_contact);
	buf.offset += sizeof(topic.ground_contact);
	static_assert(sizeof(topic.maybe_landed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.maybe_landed, sizeof(topic.maybe_landed));
	buf.iterator += sizeof(topic.maybe_landed);
	buf.offset += sizeof(topic.maybe_landed);
	static_assert(sizeof(topic.landed) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.landed, sizeof(topic.landed));
	buf.iterator += sizeof(topic.landed);
	buf.offset += sizeof(topic.landed);
	static_assert(sizeof(topic.in_ground_effect) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.in_ground_effect, sizeof(topic.in_ground_effect));
	buf.iterator += sizeof(topic.in_ground_effect);
	buf.offset += sizeof(topic.in_ground_effect);
	static_assert(sizeof(topic.in_descend) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.in_descend, sizeof(topic.in_descend));
	buf.iterator += sizeof(topic.in_descend);
	buf.offset += sizeof(topic.in_descend);
	static_assert(sizeof(topic.has_low_throttle) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.has_low_throttle, sizeof(topic.has_low_throttle));
	buf.iterator += sizeof(topic.has_low_throttle);
	buf.offset += sizeof(topic.has_low_throttle);
	static_assert(sizeof(topic.vertical_movement) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.vertical_movement, sizeof(topic.vertical_movement));
	buf.iterator += sizeof(topic.vertical_movement);
	buf.offset += sizeof(topic.vertical_movement);
	static_assert(sizeof(topic.horizontal_movement) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.horizontal_movement, sizeof(topic.horizontal_movement));
	buf.iterator += sizeof(topic.horizontal_movement);
	buf.offset += sizeof(topic.horizontal_movement);
	static_assert(sizeof(topic.rotational_movement) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rotational_movement, sizeof(topic.rotational_movement));
	buf.iterator += sizeof(topic.rotational_movement);
	buf.offset += sizeof(topic.rotational_movement);
	static_assert(sizeof(topic.close_to_ground_or_skipped_check) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.close_to_ground_or_skipped_check, sizeof(topic.close_to_ground_or_skipped_check));
	buf.iterator += sizeof(topic.close_to_ground_or_skipped_check);
	buf.offset += sizeof(topic.close_to_ground_or_skipped_check);
	static_assert(sizeof(topic.at_rest) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.at_rest, sizeof(topic.at_rest));
	buf.iterator += sizeof(topic.at_rest);
	buf.offset += sizeof(topic.at_rest);
	return true;
}

bool ucdr_deserialize_vehicle_land_detected(ucdrBuffer& buf, vehicle_land_detected_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.freefall) == 1, "size mismatch");
	memcpy(&topic.freefall, buf.iterator, sizeof(topic.freefall));
	buf.iterator += sizeof(topic.freefall);
	buf.offset += sizeof(topic.freefall);
	static_assert(sizeof(topic.ground_contact) == 1, "size mismatch");
	memcpy(&topic.ground_contact, buf.iterator, sizeof(topic.ground_contact));
	buf.iterator += sizeof(topic.ground_contact);
	buf.offset += sizeof(topic.ground_contact);
	static_assert(sizeof(topic.maybe_landed) == 1, "size mismatch");
	memcpy(&topic.maybe_landed, buf.iterator, sizeof(topic.maybe_landed));
	buf.iterator += sizeof(topic.maybe_landed);
	buf.offset += sizeof(topic.maybe_landed);
	static_assert(sizeof(topic.landed) == 1, "size mismatch");
	memcpy(&topic.landed, buf.iterator, sizeof(topic.landed));
	buf.iterator += sizeof(topic.landed);
	buf.offset += sizeof(topic.landed);
	static_assert(sizeof(topic.in_ground_effect) == 1, "size mismatch");
	memcpy(&topic.in_ground_effect, buf.iterator, sizeof(topic.in_ground_effect));
	buf.iterator += sizeof(topic.in_ground_effect);
	buf.offset += sizeof(topic.in_ground_effect);
	static_assert(sizeof(topic.in_descend) == 1, "size mismatch");
	memcpy(&topic.in_descend, buf.iterator, sizeof(topic.in_descend));
	buf.iterator += sizeof(topic.in_descend);
	buf.offset += sizeof(topic.in_descend);
	static_assert(sizeof(topic.has_low_throttle) == 1, "size mismatch");
	memcpy(&topic.has_low_throttle, buf.iterator, sizeof(topic.has_low_throttle));
	buf.iterator += sizeof(topic.has_low_throttle);
	buf.offset += sizeof(topic.has_low_throttle);
	static_assert(sizeof(topic.vertical_movement) == 1, "size mismatch");
	memcpy(&topic.vertical_movement, buf.iterator, sizeof(topic.vertical_movement));
	buf.iterator += sizeof(topic.vertical_movement);
	buf.offset += sizeof(topic.vertical_movement);
	static_assert(sizeof(topic.horizontal_movement) == 1, "size mismatch");
	memcpy(&topic.horizontal_movement, buf.iterator, sizeof(topic.horizontal_movement));
	buf.iterator += sizeof(topic.horizontal_movement);
	buf.offset += sizeof(topic.horizontal_movement);
	static_assert(sizeof(topic.rotational_movement) == 1, "size mismatch");
	memcpy(&topic.rotational_movement, buf.iterator, sizeof(topic.rotational_movement));
	buf.iterator += sizeof(topic.rotational_movement);
	buf.offset += sizeof(topic.rotational_movement);
	static_assert(sizeof(topic.close_to_ground_or_skipped_check) == 1, "size mismatch");
	memcpy(&topic.close_to_ground_or_skipped_check, buf.iterator, sizeof(topic.close_to_ground_or_skipped_check));
	buf.iterator += sizeof(topic.close_to_ground_or_skipped_check);
	buf.offset += sizeof(topic.close_to_ground_or_skipped_check);
	static_assert(sizeof(topic.at_rest) == 1, "size mismatch");
	memcpy(&topic.at_rest, buf.iterator, sizeof(topic.at_rest));
	buf.iterator += sizeof(topic.at_rest);
	buf.offset += sizeof(topic.at_rest);
	return true;
}
