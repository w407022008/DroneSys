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
#include <uORB/topics/system_power.h>


static inline constexpr int ucdr_topic_size_system_power()
{
	return 37;
}

bool ucdr_serialize_system_power(const system_power_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.voltage5v_v) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.voltage5v_v, sizeof(topic.voltage5v_v));
	buf.iterator += sizeof(topic.voltage5v_v);
	buf.offset += sizeof(topic.voltage5v_v);
	static_assert(sizeof(topic.sensors3v3) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.sensors3v3, sizeof(topic.sensors3v3));
	buf.iterator += sizeof(topic.sensors3v3);
	buf.offset += sizeof(topic.sensors3v3);
	static_assert(sizeof(topic.sensors3v3_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.sensors3v3_valid, sizeof(topic.sensors3v3_valid));
	buf.iterator += sizeof(topic.sensors3v3_valid);
	buf.offset += sizeof(topic.sensors3v3_valid);
	static_assert(sizeof(topic.usb_connected) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.usb_connected, sizeof(topic.usb_connected));
	buf.iterator += sizeof(topic.usb_connected);
	buf.offset += sizeof(topic.usb_connected);
	static_assert(sizeof(topic.brick_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.brick_valid, sizeof(topic.brick_valid));
	buf.iterator += sizeof(topic.brick_valid);
	buf.offset += sizeof(topic.brick_valid);
	static_assert(sizeof(topic.usb_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.usb_valid, sizeof(topic.usb_valid));
	buf.iterator += sizeof(topic.usb_valid);
	buf.offset += sizeof(topic.usb_valid);
	static_assert(sizeof(topic.servo_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.servo_valid, sizeof(topic.servo_valid));
	buf.iterator += sizeof(topic.servo_valid);
	buf.offset += sizeof(topic.servo_valid);
	static_assert(sizeof(topic.periph_5v_oc) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.periph_5v_oc, sizeof(topic.periph_5v_oc));
	buf.iterator += sizeof(topic.periph_5v_oc);
	buf.offset += sizeof(topic.periph_5v_oc);
	static_assert(sizeof(topic.hipower_5v_oc) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.hipower_5v_oc, sizeof(topic.hipower_5v_oc));
	buf.iterator += sizeof(topic.hipower_5v_oc);
	buf.offset += sizeof(topic.hipower_5v_oc);
	static_assert(sizeof(topic.comp_5v_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.comp_5v_valid, sizeof(topic.comp_5v_valid));
	buf.iterator += sizeof(topic.comp_5v_valid);
	buf.offset += sizeof(topic.comp_5v_valid);
	static_assert(sizeof(topic.can1_gps1_5v_valid) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.can1_gps1_5v_valid, sizeof(topic.can1_gps1_5v_valid));
	buf.iterator += sizeof(topic.can1_gps1_5v_valid);
	buf.offset += sizeof(topic.can1_gps1_5v_valid);
	return true;
}

bool ucdr_deserialize_system_power(ucdrBuffer& buf, system_power_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.voltage5v_v) == 4, "size mismatch");
	memcpy(&topic.voltage5v_v, buf.iterator, sizeof(topic.voltage5v_v));
	buf.iterator += sizeof(topic.voltage5v_v);
	buf.offset += sizeof(topic.voltage5v_v);
	static_assert(sizeof(topic.sensors3v3) == 16, "size mismatch");
	memcpy(&topic.sensors3v3, buf.iterator, sizeof(topic.sensors3v3));
	buf.iterator += sizeof(topic.sensors3v3);
	buf.offset += sizeof(topic.sensors3v3);
	static_assert(sizeof(topic.sensors3v3_valid) == 1, "size mismatch");
	memcpy(&topic.sensors3v3_valid, buf.iterator, sizeof(topic.sensors3v3_valid));
	buf.iterator += sizeof(topic.sensors3v3_valid);
	buf.offset += sizeof(topic.sensors3v3_valid);
	static_assert(sizeof(topic.usb_connected) == 1, "size mismatch");
	memcpy(&topic.usb_connected, buf.iterator, sizeof(topic.usb_connected));
	buf.iterator += sizeof(topic.usb_connected);
	buf.offset += sizeof(topic.usb_connected);
	static_assert(sizeof(topic.brick_valid) == 1, "size mismatch");
	memcpy(&topic.brick_valid, buf.iterator, sizeof(topic.brick_valid));
	buf.iterator += sizeof(topic.brick_valid);
	buf.offset += sizeof(topic.brick_valid);
	static_assert(sizeof(topic.usb_valid) == 1, "size mismatch");
	memcpy(&topic.usb_valid, buf.iterator, sizeof(topic.usb_valid));
	buf.iterator += sizeof(topic.usb_valid);
	buf.offset += sizeof(topic.usb_valid);
	static_assert(sizeof(topic.servo_valid) == 1, "size mismatch");
	memcpy(&topic.servo_valid, buf.iterator, sizeof(topic.servo_valid));
	buf.iterator += sizeof(topic.servo_valid);
	buf.offset += sizeof(topic.servo_valid);
	static_assert(sizeof(topic.periph_5v_oc) == 1, "size mismatch");
	memcpy(&topic.periph_5v_oc, buf.iterator, sizeof(topic.periph_5v_oc));
	buf.iterator += sizeof(topic.periph_5v_oc);
	buf.offset += sizeof(topic.periph_5v_oc);
	static_assert(sizeof(topic.hipower_5v_oc) == 1, "size mismatch");
	memcpy(&topic.hipower_5v_oc, buf.iterator, sizeof(topic.hipower_5v_oc));
	buf.iterator += sizeof(topic.hipower_5v_oc);
	buf.offset += sizeof(topic.hipower_5v_oc);
	static_assert(sizeof(topic.comp_5v_valid) == 1, "size mismatch");
	memcpy(&topic.comp_5v_valid, buf.iterator, sizeof(topic.comp_5v_valid));
	buf.iterator += sizeof(topic.comp_5v_valid);
	buf.offset += sizeof(topic.comp_5v_valid);
	static_assert(sizeof(topic.can1_gps1_5v_valid) == 1, "size mismatch");
	memcpy(&topic.can1_gps1_5v_valid, buf.iterator, sizeof(topic.can1_gps1_5v_valid));
	buf.iterator += sizeof(topic.can1_gps1_5v_valid);
	buf.offset += sizeof(topic.can1_gps1_5v_valid);
	return true;
}
