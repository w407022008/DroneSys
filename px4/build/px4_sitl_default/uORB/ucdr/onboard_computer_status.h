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
#include <uORB/topics/onboard_computer_status.h>


static inline constexpr int ucdr_topic_size_onboard_computer_status()
{
	return 240;
}

bool ucdr_serialize_onboard_computer_status(const onboard_computer_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.uptime) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.uptime, sizeof(topic.uptime));
	buf.iterator += sizeof(topic.uptime);
	buf.offset += sizeof(topic.uptime);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.cpu_cores) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.cpu_cores, sizeof(topic.cpu_cores));
	buf.iterator += sizeof(topic.cpu_cores);
	buf.offset += sizeof(topic.cpu_cores);
	static_assert(sizeof(topic.cpu_combined) == 10, "size mismatch");
	memcpy(buf.iterator, &topic.cpu_combined, sizeof(topic.cpu_combined));
	buf.iterator += sizeof(topic.cpu_combined);
	buf.offset += sizeof(topic.cpu_combined);
	static_assert(sizeof(topic.gpu_cores) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.gpu_cores, sizeof(topic.gpu_cores));
	buf.iterator += sizeof(topic.gpu_cores);
	buf.offset += sizeof(topic.gpu_cores);
	static_assert(sizeof(topic.gpu_combined) == 10, "size mismatch");
	memcpy(buf.iterator, &topic.gpu_combined, sizeof(topic.gpu_combined));
	buf.iterator += sizeof(topic.gpu_combined);
	buf.offset += sizeof(topic.gpu_combined);
	static_assert(sizeof(topic.temperature_board) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_board, sizeof(topic.temperature_board));
	buf.iterator += sizeof(topic.temperature_board);
	buf.offset += sizeof(topic.temperature_board);
	static_assert(sizeof(topic.temperature_core) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.temperature_core, sizeof(topic.temperature_core));
	buf.iterator += sizeof(topic.temperature_core);
	buf.offset += sizeof(topic.temperature_core);
	static_assert(sizeof(topic.fan_speed) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.fan_speed, sizeof(topic.fan_speed));
	buf.iterator += sizeof(topic.fan_speed);
	buf.offset += sizeof(topic.fan_speed);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.ram_usage) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ram_usage, sizeof(topic.ram_usage));
	buf.iterator += sizeof(topic.ram_usage);
	buf.offset += sizeof(topic.ram_usage);
	static_assert(sizeof(topic.ram_total) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ram_total, sizeof(topic.ram_total));
	buf.iterator += sizeof(topic.ram_total);
	buf.offset += sizeof(topic.ram_total);
	static_assert(sizeof(topic.storage_type) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.storage_type, sizeof(topic.storage_type));
	buf.iterator += sizeof(topic.storage_type);
	buf.offset += sizeof(topic.storage_type);
	static_assert(sizeof(topic.storage_usage) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.storage_usage, sizeof(topic.storage_usage));
	buf.iterator += sizeof(topic.storage_usage);
	buf.offset += sizeof(topic.storage_usage);
	static_assert(sizeof(topic.storage_total) == 16, "size mismatch");
	memcpy(buf.iterator, &topic.storage_total, sizeof(topic.storage_total));
	buf.iterator += sizeof(topic.storage_total);
	buf.offset += sizeof(topic.storage_total);
	static_assert(sizeof(topic.link_type) == 24, "size mismatch");
	memcpy(buf.iterator, &topic.link_type, sizeof(topic.link_type));
	buf.iterator += sizeof(topic.link_type);
	buf.offset += sizeof(topic.link_type);
	static_assert(sizeof(topic.link_tx_rate) == 24, "size mismatch");
	memcpy(buf.iterator, &topic.link_tx_rate, sizeof(topic.link_tx_rate));
	buf.iterator += sizeof(topic.link_tx_rate);
	buf.offset += sizeof(topic.link_tx_rate);
	static_assert(sizeof(topic.link_rx_rate) == 24, "size mismatch");
	memcpy(buf.iterator, &topic.link_rx_rate, sizeof(topic.link_rx_rate));
	buf.iterator += sizeof(topic.link_rx_rate);
	buf.offset += sizeof(topic.link_rx_rate);
	static_assert(sizeof(topic.link_tx_max) == 24, "size mismatch");
	memcpy(buf.iterator, &topic.link_tx_max, sizeof(topic.link_tx_max));
	buf.iterator += sizeof(topic.link_tx_max);
	buf.offset += sizeof(topic.link_tx_max);
	static_assert(sizeof(topic.link_rx_max) == 24, "size mismatch");
	memcpy(buf.iterator, &topic.link_rx_max, sizeof(topic.link_rx_max));
	buf.iterator += sizeof(topic.link_rx_max);
	buf.offset += sizeof(topic.link_rx_max);
	return true;
}

bool ucdr_deserialize_onboard_computer_status(ucdrBuffer& buf, onboard_computer_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.uptime) == 4, "size mismatch");
	memcpy(&topic.uptime, buf.iterator, sizeof(topic.uptime));
	buf.iterator += sizeof(topic.uptime);
	buf.offset += sizeof(topic.uptime);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.cpu_cores) == 8, "size mismatch");
	memcpy(&topic.cpu_cores, buf.iterator, sizeof(topic.cpu_cores));
	buf.iterator += sizeof(topic.cpu_cores);
	buf.offset += sizeof(topic.cpu_cores);
	static_assert(sizeof(topic.cpu_combined) == 10, "size mismatch");
	memcpy(&topic.cpu_combined, buf.iterator, sizeof(topic.cpu_combined));
	buf.iterator += sizeof(topic.cpu_combined);
	buf.offset += sizeof(topic.cpu_combined);
	static_assert(sizeof(topic.gpu_cores) == 4, "size mismatch");
	memcpy(&topic.gpu_cores, buf.iterator, sizeof(topic.gpu_cores));
	buf.iterator += sizeof(topic.gpu_cores);
	buf.offset += sizeof(topic.gpu_cores);
	static_assert(sizeof(topic.gpu_combined) == 10, "size mismatch");
	memcpy(&topic.gpu_combined, buf.iterator, sizeof(topic.gpu_combined));
	buf.iterator += sizeof(topic.gpu_combined);
	buf.offset += sizeof(topic.gpu_combined);
	static_assert(sizeof(topic.temperature_board) == 1, "size mismatch");
	memcpy(&topic.temperature_board, buf.iterator, sizeof(topic.temperature_board));
	buf.iterator += sizeof(topic.temperature_board);
	buf.offset += sizeof(topic.temperature_board);
	static_assert(sizeof(topic.temperature_core) == 8, "size mismatch");
	memcpy(&topic.temperature_core, buf.iterator, sizeof(topic.temperature_core));
	buf.iterator += sizeof(topic.temperature_core);
	buf.offset += sizeof(topic.temperature_core);
	static_assert(sizeof(topic.fan_speed) == 8, "size mismatch");
	memcpy(&topic.fan_speed, buf.iterator, sizeof(topic.fan_speed));
	buf.iterator += sizeof(topic.fan_speed);
	buf.offset += sizeof(topic.fan_speed);
	buf.iterator += 2; // padding
	buf.offset += 2; // padding
	static_assert(sizeof(topic.ram_usage) == 4, "size mismatch");
	memcpy(&topic.ram_usage, buf.iterator, sizeof(topic.ram_usage));
	buf.iterator += sizeof(topic.ram_usage);
	buf.offset += sizeof(topic.ram_usage);
	static_assert(sizeof(topic.ram_total) == 4, "size mismatch");
	memcpy(&topic.ram_total, buf.iterator, sizeof(topic.ram_total));
	buf.iterator += sizeof(topic.ram_total);
	buf.offset += sizeof(topic.ram_total);
	static_assert(sizeof(topic.storage_type) == 16, "size mismatch");
	memcpy(&topic.storage_type, buf.iterator, sizeof(topic.storage_type));
	buf.iterator += sizeof(topic.storage_type);
	buf.offset += sizeof(topic.storage_type);
	static_assert(sizeof(topic.storage_usage) == 16, "size mismatch");
	memcpy(&topic.storage_usage, buf.iterator, sizeof(topic.storage_usage));
	buf.iterator += sizeof(topic.storage_usage);
	buf.offset += sizeof(topic.storage_usage);
	static_assert(sizeof(topic.storage_total) == 16, "size mismatch");
	memcpy(&topic.storage_total, buf.iterator, sizeof(topic.storage_total));
	buf.iterator += sizeof(topic.storage_total);
	buf.offset += sizeof(topic.storage_total);
	static_assert(sizeof(topic.link_type) == 24, "size mismatch");
	memcpy(&topic.link_type, buf.iterator, sizeof(topic.link_type));
	buf.iterator += sizeof(topic.link_type);
	buf.offset += sizeof(topic.link_type);
	static_assert(sizeof(topic.link_tx_rate) == 24, "size mismatch");
	memcpy(&topic.link_tx_rate, buf.iterator, sizeof(topic.link_tx_rate));
	buf.iterator += sizeof(topic.link_tx_rate);
	buf.offset += sizeof(topic.link_tx_rate);
	static_assert(sizeof(topic.link_rx_rate) == 24, "size mismatch");
	memcpy(&topic.link_rx_rate, buf.iterator, sizeof(topic.link_rx_rate));
	buf.iterator += sizeof(topic.link_rx_rate);
	buf.offset += sizeof(topic.link_rx_rate);
	static_assert(sizeof(topic.link_tx_max) == 24, "size mismatch");
	memcpy(&topic.link_tx_max, buf.iterator, sizeof(topic.link_tx_max));
	buf.iterator += sizeof(topic.link_tx_max);
	buf.offset += sizeof(topic.link_tx_max);
	static_assert(sizeof(topic.link_rx_max) == 24, "size mismatch");
	memcpy(&topic.link_rx_max, buf.iterator, sizeof(topic.link_rx_max));
	buf.iterator += sizeof(topic.link_rx_max);
	buf.offset += sizeof(topic.link_rx_max);
	return true;
}
