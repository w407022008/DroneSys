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
#include <uORB/topics/iridiumsbd_status.h>


static inline constexpr int ucdr_topic_size_iridiumsbd_status()
{
	return 35;
}

bool ucdr_serialize_iridiumsbd_status(const iridiumsbd_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.last_heartbeat) == 8, "size mismatch");
	memcpy(buf.iterator, &topic.last_heartbeat, sizeof(topic.last_heartbeat));
	buf.iterator += sizeof(topic.last_heartbeat);
	buf.offset += sizeof(topic.last_heartbeat);
	static_assert(sizeof(topic.tx_buf_write_index) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.tx_buf_write_index, sizeof(topic.tx_buf_write_index));
	buf.iterator += sizeof(topic.tx_buf_write_index);
	buf.offset += sizeof(topic.tx_buf_write_index);
	static_assert(sizeof(topic.rx_buf_read_index) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rx_buf_read_index, sizeof(topic.rx_buf_read_index));
	buf.iterator += sizeof(topic.rx_buf_read_index);
	buf.offset += sizeof(topic.rx_buf_read_index);
	static_assert(sizeof(topic.rx_buf_end_index) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.rx_buf_end_index, sizeof(topic.rx_buf_end_index));
	buf.iterator += sizeof(topic.rx_buf_end_index);
	buf.offset += sizeof(topic.rx_buf_end_index);
	static_assert(sizeof(topic.failed_sbd_sessions) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.failed_sbd_sessions, sizeof(topic.failed_sbd_sessions));
	buf.iterator += sizeof(topic.failed_sbd_sessions);
	buf.offset += sizeof(topic.failed_sbd_sessions);
	static_assert(sizeof(topic.successful_sbd_sessions) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.successful_sbd_sessions, sizeof(topic.successful_sbd_sessions));
	buf.iterator += sizeof(topic.successful_sbd_sessions);
	buf.offset += sizeof(topic.successful_sbd_sessions);
	static_assert(sizeof(topic.num_tx_buf_reset) == 2, "size mismatch");
	memcpy(buf.iterator, &topic.num_tx_buf_reset, sizeof(topic.num_tx_buf_reset));
	buf.iterator += sizeof(topic.num_tx_buf_reset);
	buf.offset += sizeof(topic.num_tx_buf_reset);
	static_assert(sizeof(topic.signal_quality) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.signal_quality, sizeof(topic.signal_quality));
	buf.iterator += sizeof(topic.signal_quality);
	buf.offset += sizeof(topic.signal_quality);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.state, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	static_assert(sizeof(topic.ring_pending) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.ring_pending, sizeof(topic.ring_pending));
	buf.iterator += sizeof(topic.ring_pending);
	buf.offset += sizeof(topic.ring_pending);
	static_assert(sizeof(topic.tx_buf_write_pending) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.tx_buf_write_pending, sizeof(topic.tx_buf_write_pending));
	buf.iterator += sizeof(topic.tx_buf_write_pending);
	buf.offset += sizeof(topic.tx_buf_write_pending);
	static_assert(sizeof(topic.tx_session_pending) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.tx_session_pending, sizeof(topic.tx_session_pending));
	buf.iterator += sizeof(topic.tx_session_pending);
	buf.offset += sizeof(topic.tx_session_pending);
	static_assert(sizeof(topic.rx_read_pending) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rx_read_pending, sizeof(topic.rx_read_pending));
	buf.iterator += sizeof(topic.rx_read_pending);
	buf.offset += sizeof(topic.rx_read_pending);
	static_assert(sizeof(topic.rx_session_pending) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.rx_session_pending, sizeof(topic.rx_session_pending));
	buf.iterator += sizeof(topic.rx_session_pending);
	buf.offset += sizeof(topic.rx_session_pending);
	return true;
}

bool ucdr_deserialize_iridiumsbd_status(ucdrBuffer& buf, iridiumsbd_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.last_heartbeat) == 8, "size mismatch");
	memcpy(&topic.last_heartbeat, buf.iterator, sizeof(topic.last_heartbeat));
	buf.iterator += sizeof(topic.last_heartbeat);
	buf.offset += sizeof(topic.last_heartbeat);
	static_assert(sizeof(topic.tx_buf_write_index) == 2, "size mismatch");
	memcpy(&topic.tx_buf_write_index, buf.iterator, sizeof(topic.tx_buf_write_index));
	buf.iterator += sizeof(topic.tx_buf_write_index);
	buf.offset += sizeof(topic.tx_buf_write_index);
	static_assert(sizeof(topic.rx_buf_read_index) == 2, "size mismatch");
	memcpy(&topic.rx_buf_read_index, buf.iterator, sizeof(topic.rx_buf_read_index));
	buf.iterator += sizeof(topic.rx_buf_read_index);
	buf.offset += sizeof(topic.rx_buf_read_index);
	static_assert(sizeof(topic.rx_buf_end_index) == 2, "size mismatch");
	memcpy(&topic.rx_buf_end_index, buf.iterator, sizeof(topic.rx_buf_end_index));
	buf.iterator += sizeof(topic.rx_buf_end_index);
	buf.offset += sizeof(topic.rx_buf_end_index);
	static_assert(sizeof(topic.failed_sbd_sessions) == 2, "size mismatch");
	memcpy(&topic.failed_sbd_sessions, buf.iterator, sizeof(topic.failed_sbd_sessions));
	buf.iterator += sizeof(topic.failed_sbd_sessions);
	buf.offset += sizeof(topic.failed_sbd_sessions);
	static_assert(sizeof(topic.successful_sbd_sessions) == 2, "size mismatch");
	memcpy(&topic.successful_sbd_sessions, buf.iterator, sizeof(topic.successful_sbd_sessions));
	buf.iterator += sizeof(topic.successful_sbd_sessions);
	buf.offset += sizeof(topic.successful_sbd_sessions);
	static_assert(sizeof(topic.num_tx_buf_reset) == 2, "size mismatch");
	memcpy(&topic.num_tx_buf_reset, buf.iterator, sizeof(topic.num_tx_buf_reset));
	buf.iterator += sizeof(topic.num_tx_buf_reset);
	buf.offset += sizeof(topic.num_tx_buf_reset);
	static_assert(sizeof(topic.signal_quality) == 1, "size mismatch");
	memcpy(&topic.signal_quality, buf.iterator, sizeof(topic.signal_quality));
	buf.iterator += sizeof(topic.signal_quality);
	buf.offset += sizeof(topic.signal_quality);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(&topic.state, buf.iterator, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	static_assert(sizeof(topic.ring_pending) == 1, "size mismatch");
	memcpy(&topic.ring_pending, buf.iterator, sizeof(topic.ring_pending));
	buf.iterator += sizeof(topic.ring_pending);
	buf.offset += sizeof(topic.ring_pending);
	static_assert(sizeof(topic.tx_buf_write_pending) == 1, "size mismatch");
	memcpy(&topic.tx_buf_write_pending, buf.iterator, sizeof(topic.tx_buf_write_pending));
	buf.iterator += sizeof(topic.tx_buf_write_pending);
	buf.offset += sizeof(topic.tx_buf_write_pending);
	static_assert(sizeof(topic.tx_session_pending) == 1, "size mismatch");
	memcpy(&topic.tx_session_pending, buf.iterator, sizeof(topic.tx_session_pending));
	buf.iterator += sizeof(topic.tx_session_pending);
	buf.offset += sizeof(topic.tx_session_pending);
	static_assert(sizeof(topic.rx_read_pending) == 1, "size mismatch");
	memcpy(&topic.rx_read_pending, buf.iterator, sizeof(topic.rx_read_pending));
	buf.iterator += sizeof(topic.rx_read_pending);
	buf.offset += sizeof(topic.rx_read_pending);
	static_assert(sizeof(topic.rx_session_pending) == 1, "size mismatch");
	memcpy(&topic.rx_session_pending, buf.iterator, sizeof(topic.rx_session_pending));
	buf.iterator += sizeof(topic.rx_session_pending);
	buf.offset += sizeof(topic.rx_session_pending);
	return true;
}
