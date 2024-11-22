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
#include <uORB/topics/telemetry_status.h>


static inline constexpr int ucdr_topic_size_telemetry_status()
{
	return 87;
}

bool ucdr_serialize_telemetry_status(const telemetry_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.type, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mode, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	static_assert(sizeof(topic.flow_control) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.flow_control, sizeof(topic.flow_control));
	buf.iterator += sizeof(topic.flow_control);
	buf.offset += sizeof(topic.flow_control);
	static_assert(sizeof(topic.forwarding) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.forwarding, sizeof(topic.forwarding));
	buf.iterator += sizeof(topic.forwarding);
	buf.offset += sizeof(topic.forwarding);
	static_assert(sizeof(topic.mavlink_v2) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.mavlink_v2, sizeof(topic.mavlink_v2));
	buf.iterator += sizeof(topic.mavlink_v2);
	buf.offset += sizeof(topic.mavlink_v2);
	static_assert(sizeof(topic.ftp) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.ftp, sizeof(topic.ftp));
	buf.iterator += sizeof(topic.ftp);
	buf.offset += sizeof(topic.ftp);
	static_assert(sizeof(topic.streams) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.streams, sizeof(topic.streams));
	buf.iterator += sizeof(topic.streams);
	buf.offset += sizeof(topic.streams);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.data_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.data_rate, sizeof(topic.data_rate));
	buf.iterator += sizeof(topic.data_rate);
	buf.offset += sizeof(topic.data_rate);
	static_assert(sizeof(topic.rate_multiplier) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rate_multiplier, sizeof(topic.rate_multiplier));
	buf.iterator += sizeof(topic.rate_multiplier);
	buf.offset += sizeof(topic.rate_multiplier);
	static_assert(sizeof(topic.tx_rate_avg) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tx_rate_avg, sizeof(topic.tx_rate_avg));
	buf.iterator += sizeof(topic.tx_rate_avg);
	buf.offset += sizeof(topic.tx_rate_avg);
	static_assert(sizeof(topic.tx_error_rate_avg) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tx_error_rate_avg, sizeof(topic.tx_error_rate_avg));
	buf.iterator += sizeof(topic.tx_error_rate_avg);
	buf.offset += sizeof(topic.tx_error_rate_avg);
	static_assert(sizeof(topic.tx_message_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tx_message_count, sizeof(topic.tx_message_count));
	buf.iterator += sizeof(topic.tx_message_count);
	buf.offset += sizeof(topic.tx_message_count);
	static_assert(sizeof(topic.tx_buffer_overruns) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.tx_buffer_overruns, sizeof(topic.tx_buffer_overruns));
	buf.iterator += sizeof(topic.tx_buffer_overruns);
	buf.offset += sizeof(topic.tx_buffer_overruns);
	static_assert(sizeof(topic.rx_rate_avg) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_rate_avg, sizeof(topic.rx_rate_avg));
	buf.iterator += sizeof(topic.rx_rate_avg);
	buf.offset += sizeof(topic.rx_rate_avg);
	static_assert(sizeof(topic.rx_message_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_message_count, sizeof(topic.rx_message_count));
	buf.iterator += sizeof(topic.rx_message_count);
	buf.offset += sizeof(topic.rx_message_count);
	static_assert(sizeof(topic.rx_message_lost_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_message_lost_count, sizeof(topic.rx_message_lost_count));
	buf.iterator += sizeof(topic.rx_message_lost_count);
	buf.offset += sizeof(topic.rx_message_lost_count);
	static_assert(sizeof(topic.rx_buffer_overruns) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_buffer_overruns, sizeof(topic.rx_buffer_overruns));
	buf.iterator += sizeof(topic.rx_buffer_overruns);
	buf.offset += sizeof(topic.rx_buffer_overruns);
	static_assert(sizeof(topic.rx_parse_errors) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_parse_errors, sizeof(topic.rx_parse_errors));
	buf.iterator += sizeof(topic.rx_parse_errors);
	buf.offset += sizeof(topic.rx_parse_errors);
	static_assert(sizeof(topic.rx_packet_drop_count) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_packet_drop_count, sizeof(topic.rx_packet_drop_count));
	buf.iterator += sizeof(topic.rx_packet_drop_count);
	buf.offset += sizeof(topic.rx_packet_drop_count);
	static_assert(sizeof(topic.rx_message_lost_rate) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.rx_message_lost_rate, sizeof(topic.rx_message_lost_rate));
	buf.iterator += sizeof(topic.rx_message_lost_rate);
	buf.offset += sizeof(topic.rx_message_lost_rate);
	static_assert(sizeof(topic.heartbeat_type_antenna_tracker) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_antenna_tracker, sizeof(topic.heartbeat_type_antenna_tracker));
	buf.iterator += sizeof(topic.heartbeat_type_antenna_tracker);
	buf.offset += sizeof(topic.heartbeat_type_antenna_tracker);
	static_assert(sizeof(topic.heartbeat_type_gcs) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_gcs, sizeof(topic.heartbeat_type_gcs));
	buf.iterator += sizeof(topic.heartbeat_type_gcs);
	buf.offset += sizeof(topic.heartbeat_type_gcs);
	static_assert(sizeof(topic.heartbeat_type_onboard_controller) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_onboard_controller, sizeof(topic.heartbeat_type_onboard_controller));
	buf.iterator += sizeof(topic.heartbeat_type_onboard_controller);
	buf.offset += sizeof(topic.heartbeat_type_onboard_controller);
	static_assert(sizeof(topic.heartbeat_type_gimbal) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_gimbal, sizeof(topic.heartbeat_type_gimbal));
	buf.iterator += sizeof(topic.heartbeat_type_gimbal);
	buf.offset += sizeof(topic.heartbeat_type_gimbal);
	static_assert(sizeof(topic.heartbeat_type_adsb) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_adsb, sizeof(topic.heartbeat_type_adsb));
	buf.iterator += sizeof(topic.heartbeat_type_adsb);
	buf.offset += sizeof(topic.heartbeat_type_adsb);
	static_assert(sizeof(topic.heartbeat_type_camera) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_camera, sizeof(topic.heartbeat_type_camera));
	buf.iterator += sizeof(topic.heartbeat_type_camera);
	buf.offset += sizeof(topic.heartbeat_type_camera);
	static_assert(sizeof(topic.heartbeat_type_parachute) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_parachute, sizeof(topic.heartbeat_type_parachute));
	buf.iterator += sizeof(topic.heartbeat_type_parachute);
	buf.offset += sizeof(topic.heartbeat_type_parachute);
	static_assert(sizeof(topic.heartbeat_type_open_drone_id) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_type_open_drone_id, sizeof(topic.heartbeat_type_open_drone_id));
	buf.iterator += sizeof(topic.heartbeat_type_open_drone_id);
	buf.offset += sizeof(topic.heartbeat_type_open_drone_id);
	static_assert(sizeof(topic.heartbeat_component_telemetry_radio) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_telemetry_radio, sizeof(topic.heartbeat_component_telemetry_radio));
	buf.iterator += sizeof(topic.heartbeat_component_telemetry_radio);
	buf.offset += sizeof(topic.heartbeat_component_telemetry_radio);
	static_assert(sizeof(topic.heartbeat_component_log) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_log, sizeof(topic.heartbeat_component_log));
	buf.iterator += sizeof(topic.heartbeat_component_log);
	buf.offset += sizeof(topic.heartbeat_component_log);
	static_assert(sizeof(topic.heartbeat_component_osd) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_osd, sizeof(topic.heartbeat_component_osd));
	buf.iterator += sizeof(topic.heartbeat_component_osd);
	buf.offset += sizeof(topic.heartbeat_component_osd);
	static_assert(sizeof(topic.heartbeat_component_obstacle_avoidance) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_obstacle_avoidance, sizeof(topic.heartbeat_component_obstacle_avoidance));
	buf.iterator += sizeof(topic.heartbeat_component_obstacle_avoidance);
	buf.offset += sizeof(topic.heartbeat_component_obstacle_avoidance);
	static_assert(sizeof(topic.heartbeat_component_vio) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_vio, sizeof(topic.heartbeat_component_vio));
	buf.iterator += sizeof(topic.heartbeat_component_vio);
	buf.offset += sizeof(topic.heartbeat_component_vio);
	static_assert(sizeof(topic.heartbeat_component_pairing_manager) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_pairing_manager, sizeof(topic.heartbeat_component_pairing_manager));
	buf.iterator += sizeof(topic.heartbeat_component_pairing_manager);
	buf.offset += sizeof(topic.heartbeat_component_pairing_manager);
	static_assert(sizeof(topic.heartbeat_component_udp_bridge) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_udp_bridge, sizeof(topic.heartbeat_component_udp_bridge));
	buf.iterator += sizeof(topic.heartbeat_component_udp_bridge);
	buf.offset += sizeof(topic.heartbeat_component_udp_bridge);
	static_assert(sizeof(topic.heartbeat_component_uart_bridge) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.heartbeat_component_uart_bridge, sizeof(topic.heartbeat_component_uart_bridge));
	buf.iterator += sizeof(topic.heartbeat_component_uart_bridge);
	buf.offset += sizeof(topic.heartbeat_component_uart_bridge);
	static_assert(sizeof(topic.avoidance_system_healthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.avoidance_system_healthy, sizeof(topic.avoidance_system_healthy));
	buf.iterator += sizeof(topic.avoidance_system_healthy);
	buf.offset += sizeof(topic.avoidance_system_healthy);
	static_assert(sizeof(topic.open_drone_id_system_healthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.open_drone_id_system_healthy, sizeof(topic.open_drone_id_system_healthy));
	buf.iterator += sizeof(topic.open_drone_id_system_healthy);
	buf.offset += sizeof(topic.open_drone_id_system_healthy);
	static_assert(sizeof(topic.parachute_system_healthy) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.parachute_system_healthy, sizeof(topic.parachute_system_healthy));
	buf.iterator += sizeof(topic.parachute_system_healthy);
	buf.offset += sizeof(topic.parachute_system_healthy);
	return true;
}

bool ucdr_deserialize_telemetry_status(ucdrBuffer& buf, telemetry_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.type) == 1, "size mismatch");
	memcpy(&topic.type, buf.iterator, sizeof(topic.type));
	buf.iterator += sizeof(topic.type);
	buf.offset += sizeof(topic.type);
	static_assert(sizeof(topic.mode) == 1, "size mismatch");
	memcpy(&topic.mode, buf.iterator, sizeof(topic.mode));
	buf.iterator += sizeof(topic.mode);
	buf.offset += sizeof(topic.mode);
	static_assert(sizeof(topic.flow_control) == 1, "size mismatch");
	memcpy(&topic.flow_control, buf.iterator, sizeof(topic.flow_control));
	buf.iterator += sizeof(topic.flow_control);
	buf.offset += sizeof(topic.flow_control);
	static_assert(sizeof(topic.forwarding) == 1, "size mismatch");
	memcpy(&topic.forwarding, buf.iterator, sizeof(topic.forwarding));
	buf.iterator += sizeof(topic.forwarding);
	buf.offset += sizeof(topic.forwarding);
	static_assert(sizeof(topic.mavlink_v2) == 1, "size mismatch");
	memcpy(&topic.mavlink_v2, buf.iterator, sizeof(topic.mavlink_v2));
	buf.iterator += sizeof(topic.mavlink_v2);
	buf.offset += sizeof(topic.mavlink_v2);
	static_assert(sizeof(topic.ftp) == 1, "size mismatch");
	memcpy(&topic.ftp, buf.iterator, sizeof(topic.ftp));
	buf.iterator += sizeof(topic.ftp);
	buf.offset += sizeof(topic.ftp);
	static_assert(sizeof(topic.streams) == 1, "size mismatch");
	memcpy(&topic.streams, buf.iterator, sizeof(topic.streams));
	buf.iterator += sizeof(topic.streams);
	buf.offset += sizeof(topic.streams);
	buf.iterator += 1; // padding
	buf.offset += 1; // padding
	static_assert(sizeof(topic.data_rate) == 4, "size mismatch");
	memcpy(&topic.data_rate, buf.iterator, sizeof(topic.data_rate));
	buf.iterator += sizeof(topic.data_rate);
	buf.offset += sizeof(topic.data_rate);
	static_assert(sizeof(topic.rate_multiplier) == 4, "size mismatch");
	memcpy(&topic.rate_multiplier, buf.iterator, sizeof(topic.rate_multiplier));
	buf.iterator += sizeof(topic.rate_multiplier);
	buf.offset += sizeof(topic.rate_multiplier);
	static_assert(sizeof(topic.tx_rate_avg) == 4, "size mismatch");
	memcpy(&topic.tx_rate_avg, buf.iterator, sizeof(topic.tx_rate_avg));
	buf.iterator += sizeof(topic.tx_rate_avg);
	buf.offset += sizeof(topic.tx_rate_avg);
	static_assert(sizeof(topic.tx_error_rate_avg) == 4, "size mismatch");
	memcpy(&topic.tx_error_rate_avg, buf.iterator, sizeof(topic.tx_error_rate_avg));
	buf.iterator += sizeof(topic.tx_error_rate_avg);
	buf.offset += sizeof(topic.tx_error_rate_avg);
	static_assert(sizeof(topic.tx_message_count) == 4, "size mismatch");
	memcpy(&topic.tx_message_count, buf.iterator, sizeof(topic.tx_message_count));
	buf.iterator += sizeof(topic.tx_message_count);
	buf.offset += sizeof(topic.tx_message_count);
	static_assert(sizeof(topic.tx_buffer_overruns) == 4, "size mismatch");
	memcpy(&topic.tx_buffer_overruns, buf.iterator, sizeof(topic.tx_buffer_overruns));
	buf.iterator += sizeof(topic.tx_buffer_overruns);
	buf.offset += sizeof(topic.tx_buffer_overruns);
	static_assert(sizeof(topic.rx_rate_avg) == 4, "size mismatch");
	memcpy(&topic.rx_rate_avg, buf.iterator, sizeof(topic.rx_rate_avg));
	buf.iterator += sizeof(topic.rx_rate_avg);
	buf.offset += sizeof(topic.rx_rate_avg);
	static_assert(sizeof(topic.rx_message_count) == 4, "size mismatch");
	memcpy(&topic.rx_message_count, buf.iterator, sizeof(topic.rx_message_count));
	buf.iterator += sizeof(topic.rx_message_count);
	buf.offset += sizeof(topic.rx_message_count);
	static_assert(sizeof(topic.rx_message_lost_count) == 4, "size mismatch");
	memcpy(&topic.rx_message_lost_count, buf.iterator, sizeof(topic.rx_message_lost_count));
	buf.iterator += sizeof(topic.rx_message_lost_count);
	buf.offset += sizeof(topic.rx_message_lost_count);
	static_assert(sizeof(topic.rx_buffer_overruns) == 4, "size mismatch");
	memcpy(&topic.rx_buffer_overruns, buf.iterator, sizeof(topic.rx_buffer_overruns));
	buf.iterator += sizeof(topic.rx_buffer_overruns);
	buf.offset += sizeof(topic.rx_buffer_overruns);
	static_assert(sizeof(topic.rx_parse_errors) == 4, "size mismatch");
	memcpy(&topic.rx_parse_errors, buf.iterator, sizeof(topic.rx_parse_errors));
	buf.iterator += sizeof(topic.rx_parse_errors);
	buf.offset += sizeof(topic.rx_parse_errors);
	static_assert(sizeof(topic.rx_packet_drop_count) == 4, "size mismatch");
	memcpy(&topic.rx_packet_drop_count, buf.iterator, sizeof(topic.rx_packet_drop_count));
	buf.iterator += sizeof(topic.rx_packet_drop_count);
	buf.offset += sizeof(topic.rx_packet_drop_count);
	static_assert(sizeof(topic.rx_message_lost_rate) == 4, "size mismatch");
	memcpy(&topic.rx_message_lost_rate, buf.iterator, sizeof(topic.rx_message_lost_rate));
	buf.iterator += sizeof(topic.rx_message_lost_rate);
	buf.offset += sizeof(topic.rx_message_lost_rate);
	static_assert(sizeof(topic.heartbeat_type_antenna_tracker) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_antenna_tracker, buf.iterator, sizeof(topic.heartbeat_type_antenna_tracker));
	buf.iterator += sizeof(topic.heartbeat_type_antenna_tracker);
	buf.offset += sizeof(topic.heartbeat_type_antenna_tracker);
	static_assert(sizeof(topic.heartbeat_type_gcs) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_gcs, buf.iterator, sizeof(topic.heartbeat_type_gcs));
	buf.iterator += sizeof(topic.heartbeat_type_gcs);
	buf.offset += sizeof(topic.heartbeat_type_gcs);
	static_assert(sizeof(topic.heartbeat_type_onboard_controller) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_onboard_controller, buf.iterator, sizeof(topic.heartbeat_type_onboard_controller));
	buf.iterator += sizeof(topic.heartbeat_type_onboard_controller);
	buf.offset += sizeof(topic.heartbeat_type_onboard_controller);
	static_assert(sizeof(topic.heartbeat_type_gimbal) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_gimbal, buf.iterator, sizeof(topic.heartbeat_type_gimbal));
	buf.iterator += sizeof(topic.heartbeat_type_gimbal);
	buf.offset += sizeof(topic.heartbeat_type_gimbal);
	static_assert(sizeof(topic.heartbeat_type_adsb) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_adsb, buf.iterator, sizeof(topic.heartbeat_type_adsb));
	buf.iterator += sizeof(topic.heartbeat_type_adsb);
	buf.offset += sizeof(topic.heartbeat_type_adsb);
	static_assert(sizeof(topic.heartbeat_type_camera) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_camera, buf.iterator, sizeof(topic.heartbeat_type_camera));
	buf.iterator += sizeof(topic.heartbeat_type_camera);
	buf.offset += sizeof(topic.heartbeat_type_camera);
	static_assert(sizeof(topic.heartbeat_type_parachute) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_parachute, buf.iterator, sizeof(topic.heartbeat_type_parachute));
	buf.iterator += sizeof(topic.heartbeat_type_parachute);
	buf.offset += sizeof(topic.heartbeat_type_parachute);
	static_assert(sizeof(topic.heartbeat_type_open_drone_id) == 1, "size mismatch");
	memcpy(&topic.heartbeat_type_open_drone_id, buf.iterator, sizeof(topic.heartbeat_type_open_drone_id));
	buf.iterator += sizeof(topic.heartbeat_type_open_drone_id);
	buf.offset += sizeof(topic.heartbeat_type_open_drone_id);
	static_assert(sizeof(topic.heartbeat_component_telemetry_radio) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_telemetry_radio, buf.iterator, sizeof(topic.heartbeat_component_telemetry_radio));
	buf.iterator += sizeof(topic.heartbeat_component_telemetry_radio);
	buf.offset += sizeof(topic.heartbeat_component_telemetry_radio);
	static_assert(sizeof(topic.heartbeat_component_log) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_log, buf.iterator, sizeof(topic.heartbeat_component_log));
	buf.iterator += sizeof(topic.heartbeat_component_log);
	buf.offset += sizeof(topic.heartbeat_component_log);
	static_assert(sizeof(topic.heartbeat_component_osd) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_osd, buf.iterator, sizeof(topic.heartbeat_component_osd));
	buf.iterator += sizeof(topic.heartbeat_component_osd);
	buf.offset += sizeof(topic.heartbeat_component_osd);
	static_assert(sizeof(topic.heartbeat_component_obstacle_avoidance) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_obstacle_avoidance, buf.iterator, sizeof(topic.heartbeat_component_obstacle_avoidance));
	buf.iterator += sizeof(topic.heartbeat_component_obstacle_avoidance);
	buf.offset += sizeof(topic.heartbeat_component_obstacle_avoidance);
	static_assert(sizeof(topic.heartbeat_component_vio) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_vio, buf.iterator, sizeof(topic.heartbeat_component_vio));
	buf.iterator += sizeof(topic.heartbeat_component_vio);
	buf.offset += sizeof(topic.heartbeat_component_vio);
	static_assert(sizeof(topic.heartbeat_component_pairing_manager) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_pairing_manager, buf.iterator, sizeof(topic.heartbeat_component_pairing_manager));
	buf.iterator += sizeof(topic.heartbeat_component_pairing_manager);
	buf.offset += sizeof(topic.heartbeat_component_pairing_manager);
	static_assert(sizeof(topic.heartbeat_component_udp_bridge) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_udp_bridge, buf.iterator, sizeof(topic.heartbeat_component_udp_bridge));
	buf.iterator += sizeof(topic.heartbeat_component_udp_bridge);
	buf.offset += sizeof(topic.heartbeat_component_udp_bridge);
	static_assert(sizeof(topic.heartbeat_component_uart_bridge) == 1, "size mismatch");
	memcpy(&topic.heartbeat_component_uart_bridge, buf.iterator, sizeof(topic.heartbeat_component_uart_bridge));
	buf.iterator += sizeof(topic.heartbeat_component_uart_bridge);
	buf.offset += sizeof(topic.heartbeat_component_uart_bridge);
	static_assert(sizeof(topic.avoidance_system_healthy) == 1, "size mismatch");
	memcpy(&topic.avoidance_system_healthy, buf.iterator, sizeof(topic.avoidance_system_healthy));
	buf.iterator += sizeof(topic.avoidance_system_healthy);
	buf.offset += sizeof(topic.avoidance_system_healthy);
	static_assert(sizeof(topic.open_drone_id_system_healthy) == 1, "size mismatch");
	memcpy(&topic.open_drone_id_system_healthy, buf.iterator, sizeof(topic.open_drone_id_system_healthy));
	buf.iterator += sizeof(topic.open_drone_id_system_healthy);
	buf.offset += sizeof(topic.open_drone_id_system_healthy);
	static_assert(sizeof(topic.parachute_system_healthy) == 1, "size mismatch");
	memcpy(&topic.parachute_system_healthy, buf.iterator, sizeof(topic.parachute_system_healthy));
	buf.iterator += sizeof(topic.parachute_system_healthy);
	buf.offset += sizeof(topic.parachute_system_healthy);
	return true;
}
