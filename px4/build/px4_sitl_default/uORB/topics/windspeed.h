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

/* Auto-generated by genmsg_cpp from file /home/sique/src/PX4_v1.14.2/msg/Windspeed.msg */


#pragma once


#include <uORB/uORB.h>


#ifndef __cplusplus
#define WINDSPEED_ROTATION_YAW_0 0
#define WINDSPEED_ROTATION_YAW_45 1
#define WINDSPEED_ROTATION_YAW_90 2
#define WINDSPEED_ROTATION_YAW_135 3
#define WINDSPEED_ROTATION_YAW_180 4
#define WINDSPEED_ROTATION_YAW_225 5
#define WINDSPEED_ROTATION_YAW_270 6
#define WINDSPEED_ROTATION_YAW_315 7
#define WINDSPEED_ROTATION_FORWARD_FACING 0
#define WINDSPEED_ROTATION_RIGHT_FACING 2
#define WINDSPEED_ROTATION_BACKWARD_FACING 4
#define WINDSPEED_ROTATION_LEFT_FACING 6
#define WINDSPEED_ROTATION_UPWARD_FACING 24
#define WINDSPEED_ROTATION_DOWNWARD_FACING 25
#define WINDSPEED_ROTATION_DOWNWARD_FACING_SECOND 100

#endif


#ifdef __cplusplus
struct __EXPORT windspeed_s {
#else
struct windspeed_s {
#endif
	uint64_t timestamp;
	uint64_t timestamp_sample;
	float measurement_windspeed_x_m_s;
	float measurement_windspeed_y_m_s;
	float measurement_windspeed_z_m_s;
	float measurement_windspeed_zs_m_s;
	float filtered_windspeed_x_m_s;
	float filtered_windspeed_y_m_s;
	float filtered_windspeed_z_m_s;
	float filtered_windspeed_zs_m_s;
	float air_temperature_celsius;
	float confidence_x;
	float confidence_y;
	float confidence_z;
	float confidence_zs;
	uint8_t id;
	uint8_t orientation;
	uint8_t _padding0[2]; // required for logger


#ifdef __cplusplus
	static constexpr uint8_t ROTATION_YAW_0 = 0;
	static constexpr uint8_t ROTATION_YAW_45 = 1;
	static constexpr uint8_t ROTATION_YAW_90 = 2;
	static constexpr uint8_t ROTATION_YAW_135 = 3;
	static constexpr uint8_t ROTATION_YAW_180 = 4;
	static constexpr uint8_t ROTATION_YAW_225 = 5;
	static constexpr uint8_t ROTATION_YAW_270 = 6;
	static constexpr uint8_t ROTATION_YAW_315 = 7;
	static constexpr uint8_t ROTATION_FORWARD_FACING = 0;
	static constexpr uint8_t ROTATION_RIGHT_FACING = 2;
	static constexpr uint8_t ROTATION_BACKWARD_FACING = 4;
	static constexpr uint8_t ROTATION_LEFT_FACING = 6;
	static constexpr uint8_t ROTATION_UPWARD_FACING = 24;
	static constexpr uint8_t ROTATION_DOWNWARD_FACING = 25;
	static constexpr uint8_t ROTATION_DOWNWARD_FACING_SECOND = 100;

#endif
};

#ifdef __cplusplus
namespace px4 {
	namespace msg {
		using Windspeed = windspeed_s;
	} // namespace msg
} // namespace px4
#endif

/* register this as object request broker structure */
ORB_DECLARE(windspeed);


#ifdef __cplusplus
void print_message(const orb_metadata *meta, const windspeed_s& message);
#endif