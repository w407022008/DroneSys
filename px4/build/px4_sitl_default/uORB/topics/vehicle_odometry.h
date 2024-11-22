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

/* Auto-generated by genmsg_cpp from file /home/sique/src/PX4_v1.14.2/msg/VehicleOdometry.msg */


#pragma once


#include <uORB/uORB.h>


#ifndef __cplusplus
#define VEHICLE_ODOMETRY_POSE_FRAME_UNKNOWN 0
#define VEHICLE_ODOMETRY_POSE_FRAME_NED 1
#define VEHICLE_ODOMETRY_POSE_FRAME_FRD 2
#define VEHICLE_ODOMETRY_VELOCITY_FRAME_UNKNOWN 0
#define VEHICLE_ODOMETRY_VELOCITY_FRAME_NED 1
#define VEHICLE_ODOMETRY_VELOCITY_FRAME_FRD 2
#define VEHICLE_ODOMETRY_VELOCITY_FRAME_BODY_FRD 3

#endif


#ifdef __cplusplus
struct __EXPORT vehicle_odometry_s {
#else
struct vehicle_odometry_s {
#endif
	uint64_t timestamp;
	uint64_t timestamp_sample;
	float position[3];
	float q[4];
	float velocity[3];
	float angular_velocity[3];
	float position_variance[3];
	float orientation_variance[3];
	float velocity_variance[3];
	uint8_t pose_frame;
	uint8_t velocity_frame;
	uint8_t reset_counter;
	int8_t quality;
	uint8_t _padding0[4]; // required for logger


#ifdef __cplusplus
	static constexpr uint8_t POSE_FRAME_UNKNOWN = 0;
	static constexpr uint8_t POSE_FRAME_NED = 1;
	static constexpr uint8_t POSE_FRAME_FRD = 2;
	static constexpr uint8_t VELOCITY_FRAME_UNKNOWN = 0;
	static constexpr uint8_t VELOCITY_FRAME_NED = 1;
	static constexpr uint8_t VELOCITY_FRAME_FRD = 2;
	static constexpr uint8_t VELOCITY_FRAME_BODY_FRD = 3;

#endif
};

#ifdef __cplusplus
namespace px4 {
	namespace msg {
		using VehicleOdometry = vehicle_odometry_s;
	} // namespace msg
} // namespace px4
#endif

/* register this as object request broker structure */
ORB_DECLARE(vehicle_odometry);
ORB_DECLARE(vehicle_mocap_odometry);
ORB_DECLARE(vehicle_visual_odometry);
ORB_DECLARE(estimator_odometry);


#ifdef __cplusplus
void print_message(const orb_metadata *meta, const vehicle_odometry_s& message);
#endif
