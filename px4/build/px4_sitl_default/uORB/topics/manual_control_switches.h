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

/* Auto-generated by genmsg_cpp from file /home/sique/src/PX4_v1.14.2/msg/ManualControlSwitches.msg */


#pragma once


#include <uORB/uORB.h>


#ifndef __cplusplus
#define MANUAL_CONTROL_SWITCHES_SWITCH_POS_NONE 0
#define MANUAL_CONTROL_SWITCHES_SWITCH_POS_ON 1
#define MANUAL_CONTROL_SWITCHES_SWITCH_POS_MIDDLE 2
#define MANUAL_CONTROL_SWITCHES_SWITCH_POS_OFF 3
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_NONE 0
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_1 1
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_2 2
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_3 3
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_4 4
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_5 5
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_6 6
#define MANUAL_CONTROL_SWITCHES_MODE_SLOT_NUM 6

#endif


#ifdef __cplusplus
struct __EXPORT manual_control_switches_s {
#else
struct manual_control_switches_s {
#endif
	uint64_t timestamp;
	uint64_t timestamp_sample;
	uint32_t switch_changes;
	uint8_t mode_slot;
	uint8_t arm_switch;
	uint8_t return_switch;
	uint8_t loiter_switch;
	uint8_t offboard_switch;
	uint8_t kill_switch;
	uint8_t gear_switch;
	uint8_t transition_switch;
	uint8_t photo_switch;
	uint8_t video_switch;
	uint8_t engage_main_motor_switch;
	uint8_t _padding0[1]; // required for logger


#ifdef __cplusplus
	static constexpr uint8_t SWITCH_POS_NONE = 0;
	static constexpr uint8_t SWITCH_POS_ON = 1;
	static constexpr uint8_t SWITCH_POS_MIDDLE = 2;
	static constexpr uint8_t SWITCH_POS_OFF = 3;
	static constexpr uint8_t MODE_SLOT_NONE = 0;
	static constexpr uint8_t MODE_SLOT_1 = 1;
	static constexpr uint8_t MODE_SLOT_2 = 2;
	static constexpr uint8_t MODE_SLOT_3 = 3;
	static constexpr uint8_t MODE_SLOT_4 = 4;
	static constexpr uint8_t MODE_SLOT_5 = 5;
	static constexpr uint8_t MODE_SLOT_6 = 6;
	static constexpr uint8_t MODE_SLOT_NUM = 6;

#endif
};

#ifdef __cplusplus
namespace px4 {
	namespace msg {
		using ManualControlSwitches = manual_control_switches_s;
	} // namespace msg
} // namespace px4
#endif

/* register this as object request broker structure */
ORB_DECLARE(manual_control_switches);


#ifdef __cplusplus
void print_message(const orb_metadata *meta, const manual_control_switches_s& message);
#endif
