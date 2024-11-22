
#pragma once

// This file is auto-generated by generate_function_header.py from /home/sique/src/PX4_v1.14.2/src/lib/mixer_module/output_functions.yaml

#include <stdint.h>

enum class OutputFunction : int32_t {
	Disabled = 0,
	Constant_Min = 1,
	Constant_Max = 2,
	Motor1 = 101,
	Motor2 = 102,
	Motor3 = 103,
	Motor4 = 104,
	Motor5 = 105,
	Motor6 = 106,
	Motor7 = 107,
	Motor8 = 108,
	Motor9 = 109,
	Motor10 = 110,
	Motor11 = 111,
	Motor12 = 112,
	MotorMax = 112,

	Servo1 = 201,
	Servo2 = 202,
	Servo3 = 203,
	Servo4 = 204,
	Servo5 = 205,
	Servo6 = 206,
	Servo7 = 207,
	Servo8 = 208,
	ServoMax = 208,

	Offboard_Actuator_Set1 = 301,
	Offboard_Actuator_Set2 = 302,
	Offboard_Actuator_Set3 = 303,
	Offboard_Actuator_Set4 = 304,
	Offboard_Actuator_Set5 = 305,
	Offboard_Actuator_Set6 = 306,
	Offboard_Actuator_SetMax = 306,

	Landing_Gear = 400,
	Parachute = 401,
	RC_Roll = 402,
	RC_Pitch = 403,
	RC_Throttle = 404,
	RC_Yaw = 405,
	RC_Flaps = 406,
	RC_AUX1 = 407,
	RC_AUX2 = 408,
	RC_AUX3 = 409,
	RC_AUX4 = 410,
	RC_AUX5 = 411,
	RC_AUX6 = 412,
	RC_AUXMax = 412,

	Gimbal_Roll = 420,
	Gimbal_Pitch = 421,
	Gimbal_Yaw = 422,
	Gripper = 430,
	Landing_Gear_Wheel = 440,
	Camera_Trigger = 2000,
	Camera_Capture = 2032,
	PPS_Input = 2064,

};
