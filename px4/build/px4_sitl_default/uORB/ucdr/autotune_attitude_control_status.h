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
#include <uORB/topics/autotune_attitude_control_status.h>


static inline constexpr int ucdr_topic_size_autotune_attitude_control_status()
{
	return 101;
}

bool ucdr_serialize_autotune_attitude_control_status(const autotune_attitude_control_status_s& topic, ucdrBuffer& buf, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	const uint64_t timestamp_adjusted = topic.timestamp + time_offset;
	memcpy(buf.iterator, &timestamp_adjusted, sizeof(topic.timestamp));
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.coeff) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.coeff, sizeof(topic.coeff));
	buf.iterator += sizeof(topic.coeff);
	buf.offset += sizeof(topic.coeff);
	static_assert(sizeof(topic.coeff_var) == 20, "size mismatch");
	memcpy(buf.iterator, &topic.coeff_var, sizeof(topic.coeff_var));
	buf.iterator += sizeof(topic.coeff_var);
	buf.offset += sizeof(topic.coeff_var);
	static_assert(sizeof(topic.fitness) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.fitness, sizeof(topic.fitness));
	buf.iterator += sizeof(topic.fitness);
	buf.offset += sizeof(topic.fitness);
	static_assert(sizeof(topic.innov) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.innov, sizeof(topic.innov));
	buf.iterator += sizeof(topic.innov);
	buf.offset += sizeof(topic.innov);
	static_assert(sizeof(topic.dt_model) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.dt_model, sizeof(topic.dt_model));
	buf.iterator += sizeof(topic.dt_model);
	buf.offset += sizeof(topic.dt_model);
	static_assert(sizeof(topic.kc) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.kc, sizeof(topic.kc));
	buf.iterator += sizeof(topic.kc);
	buf.offset += sizeof(topic.kc);
	static_assert(sizeof(topic.ki) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.ki, sizeof(topic.ki));
	buf.iterator += sizeof(topic.ki);
	buf.offset += sizeof(topic.ki);
	static_assert(sizeof(topic.kd) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.kd, sizeof(topic.kd));
	buf.iterator += sizeof(topic.kd);
	buf.offset += sizeof(topic.kd);
	static_assert(sizeof(topic.kff) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.kff, sizeof(topic.kff));
	buf.iterator += sizeof(topic.kff);
	buf.offset += sizeof(topic.kff);
	static_assert(sizeof(topic.att_p) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.att_p, sizeof(topic.att_p));
	buf.iterator += sizeof(topic.att_p);
	buf.offset += sizeof(topic.att_p);
	static_assert(sizeof(topic.rate_sp) == 12, "size mismatch");
	memcpy(buf.iterator, &topic.rate_sp, sizeof(topic.rate_sp));
	buf.iterator += sizeof(topic.rate_sp);
	buf.offset += sizeof(topic.rate_sp);
	static_assert(sizeof(topic.u_filt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.u_filt, sizeof(topic.u_filt));
	buf.iterator += sizeof(topic.u_filt);
	buf.offset += sizeof(topic.u_filt);
	static_assert(sizeof(topic.y_filt) == 4, "size mismatch");
	memcpy(buf.iterator, &topic.y_filt, sizeof(topic.y_filt));
	buf.iterator += sizeof(topic.y_filt);
	buf.offset += sizeof(topic.y_filt);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(buf.iterator, &topic.state, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	return true;
}

bool ucdr_deserialize_autotune_attitude_control_status(ucdrBuffer& buf, autotune_attitude_control_status_s& topic, int64_t time_offset = 0)
{
	static_assert(sizeof(topic.timestamp) == 8, "size mismatch");
	memcpy(&topic.timestamp, buf.iterator, sizeof(topic.timestamp));
	if (topic.timestamp == 0) topic.timestamp = hrt_absolute_time();
	else topic.timestamp = math::min(topic.timestamp - time_offset, hrt_absolute_time());
	buf.iterator += sizeof(topic.timestamp);
	buf.offset += sizeof(topic.timestamp);
	static_assert(sizeof(topic.coeff) == 20, "size mismatch");
	memcpy(&topic.coeff, buf.iterator, sizeof(topic.coeff));
	buf.iterator += sizeof(topic.coeff);
	buf.offset += sizeof(topic.coeff);
	static_assert(sizeof(topic.coeff_var) == 20, "size mismatch");
	memcpy(&topic.coeff_var, buf.iterator, sizeof(topic.coeff_var));
	buf.iterator += sizeof(topic.coeff_var);
	buf.offset += sizeof(topic.coeff_var);
	static_assert(sizeof(topic.fitness) == 4, "size mismatch");
	memcpy(&topic.fitness, buf.iterator, sizeof(topic.fitness));
	buf.iterator += sizeof(topic.fitness);
	buf.offset += sizeof(topic.fitness);
	static_assert(sizeof(topic.innov) == 4, "size mismatch");
	memcpy(&topic.innov, buf.iterator, sizeof(topic.innov));
	buf.iterator += sizeof(topic.innov);
	buf.offset += sizeof(topic.innov);
	static_assert(sizeof(topic.dt_model) == 4, "size mismatch");
	memcpy(&topic.dt_model, buf.iterator, sizeof(topic.dt_model));
	buf.iterator += sizeof(topic.dt_model);
	buf.offset += sizeof(topic.dt_model);
	static_assert(sizeof(topic.kc) == 4, "size mismatch");
	memcpy(&topic.kc, buf.iterator, sizeof(topic.kc));
	buf.iterator += sizeof(topic.kc);
	buf.offset += sizeof(topic.kc);
	static_assert(sizeof(topic.ki) == 4, "size mismatch");
	memcpy(&topic.ki, buf.iterator, sizeof(topic.ki));
	buf.iterator += sizeof(topic.ki);
	buf.offset += sizeof(topic.ki);
	static_assert(sizeof(topic.kd) == 4, "size mismatch");
	memcpy(&topic.kd, buf.iterator, sizeof(topic.kd));
	buf.iterator += sizeof(topic.kd);
	buf.offset += sizeof(topic.kd);
	static_assert(sizeof(topic.kff) == 4, "size mismatch");
	memcpy(&topic.kff, buf.iterator, sizeof(topic.kff));
	buf.iterator += sizeof(topic.kff);
	buf.offset += sizeof(topic.kff);
	static_assert(sizeof(topic.att_p) == 4, "size mismatch");
	memcpy(&topic.att_p, buf.iterator, sizeof(topic.att_p));
	buf.iterator += sizeof(topic.att_p);
	buf.offset += sizeof(topic.att_p);
	static_assert(sizeof(topic.rate_sp) == 12, "size mismatch");
	memcpy(&topic.rate_sp, buf.iterator, sizeof(topic.rate_sp));
	buf.iterator += sizeof(topic.rate_sp);
	buf.offset += sizeof(topic.rate_sp);
	static_assert(sizeof(topic.u_filt) == 4, "size mismatch");
	memcpy(&topic.u_filt, buf.iterator, sizeof(topic.u_filt));
	buf.iterator += sizeof(topic.u_filt);
	buf.offset += sizeof(topic.u_filt);
	static_assert(sizeof(topic.y_filt) == 4, "size mismatch");
	memcpy(&topic.y_filt, buf.iterator, sizeof(topic.y_filt));
	buf.iterator += sizeof(topic.y_filt);
	buf.offset += sizeof(topic.y_filt);
	static_assert(sizeof(topic.state) == 1, "size mismatch");
	memcpy(&topic.state, buf.iterator, sizeof(topic.state));
	buf.iterator += sizeof(topic.state);
	buf.offset += sizeof(topic.state);
	return true;
}
