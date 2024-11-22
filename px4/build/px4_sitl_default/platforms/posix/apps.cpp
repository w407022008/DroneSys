/* definitions of builtin command list - automatically generated, do not edit */
#include <px4_platform_common/time.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/log.h>

#include "apps.h"

#include <cstdio>
#include <map>
#include <string>

#include <cstdlib>

#define MODULE_NAME "px4"

extern "C" {

int cdev_test_main(int argc, char *argv[]);
int controllib_test_main(int argc, char *argv[]);
int rc_tests_main(int argc, char *argv[]);
int uorb_tests_main(int argc, char *argv[]);
int wqueue_test_main(int argc, char *argv[]);
int camera_trigger_main(int argc, char *argv[]);
int gps_main(int argc, char *argv[]);
int msp_osd_main(int argc, char *argv[]);
int tone_alarm_main(int argc, char *argv[]);
int airship_att_control_main(int argc, char *argv[]);
int airspeed_selector_main(int argc, char *argv[]);
int attitude_estimator_q_main(int argc, char *argv[]);
int camera_feedback_main(int argc, char *argv[]);
int commander_main(int argc, char *argv[]);
int control_allocator_main(int argc, char *argv[]);
int dataman_main(int argc, char *argv[]);
int ekf2_main(int argc, char *argv[]);
int send_event_main(int argc, char *argv[]);
int flight_mode_manager_main(int argc, char *argv[]);
int fw_att_control_main(int argc, char *argv[]);
int fw_autotune_attitude_control_main(int argc, char *argv[]);
int fw_pos_control_main(int argc, char *argv[]);
int fw_rate_control_main(int argc, char *argv[]);
int gimbal_main(int argc, char *argv[]);
int gyro_calibration_main(int argc, char *argv[]);
int gyro_fft_main(int argc, char *argv[]);
int land_detector_main(int argc, char *argv[]);
int landing_target_estimator_main(int argc, char *argv[]);
int load_mon_main(int argc, char *argv[]);
int local_position_estimator_main(int argc, char *argv[]);
int logger_main(int argc, char *argv[]);
int mag_bias_estimator_main(int argc, char *argv[]);
int manual_control_main(int argc, char *argv[]);
int mavlink_main(int argc, char *argv[]);
int mavlink_tests_main(int argc, char *argv[]);
int mc_att_control_main(int argc, char *argv[]);
int mc_autotune_attitude_control_main(int argc, char *argv[]);
int mc_hover_thrust_estimator_main(int argc, char *argv[]);
int mc_pos_control_main(int argc, char *argv[]);
int mc_rate_control_main(int argc, char *argv[]);
int navigator_main(int argc, char *argv[]);
int payload_deliverer_main(int argc, char *argv[]);
int rc_update_main(int argc, char *argv[]);
int replay_main(int argc, char *argv[]);
int rover_pos_control_main(int argc, char *argv[]);
int sensors_main(int argc, char *argv[]);
int battery_simulator_main(int argc, char *argv[]);
int pwm_out_sim_main(int argc, char *argv[]);
int sensor_airspeed_sim_main(int argc, char *argv[]);
int sensor_baro_sim_main(int argc, char *argv[]);
int sensor_gps_sim_main(int argc, char *argv[]);
int sensor_mag_sim_main(int argc, char *argv[]);
int simulator_mavlink_main(int argc, char *argv[]);
int simulator_sih_main(int argc, char *argv[]);
int temperature_compensation_main(int argc, char *argv[]);
int uuv_att_control_main(int argc, char *argv[]);
int uuv_pos_control_main(int argc, char *argv[]);
int uxrce_dds_client_main(int argc, char *argv[]);
int vtol_att_control_main(int argc, char *argv[]);
int actuator_test_main(int argc, char *argv[]);
int bsondump_main(int argc, char *argv[]);
int dyn_main(int argc, char *argv[]);
int failure_main(int argc, char *argv[]);
int led_control_main(int argc, char *argv[]);
int param_main(int argc, char *argv[]);
int perf_main(int argc, char *argv[]);
int sd_bench_main(int argc, char *argv[]);
int shutdown_main(int argc, char *argv[]);
int system_time_main(int argc, char *argv[]);
int tests_main(int argc, char *argv[]);
int hrt_test_main(int argc, char *argv[]);
int listener_main(int argc, char *argv[]);
int tune_control_main(int argc, char *argv[]);
int uorb_main(int argc, char *argv[]);
int ver_main(int argc, char *argv[]);
int work_queue_main(int argc, char *argv[]);
int fake_gps_main(int argc, char *argv[]);
int fake_imu_main(int argc, char *argv[]);
int fake_magnetometer_main(int argc, char *argv[]);
int hello_main(int argc, char *argv[]);
int px4_mavlink_debug_main(int argc, char *argv[]);
int px4_simple_app_main(int argc, char *argv[]);
int work_item_example_main(int argc, char *argv[]);

int shutdown_main(int argc, char *argv[]);
int list_tasks_main(int argc, char *argv[]);
int list_files_main(int argc, char *argv[]);
int sleep_main(int argc, char *argv[]);

}

void init_app_map(apps_map_type &apps)
{
		apps["cdev_test"] = cdev_test_main;
	apps["controllib_test"] = controllib_test_main;
	apps["rc_tests"] = rc_tests_main;
	apps["uorb_tests"] = uorb_tests_main;
	apps["wqueue_test"] = wqueue_test_main;
	apps["camera_trigger"] = camera_trigger_main;
	apps["gps"] = gps_main;
	apps["msp_osd"] = msp_osd_main;
	apps["tone_alarm"] = tone_alarm_main;
	apps["airship_att_control"] = airship_att_control_main;
	apps["airspeed_selector"] = airspeed_selector_main;
	apps["attitude_estimator_q"] = attitude_estimator_q_main;
	apps["camera_feedback"] = camera_feedback_main;
	apps["commander"] = commander_main;
	apps["control_allocator"] = control_allocator_main;
	apps["dataman"] = dataman_main;
	apps["ekf2"] = ekf2_main;
	apps["send_event"] = send_event_main;
	apps["flight_mode_manager"] = flight_mode_manager_main;
	apps["fw_att_control"] = fw_att_control_main;
	apps["fw_autotune_attitude_control"] = fw_autotune_attitude_control_main;
	apps["fw_pos_control"] = fw_pos_control_main;
	apps["fw_rate_control"] = fw_rate_control_main;
	apps["gimbal"] = gimbal_main;
	apps["gyro_calibration"] = gyro_calibration_main;
	apps["gyro_fft"] = gyro_fft_main;
	apps["land_detector"] = land_detector_main;
	apps["landing_target_estimator"] = landing_target_estimator_main;
	apps["load_mon"] = load_mon_main;
	apps["local_position_estimator"] = local_position_estimator_main;
	apps["logger"] = logger_main;
	apps["mag_bias_estimator"] = mag_bias_estimator_main;
	apps["manual_control"] = manual_control_main;
	apps["mavlink"] = mavlink_main;
	apps["mavlink_tests"] = mavlink_tests_main;
	apps["mc_att_control"] = mc_att_control_main;
	apps["mc_autotune_attitude_control"] = mc_autotune_attitude_control_main;
	apps["mc_hover_thrust_estimator"] = mc_hover_thrust_estimator_main;
	apps["mc_pos_control"] = mc_pos_control_main;
	apps["mc_rate_control"] = mc_rate_control_main;
	apps["navigator"] = navigator_main;
	apps["payload_deliverer"] = payload_deliverer_main;
	apps["rc_update"] = rc_update_main;
	apps["replay"] = replay_main;
	apps["rover_pos_control"] = rover_pos_control_main;
	apps["sensors"] = sensors_main;
	apps["battery_simulator"] = battery_simulator_main;
	apps["pwm_out_sim"] = pwm_out_sim_main;
	apps["sensor_airspeed_sim"] = sensor_airspeed_sim_main;
	apps["sensor_baro_sim"] = sensor_baro_sim_main;
	apps["sensor_gps_sim"] = sensor_gps_sim_main;
	apps["sensor_mag_sim"] = sensor_mag_sim_main;
	apps["simulator_mavlink"] = simulator_mavlink_main;
	apps["simulator_sih"] = simulator_sih_main;
	apps["temperature_compensation"] = temperature_compensation_main;
	apps["uuv_att_control"] = uuv_att_control_main;
	apps["uuv_pos_control"] = uuv_pos_control_main;
	apps["uxrce_dds_client"] = uxrce_dds_client_main;
	apps["vtol_att_control"] = vtol_att_control_main;
	apps["actuator_test"] = actuator_test_main;
	apps["bsondump"] = bsondump_main;
	apps["dyn"] = dyn_main;
	apps["failure"] = failure_main;
	apps["led_control"] = led_control_main;
	apps["param"] = param_main;
	apps["perf"] = perf_main;
	apps["sd_bench"] = sd_bench_main;
	apps["shutdown"] = shutdown_main;
	apps["system_time"] = system_time_main;
	apps["tests"] = tests_main;
	apps["hrt_test"] = hrt_test_main;
	apps["listener"] = listener_main;
	apps["tune_control"] = tune_control_main;
	apps["uorb"] = uorb_main;
	apps["ver"] = ver_main;
	apps["work_queue"] = work_queue_main;
	apps["fake_gps"] = fake_gps_main;
	apps["fake_imu"] = fake_imu_main;
	apps["fake_magnetometer"] = fake_magnetometer_main;
	apps["hello"] = hello_main;
	apps["px4_mavlink_debug"] = px4_mavlink_debug_main;
	apps["px4_simple_app"] = px4_simple_app_main;
	apps["work_item_example"] = work_item_example_main;

	apps["shutdown"] = shutdown_main;
	apps["list_tasks"] = list_tasks_main;
	apps["list_files"] = list_files_main;
	apps["sleep"] = sleep_main;
}

void list_builtins(apps_map_type &apps)
{
	printf("Builtin Commands:\n");
	for (apps_map_type::iterator it = apps.begin(); it != apps.end(); ++it) {
		printf("  %s\n", it->first.c_str());
	}
}

int shutdown_main(int argc, char *argv[])
{
	printf("Exiting NOW.\n");
	system_exit(0);
}

int list_tasks_main(int argc, char *argv[])
{
	px4_show_tasks();
	return 0;
}

int list_files_main(int argc, char *argv[])
{
	px4_show_files();
	return 0;
}

int sleep_main(int argc, char *argv[])
{
        if (argc != 2) {
           PX4_WARN( "Usage: sleep <seconds>" );
           return 1;
        }

        unsigned long usecs = 1000000UL * atol(argv[1]);
        printf("Sleeping for %s s; (%lu us).\n", argv[1], usecs);
        px4_usleep(usecs);
        return 0;
}
