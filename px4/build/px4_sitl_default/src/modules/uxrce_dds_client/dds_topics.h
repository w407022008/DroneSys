
#include <utilities.hpp>

#include <uxr/client/client.h>
#include <ucdr/microcdr.h>

#include <uORB/Subscription.hpp>
#include <uORB/Publication.hpp>
#include <uORB/ucdr/collision_constraints.h>
#include <uORB/ucdr/failsafe_flags.h>
#include <uORB/ucdr/obstacle_distance.h>
#include <uORB/ucdr/offboard_control_mode.h>
#include <uORB/ucdr/onboard_computer_status.h>
#include <uORB/ucdr/position_setpoint_triplet.h>
#include <uORB/ucdr/sensor_combined.h>
#include <uORB/ucdr/sensor_gps.h>
#include <uORB/ucdr/sensor_optical_flow.h>
#include <uORB/ucdr/telemetry_status.h>
#include <uORB/ucdr/timesync_status.h>
#include <uORB/ucdr/trajectory_setpoint.h>
#include <uORB/ucdr/vehicle_attitude.h>
#include <uORB/ucdr/vehicle_attitude_setpoint.h>
#include <uORB/ucdr/vehicle_command.h>
#include <uORB/ucdr/vehicle_control_mode.h>
#include <uORB/ucdr/vehicle_global_position.h>
#include <uORB/ucdr/vehicle_local_position.h>
#include <uORB/ucdr/vehicle_odometry.h>
#include <uORB/ucdr/vehicle_rates_setpoint.h>
#include <uORB/ucdr/vehicle_status.h>
#include <uORB/ucdr/vehicle_trajectory_bezier.h>
#include <uORB/ucdr/vehicle_trajectory_waypoint.h>

// Subscribers for messages to send
struct SendTopicsSubs {
	uORB::Subscription collision_constraints_sub{ORB_ID(collision_constraints)};
	uxrObjectId collision_constraints_data_writer{};
	uORB::Subscription failsafe_flags_sub{ORB_ID(failsafe_flags)};
	uxrObjectId failsafe_flags_data_writer{};
	uORB::Subscription position_setpoint_triplet_sub{ORB_ID(position_setpoint_triplet)};
	uxrObjectId position_setpoint_triplet_data_writer{};
	uORB::Subscription sensor_combined_sub{ORB_ID(sensor_combined)};
	uxrObjectId sensor_combined_data_writer{};
	uORB::Subscription timesync_status_sub{ORB_ID(timesync_status)};
	uxrObjectId timesync_status_data_writer{};
	uORB::Subscription vehicle_attitude_sub{ORB_ID(vehicle_attitude)};
	uxrObjectId vehicle_attitude_data_writer{};
	uORB::Subscription vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uxrObjectId vehicle_control_mode_data_writer{};
	uORB::Subscription vehicle_global_position_sub{ORB_ID(vehicle_global_position)};
	uxrObjectId vehicle_global_position_data_writer{};
	uORB::Subscription vehicle_gps_position_sub{ORB_ID(vehicle_gps_position)};
	uxrObjectId vehicle_gps_position_data_writer{};
	uORB::Subscription vehicle_local_position_sub{ORB_ID(vehicle_local_position)};
	uxrObjectId vehicle_local_position_data_writer{};
	uORB::Subscription vehicle_odometry_sub{ORB_ID(vehicle_odometry)};
	uxrObjectId vehicle_odometry_data_writer{};
	uORB::Subscription vehicle_status_sub{ORB_ID(vehicle_status)};
	uxrObjectId vehicle_status_data_writer{};
	uORB::Subscription vehicle_trajectory_waypoint_desired_sub{ORB_ID(vehicle_trajectory_waypoint_desired)};
	uxrObjectId vehicle_trajectory_waypoint_desired_data_writer{};

	uint32_t num_payload_sent{};

	void update(uxrSession *session, uxrStreamId reliable_out_stream_id, uxrStreamId best_effort_stream_id, uxrObjectId participant_id, const char *client_namespace);
	void reset();
};

void SendTopicsSubs::reset() {
	num_payload_sent = 0;
	collision_constraints_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	failsafe_flags_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	position_setpoint_triplet_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	sensor_combined_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	timesync_status_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_attitude_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_control_mode_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_global_position_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_gps_position_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_local_position_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_odometry_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_status_data_writer = uxr_object_id(0, UXR_INVALID_ID);
	vehicle_trajectory_waypoint_desired_data_writer = uxr_object_id(0, UXR_INVALID_ID);
};

void SendTopicsSubs::update(uxrSession *session, uxrStreamId reliable_out_stream_id, uxrStreamId best_effort_stream_id, uxrObjectId participant_id, const char *client_namespace)
{
	int64_t time_offset_us = session->time_offset / 1000; // ns -> us

	{
		collision_constraints_s data;

		if (collision_constraints_sub.update(&data)) {

			if (collision_constraints_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::collision_constraints, client_namespace, "collision_constraints", "px4_msgs::msg::dds_::CollisionConstraints_", collision_constraints_data_writer);
			}

			if (collision_constraints_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_collision_constraints();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, collision_constraints_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_collision_constraints(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID collision_constraints");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID collision_constraints");
			}

		}
	}

	{
		failsafe_flags_s data;

		if (failsafe_flags_sub.update(&data)) {

			if (failsafe_flags_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::failsafe_flags, client_namespace, "failsafe_flags", "px4_msgs::msg::dds_::FailsafeFlags_", failsafe_flags_data_writer);
			}

			if (failsafe_flags_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_failsafe_flags();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, failsafe_flags_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_failsafe_flags(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID failsafe_flags");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID failsafe_flags");
			}

		}
	}

	{
		position_setpoint_triplet_s data;

		if (position_setpoint_triplet_sub.update(&data)) {

			if (position_setpoint_triplet_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::position_setpoint_triplet, client_namespace, "position_setpoint_triplet", "px4_msgs::msg::dds_::PositionSetpointTriplet_", position_setpoint_triplet_data_writer);
			}

			if (position_setpoint_triplet_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_position_setpoint_triplet();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, position_setpoint_triplet_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_position_setpoint_triplet(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID position_setpoint_triplet");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID position_setpoint_triplet");
			}

		}
	}

	{
		sensor_combined_s data;

		if (sensor_combined_sub.update(&data)) {

			if (sensor_combined_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::sensor_combined, client_namespace, "sensor_combined", "px4_msgs::msg::dds_::SensorCombined_", sensor_combined_data_writer);
			}

			if (sensor_combined_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_sensor_combined();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, sensor_combined_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_sensor_combined(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID sensor_combined");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID sensor_combined");
			}

		}
	}

	{
		timesync_status_s data;

		if (timesync_status_sub.update(&data)) {

			if (timesync_status_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::timesync_status, client_namespace, "timesync_status", "px4_msgs::msg::dds_::TimesyncStatus_", timesync_status_data_writer);
			}

			if (timesync_status_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_timesync_status();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, timesync_status_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_timesync_status(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID timesync_status");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID timesync_status");
			}

		}
	}

	{
		vehicle_attitude_s data;

		if (vehicle_attitude_sub.update(&data)) {

			if (vehicle_attitude_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_attitude, client_namespace, "vehicle_attitude", "px4_msgs::msg::dds_::VehicleAttitude_", vehicle_attitude_data_writer);
			}

			if (vehicle_attitude_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_attitude();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_attitude_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_attitude(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_attitude");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_attitude");
			}

		}
	}

	{
		vehicle_control_mode_s data;

		if (vehicle_control_mode_sub.update(&data)) {

			if (vehicle_control_mode_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_control_mode, client_namespace, "vehicle_control_mode", "px4_msgs::msg::dds_::VehicleControlMode_", vehicle_control_mode_data_writer);
			}

			if (vehicle_control_mode_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_control_mode();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_control_mode_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_control_mode(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_control_mode");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_control_mode");
			}

		}
	}

	{
		vehicle_global_position_s data;

		if (vehicle_global_position_sub.update(&data)) {

			if (vehicle_global_position_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_global_position, client_namespace, "vehicle_global_position", "px4_msgs::msg::dds_::VehicleGlobalPosition_", vehicle_global_position_data_writer);
			}

			if (vehicle_global_position_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_global_position();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_global_position_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_global_position(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_global_position");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_global_position");
			}

		}
	}

	{
		sensor_gps_s data;

		if (vehicle_gps_position_sub.update(&data)) {

			if (vehicle_gps_position_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_gps_position, client_namespace, "vehicle_gps_position", "px4_msgs::msg::dds_::SensorGps_", vehicle_gps_position_data_writer);
			}

			if (vehicle_gps_position_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_sensor_gps();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_gps_position_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_sensor_gps(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_gps_position");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_gps_position");
			}

		}
	}

	{
		vehicle_local_position_s data;

		if (vehicle_local_position_sub.update(&data)) {

			if (vehicle_local_position_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_local_position, client_namespace, "vehicle_local_position", "px4_msgs::msg::dds_::VehicleLocalPosition_", vehicle_local_position_data_writer);
			}

			if (vehicle_local_position_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_local_position();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_local_position_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_local_position(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_local_position");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_local_position");
			}

		}
	}

	{
		vehicle_odometry_s data;

		if (vehicle_odometry_sub.update(&data)) {

			if (vehicle_odometry_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_odometry, client_namespace, "vehicle_odometry", "px4_msgs::msg::dds_::VehicleOdometry_", vehicle_odometry_data_writer);
			}

			if (vehicle_odometry_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_odometry();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_odometry_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_odometry(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_odometry");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_odometry");
			}

		}
	}

	{
		vehicle_status_s data;

		if (vehicle_status_sub.update(&data)) {

			if (vehicle_status_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_status, client_namespace, "vehicle_status", "px4_msgs::msg::dds_::VehicleStatus_", vehicle_status_data_writer);
			}

			if (vehicle_status_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_status();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_status_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_status(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_status");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_status");
			}

		}
	}

	{
		vehicle_trajectory_waypoint_s data;

		if (vehicle_trajectory_waypoint_desired_sub.update(&data)) {

			if (vehicle_trajectory_waypoint_desired_data_writer.id == UXR_INVALID_ID) {
				// data writer not created yet
				create_data_writer(session, reliable_out_stream_id, participant_id, ORB_ID::vehicle_trajectory_waypoint_desired, client_namespace, "vehicle_trajectory_waypoint_desired", "px4_msgs::msg::dds_::VehicleTrajectoryWaypoint_", vehicle_trajectory_waypoint_desired_data_writer);
			}

			if (vehicle_trajectory_waypoint_desired_data_writer.id != UXR_INVALID_ID) {

				ucdrBuffer ub;
				uint32_t topic_size = ucdr_topic_size_vehicle_trajectory_waypoint();
				if (uxr_prepare_output_stream(session, best_effort_stream_id, vehicle_trajectory_waypoint_desired_data_writer, &ub, topic_size) != UXR_INVALID_REQUEST_ID) {
					ucdr_serialize_vehicle_trajectory_waypoint(data, ub, time_offset_us);
					// TODO: fill up the MTU and then flush, which reduces the packet overhead
					uxr_flash_output_streams(session);
					num_payload_sent += topic_size;

				} else {
					//PX4_ERR("Error uxr_prepare_output_stream UXR_INVALID_REQUEST_ID vehicle_trajectory_waypoint_desired");
				}

			} else {
				//PX4_ERR("Error UXR_INVALID_ID vehicle_trajectory_waypoint_desired");
			}

		}
	}
}

// Publishers for received messages
struct RcvTopicsPubs {
	uORB::Publication<offboard_control_mode_s> offboard_control_mode_pub{ORB_ID(offboard_control_mode)};
	uORB::Publication<onboard_computer_status_s> onboard_computer_status_pub{ORB_ID(onboard_computer_status)};
	uORB::Publication<obstacle_distance_s> obstacle_distance_pub{ORB_ID(obstacle_distance)};
	uORB::Publication<sensor_optical_flow_s> sensor_optical_flow_pub{ORB_ID(sensor_optical_flow)};
	uORB::Publication<telemetry_status_s> telemetry_status_pub{ORB_ID(telemetry_status)};
	uORB::Publication<trajectory_setpoint_s> trajectory_setpoint_pub{ORB_ID(trajectory_setpoint)};
	uORB::Publication<vehicle_attitude_setpoint_s> vehicle_attitude_setpoint_pub{ORB_ID(vehicle_attitude_setpoint)};
	uORB::Publication<vehicle_odometry_s> vehicle_mocap_odometry_pub{ORB_ID(vehicle_mocap_odometry)};
	uORB::Publication<vehicle_rates_setpoint_s> vehicle_rates_setpoint_pub{ORB_ID(vehicle_rates_setpoint)};
	uORB::Publication<vehicle_odometry_s> vehicle_visual_odometry_pub{ORB_ID(vehicle_visual_odometry)};
	uORB::Publication<vehicle_command_s> vehicle_command_pub{ORB_ID(vehicle_command)};
	uORB::Publication<vehicle_trajectory_bezier_s> vehicle_trajectory_bezier_pub{ORB_ID(vehicle_trajectory_bezier)};
	uORB::Publication<vehicle_trajectory_waypoint_s> vehicle_trajectory_waypoint_pub{ORB_ID(vehicle_trajectory_waypoint)};

	uint32_t num_payload_received{};

	bool init(uxrSession *session, uxrStreamId reliable_out_stream_id, uxrStreamId reliable_in_stream_id, uxrStreamId best_effort_in_stream_id, uxrObjectId participant_id, const char *client_namespace);
};

static void on_topic_update(uxrSession *session, uxrObjectId object_id, uint16_t request_id, uxrStreamId stream_id,
		     struct ucdrBuffer *ub, uint16_t length, void *args)
{
	RcvTopicsPubs *pubs = (RcvTopicsPubs *)args;
	const int64_t time_offset_us = session->time_offset / 1000; // ns -> us
	pubs->num_payload_received += length;

	switch (object_id.id) {
	case 0+1000: {
			offboard_control_mode_s data;

			if (ucdr_deserialize_offboard_control_mode(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(offboard_control_mode), data);
				pubs->offboard_control_mode_pub.publish(data);
			}
		}
		break;

	case 1+1000: {
			onboard_computer_status_s data;

			if (ucdr_deserialize_onboard_computer_status(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(onboard_computer_status), data);
				pubs->onboard_computer_status_pub.publish(data);
			}
		}
		break;

	case 2+1000: {
			obstacle_distance_s data;

			if (ucdr_deserialize_obstacle_distance(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(obstacle_distance), data);
				pubs->obstacle_distance_pub.publish(data);
			}
		}
		break;

	case 3+1000: {
			sensor_optical_flow_s data;

			if (ucdr_deserialize_sensor_optical_flow(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(sensor_optical_flow), data);
				pubs->sensor_optical_flow_pub.publish(data);
			}
		}
		break;

	case 4+1000: {
			telemetry_status_s data;

			if (ucdr_deserialize_telemetry_status(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(telemetry_status), data);
				pubs->telemetry_status_pub.publish(data);
			}
		}
		break;

	case 5+1000: {
			trajectory_setpoint_s data;

			if (ucdr_deserialize_trajectory_setpoint(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(trajectory_setpoint), data);
				pubs->trajectory_setpoint_pub.publish(data);
			}
		}
		break;

	case 6+1000: {
			vehicle_attitude_setpoint_s data;

			if (ucdr_deserialize_vehicle_attitude_setpoint(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_attitude_setpoint), data);
				pubs->vehicle_attitude_setpoint_pub.publish(data);
			}
		}
		break;

	case 7+1000: {
			vehicle_odometry_s data;

			if (ucdr_deserialize_vehicle_odometry(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_odometry), data);
				pubs->vehicle_mocap_odometry_pub.publish(data);
			}
		}
		break;

	case 8+1000: {
			vehicle_rates_setpoint_s data;

			if (ucdr_deserialize_vehicle_rates_setpoint(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_rates_setpoint), data);
				pubs->vehicle_rates_setpoint_pub.publish(data);
			}
		}
		break;

	case 9+1000: {
			vehicle_odometry_s data;

			if (ucdr_deserialize_vehicle_odometry(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_odometry), data);
				pubs->vehicle_visual_odometry_pub.publish(data);
			}
		}
		break;

	case 10+1000: {
			vehicle_command_s data;

			if (ucdr_deserialize_vehicle_command(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_command), data);
				pubs->vehicle_command_pub.publish(data);
			}
		}
		break;

	case 11+1000: {
			vehicle_trajectory_bezier_s data;

			if (ucdr_deserialize_vehicle_trajectory_bezier(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_trajectory_bezier), data);
				pubs->vehicle_trajectory_bezier_pub.publish(data);
			}
		}
		break;

	case 12+1000: {
			vehicle_trajectory_waypoint_s data;

			if (ucdr_deserialize_vehicle_trajectory_waypoint(*ub, data, time_offset_us)) {
				//print_message(ORB_ID(vehicle_trajectory_waypoint), data);
				pubs->vehicle_trajectory_waypoint_pub.publish(data);
			}
		}
		break;


	default:
		PX4_ERR("unknown object id: %i", object_id.id);
		break;
	}
}

bool RcvTopicsPubs::init(uxrSession *session, uxrStreamId reliable_out_stream_id, uxrStreamId reliable_in_stream_id, uxrStreamId best_effort_in_stream_id, uxrObjectId participant_id, const char *client_namespace)
{
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<offboard_control_mode_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 0, client_namespace, "offboard_control_mode", "px4_msgs::msg::dds_::OffboardControlMode_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<onboard_computer_status_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 1, client_namespace, "onboard_computer_status", "px4_msgs::msg::dds_::OnboardComputerStatus_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<obstacle_distance_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 2, client_namespace, "obstacle_distance", "px4_msgs::msg::dds_::ObstacleDistance_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<sensor_optical_flow_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 3, client_namespace, "sensor_optical_flow", "px4_msgs::msg::dds_::SensorOpticalFlow_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<telemetry_status_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 4, client_namespace, "telemetry_status", "px4_msgs::msg::dds_::TelemetryStatus_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<trajectory_setpoint_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 5, client_namespace, "trajectory_setpoint", "px4_msgs::msg::dds_::TrajectorySetpoint_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_attitude_setpoint_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 6, client_namespace, "vehicle_attitude_setpoint", "px4_msgs::msg::dds_::VehicleAttitudeSetpoint_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_odometry_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 7, client_namespace, "vehicle_mocap_odometry", "px4_msgs::msg::dds_::VehicleOdometry_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_rates_setpoint_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 8, client_namespace, "vehicle_rates_setpoint", "px4_msgs::msg::dds_::VehicleRatesSetpoint_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_odometry_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 9, client_namespace, "vehicle_visual_odometry", "px4_msgs::msg::dds_::VehicleOdometry_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_command_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 10, client_namespace, "vehicle_command", "px4_msgs::msg::dds_::VehicleCommand_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_trajectory_bezier_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 11, client_namespace, "vehicle_trajectory_bezier", "px4_msgs::msg::dds_::VehicleTrajectoryBezier_", queue_depth);
	}
	{
			uint16_t queue_depth = uORB::DefaultQueueSize<vehicle_trajectory_waypoint_s>::value * 2; // use a bit larger queue size than internal
			create_data_reader(session, reliable_out_stream_id, best_effort_in_stream_id, participant_id, 12, client_namespace, "vehicle_trajectory_waypoint", "px4_msgs::msg::dds_::VehicleTrajectoryWaypoint_", queue_depth);
	}

	uxr_set_topic_callback(session, on_topic_update, this);

	return true;
}
