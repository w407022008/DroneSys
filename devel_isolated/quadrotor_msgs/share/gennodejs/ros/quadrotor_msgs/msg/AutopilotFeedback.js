// Auto-generated. Do not edit!

// (in-package quadrotor_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let LowLevelFeedback = require('./LowLevelFeedback.js');
let TrajectoryPoint = require('./TrajectoryPoint.js');
let nav_msgs = _finder('nav_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class AutopilotFeedback {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.autopilot_state = null;
      this.control_command_delay = null;
      this.control_computation_time = null;
      this.trajectory_execution_left_duration = null;
      this.trajectories_left_in_queue = null;
      this.low_level_feedback = null;
      this.reference_state = null;
      this.state_estimate = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('autopilot_state')) {
        this.autopilot_state = initObj.autopilot_state
      }
      else {
        this.autopilot_state = 0;
      }
      if (initObj.hasOwnProperty('control_command_delay')) {
        this.control_command_delay = initObj.control_command_delay
      }
      else {
        this.control_command_delay = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('control_computation_time')) {
        this.control_computation_time = initObj.control_computation_time
      }
      else {
        this.control_computation_time = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('trajectory_execution_left_duration')) {
        this.trajectory_execution_left_duration = initObj.trajectory_execution_left_duration
      }
      else {
        this.trajectory_execution_left_duration = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('trajectories_left_in_queue')) {
        this.trajectories_left_in_queue = initObj.trajectories_left_in_queue
      }
      else {
        this.trajectories_left_in_queue = 0;
      }
      if (initObj.hasOwnProperty('low_level_feedback')) {
        this.low_level_feedback = initObj.low_level_feedback
      }
      else {
        this.low_level_feedback = new LowLevelFeedback();
      }
      if (initObj.hasOwnProperty('reference_state')) {
        this.reference_state = initObj.reference_state
      }
      else {
        this.reference_state = new TrajectoryPoint();
      }
      if (initObj.hasOwnProperty('state_estimate')) {
        this.state_estimate = initObj.state_estimate
      }
      else {
        this.state_estimate = new nav_msgs.msg.Odometry();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type AutopilotFeedback
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [autopilot_state]
    bufferOffset = _serializer.uint8(obj.autopilot_state, buffer, bufferOffset);
    // Serialize message field [control_command_delay]
    bufferOffset = _serializer.duration(obj.control_command_delay, buffer, bufferOffset);
    // Serialize message field [control_computation_time]
    bufferOffset = _serializer.duration(obj.control_computation_time, buffer, bufferOffset);
    // Serialize message field [trajectory_execution_left_duration]
    bufferOffset = _serializer.duration(obj.trajectory_execution_left_duration, buffer, bufferOffset);
    // Serialize message field [trajectories_left_in_queue]
    bufferOffset = _serializer.uint8(obj.trajectories_left_in_queue, buffer, bufferOffset);
    // Serialize message field [low_level_feedback]
    bufferOffset = LowLevelFeedback.serialize(obj.low_level_feedback, buffer, bufferOffset);
    // Serialize message field [reference_state]
    bufferOffset = TrajectoryPoint.serialize(obj.reference_state, buffer, bufferOffset);
    // Serialize message field [state_estimate]
    bufferOffset = nav_msgs.msg.Odometry.serialize(obj.state_estimate, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type AutopilotFeedback
    let len;
    let data = new AutopilotFeedback(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [autopilot_state]
    data.autopilot_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [control_command_delay]
    data.control_command_delay = _deserializer.duration(buffer, bufferOffset);
    // Deserialize message field [control_computation_time]
    data.control_computation_time = _deserializer.duration(buffer, bufferOffset);
    // Deserialize message field [trajectory_execution_left_duration]
    data.trajectory_execution_left_duration = _deserializer.duration(buffer, bufferOffset);
    // Deserialize message field [trajectories_left_in_queue]
    data.trajectories_left_in_queue = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [low_level_feedback]
    data.low_level_feedback = LowLevelFeedback.deserialize(buffer, bufferOffset);
    // Deserialize message field [reference_state]
    data.reference_state = TrajectoryPoint.deserialize(buffer, bufferOffset);
    // Deserialize message field [state_estimate]
    data.state_estimate = nav_msgs.msg.Odometry.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += LowLevelFeedback.getMessageSize(object.low_level_feedback);
    length += nav_msgs.msg.Odometry.getMessageSize(object.state_estimate);
    return length + 314;
  }

  static datatype() {
    // Returns string type for a message object
    return 'quadrotor_msgs/AutopilotFeedback';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8c8e08f7c3465bc93596097f7c8ecc39';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Autopilot state enums
    uint8 OFF=0
    uint8 START=1
    uint8 HOVER=2
    uint8 LAND=3
    uint8 EMERGENCY_LAND=4
    uint8 BREAKING=5
    uint8 GO_TO_POSE=6
    uint8 VELOCITY_CONTROL=7
    uint8 REFERENCE_CONTROL=8
    uint8 TRAJECTORY_CONTROL=9
    uint8 COMMAND_FEEDTHROUGH=10
    uint8 RC_MANUAL=11
    
    
    Header header
    
    # Autopilot state as defined above. This reflects what is implemented in
    # autopilot/include/autopilot/autopilot.h
    uint8 autopilot_state
    
    # Control command delay
    duration control_command_delay
    
    # Controller computation time [s]
    duration control_computation_time
    
    # Duration left of the trajectories in the queue
    # Only valid in TRAJECTORY_CONTROL mode
    duration trajectory_execution_left_duration
    
    # Number of trajectories that were sent to the autopilot and are stored in its
    # queue. Only valid in TRAJECTORY_CONTROL mode
    uint8 trajectories_left_in_queue
    
    # Low level feedback
    quadrotor_msgs/LowLevelFeedback low_level_feedback
    
    # Desired state used to compute the control command
    quadrotor_msgs/TrajectoryPoint reference_state
    
    # State estimate used to compute the control command
    nav_msgs/Odometry state_estimate
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: quadrotor_msgs/LowLevelFeedback
    # battery state enums
    uint8 BAT_INVALID=0
    uint8 BAT_GOOD=1
    uint8 BAT_LOW=2
    uint8 BAT_CRITICAL=3
    
    # control mode enums as defined in ControlCommand.msg
    uint8 NONE=0
    uint8 ATTITUDE=1
    uint8 BODY_RATES=2
    uint8 ANGULAR_ACCELERATION=3
    uint8 ROTOR_THRUSTS=4
    # Additionally to the control command we want to know whether an RC has taken
    # over from the low level feedback
    uint8 RC_MANUAL=10
    
    Header header
    
    # Battery information
    float32 battery_voltage
    uint8 battery_state
    
    # Control mode as defined above
    uint8 control_mode
    
    # Motor speed feedback [rpm]
    int16[] motor_speeds
    
    # Thrust mapping coefficients
    # thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +
    #     thrust_mapping_coeffs[0]
    float64[] thrust_mapping_coeffs
    
    ================================================================================
    MSG: quadrotor_msgs/TrajectoryPoint
    duration time_from_start
    
    geometry_msgs/Pose pose
    
    geometry_msgs/Twist velocity
    
    geometry_msgs/Twist acceleration
    
    geometry_msgs/Twist jerk
    
    geometry_msgs/Twist snap
    
    # Heading angle with respect to world frame [rad]
    float64 heading
    
    # First derivative of the heading angle [rad/s]
    float64 heading_rate
    
    # Second derivative of the heading angle [rad/s^2]
    float64 heading_acceleration
    
    # Collective thrust [m/s^2]
    float64 thrust
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Twist
    # This expresses velocity in free space broken into its linear and angular parts.
    Vector3  linear
    Vector3  angular
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    ================================================================================
    MSG: nav_msgs/Odometry
    # This represents an estimate of a position and velocity in free space.  
    # The pose in this message should be specified in the coordinate frame given by header.frame_id.
    # The twist in this message should be specified in the coordinate frame given by the child_frame_id
    Header header
    string child_frame_id
    geometry_msgs/PoseWithCovariance pose
    geometry_msgs/TwistWithCovariance twist
    
    ================================================================================
    MSG: geometry_msgs/PoseWithCovariance
    # This represents a pose in free space with uncertainty.
    
    Pose pose
    
    # Row-major representation of the 6x6 covariance matrix
    # The orientation parameters use a fixed-axis representation.
    # In order, the parameters are:
    # (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    float64[36] covariance
    
    ================================================================================
    MSG: geometry_msgs/TwistWithCovariance
    # This expresses velocity in free space with uncertainty.
    
    Twist twist
    
    # Row-major representation of the 6x6 covariance matrix
    # The orientation parameters use a fixed-axis representation.
    # In order, the parameters are:
    # (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    float64[36] covariance
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new AutopilotFeedback(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.autopilot_state !== undefined) {
      resolved.autopilot_state = msg.autopilot_state;
    }
    else {
      resolved.autopilot_state = 0
    }

    if (msg.control_command_delay !== undefined) {
      resolved.control_command_delay = msg.control_command_delay;
    }
    else {
      resolved.control_command_delay = {secs: 0, nsecs: 0}
    }

    if (msg.control_computation_time !== undefined) {
      resolved.control_computation_time = msg.control_computation_time;
    }
    else {
      resolved.control_computation_time = {secs: 0, nsecs: 0}
    }

    if (msg.trajectory_execution_left_duration !== undefined) {
      resolved.trajectory_execution_left_duration = msg.trajectory_execution_left_duration;
    }
    else {
      resolved.trajectory_execution_left_duration = {secs: 0, nsecs: 0}
    }

    if (msg.trajectories_left_in_queue !== undefined) {
      resolved.trajectories_left_in_queue = msg.trajectories_left_in_queue;
    }
    else {
      resolved.trajectories_left_in_queue = 0
    }

    if (msg.low_level_feedback !== undefined) {
      resolved.low_level_feedback = LowLevelFeedback.Resolve(msg.low_level_feedback)
    }
    else {
      resolved.low_level_feedback = new LowLevelFeedback()
    }

    if (msg.reference_state !== undefined) {
      resolved.reference_state = TrajectoryPoint.Resolve(msg.reference_state)
    }
    else {
      resolved.reference_state = new TrajectoryPoint()
    }

    if (msg.state_estimate !== undefined) {
      resolved.state_estimate = nav_msgs.msg.Odometry.Resolve(msg.state_estimate)
    }
    else {
      resolved.state_estimate = new nav_msgs.msg.Odometry()
    }

    return resolved;
    }
};

// Constants for message
AutopilotFeedback.Constants = {
  OFF: 0,
  START: 1,
  HOVER: 2,
  LAND: 3,
  EMERGENCY_LAND: 4,
  BREAKING: 5,
  GO_TO_POSE: 6,
  VELOCITY_CONTROL: 7,
  REFERENCE_CONTROL: 8,
  TRAJECTORY_CONTROL: 9,
  COMMAND_FEEDTHROUGH: 10,
  RC_MANUAL: 11,
}

module.exports = AutopilotFeedback;
