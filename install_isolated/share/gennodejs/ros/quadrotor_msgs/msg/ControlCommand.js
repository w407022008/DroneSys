// Auto-generated. Do not edit!

// (in-package quadrotor_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class ControlCommand {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.control_mode = null;
      this.armed = null;
      this.expected_execution_time = null;
      this.orientation = null;
      this.bodyrates = null;
      this.angular_accelerations = null;
      this.collective_thrust = null;
      this.rotor_thrusts = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('control_mode')) {
        this.control_mode = initObj.control_mode
      }
      else {
        this.control_mode = 0;
      }
      if (initObj.hasOwnProperty('armed')) {
        this.armed = initObj.armed
      }
      else {
        this.armed = false;
      }
      if (initObj.hasOwnProperty('expected_execution_time')) {
        this.expected_execution_time = initObj.expected_execution_time
      }
      else {
        this.expected_execution_time = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('orientation')) {
        this.orientation = initObj.orientation
      }
      else {
        this.orientation = new geometry_msgs.msg.Quaternion();
      }
      if (initObj.hasOwnProperty('bodyrates')) {
        this.bodyrates = initObj.bodyrates
      }
      else {
        this.bodyrates = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('angular_accelerations')) {
        this.angular_accelerations = initObj.angular_accelerations
      }
      else {
        this.angular_accelerations = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('collective_thrust')) {
        this.collective_thrust = initObj.collective_thrust
      }
      else {
        this.collective_thrust = 0.0;
      }
      if (initObj.hasOwnProperty('rotor_thrusts')) {
        this.rotor_thrusts = initObj.rotor_thrusts
      }
      else {
        this.rotor_thrusts = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControlCommand
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [control_mode]
    bufferOffset = _serializer.uint8(obj.control_mode, buffer, bufferOffset);
    // Serialize message field [armed]
    bufferOffset = _serializer.bool(obj.armed, buffer, bufferOffset);
    // Serialize message field [expected_execution_time]
    bufferOffset = _serializer.time(obj.expected_execution_time, buffer, bufferOffset);
    // Serialize message field [orientation]
    bufferOffset = geometry_msgs.msg.Quaternion.serialize(obj.orientation, buffer, bufferOffset);
    // Serialize message field [bodyrates]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.bodyrates, buffer, bufferOffset);
    // Serialize message field [angular_accelerations]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.angular_accelerations, buffer, bufferOffset);
    // Serialize message field [collective_thrust]
    bufferOffset = _serializer.float64(obj.collective_thrust, buffer, bufferOffset);
    // Serialize message field [rotor_thrusts]
    bufferOffset = _arraySerializer.float64(obj.rotor_thrusts, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControlCommand
    let len;
    let data = new ControlCommand(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [control_mode]
    data.control_mode = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [armed]
    data.armed = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [expected_execution_time]
    data.expected_execution_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [orientation]
    data.orientation = geometry_msgs.msg.Quaternion.deserialize(buffer, bufferOffset);
    // Deserialize message field [bodyrates]
    data.bodyrates = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [angular_accelerations]
    data.angular_accelerations = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [collective_thrust]
    data.collective_thrust = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [rotor_thrusts]
    data.rotor_thrusts = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 8 * object.rotor_thrusts.length;
    return length + 102;
  }

  static datatype() {
    // Returns string type for a message object
    return 'quadrotor_msgs/ControlCommand';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a1918a34164f6647c898e4d55efbfcef';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Quadrotor control command
    
    # control mode enums
    uint8 NONE=0
    uint8 ATTITUDE=1
    uint8 BODY_RATES=2
    uint8 ANGULAR_ACCELERATIONS=3
    uint8 ROTOR_THRUSTS=4
    
    Header header
    
    # Control mode as defined above
    uint8 control_mode
    
    # Flag whether controller is allowed to arm
    bool armed
    
    # Time at which this command should be executed
    time expected_execution_time
    
    # Orientation of the body frame with respect to the world frame [-]
    geometry_msgs/Quaternion orientation
    
    # Body rates in body frame [rad/s]
    # Note that in ATTITUDE mode the x-y-bodyrates are only feed forward terms 
    # computed from a reference trajectory
    # Also in ATTITUDE mode, the z-bodyrate has to be from feedback control
    geometry_msgs/Vector3 bodyrates
    
    # Angular accelerations in body frame [rad/s^2]
    geometry_msgs/Vector3 angular_accelerations
    
    # Collective mass normalized thrust [m/s^2]
    float64 collective_thrust
    
    # Single rotor thrusts [N]
    # These are only considered in the ROTOR_THRUSTS control mode
    float64[] rotor_thrusts
    
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
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ControlCommand(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.control_mode !== undefined) {
      resolved.control_mode = msg.control_mode;
    }
    else {
      resolved.control_mode = 0
    }

    if (msg.armed !== undefined) {
      resolved.armed = msg.armed;
    }
    else {
      resolved.armed = false
    }

    if (msg.expected_execution_time !== undefined) {
      resolved.expected_execution_time = msg.expected_execution_time;
    }
    else {
      resolved.expected_execution_time = {secs: 0, nsecs: 0}
    }

    if (msg.orientation !== undefined) {
      resolved.orientation = geometry_msgs.msg.Quaternion.Resolve(msg.orientation)
    }
    else {
      resolved.orientation = new geometry_msgs.msg.Quaternion()
    }

    if (msg.bodyrates !== undefined) {
      resolved.bodyrates = geometry_msgs.msg.Vector3.Resolve(msg.bodyrates)
    }
    else {
      resolved.bodyrates = new geometry_msgs.msg.Vector3()
    }

    if (msg.angular_accelerations !== undefined) {
      resolved.angular_accelerations = geometry_msgs.msg.Vector3.Resolve(msg.angular_accelerations)
    }
    else {
      resolved.angular_accelerations = new geometry_msgs.msg.Vector3()
    }

    if (msg.collective_thrust !== undefined) {
      resolved.collective_thrust = msg.collective_thrust;
    }
    else {
      resolved.collective_thrust = 0.0
    }

    if (msg.rotor_thrusts !== undefined) {
      resolved.rotor_thrusts = msg.rotor_thrusts;
    }
    else {
      resolved.rotor_thrusts = []
    }

    return resolved;
    }
};

// Constants for message
ControlCommand.Constants = {
  NONE: 0,
  ATTITUDE: 1,
  BODY_RATES: 2,
  ANGULAR_ACCELERATIONS: 3,
  ROTOR_THRUSTS: 4,
}

module.exports = ControlCommand;
