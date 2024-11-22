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

//-----------------------------------------------------------

class LowLevelFeedback {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.battery_voltage = null;
      this.battery_state = null;
      this.control_mode = null;
      this.motor_speeds = null;
      this.thrust_mapping_coeffs = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('battery_voltage')) {
        this.battery_voltage = initObj.battery_voltage
      }
      else {
        this.battery_voltage = 0.0;
      }
      if (initObj.hasOwnProperty('battery_state')) {
        this.battery_state = initObj.battery_state
      }
      else {
        this.battery_state = 0;
      }
      if (initObj.hasOwnProperty('control_mode')) {
        this.control_mode = initObj.control_mode
      }
      else {
        this.control_mode = 0;
      }
      if (initObj.hasOwnProperty('motor_speeds')) {
        this.motor_speeds = initObj.motor_speeds
      }
      else {
        this.motor_speeds = [];
      }
      if (initObj.hasOwnProperty('thrust_mapping_coeffs')) {
        this.thrust_mapping_coeffs = initObj.thrust_mapping_coeffs
      }
      else {
        this.thrust_mapping_coeffs = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type LowLevelFeedback
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [battery_voltage]
    bufferOffset = _serializer.float32(obj.battery_voltage, buffer, bufferOffset);
    // Serialize message field [battery_state]
    bufferOffset = _serializer.uint8(obj.battery_state, buffer, bufferOffset);
    // Serialize message field [control_mode]
    bufferOffset = _serializer.uint8(obj.control_mode, buffer, bufferOffset);
    // Serialize message field [motor_speeds]
    bufferOffset = _arraySerializer.int16(obj.motor_speeds, buffer, bufferOffset, null);
    // Serialize message field [thrust_mapping_coeffs]
    bufferOffset = _arraySerializer.float64(obj.thrust_mapping_coeffs, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type LowLevelFeedback
    let len;
    let data = new LowLevelFeedback(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [battery_voltage]
    data.battery_voltage = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [battery_state]
    data.battery_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [control_mode]
    data.control_mode = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [motor_speeds]
    data.motor_speeds = _arrayDeserializer.int16(buffer, bufferOffset, null)
    // Deserialize message field [thrust_mapping_coeffs]
    data.thrust_mapping_coeffs = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 2 * object.motor_speeds.length;
    length += 8 * object.thrust_mapping_coeffs.length;
    return length + 14;
  }

  static datatype() {
    // Returns string type for a message object
    return 'quadrotor_msgs/LowLevelFeedback';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e3cfad3ba98dfdc505bcf1fe91833d87';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new LowLevelFeedback(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.battery_voltage !== undefined) {
      resolved.battery_voltage = msg.battery_voltage;
    }
    else {
      resolved.battery_voltage = 0.0
    }

    if (msg.battery_state !== undefined) {
      resolved.battery_state = msg.battery_state;
    }
    else {
      resolved.battery_state = 0
    }

    if (msg.control_mode !== undefined) {
      resolved.control_mode = msg.control_mode;
    }
    else {
      resolved.control_mode = 0
    }

    if (msg.motor_speeds !== undefined) {
      resolved.motor_speeds = msg.motor_speeds;
    }
    else {
      resolved.motor_speeds = []
    }

    if (msg.thrust_mapping_coeffs !== undefined) {
      resolved.thrust_mapping_coeffs = msg.thrust_mapping_coeffs;
    }
    else {
      resolved.thrust_mapping_coeffs = []
    }

    return resolved;
    }
};

// Constants for message
LowLevelFeedback.Constants = {
  BAT_INVALID: 0,
  BAT_GOOD: 1,
  BAT_LOW: 2,
  BAT_CRITICAL: 3,
  NONE: 0,
  ATTITUDE: 1,
  BODY_RATES: 2,
  ANGULAR_ACCELERATION: 3,
  ROTOR_THRUSTS: 4,
  RC_MANUAL: 10,
}

module.exports = LowLevelFeedback;
