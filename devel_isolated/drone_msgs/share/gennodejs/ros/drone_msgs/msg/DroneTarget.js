// Auto-generated. Do not edit!

// (in-package drone_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');
let std_msgs = _finder('std_msgs');
let mavros_msgs = _finder('mavros_msgs');

//-----------------------------------------------------------

class DroneTarget {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.position_target = null;
      this.velocity_target = null;
      this.acceleration_target = null;
      this.q_target = null;
      this.euler_target = null;
      this.rate_target = null;
      this.thrust_target = null;
      this.actuator_target = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('position_target')) {
        this.position_target = initObj.position_target
      }
      else {
        this.position_target = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('velocity_target')) {
        this.velocity_target = initObj.velocity_target
      }
      else {
        this.velocity_target = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('acceleration_target')) {
        this.acceleration_target = initObj.acceleration_target
      }
      else {
        this.acceleration_target = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('q_target')) {
        this.q_target = initObj.q_target
      }
      else {
        this.q_target = new geometry_msgs.msg.Quaternion();
      }
      if (initObj.hasOwnProperty('euler_target')) {
        this.euler_target = initObj.euler_target
      }
      else {
        this.euler_target = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('rate_target')) {
        this.rate_target = initObj.rate_target
      }
      else {
        this.rate_target = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('thrust_target')) {
        this.thrust_target = initObj.thrust_target
      }
      else {
        this.thrust_target = 0.0;
      }
      if (initObj.hasOwnProperty('actuator_target')) {
        this.actuator_target = initObj.actuator_target
      }
      else {
        this.actuator_target = new mavros_msgs.msg.ActuatorControl();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DroneTarget
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Check that the constant length array field [position_target] has the right length
    if (obj.position_target.length !== 3) {
      throw new Error('Unable to serialize array field position_target - length must be 3')
    }
    // Serialize message field [position_target]
    bufferOffset = _arraySerializer.float32(obj.position_target, buffer, bufferOffset, 3);
    // Check that the constant length array field [velocity_target] has the right length
    if (obj.velocity_target.length !== 3) {
      throw new Error('Unable to serialize array field velocity_target - length must be 3')
    }
    // Serialize message field [velocity_target]
    bufferOffset = _arraySerializer.float32(obj.velocity_target, buffer, bufferOffset, 3);
    // Check that the constant length array field [acceleration_target] has the right length
    if (obj.acceleration_target.length !== 3) {
      throw new Error('Unable to serialize array field acceleration_target - length must be 3')
    }
    // Serialize message field [acceleration_target]
    bufferOffset = _arraySerializer.float32(obj.acceleration_target, buffer, bufferOffset, 3);
    // Serialize message field [q_target]
    bufferOffset = geometry_msgs.msg.Quaternion.serialize(obj.q_target, buffer, bufferOffset);
    // Check that the constant length array field [euler_target] has the right length
    if (obj.euler_target.length !== 3) {
      throw new Error('Unable to serialize array field euler_target - length must be 3')
    }
    // Serialize message field [euler_target]
    bufferOffset = _arraySerializer.float32(obj.euler_target, buffer, bufferOffset, 3);
    // Check that the constant length array field [rate_target] has the right length
    if (obj.rate_target.length !== 3) {
      throw new Error('Unable to serialize array field rate_target - length must be 3')
    }
    // Serialize message field [rate_target]
    bufferOffset = _arraySerializer.float32(obj.rate_target, buffer, bufferOffset, 3);
    // Serialize message field [thrust_target]
    bufferOffset = _serializer.float32(obj.thrust_target, buffer, bufferOffset);
    // Serialize message field [actuator_target]
    bufferOffset = mavros_msgs.msg.ActuatorControl.serialize(obj.actuator_target, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DroneTarget
    let len;
    let data = new DroneTarget(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [position_target]
    data.position_target = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [velocity_target]
    data.velocity_target = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [acceleration_target]
    data.acceleration_target = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [q_target]
    data.q_target = geometry_msgs.msg.Quaternion.deserialize(buffer, bufferOffset);
    // Deserialize message field [euler_target]
    data.euler_target = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [rate_target]
    data.rate_target = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [thrust_target]
    data.thrust_target = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [actuator_target]
    data.actuator_target = mavros_msgs.msg.ActuatorControl.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += mavros_msgs.msg.ActuatorControl.getMessageSize(object.actuator_target);
    return length + 96;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/DroneTarget';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b13c4477f8a36524e314a3b537e64de4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    float32[3] position_target          ## [m]
    float32[3] velocity_target          ## [m/s]
    float32[3] acceleration_target      ## [m/s/s]
    geometry_msgs/Quaternion q_target   ## quat
    float32[3] euler_target             ## [rad]
    float32[3] rate_target              ## [rad/s]
    float32 thrust_target
    mavros_msgs/ActuatorControl actuator_target
    
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
    MSG: mavros_msgs/ActuatorControl
    # raw servo values for direct actuator controls
    #
    # about groups, mixing and channels:
    # https://pixhawk.org/dev/mixing
    
    # constant for mixer group
    uint8 PX4_MIX_FLIGHT_CONTROL = 0
    uint8 PX4_MIX_FLIGHT_CONTROL_VTOL_ALT = 1
    uint8 PX4_MIX_PAYLOAD = 2
    uint8 PX4_MIX_MANUAL_PASSTHROUGH = 3
    #uint8 PX4_MIX_FC_MC_VIRT = 4
    #uint8 PX4_MIX_FC_FW_VIRT = 5
    
    std_msgs/Header header
    uint8 group_mix
    float32[8] controls
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DroneTarget(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.position_target !== undefined) {
      resolved.position_target = msg.position_target;
    }
    else {
      resolved.position_target = new Array(3).fill(0)
    }

    if (msg.velocity_target !== undefined) {
      resolved.velocity_target = msg.velocity_target;
    }
    else {
      resolved.velocity_target = new Array(3).fill(0)
    }

    if (msg.acceleration_target !== undefined) {
      resolved.acceleration_target = msg.acceleration_target;
    }
    else {
      resolved.acceleration_target = new Array(3).fill(0)
    }

    if (msg.q_target !== undefined) {
      resolved.q_target = geometry_msgs.msg.Quaternion.Resolve(msg.q_target)
    }
    else {
      resolved.q_target = new geometry_msgs.msg.Quaternion()
    }

    if (msg.euler_target !== undefined) {
      resolved.euler_target = msg.euler_target;
    }
    else {
      resolved.euler_target = new Array(3).fill(0)
    }

    if (msg.rate_target !== undefined) {
      resolved.rate_target = msg.rate_target;
    }
    else {
      resolved.rate_target = new Array(3).fill(0)
    }

    if (msg.thrust_target !== undefined) {
      resolved.thrust_target = msg.thrust_target;
    }
    else {
      resolved.thrust_target = 0.0
    }

    if (msg.actuator_target !== undefined) {
      resolved.actuator_target = mavros_msgs.msg.ActuatorControl.Resolve(msg.actuator_target)
    }
    else {
      resolved.actuator_target = new mavros_msgs.msg.ActuatorControl()
    }

    return resolved;
    }
};

module.exports = DroneTarget;
