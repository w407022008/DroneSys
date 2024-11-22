// Auto-generated. Do not edit!

// (in-package mavros_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class SysStatus {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.sensors_present = null;
      this.sensors_enabled = null;
      this.sensors_health = null;
      this.load = null;
      this.voltage_battery = null;
      this.current_battery = null;
      this.battery_remaining = null;
      this.drop_rate_comm = null;
      this.errors_comm = null;
      this.errors_count1 = null;
      this.errors_count2 = null;
      this.errors_count3 = null;
      this.errors_count4 = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('sensors_present')) {
        this.sensors_present = initObj.sensors_present
      }
      else {
        this.sensors_present = 0;
      }
      if (initObj.hasOwnProperty('sensors_enabled')) {
        this.sensors_enabled = initObj.sensors_enabled
      }
      else {
        this.sensors_enabled = 0;
      }
      if (initObj.hasOwnProperty('sensors_health')) {
        this.sensors_health = initObj.sensors_health
      }
      else {
        this.sensors_health = 0;
      }
      if (initObj.hasOwnProperty('load')) {
        this.load = initObj.load
      }
      else {
        this.load = 0;
      }
      if (initObj.hasOwnProperty('voltage_battery')) {
        this.voltage_battery = initObj.voltage_battery
      }
      else {
        this.voltage_battery = 0;
      }
      if (initObj.hasOwnProperty('current_battery')) {
        this.current_battery = initObj.current_battery
      }
      else {
        this.current_battery = 0;
      }
      if (initObj.hasOwnProperty('battery_remaining')) {
        this.battery_remaining = initObj.battery_remaining
      }
      else {
        this.battery_remaining = 0;
      }
      if (initObj.hasOwnProperty('drop_rate_comm')) {
        this.drop_rate_comm = initObj.drop_rate_comm
      }
      else {
        this.drop_rate_comm = 0;
      }
      if (initObj.hasOwnProperty('errors_comm')) {
        this.errors_comm = initObj.errors_comm
      }
      else {
        this.errors_comm = 0;
      }
      if (initObj.hasOwnProperty('errors_count1')) {
        this.errors_count1 = initObj.errors_count1
      }
      else {
        this.errors_count1 = 0;
      }
      if (initObj.hasOwnProperty('errors_count2')) {
        this.errors_count2 = initObj.errors_count2
      }
      else {
        this.errors_count2 = 0;
      }
      if (initObj.hasOwnProperty('errors_count3')) {
        this.errors_count3 = initObj.errors_count3
      }
      else {
        this.errors_count3 = 0;
      }
      if (initObj.hasOwnProperty('errors_count4')) {
        this.errors_count4 = initObj.errors_count4
      }
      else {
        this.errors_count4 = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SysStatus
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [sensors_present]
    bufferOffset = _serializer.uint32(obj.sensors_present, buffer, bufferOffset);
    // Serialize message field [sensors_enabled]
    bufferOffset = _serializer.uint32(obj.sensors_enabled, buffer, bufferOffset);
    // Serialize message field [sensors_health]
    bufferOffset = _serializer.uint32(obj.sensors_health, buffer, bufferOffset);
    // Serialize message field [load]
    bufferOffset = _serializer.uint16(obj.load, buffer, bufferOffset);
    // Serialize message field [voltage_battery]
    bufferOffset = _serializer.uint16(obj.voltage_battery, buffer, bufferOffset);
    // Serialize message field [current_battery]
    bufferOffset = _serializer.int16(obj.current_battery, buffer, bufferOffset);
    // Serialize message field [battery_remaining]
    bufferOffset = _serializer.int8(obj.battery_remaining, buffer, bufferOffset);
    // Serialize message field [drop_rate_comm]
    bufferOffset = _serializer.uint16(obj.drop_rate_comm, buffer, bufferOffset);
    // Serialize message field [errors_comm]
    bufferOffset = _serializer.uint16(obj.errors_comm, buffer, bufferOffset);
    // Serialize message field [errors_count1]
    bufferOffset = _serializer.uint16(obj.errors_count1, buffer, bufferOffset);
    // Serialize message field [errors_count2]
    bufferOffset = _serializer.uint16(obj.errors_count2, buffer, bufferOffset);
    // Serialize message field [errors_count3]
    bufferOffset = _serializer.uint16(obj.errors_count3, buffer, bufferOffset);
    // Serialize message field [errors_count4]
    bufferOffset = _serializer.uint16(obj.errors_count4, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SysStatus
    let len;
    let data = new SysStatus(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [sensors_present]
    data.sensors_present = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [sensors_enabled]
    data.sensors_enabled = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [sensors_health]
    data.sensors_health = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [load]
    data.load = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [voltage_battery]
    data.voltage_battery = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [current_battery]
    data.current_battery = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [battery_remaining]
    data.battery_remaining = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [drop_rate_comm]
    data.drop_rate_comm = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [errors_comm]
    data.errors_comm = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [errors_count1]
    data.errors_count1 = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [errors_count2]
    data.errors_count2 = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [errors_count3]
    data.errors_count3 = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [errors_count4]
    data.errors_count4 = _deserializer.uint16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 31;
  }

  static datatype() {
    // Returns string type for a message object
    return 'mavros_msgs/SysStatus';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4039be26d76b32d20c569c754da6e25c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    uint32 sensors_present
    uint32 sensors_enabled
    uint32 sensors_health
    uint16 load
    uint16 voltage_battery
    int16 current_battery
    int8 battery_remaining
    uint16 drop_rate_comm
    uint16 errors_comm
    uint16 errors_count1
    uint16 errors_count2
    uint16 errors_count3
    uint16 errors_count4
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
    const resolved = new SysStatus(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.sensors_present !== undefined) {
      resolved.sensors_present = msg.sensors_present;
    }
    else {
      resolved.sensors_present = 0
    }

    if (msg.sensors_enabled !== undefined) {
      resolved.sensors_enabled = msg.sensors_enabled;
    }
    else {
      resolved.sensors_enabled = 0
    }

    if (msg.sensors_health !== undefined) {
      resolved.sensors_health = msg.sensors_health;
    }
    else {
      resolved.sensors_health = 0
    }

    if (msg.load !== undefined) {
      resolved.load = msg.load;
    }
    else {
      resolved.load = 0
    }

    if (msg.voltage_battery !== undefined) {
      resolved.voltage_battery = msg.voltage_battery;
    }
    else {
      resolved.voltage_battery = 0
    }

    if (msg.current_battery !== undefined) {
      resolved.current_battery = msg.current_battery;
    }
    else {
      resolved.current_battery = 0
    }

    if (msg.battery_remaining !== undefined) {
      resolved.battery_remaining = msg.battery_remaining;
    }
    else {
      resolved.battery_remaining = 0
    }

    if (msg.drop_rate_comm !== undefined) {
      resolved.drop_rate_comm = msg.drop_rate_comm;
    }
    else {
      resolved.drop_rate_comm = 0
    }

    if (msg.errors_comm !== undefined) {
      resolved.errors_comm = msg.errors_comm;
    }
    else {
      resolved.errors_comm = 0
    }

    if (msg.errors_count1 !== undefined) {
      resolved.errors_count1 = msg.errors_count1;
    }
    else {
      resolved.errors_count1 = 0
    }

    if (msg.errors_count2 !== undefined) {
      resolved.errors_count2 = msg.errors_count2;
    }
    else {
      resolved.errors_count2 = 0
    }

    if (msg.errors_count3 !== undefined) {
      resolved.errors_count3 = msg.errors_count3;
    }
    else {
      resolved.errors_count3 = 0
    }

    if (msg.errors_count4 !== undefined) {
      resolved.errors_count4 = msg.errors_count4;
    }
    else {
      resolved.errors_count4 = 0
    }

    return resolved;
    }
};

module.exports = SysStatus;
