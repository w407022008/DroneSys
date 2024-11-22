// Auto-generated. Do not edit!

// (in-package drone_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class RCInput {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.rc_x = null;
      this.rc_y = null;
      this.rc_z = null;
      this.rc_r = null;
      this.buttons = null;
      this.goal_enable = null;
      this.data_source = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('rc_x')) {
        this.rc_x = initObj.rc_x
      }
      else {
        this.rc_x = 0.0;
      }
      if (initObj.hasOwnProperty('rc_y')) {
        this.rc_y = initObj.rc_y
      }
      else {
        this.rc_y = 0.0;
      }
      if (initObj.hasOwnProperty('rc_z')) {
        this.rc_z = initObj.rc_z
      }
      else {
        this.rc_z = 0.0;
      }
      if (initObj.hasOwnProperty('rc_r')) {
        this.rc_r = initObj.rc_r
      }
      else {
        this.rc_r = 0.0;
      }
      if (initObj.hasOwnProperty('buttons')) {
        this.buttons = initObj.buttons
      }
      else {
        this.buttons = 0;
      }
      if (initObj.hasOwnProperty('goal_enable')) {
        this.goal_enable = initObj.goal_enable
      }
      else {
        this.goal_enable = 0;
      }
      if (initObj.hasOwnProperty('data_source')) {
        this.data_source = initObj.data_source
      }
      else {
        this.data_source = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RCInput
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [rc_x]
    bufferOffset = _serializer.float32(obj.rc_x, buffer, bufferOffset);
    // Serialize message field [rc_y]
    bufferOffset = _serializer.float32(obj.rc_y, buffer, bufferOffset);
    // Serialize message field [rc_z]
    bufferOffset = _serializer.float32(obj.rc_z, buffer, bufferOffset);
    // Serialize message field [rc_r]
    bufferOffset = _serializer.float32(obj.rc_r, buffer, bufferOffset);
    // Serialize message field [buttons]
    bufferOffset = _serializer.uint32(obj.buttons, buffer, bufferOffset);
    // Serialize message field [goal_enable]
    bufferOffset = _serializer.uint8(obj.goal_enable, buffer, bufferOffset);
    // Serialize message field [data_source]
    bufferOffset = _serializer.int32(obj.data_source, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RCInput
    let len;
    let data = new RCInput(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [rc_x]
    data.rc_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rc_y]
    data.rc_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rc_z]
    data.rc_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rc_r]
    data.rc_r = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [buttons]
    data.buttons = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [goal_enable]
    data.goal_enable = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [data_source]
    data.data_source = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 25;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/RCInput';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '13d10a65cefb07444f918f9ce0babb28';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    ## Radio Control Input
    std_msgs/Header header
    
    # Data Source
    uint8 DISABLE = 0
    uint8 MAVROS_MANUAL_CONTROL = 1
    uint8 DRIVER_JOYSTICK = 2
    
    float32 rc_x             # stick position in x direction -1..1
                             # in general corresponds to forward/back motion or pitch of vehicle,
                             # in general a positive value means forward or negative pitch and
                             # a negative value means backward or positive pitch
    float32 rc_y             # stick position in y direction -1..1
                             # in general corresponds to right/left motion or roll of vehicle,
                             # in general a positive value means right or positive roll and
                             # a negative value means left or negative roll
    float32 rc_z             # throttle stick position 0..1
                             # in general corresponds to up/down motion or thrust of vehicle,
                             # in general the value corresponds to the demanded throttle by the user,
                             # if the input is used for setting the setpoint of a vertical position
                             # controller any value > 0.5 means up and any value < 0.5 means down
    float32 rc_r             # yaw stick/twist position, -1..1
                             # in general corresponds to the righthand rotation around the vertical
                             # (downwards) axis of the vehicle
    uint32 buttons           # Binary
    
    uint8 goal_enable        # push down(1):enable; release(0):disable
    
    int32 data_source # determin the data source
    
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
    const resolved = new RCInput(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.rc_x !== undefined) {
      resolved.rc_x = msg.rc_x;
    }
    else {
      resolved.rc_x = 0.0
    }

    if (msg.rc_y !== undefined) {
      resolved.rc_y = msg.rc_y;
    }
    else {
      resolved.rc_y = 0.0
    }

    if (msg.rc_z !== undefined) {
      resolved.rc_z = msg.rc_z;
    }
    else {
      resolved.rc_z = 0.0
    }

    if (msg.rc_r !== undefined) {
      resolved.rc_r = msg.rc_r;
    }
    else {
      resolved.rc_r = 0.0
    }

    if (msg.buttons !== undefined) {
      resolved.buttons = msg.buttons;
    }
    else {
      resolved.buttons = 0
    }

    if (msg.goal_enable !== undefined) {
      resolved.goal_enable = msg.goal_enable;
    }
    else {
      resolved.goal_enable = 0
    }

    if (msg.data_source !== undefined) {
      resolved.data_source = msg.data_source;
    }
    else {
      resolved.data_source = 0
    }

    return resolved;
    }
};

// Constants for message
RCInput.Constants = {
  DISABLE: 0,
  MAVROS_MANUAL_CONTROL: 1,
  DRIVER_JOYSTICK: 2,
}

module.exports = RCInput;
