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

class ControlOutput {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.Thrust = null;
      this.Throttle = null;
      this.u_l = null;
      this.u_d = null;
      this.NE = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('Thrust')) {
        this.Thrust = initObj.Thrust
      }
      else {
        this.Thrust = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('Throttle')) {
        this.Throttle = initObj.Throttle
      }
      else {
        this.Throttle = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('u_l')) {
        this.u_l = initObj.u_l
      }
      else {
        this.u_l = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('u_d')) {
        this.u_d = initObj.u_d
      }
      else {
        this.u_d = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('NE')) {
        this.NE = initObj.NE
      }
      else {
        this.NE = new Array(3).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControlOutput
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Check that the constant length array field [Thrust] has the right length
    if (obj.Thrust.length !== 3) {
      throw new Error('Unable to serialize array field Thrust - length must be 3')
    }
    // Serialize message field [Thrust]
    bufferOffset = _arraySerializer.float32(obj.Thrust, buffer, bufferOffset, 3);
    // Check that the constant length array field [Throttle] has the right length
    if (obj.Throttle.length !== 3) {
      throw new Error('Unable to serialize array field Throttle - length must be 3')
    }
    // Serialize message field [Throttle]
    bufferOffset = _arraySerializer.float32(obj.Throttle, buffer, bufferOffset, 3);
    // Check that the constant length array field [u_l] has the right length
    if (obj.u_l.length !== 3) {
      throw new Error('Unable to serialize array field u_l - length must be 3')
    }
    // Serialize message field [u_l]
    bufferOffset = _arraySerializer.float32(obj.u_l, buffer, bufferOffset, 3);
    // Check that the constant length array field [u_d] has the right length
    if (obj.u_d.length !== 3) {
      throw new Error('Unable to serialize array field u_d - length must be 3')
    }
    // Serialize message field [u_d]
    bufferOffset = _arraySerializer.float32(obj.u_d, buffer, bufferOffset, 3);
    // Check that the constant length array field [NE] has the right length
    if (obj.NE.length !== 3) {
      throw new Error('Unable to serialize array field NE - length must be 3')
    }
    // Serialize message field [NE]
    bufferOffset = _arraySerializer.float32(obj.NE, buffer, bufferOffset, 3);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControlOutput
    let len;
    let data = new ControlOutput(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [Thrust]
    data.Thrust = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [Throttle]
    data.Throttle = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [u_l]
    data.u_l = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [u_d]
    data.u_d = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [NE]
    data.NE = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 60;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/ControlOutput';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '08f4e53b4980f9738cc0255cfbfcc182';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    float32[3] Thrust               
    float32[3] Throttle
    
    float32[3] u_l                 ## [0-1]
    float32[3] u_d                 ## [rad]
    float32[3] NE                  ## [rad]
      
    
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
    const resolved = new ControlOutput(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.Thrust !== undefined) {
      resolved.Thrust = msg.Thrust;
    }
    else {
      resolved.Thrust = new Array(3).fill(0)
    }

    if (msg.Throttle !== undefined) {
      resolved.Throttle = msg.Throttle;
    }
    else {
      resolved.Throttle = new Array(3).fill(0)
    }

    if (msg.u_l !== undefined) {
      resolved.u_l = msg.u_l;
    }
    else {
      resolved.u_l = new Array(3).fill(0)
    }

    if (msg.u_d !== undefined) {
      resolved.u_d = msg.u_d;
    }
    else {
      resolved.u_d = new Array(3).fill(0)
    }

    if (msg.NE !== undefined) {
      resolved.NE = msg.NE;
    }
    else {
      resolved.NE = new Array(3).fill(0)
    }

    return resolved;
    }
};

module.exports = ControlOutput;
