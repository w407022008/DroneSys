// Auto-generated. Do not edit!

// (in-package mavros_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class CellularStatus {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.status = null;
      this.failure_reason = null;
      this.type = null;
      this.quality = null;
      this.mcc = null;
      this.mnc = null;
      this.lac = null;
    }
    else {
      if (initObj.hasOwnProperty('status')) {
        this.status = initObj.status
      }
      else {
        this.status = 0;
      }
      if (initObj.hasOwnProperty('failure_reason')) {
        this.failure_reason = initObj.failure_reason
      }
      else {
        this.failure_reason = 0;
      }
      if (initObj.hasOwnProperty('type')) {
        this.type = initObj.type
      }
      else {
        this.type = 0;
      }
      if (initObj.hasOwnProperty('quality')) {
        this.quality = initObj.quality
      }
      else {
        this.quality = 0;
      }
      if (initObj.hasOwnProperty('mcc')) {
        this.mcc = initObj.mcc
      }
      else {
        this.mcc = 0;
      }
      if (initObj.hasOwnProperty('mnc')) {
        this.mnc = initObj.mnc
      }
      else {
        this.mnc = 0;
      }
      if (initObj.hasOwnProperty('lac')) {
        this.lac = initObj.lac
      }
      else {
        this.lac = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CellularStatus
    // Serialize message field [status]
    bufferOffset = _serializer.uint8(obj.status, buffer, bufferOffset);
    // Serialize message field [failure_reason]
    bufferOffset = _serializer.uint8(obj.failure_reason, buffer, bufferOffset);
    // Serialize message field [type]
    bufferOffset = _serializer.uint8(obj.type, buffer, bufferOffset);
    // Serialize message field [quality]
    bufferOffset = _serializer.uint8(obj.quality, buffer, bufferOffset);
    // Serialize message field [mcc]
    bufferOffset = _serializer.uint16(obj.mcc, buffer, bufferOffset);
    // Serialize message field [mnc]
    bufferOffset = _serializer.uint16(obj.mnc, buffer, bufferOffset);
    // Serialize message field [lac]
    bufferOffset = _serializer.uint16(obj.lac, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CellularStatus
    let len;
    let data = new CellularStatus(null);
    // Deserialize message field [status]
    data.status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [failure_reason]
    data.failure_reason = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [type]
    data.type = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [quality]
    data.quality = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [mcc]
    data.mcc = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [mnc]
    data.mnc = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [lac]
    data.lac = _deserializer.uint16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 10;
  }

  static datatype() {
    // Returns string type for a message object
    return 'mavros_msgs/CellularStatus';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a474bdb9df111b4e16fab4c29f7a6956';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #Follows https://mavlink.io/en/messages/common.html#CELLULAR_STATUS specification
    
    uint8 status
    uint8 failure_reason
    uint8 type
    uint8 quality
    uint16 mcc
    uint16 mnc
    uint16 lac
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CellularStatus(null);
    if (msg.status !== undefined) {
      resolved.status = msg.status;
    }
    else {
      resolved.status = 0
    }

    if (msg.failure_reason !== undefined) {
      resolved.failure_reason = msg.failure_reason;
    }
    else {
      resolved.failure_reason = 0
    }

    if (msg.type !== undefined) {
      resolved.type = msg.type;
    }
    else {
      resolved.type = 0
    }

    if (msg.quality !== undefined) {
      resolved.quality = msg.quality;
    }
    else {
      resolved.quality = 0
    }

    if (msg.mcc !== undefined) {
      resolved.mcc = msg.mcc;
    }
    else {
      resolved.mcc = 0
    }

    if (msg.mnc !== undefined) {
      resolved.mnc = msg.mnc;
    }
    else {
      resolved.mnc = 0
    }

    if (msg.lac !== undefined) {
      resolved.lac = msg.lac;
    }
    else {
      resolved.lac = 0
    }

    return resolved;
    }
};

module.exports = CellularStatus;
