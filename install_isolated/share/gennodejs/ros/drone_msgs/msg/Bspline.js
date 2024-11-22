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

//-----------------------------------------------------------

class Bspline {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.order = null;
      this.traj_id = null;
      this.knots = null;
      this.pts = null;
      this.start_time = null;
    }
    else {
      if (initObj.hasOwnProperty('order')) {
        this.order = initObj.order
      }
      else {
        this.order = 0;
      }
      if (initObj.hasOwnProperty('traj_id')) {
        this.traj_id = initObj.traj_id
      }
      else {
        this.traj_id = 0;
      }
      if (initObj.hasOwnProperty('knots')) {
        this.knots = initObj.knots
      }
      else {
        this.knots = [];
      }
      if (initObj.hasOwnProperty('pts')) {
        this.pts = initObj.pts
      }
      else {
        this.pts = [];
      }
      if (initObj.hasOwnProperty('start_time')) {
        this.start_time = initObj.start_time
      }
      else {
        this.start_time = {secs: 0, nsecs: 0};
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Bspline
    // Serialize message field [order]
    bufferOffset = _serializer.int32(obj.order, buffer, bufferOffset);
    // Serialize message field [traj_id]
    bufferOffset = _serializer.int64(obj.traj_id, buffer, bufferOffset);
    // Serialize message field [knots]
    bufferOffset = _arraySerializer.float64(obj.knots, buffer, bufferOffset, null);
    // Serialize message field [pts]
    // Serialize the length for message field [pts]
    bufferOffset = _serializer.uint32(obj.pts.length, buffer, bufferOffset);
    obj.pts.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [start_time]
    bufferOffset = _serializer.time(obj.start_time, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Bspline
    let len;
    let data = new Bspline(null);
    // Deserialize message field [order]
    data.order = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [traj_id]
    data.traj_id = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [knots]
    data.knots = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [pts]
    // Deserialize array length for message field [pts]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.pts = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.pts[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [start_time]
    data.start_time = _deserializer.time(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.knots.length;
    length += 24 * object.pts.length;
    return length + 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/Bspline';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8ae0a2f5019a8a20108147f57eefee55';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 order                 ## 
    int64 traj_id               ## id of trajecotry
    float64[] knots             ## knots list
    geometry_msgs/Point[] pts   ## control points list
    time start_time             ## time stamp
    
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
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
    const resolved = new Bspline(null);
    if (msg.order !== undefined) {
      resolved.order = msg.order;
    }
    else {
      resolved.order = 0
    }

    if (msg.traj_id !== undefined) {
      resolved.traj_id = msg.traj_id;
    }
    else {
      resolved.traj_id = 0
    }

    if (msg.knots !== undefined) {
      resolved.knots = msg.knots;
    }
    else {
      resolved.knots = []
    }

    if (msg.pts !== undefined) {
      resolved.pts = new Array(msg.pts.length);
      for (let i = 0; i < resolved.pts.length; ++i) {
        resolved.pts[i] = geometry_msgs.msg.Point.Resolve(msg.pts[i]);
      }
    }
    else {
      resolved.pts = []
    }

    if (msg.start_time !== undefined) {
      resolved.start_time = msg.start_time;
    }
    else {
      resolved.start_time = {secs: 0, nsecs: 0}
    }

    return resolved;
    }
};

module.exports = Bspline;
