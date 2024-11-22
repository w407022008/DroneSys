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

//-----------------------------------------------------------

class AttitudeReference {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.thrust_sp = null;
      this.collective_accel = null;
      this.desired_attitude = null;
      this.desired_att_q = null;
      this.body_rate = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('thrust_sp')) {
        this.thrust_sp = initObj.thrust_sp
      }
      else {
        this.thrust_sp = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('collective_accel')) {
        this.collective_accel = initObj.collective_accel
      }
      else {
        this.collective_accel = 0.0;
      }
      if (initObj.hasOwnProperty('desired_attitude')) {
        this.desired_attitude = initObj.desired_attitude
      }
      else {
        this.desired_attitude = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('desired_att_q')) {
        this.desired_att_q = initObj.desired_att_q
      }
      else {
        this.desired_att_q = new geometry_msgs.msg.Quaternion();
      }
      if (initObj.hasOwnProperty('body_rate')) {
        this.body_rate = initObj.body_rate
      }
      else {
        this.body_rate = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type AttitudeReference
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Check that the constant length array field [thrust_sp] has the right length
    if (obj.thrust_sp.length !== 3) {
      throw new Error('Unable to serialize array field thrust_sp - length must be 3')
    }
    // Serialize message field [thrust_sp]
    bufferOffset = _arraySerializer.float32(obj.thrust_sp, buffer, bufferOffset, 3);
    // Serialize message field [collective_accel]
    bufferOffset = _serializer.float32(obj.collective_accel, buffer, bufferOffset);
    // Check that the constant length array field [desired_attitude] has the right length
    if (obj.desired_attitude.length !== 3) {
      throw new Error('Unable to serialize array field desired_attitude - length must be 3')
    }
    // Serialize message field [desired_attitude]
    bufferOffset = _arraySerializer.float32(obj.desired_attitude, buffer, bufferOffset, 3);
    // Serialize message field [desired_att_q]
    bufferOffset = geometry_msgs.msg.Quaternion.serialize(obj.desired_att_q, buffer, bufferOffset);
    // Serialize message field [body_rate]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.body_rate, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type AttitudeReference
    let len;
    let data = new AttitudeReference(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [thrust_sp]
    data.thrust_sp = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [collective_accel]
    data.collective_accel = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [desired_attitude]
    data.desired_attitude = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [desired_att_q]
    data.desired_att_q = geometry_msgs.msg.Quaternion.deserialize(buffer, bufferOffset);
    // Deserialize message field [body_rate]
    data.body_rate = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 84;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/AttitudeReference';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ad65c8727b64e262c550df8ad8b37905';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    ## Setpoint Attitude + T
    float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
    float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint
    float32[3] desired_attitude            ## [rad] Eurler angle setpoint
    geometry_msgs/Quaternion desired_att_q ## quat setpoint
    geometry_msgs/Vector3 body_rate  ## [rad/s]
    
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
    const resolved = new AttitudeReference(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.thrust_sp !== undefined) {
      resolved.thrust_sp = msg.thrust_sp;
    }
    else {
      resolved.thrust_sp = new Array(3).fill(0)
    }

    if (msg.collective_accel !== undefined) {
      resolved.collective_accel = msg.collective_accel;
    }
    else {
      resolved.collective_accel = 0.0
    }

    if (msg.desired_attitude !== undefined) {
      resolved.desired_attitude = msg.desired_attitude;
    }
    else {
      resolved.desired_attitude = new Array(3).fill(0)
    }

    if (msg.desired_att_q !== undefined) {
      resolved.desired_att_q = geometry_msgs.msg.Quaternion.Resolve(msg.desired_att_q)
    }
    else {
      resolved.desired_att_q = new geometry_msgs.msg.Quaternion()
    }

    if (msg.body_rate !== undefined) {
      resolved.body_rate = geometry_msgs.msg.Vector3.Resolve(msg.body_rate)
    }
    else {
      resolved.body_rate = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

module.exports = AttitudeReference;
