// Auto-generated. Do not edit!

// (in-package drone_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PositionReference = require('./PositionReference.js');
let AttitudeReference = require('./AttitudeReference.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class ControlCommand {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.Command_ID = null;
      this.source = null;
      this.Mode = null;
      this.Reference_State = null;
      this.Attitude_sp = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('Command_ID')) {
        this.Command_ID = initObj.Command_ID
      }
      else {
        this.Command_ID = 0;
      }
      if (initObj.hasOwnProperty('source')) {
        this.source = initObj.source
      }
      else {
        this.source = '';
      }
      if (initObj.hasOwnProperty('Mode')) {
        this.Mode = initObj.Mode
      }
      else {
        this.Mode = 0;
      }
      if (initObj.hasOwnProperty('Reference_State')) {
        this.Reference_State = initObj.Reference_State
      }
      else {
        this.Reference_State = new PositionReference();
      }
      if (initObj.hasOwnProperty('Attitude_sp')) {
        this.Attitude_sp = initObj.Attitude_sp
      }
      else {
        this.Attitude_sp = new AttitudeReference();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControlCommand
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [Command_ID]
    bufferOffset = _serializer.uint32(obj.Command_ID, buffer, bufferOffset);
    // Serialize message field [source]
    bufferOffset = _serializer.string(obj.source, buffer, bufferOffset);
    // Serialize message field [Mode]
    bufferOffset = _serializer.uint8(obj.Mode, buffer, bufferOffset);
    // Serialize message field [Reference_State]
    bufferOffset = PositionReference.serialize(obj.Reference_State, buffer, bufferOffset);
    // Serialize message field [Attitude_sp]
    bufferOffset = AttitudeReference.serialize(obj.Attitude_sp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControlCommand
    let len;
    let data = new ControlCommand(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [Command_ID]
    data.Command_ID = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [source]
    data.source = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [Mode]
    data.Mode = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [Reference_State]
    data.Reference_State = PositionReference.deserialize(buffer, bufferOffset);
    // Deserialize message field [Attitude_sp]
    data.Attitude_sp = AttitudeReference.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += _getByteLength(object.source);
    length += PositionReference.getMessageSize(object.Reference_State);
    length += AttitudeReference.getMessageSize(object.Attitude_sp);
    return length + 9;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/ControlCommand';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '969640b304f3a446799efdd5c334e9b7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    ## ID should increased self
    uint32 Command_ID
    
    string source
    
    uint8 Mode
    # enum
    uint8 Idle=0
    uint8 Takeoff=1
    uint8 Hold=2
    uint8 Land=3
    uint8 Move=4
    uint8 Disarm=5
    uint8 Attitude=6
    uint8 AttitudeRate=7
    uint8 Rate=8
    
    ## Setpoint Reference
    PositionReference Reference_State
    AttitudeReference Attitude_sp
    
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
    MSG: drone_msgs/PositionReference
    std_msgs/Header header
    
    ## Setpoint position reference for PX4 Control
    
    ## Setpoint Mode
    uint8 Move_mode
    
    uint8 XYZ_POS      = 0  ##0b00
    uint8 XY_POS_Z_VEL = 1  ##0b01
    uint8 XY_VEL_Z_POS = 2  ##0b10
    uint8 XYZ_VEL = 3       ##0b11
    uint8 XYZ_ACC = 4
    uint8 XYZ_POS_VEL   = 5  
    uint8 TRAJECTORY   = 6
    
    ## Reference Frame
    uint8 Move_frame
    
    uint8 ENU_FRAME  = 0
    uint8 BODY_FRAME = 1
    
    
    
    ## Tracking life
    float32 time_from_start          ## [s]
    
    float32[3] position_ref          ## [m]
    float32[3] velocity_ref          ## [m/s]
    float32[3] acceleration_ref      ## [m/s^2]
    
    bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率
    float32 yaw_ref                  ## [rad]
    float32 yaw_rate_ref             ## [rad/s] 
    
    Bspline bspline
    ================================================================================
    MSG: drone_msgs/Bspline
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
    
    ================================================================================
    MSG: drone_msgs/AttitudeReference
    std_msgs/Header header
    
    ## Setpoint Attitude + T
    float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
    float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint
    float32[3] desired_attitude            ## [rad] Eurler angle setpoint
    geometry_msgs/Quaternion desired_att_q ## quat setpoint
    geometry_msgs/Vector3 body_rate  ## [rad/s]
    
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

    if (msg.Command_ID !== undefined) {
      resolved.Command_ID = msg.Command_ID;
    }
    else {
      resolved.Command_ID = 0
    }

    if (msg.source !== undefined) {
      resolved.source = msg.source;
    }
    else {
      resolved.source = ''
    }

    if (msg.Mode !== undefined) {
      resolved.Mode = msg.Mode;
    }
    else {
      resolved.Mode = 0
    }

    if (msg.Reference_State !== undefined) {
      resolved.Reference_State = PositionReference.Resolve(msg.Reference_State)
    }
    else {
      resolved.Reference_State = new PositionReference()
    }

    if (msg.Attitude_sp !== undefined) {
      resolved.Attitude_sp = AttitudeReference.Resolve(msg.Attitude_sp)
    }
    else {
      resolved.Attitude_sp = new AttitudeReference()
    }

    return resolved;
    }
};

// Constants for message
ControlCommand.Constants = {
  IDLE: 0,
  TAKEOFF: 1,
  HOLD: 2,
  LAND: 3,
  MOVE: 4,
  DISARM: 5,
  ATTITUDE: 6,
  ATTITUDERATE: 7,
  RATE: 8,
}

module.exports = ControlCommand;
