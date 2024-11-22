// Auto-generated. Do not edit!

// (in-package mavros_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetMavFrameRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mav_frame = null;
    }
    else {
      if (initObj.hasOwnProperty('mav_frame')) {
        this.mav_frame = initObj.mav_frame
      }
      else {
        this.mav_frame = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetMavFrameRequest
    // Serialize message field [mav_frame]
    bufferOffset = _serializer.uint8(obj.mav_frame, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetMavFrameRequest
    let len;
    let data = new SetMavFrameRequest(null);
    // Deserialize message field [mav_frame]
    data.mav_frame = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'mavros_msgs/SetMavFrameRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '10d9e03dcd8d648e58b34bad2a28091f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Set MAV_FRAME for setpoints
    
    # [[[cog:
    # from pymavlink.dialects.v20 import common
    #
    # def decl_enum(ename, pfx='', bsz=8):
    #     enum = sorted(common.enums[ename].items())
    #     enum.pop() # remove ENUM_END
    #
    #     cog.outl("# " + ename)
    #     for k, e in enum:
    #         sn = e.name[len(ename) + 1:]
    #         l = "uint{bsz} {pfx}{sn} = {k}".format(**locals())
    #         if e.description:
    #             l += ' ' * (40 - len(l)) + ' # ' + e.description
    #         cog.outl(l)
    #
    # decl_enum('MAV_FRAME', 'FRAME_')
    # ]]]
    # MAV_FRAME
    uint8 FRAME_GLOBAL = 0                   # Global (WGS84) coordinate frame + MSL altitude. First value / x: latitude, second value / y: longitude, third value / z: positive altitude over mean sea level (MSL).
    uint8 FRAME_LOCAL_NED = 1                # NED local tangent frame (x: North, y: East, z: Down) with origin fixed relative to earth.
    uint8 FRAME_MISSION = 2                  # NOT a coordinate frame, indicates a mission command.
    uint8 FRAME_GLOBAL_RELATIVE_ALT = 3      # Global (WGS84) coordinate frame + altitude relative to the home position. First value / x: latitude, second value / y: longitude, third value / z: positive altitude with 0 being at the altitude of the home location.
    uint8 FRAME_LOCAL_ENU = 4                # ENU local tangent frame (x: East, y: North, z: Up) with origin fixed relative to earth.
    uint8 FRAME_GLOBAL_INT = 5               # Global (WGS84) coordinate frame (scaled) + MSL altitude. First value / x: latitude in degrees*1E7, second value / y: longitude in degrees*1E7, third value / z: positive altitude over mean sea level (MSL).
    uint8 FRAME_GLOBAL_RELATIVE_ALT_INT = 6  # Global (WGS84) coordinate frame (scaled) + altitude relative to the home position. First value / x: latitude in degrees*1E7, second value / y: longitude in degrees*1E7, third value / z: positive altitude with 0 being at the altitude of the home location.
    uint8 FRAME_LOCAL_OFFSET_NED = 7         # NED local tangent frame (x: North, y: East, z: Down) with origin that travels with the vehicle.
    uint8 FRAME_BODY_NED = 8                 # Same as MAV_FRAME_LOCAL_NED when used to represent position values. Same as MAV_FRAME_BODY_FRD when used with velocity/accelaration values.
    uint8 FRAME_BODY_OFFSET_NED = 9          # This is the same as MAV_FRAME_BODY_FRD.
    uint8 FRAME_GLOBAL_TERRAIN_ALT = 10      # Global (WGS84) coordinate frame with AGL altitude (at the waypoint coordinate). First value / x: latitude in degrees, second value / y: longitude in degrees, third value / z: positive altitude in meters with 0 being at ground level in terrain model.
    uint8 FRAME_GLOBAL_TERRAIN_ALT_INT = 11  # Global (WGS84) coordinate frame (scaled) with AGL altitude (at the waypoint coordinate). First value / x: latitude in degrees*1E7, second value / y: longitude in degrees*1E7, third value / z: positive altitude in meters with 0 being at ground level in terrain model.
    uint8 FRAME_BODY_FRD = 12                # FRD local tangent frame (x: Forward, y: Right, z: Down) with origin that travels with vehicle. The forward axis is aligned to the front of the vehicle in the horizontal plane.
    uint8 FRAME_RESERVED_13 = 13             # MAV_FRAME_BODY_FLU - Body fixed frame of reference, Z-up (x: Forward, y: Left, z: Up).
    uint8 FRAME_RESERVED_14 = 14             # MAV_FRAME_MOCAP_NED - Odometry local coordinate frame of data given by a motion capture system, Z-down (x: North, y: East, z: Down).
    uint8 FRAME_RESERVED_15 = 15             # MAV_FRAME_MOCAP_ENU - Odometry local coordinate frame of data given by a motion capture system, Z-up (x: East, y: North, z: Up).
    uint8 FRAME_RESERVED_16 = 16             # MAV_FRAME_VISION_NED - Odometry local coordinate frame of data given by a vision estimation system, Z-down (x: North, y: East, z: Down).
    uint8 FRAME_RESERVED_17 = 17             # MAV_FRAME_VISION_ENU - Odometry local coordinate frame of data given by a vision estimation system, Z-up (x: East, y: North, z: Up).
    uint8 FRAME_RESERVED_18 = 18             # MAV_FRAME_ESTIM_NED - Odometry local coordinate frame of data given by an estimator running onboard the vehicle, Z-down (x: North, y: East, z: Down).
    uint8 FRAME_RESERVED_19 = 19             # MAV_FRAME_ESTIM_ENU - Odometry local coordinate frame of data given by an estimator running onboard the vehicle, Z-up (x: East, y: North, z: Up).
    uint8 FRAME_LOCAL_FRD = 20               # FRD local tangent frame (x: Forward, y: Right, z: Down) with origin fixed relative to earth. The forward axis is aligned to the front of the vehicle in the horizontal plane.
    uint8 FRAME_LOCAL_FLU = 21               # FLU local tangent frame (x: Forward, y: Left, z: Up) with origin fixed relative to earth. The forward axis is aligned to the front of the vehicle in the horizontal plane.
    # [[[end]]] (checksum: c5ddb537c91e87c4efba8b24c9cde50e)
    
    uint8 mav_frame
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetMavFrameRequest(null);
    if (msg.mav_frame !== undefined) {
      resolved.mav_frame = msg.mav_frame;
    }
    else {
      resolved.mav_frame = 0
    }

    return resolved;
    }
};

// Constants for message
SetMavFrameRequest.Constants = {
  FRAME_GLOBAL: 0,
  FRAME_LOCAL_NED: 1,
  FRAME_MISSION: 2,
  FRAME_GLOBAL_RELATIVE_ALT: 3,
  FRAME_LOCAL_ENU: 4,
  FRAME_GLOBAL_INT: 5,
  FRAME_GLOBAL_RELATIVE_ALT_INT: 6,
  FRAME_LOCAL_OFFSET_NED: 7,
  FRAME_BODY_NED: 8,
  FRAME_BODY_OFFSET_NED: 9,
  FRAME_GLOBAL_TERRAIN_ALT: 10,
  FRAME_GLOBAL_TERRAIN_ALT_INT: 11,
  FRAME_BODY_FRD: 12,
  FRAME_RESERVED_13: 13,
  FRAME_RESERVED_14: 14,
  FRAME_RESERVED_15: 15,
  FRAME_RESERVED_16: 16,
  FRAME_RESERVED_17: 17,
  FRAME_RESERVED_18: 18,
  FRAME_RESERVED_19: 19,
  FRAME_LOCAL_FRD: 20,
  FRAME_LOCAL_FLU: 21,
}

class SetMavFrameResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetMavFrameResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetMavFrameResponse
    let len;
    let data = new SetMavFrameResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'mavros_msgs/SetMavFrameResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetMavFrameResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: SetMavFrameRequest,
  Response: SetMavFrameResponse,
  md5sum() { return 'bda5ad49b9b82fbf5d1eeb3c9cdc0bfa'; },
  datatype() { return 'mavros_msgs/SetMavFrame'; }
};
