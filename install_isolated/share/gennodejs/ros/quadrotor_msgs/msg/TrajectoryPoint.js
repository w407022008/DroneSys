// Auto-generated. Do not edit!

// (in-package quadrotor_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class TrajectoryPoint {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.time_from_start = null;
      this.pose = null;
      this.velocity = null;
      this.acceleration = null;
      this.jerk = null;
      this.snap = null;
      this.heading = null;
      this.heading_rate = null;
      this.heading_acceleration = null;
      this.thrust = null;
    }
    else {
      if (initObj.hasOwnProperty('time_from_start')) {
        this.time_from_start = initObj.time_from_start
      }
      else {
        this.time_from_start = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('pose')) {
        this.pose = initObj.pose
      }
      else {
        this.pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('acceleration')) {
        this.acceleration = initObj.acceleration
      }
      else {
        this.acceleration = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('jerk')) {
        this.jerk = initObj.jerk
      }
      else {
        this.jerk = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('snap')) {
        this.snap = initObj.snap
      }
      else {
        this.snap = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('heading')) {
        this.heading = initObj.heading
      }
      else {
        this.heading = 0.0;
      }
      if (initObj.hasOwnProperty('heading_rate')) {
        this.heading_rate = initObj.heading_rate
      }
      else {
        this.heading_rate = 0.0;
      }
      if (initObj.hasOwnProperty('heading_acceleration')) {
        this.heading_acceleration = initObj.heading_acceleration
      }
      else {
        this.heading_acceleration = 0.0;
      }
      if (initObj.hasOwnProperty('thrust')) {
        this.thrust = initObj.thrust
      }
      else {
        this.thrust = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TrajectoryPoint
    // Serialize message field [time_from_start]
    bufferOffset = _serializer.duration(obj.time_from_start, buffer, bufferOffset);
    // Serialize message field [pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.pose, buffer, bufferOffset);
    // Serialize message field [velocity]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.velocity, buffer, bufferOffset);
    // Serialize message field [acceleration]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.acceleration, buffer, bufferOffset);
    // Serialize message field [jerk]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.jerk, buffer, bufferOffset);
    // Serialize message field [snap]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.snap, buffer, bufferOffset);
    // Serialize message field [heading]
    bufferOffset = _serializer.float64(obj.heading, buffer, bufferOffset);
    // Serialize message field [heading_rate]
    bufferOffset = _serializer.float64(obj.heading_rate, buffer, bufferOffset);
    // Serialize message field [heading_acceleration]
    bufferOffset = _serializer.float64(obj.heading_acceleration, buffer, bufferOffset);
    // Serialize message field [thrust]
    bufferOffset = _serializer.float64(obj.thrust, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TrajectoryPoint
    let len;
    let data = new TrajectoryPoint(null);
    // Deserialize message field [time_from_start]
    data.time_from_start = _deserializer.duration(buffer, bufferOffset);
    // Deserialize message field [pose]
    data.pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [velocity]
    data.velocity = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [acceleration]
    data.acceleration = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [jerk]
    data.jerk = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [snap]
    data.snap = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [heading]
    data.heading = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [heading_rate]
    data.heading_rate = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [heading_acceleration]
    data.heading_acceleration = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [thrust]
    data.thrust = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 288;
  }

  static datatype() {
    // Returns string type for a message object
    return 'quadrotor_msgs/TrajectoryPoint';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1839a691c60e7ab9d8c3da0ab668b51b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    duration time_from_start
    
    geometry_msgs/Pose pose
    
    geometry_msgs/Twist velocity
    
    geometry_msgs/Twist acceleration
    
    geometry_msgs/Twist jerk
    
    geometry_msgs/Twist snap
    
    # Heading angle with respect to world frame [rad]
    float64 heading
    
    # First derivative of the heading angle [rad/s]
    float64 heading_rate
    
    # Second derivative of the heading angle [rad/s^2]
    float64 heading_acceleration
    
    # Collective thrust [m/s^2]
    float64 thrust
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Twist
    # This expresses velocity in free space broken into its linear and angular parts.
    Vector3  linear
    Vector3  angular
    
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
    const resolved = new TrajectoryPoint(null);
    if (msg.time_from_start !== undefined) {
      resolved.time_from_start = msg.time_from_start;
    }
    else {
      resolved.time_from_start = {secs: 0, nsecs: 0}
    }

    if (msg.pose !== undefined) {
      resolved.pose = geometry_msgs.msg.Pose.Resolve(msg.pose)
    }
    else {
      resolved.pose = new geometry_msgs.msg.Pose()
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = geometry_msgs.msg.Twist.Resolve(msg.velocity)
    }
    else {
      resolved.velocity = new geometry_msgs.msg.Twist()
    }

    if (msg.acceleration !== undefined) {
      resolved.acceleration = geometry_msgs.msg.Twist.Resolve(msg.acceleration)
    }
    else {
      resolved.acceleration = new geometry_msgs.msg.Twist()
    }

    if (msg.jerk !== undefined) {
      resolved.jerk = geometry_msgs.msg.Twist.Resolve(msg.jerk)
    }
    else {
      resolved.jerk = new geometry_msgs.msg.Twist()
    }

    if (msg.snap !== undefined) {
      resolved.snap = geometry_msgs.msg.Twist.Resolve(msg.snap)
    }
    else {
      resolved.snap = new geometry_msgs.msg.Twist()
    }

    if (msg.heading !== undefined) {
      resolved.heading = msg.heading;
    }
    else {
      resolved.heading = 0.0
    }

    if (msg.heading_rate !== undefined) {
      resolved.heading_rate = msg.heading_rate;
    }
    else {
      resolved.heading_rate = 0.0
    }

    if (msg.heading_acceleration !== undefined) {
      resolved.heading_acceleration = msg.heading_acceleration;
    }
    else {
      resolved.heading_acceleration = 0.0
    }

    if (msg.thrust !== undefined) {
      resolved.thrust = msg.thrust;
    }
    else {
      resolved.thrust = 0.0
    }

    return resolved;
    }
};

module.exports = TrajectoryPoint;
