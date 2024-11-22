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
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class Arduino {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.message_type = null;
      this.airflow_sensor_num = null;
      this.current = null;
      this.voltage = null;
      this.power = null;
      this.pow_diff = null;
      this.diff_volt = null;
      this.quaternion = null;
      this.eular_angle = null;
      this.acc = null;
      this.mag = null;
      this.gyro = null;
      this.baro = null;
      this.temp = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('message_type')) {
        this.message_type = initObj.message_type
      }
      else {
        this.message_type = 0;
      }
      if (initObj.hasOwnProperty('airflow_sensor_num')) {
        this.airflow_sensor_num = initObj.airflow_sensor_num
      }
      else {
        this.airflow_sensor_num = 0;
      }
      if (initObj.hasOwnProperty('current')) {
        this.current = initObj.current
      }
      else {
        this.current = new Array(16).fill(0);
      }
      if (initObj.hasOwnProperty('voltage')) {
        this.voltage = initObj.voltage
      }
      else {
        this.voltage = new Array(16).fill(0);
      }
      if (initObj.hasOwnProperty('power')) {
        this.power = initObj.power
      }
      else {
        this.power = new Array(16).fill(0);
      }
      if (initObj.hasOwnProperty('pow_diff')) {
        this.pow_diff = initObj.pow_diff
      }
      else {
        this.pow_diff = new Array(8).fill(0);
      }
      if (initObj.hasOwnProperty('diff_volt')) {
        this.diff_volt = initObj.diff_volt
      }
      else {
        this.diff_volt = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('quaternion')) {
        this.quaternion = initObj.quaternion
      }
      else {
        this.quaternion = new geometry_msgs.msg.Quaternion();
      }
      if (initObj.hasOwnProperty('eular_angle')) {
        this.eular_angle = initObj.eular_angle
      }
      else {
        this.eular_angle = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('acc')) {
        this.acc = initObj.acc
      }
      else {
        this.acc = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('mag')) {
        this.mag = initObj.mag
      }
      else {
        this.mag = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('gyro')) {
        this.gyro = initObj.gyro
      }
      else {
        this.gyro = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('baro')) {
        this.baro = initObj.baro
      }
      else {
        this.baro = 0;
      }
      if (initObj.hasOwnProperty('temp')) {
        this.temp = initObj.temp
      }
      else {
        this.temp = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Arduino
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [message_type]
    bufferOffset = _serializer.uint8(obj.message_type, buffer, bufferOffset);
    // Serialize message field [airflow_sensor_num]
    bufferOffset = _serializer.uint8(obj.airflow_sensor_num, buffer, bufferOffset);
    // Check that the constant length array field [current] has the right length
    if (obj.current.length !== 16) {
      throw new Error('Unable to serialize array field current - length must be 16')
    }
    // Serialize message field [current]
    bufferOffset = _arraySerializer.float32(obj.current, buffer, bufferOffset, 16);
    // Check that the constant length array field [voltage] has the right length
    if (obj.voltage.length !== 16) {
      throw new Error('Unable to serialize array field voltage - length must be 16')
    }
    // Serialize message field [voltage]
    bufferOffset = _arraySerializer.float32(obj.voltage, buffer, bufferOffset, 16);
    // Check that the constant length array field [power] has the right length
    if (obj.power.length !== 16) {
      throw new Error('Unable to serialize array field power - length must be 16')
    }
    // Serialize message field [power]
    bufferOffset = _arraySerializer.float32(obj.power, buffer, bufferOffset, 16);
    // Check that the constant length array field [pow_diff] has the right length
    if (obj.pow_diff.length !== 8) {
      throw new Error('Unable to serialize array field pow_diff - length must be 8')
    }
    // Serialize message field [pow_diff]
    bufferOffset = _arraySerializer.float32(obj.pow_diff, buffer, bufferOffset, 8);
    // Check that the constant length array field [diff_volt] has the right length
    if (obj.diff_volt.length !== 4) {
      throw new Error('Unable to serialize array field diff_volt - length must be 4')
    }
    // Serialize message field [diff_volt]
    bufferOffset = _arraySerializer.float32(obj.diff_volt, buffer, bufferOffset, 4);
    // Serialize message field [quaternion]
    bufferOffset = geometry_msgs.msg.Quaternion.serialize(obj.quaternion, buffer, bufferOffset);
    // Serialize message field [eular_angle]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.eular_angle, buffer, bufferOffset);
    // Serialize message field [acc]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.acc, buffer, bufferOffset);
    // Serialize message field [mag]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.mag, buffer, bufferOffset);
    // Serialize message field [gyro]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.gyro, buffer, bufferOffset);
    // Serialize message field [baro]
    bufferOffset = _serializer.uint32(obj.baro, buffer, bufferOffset);
    // Serialize message field [temp]
    bufferOffset = _serializer.float32(obj.temp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Arduino
    let len;
    let data = new Arduino(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [message_type]
    data.message_type = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [airflow_sensor_num]
    data.airflow_sensor_num = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [current]
    data.current = _arrayDeserializer.float32(buffer, bufferOffset, 16)
    // Deserialize message field [voltage]
    data.voltage = _arrayDeserializer.float32(buffer, bufferOffset, 16)
    // Deserialize message field [power]
    data.power = _arrayDeserializer.float32(buffer, bufferOffset, 16)
    // Deserialize message field [pow_diff]
    data.pow_diff = _arrayDeserializer.float32(buffer, bufferOffset, 8)
    // Deserialize message field [diff_volt]
    data.diff_volt = _arrayDeserializer.float32(buffer, bufferOffset, 4)
    // Deserialize message field [quaternion]
    data.quaternion = geometry_msgs.msg.Quaternion.deserialize(buffer, bufferOffset);
    // Deserialize message field [eular_angle]
    data.eular_angle = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [acc]
    data.acc = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [mag]
    data.mag = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [gyro]
    data.gyro = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [baro]
    data.baro = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [temp]
    data.temp = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 378;
  }

  static datatype() {
    // Returns string type for a message object
    return 'drone_msgs/Arduino';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '75d87b27eabead7e8d84149e18bb1bd0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    
    uint8 message_type
    # enum message_type
    uint8 AIRFLOW = 0  
    uint8 FORCE   = 1  
    uint8 IMU  = 2  
    
    ## Airflow Measurement
    uint8 airflow_sensor_num        ## the number of airflow sensor
    float32[16] current             ## airflow sensor current measurement [mA]
    float32[16] voltage             ## airflow sensor voltage measurement [mV]
    float32[16] power               ## airflow sensor power measurement [mW]
    float32[8] pow_diff             ## airflow sensor power measurement difference [mW]
    
    ## Force Measurement
    float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]
    
    ## IMU Measurement
    geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.
    geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.
    geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]
    geometry_msgs/Vector3 mag                  ## [m/s^2]
    geometry_msgs/Vector3 gyro                 ## [m/s^2]
    uint32 baro                      ## [pascal]
    float32 temp                     ## [degree]
    
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
    const resolved = new Arduino(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.message_type !== undefined) {
      resolved.message_type = msg.message_type;
    }
    else {
      resolved.message_type = 0
    }

    if (msg.airflow_sensor_num !== undefined) {
      resolved.airflow_sensor_num = msg.airflow_sensor_num;
    }
    else {
      resolved.airflow_sensor_num = 0
    }

    if (msg.current !== undefined) {
      resolved.current = msg.current;
    }
    else {
      resolved.current = new Array(16).fill(0)
    }

    if (msg.voltage !== undefined) {
      resolved.voltage = msg.voltage;
    }
    else {
      resolved.voltage = new Array(16).fill(0)
    }

    if (msg.power !== undefined) {
      resolved.power = msg.power;
    }
    else {
      resolved.power = new Array(16).fill(0)
    }

    if (msg.pow_diff !== undefined) {
      resolved.pow_diff = msg.pow_diff;
    }
    else {
      resolved.pow_diff = new Array(8).fill(0)
    }

    if (msg.diff_volt !== undefined) {
      resolved.diff_volt = msg.diff_volt;
    }
    else {
      resolved.diff_volt = new Array(4).fill(0)
    }

    if (msg.quaternion !== undefined) {
      resolved.quaternion = geometry_msgs.msg.Quaternion.Resolve(msg.quaternion)
    }
    else {
      resolved.quaternion = new geometry_msgs.msg.Quaternion()
    }

    if (msg.eular_angle !== undefined) {
      resolved.eular_angle = geometry_msgs.msg.Vector3.Resolve(msg.eular_angle)
    }
    else {
      resolved.eular_angle = new geometry_msgs.msg.Vector3()
    }

    if (msg.acc !== undefined) {
      resolved.acc = geometry_msgs.msg.Vector3.Resolve(msg.acc)
    }
    else {
      resolved.acc = new geometry_msgs.msg.Vector3()
    }

    if (msg.mag !== undefined) {
      resolved.mag = geometry_msgs.msg.Vector3.Resolve(msg.mag)
    }
    else {
      resolved.mag = new geometry_msgs.msg.Vector3()
    }

    if (msg.gyro !== undefined) {
      resolved.gyro = geometry_msgs.msg.Vector3.Resolve(msg.gyro)
    }
    else {
      resolved.gyro = new geometry_msgs.msg.Vector3()
    }

    if (msg.baro !== undefined) {
      resolved.baro = msg.baro;
    }
    else {
      resolved.baro = 0
    }

    if (msg.temp !== undefined) {
      resolved.temp = msg.temp;
    }
    else {
      resolved.temp = 0.0
    }

    return resolved;
    }
};

// Constants for message
Arduino.Constants = {
  AIRFLOW: 0,
  FORCE: 1,
  IMU: 2,
}

module.exports = Arduino;
