# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from quadrotor_msgs/Trajectory.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import genpy
import geometry_msgs.msg
import quadrotor_msgs.msg
import std_msgs.msg

class Trajectory(genpy.Message):
  _md5sum = "18a34f2514fbc4cc1b109ed1c473a1d8"
  _type = "quadrotor_msgs/Trajectory"
  _has_header = True  # flag to mark the presence of a Header object
  _full_text = """# Trajectory type enums

# Undefined trajectory type
uint8 UNDEFINED=0

# General trajectory type that considers orientation from the pose and
# neglects heading values
uint8 GENERAL=1

# Trajectory types that compute orientation from acceleration and heading
# values and consider derivatives up to what is indicated by the name
uint8 ACCELERATION=2
uint8 JERK=3
uint8 SNAP=4

Header header

# Trajectory type as defined above
uint8 type

quadrotor_msgs/TrajectoryPoint[] points

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
MSG: quadrotor_msgs/TrajectoryPoint
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
float64 z"""
  # Pseudo-constants
  UNDEFINED = 0
  GENERAL = 1
  ACCELERATION = 2
  JERK = 3
  SNAP = 4

  __slots__ = ['header','type','points']
  _slot_types = ['std_msgs/Header','uint8','quadrotor_msgs/TrajectoryPoint[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,type,points

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Trajectory, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.type is None:
        self.type = 0
      if self.points is None:
        self.points = []
    else:
      self.header = std_msgs.msg.Header()
      self.type = 0
      self.points = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.type
      buff.write(_get_struct_B().pack(_x))
      length = len(self.points)
      buff.write(_struct_I.pack(length))
      for val1 in self.points:
        _v1 = val1.time_from_start
        _x = _v1
        buff.write(_get_struct_2i().pack(_x.secs, _x.nsecs))
        _v2 = val1.pose
        _v3 = _v2.position
        _x = _v3
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v4 = _v2.orientation
        _x = _v4
        buff.write(_get_struct_4d().pack(_x.x, _x.y, _x.z, _x.w))
        _v5 = val1.velocity
        _v6 = _v5.linear
        _x = _v6
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v7 = _v5.angular
        _x = _v7
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v8 = val1.acceleration
        _v9 = _v8.linear
        _x = _v9
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v10 = _v8.angular
        _x = _v10
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v11 = val1.jerk
        _v12 = _v11.linear
        _x = _v12
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v13 = _v11.angular
        _x = _v13
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v14 = val1.snap
        _v15 = _v14.linear
        _x = _v15
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v16 = _v14.angular
        _x = _v16
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _x = val1
        buff.write(_get_struct_4d().pack(_x.heading, _x.heading_rate, _x.heading_acceleration, _x.thrust))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.points is None:
        self.points = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 1
      (self.type,) = _get_struct_B().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.points = []
      for i in range(0, length):
        val1 = quadrotor_msgs.msg.TrajectoryPoint()
        _v17 = val1.time_from_start
        _x = _v17
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2i().unpack(str[start:end])
        _v18 = val1.pose
        _v19 = _v18.position
        _x = _v19
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v20 = _v18.orientation
        _x = _v20
        start = end
        end += 32
        (_x.x, _x.y, _x.z, _x.w,) = _get_struct_4d().unpack(str[start:end])
        _v21 = val1.velocity
        _v22 = _v21.linear
        _x = _v22
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v23 = _v21.angular
        _x = _v23
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v24 = val1.acceleration
        _v25 = _v24.linear
        _x = _v25
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v26 = _v24.angular
        _x = _v26
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v27 = val1.jerk
        _v28 = _v27.linear
        _x = _v28
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v29 = _v27.angular
        _x = _v29
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v30 = val1.snap
        _v31 = _v30.linear
        _x = _v31
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v32 = _v30.angular
        _x = _v32
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _x = val1
        start = end
        end += 32
        (_x.heading, _x.heading_rate, _x.heading_acceleration, _x.thrust,) = _get_struct_4d().unpack(str[start:end])
        self.points.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.type
      buff.write(_get_struct_B().pack(_x))
      length = len(self.points)
      buff.write(_struct_I.pack(length))
      for val1 in self.points:
        _v33 = val1.time_from_start
        _x = _v33
        buff.write(_get_struct_2i().pack(_x.secs, _x.nsecs))
        _v34 = val1.pose
        _v35 = _v34.position
        _x = _v35
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v36 = _v34.orientation
        _x = _v36
        buff.write(_get_struct_4d().pack(_x.x, _x.y, _x.z, _x.w))
        _v37 = val1.velocity
        _v38 = _v37.linear
        _x = _v38
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v39 = _v37.angular
        _x = _v39
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v40 = val1.acceleration
        _v41 = _v40.linear
        _x = _v41
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v42 = _v40.angular
        _x = _v42
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v43 = val1.jerk
        _v44 = _v43.linear
        _x = _v44
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v45 = _v43.angular
        _x = _v45
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v46 = val1.snap
        _v47 = _v46.linear
        _x = _v47
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v48 = _v46.angular
        _x = _v48
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _x = val1
        buff.write(_get_struct_4d().pack(_x.heading, _x.heading_rate, _x.heading_acceleration, _x.thrust))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.points is None:
        self.points = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 1
      (self.type,) = _get_struct_B().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.points = []
      for i in range(0, length):
        val1 = quadrotor_msgs.msg.TrajectoryPoint()
        _v49 = val1.time_from_start
        _x = _v49
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2i().unpack(str[start:end])
        _v50 = val1.pose
        _v51 = _v50.position
        _x = _v51
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v52 = _v50.orientation
        _x = _v52
        start = end
        end += 32
        (_x.x, _x.y, _x.z, _x.w,) = _get_struct_4d().unpack(str[start:end])
        _v53 = val1.velocity
        _v54 = _v53.linear
        _x = _v54
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v55 = _v53.angular
        _x = _v55
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v56 = val1.acceleration
        _v57 = _v56.linear
        _x = _v57
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v58 = _v56.angular
        _x = _v58
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v59 = val1.jerk
        _v60 = _v59.linear
        _x = _v60
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v61 = _v59.angular
        _x = _v61
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v62 = val1.snap
        _v63 = _v62.linear
        _x = _v63
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v64 = _v62.angular
        _x = _v64
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _x = val1
        start = end
        end += 32
        (_x.heading, _x.heading_rate, _x.heading_acceleration, _x.thrust,) = _get_struct_4d().unpack(str[start:end])
        self.points.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2i = None
def _get_struct_2i():
    global _struct_2i
    if _struct_2i is None:
        _struct_2i = struct.Struct("<2i")
    return _struct_2i
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_3d = None
def _get_struct_3d():
    global _struct_3d
    if _struct_3d is None:
        _struct_3d = struct.Struct("<3d")
    return _struct_3d
_struct_4d = None
def _get_struct_4d():
    global _struct_4d
    if _struct_4d is None:
        _struct_4d = struct.Struct("<4d")
    return _struct_4d
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
