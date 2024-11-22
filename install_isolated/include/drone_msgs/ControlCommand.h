// Generated by gencpp from file drone_msgs/ControlCommand.msg
// DO NOT EDIT!


#ifndef DRONE_MSGS_MESSAGE_CONTROLCOMMAND_H
#define DRONE_MSGS_MESSAGE_CONTROLCOMMAND_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <drone_msgs/PositionReference.h>
#include <drone_msgs/AttitudeReference.h>

namespace drone_msgs
{
template <class ContainerAllocator>
struct ControlCommand_
{
  typedef ControlCommand_<ContainerAllocator> Type;

  ControlCommand_()
    : header()
    , Command_ID(0)
    , source()
    , Mode(0)
    , Reference_State()
    , Attitude_sp()  {
    }
  ControlCommand_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , Command_ID(0)
    , source(_alloc)
    , Mode(0)
    , Reference_State(_alloc)
    , Attitude_sp(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef uint32_t _Command_ID_type;
  _Command_ID_type Command_ID;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _source_type;
  _source_type source;

   typedef uint8_t _Mode_type;
  _Mode_type Mode;

   typedef  ::drone_msgs::PositionReference_<ContainerAllocator>  _Reference_State_type;
  _Reference_State_type Reference_State;

   typedef  ::drone_msgs::AttitudeReference_<ContainerAllocator>  _Attitude_sp_type;
  _Attitude_sp_type Attitude_sp;



// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(Idle)
  #undef Idle
#endif
#if defined(_WIN32) && defined(Takeoff)
  #undef Takeoff
#endif
#if defined(_WIN32) && defined(Hold)
  #undef Hold
#endif
#if defined(_WIN32) && defined(Land)
  #undef Land
#endif
#if defined(_WIN32) && defined(Move)
  #undef Move
#endif
#if defined(_WIN32) && defined(Disarm)
  #undef Disarm
#endif
#if defined(_WIN32) && defined(Attitude)
  #undef Attitude
#endif
#if defined(_WIN32) && defined(AttitudeRate)
  #undef AttitudeRate
#endif
#if defined(_WIN32) && defined(Rate)
  #undef Rate
#endif

  enum {
    Idle = 0u,
    Takeoff = 1u,
    Hold = 2u,
    Land = 3u,
    Move = 4u,
    Disarm = 5u,
    Attitude = 6u,
    AttitudeRate = 7u,
    Rate = 8u,
  };


  typedef boost::shared_ptr< ::drone_msgs::ControlCommand_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::drone_msgs::ControlCommand_<ContainerAllocator> const> ConstPtr;

}; // struct ControlCommand_

typedef ::drone_msgs::ControlCommand_<std::allocator<void> > ControlCommand;

typedef boost::shared_ptr< ::drone_msgs::ControlCommand > ControlCommandPtr;
typedef boost::shared_ptr< ::drone_msgs::ControlCommand const> ControlCommandConstPtr;

// constants requiring out of line definition

   

   

   

   

   

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::drone_msgs::ControlCommand_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::drone_msgs::ControlCommand_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::drone_msgs::ControlCommand_<ContainerAllocator1> & lhs, const ::drone_msgs::ControlCommand_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.Command_ID == rhs.Command_ID &&
    lhs.source == rhs.source &&
    lhs.Mode == rhs.Mode &&
    lhs.Reference_State == rhs.Reference_State &&
    lhs.Attitude_sp == rhs.Attitude_sp;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::drone_msgs::ControlCommand_<ContainerAllocator1> & lhs, const ::drone_msgs::ControlCommand_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace drone_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::drone_msgs::ControlCommand_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::drone_msgs::ControlCommand_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::drone_msgs::ControlCommand_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::drone_msgs::ControlCommand_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::drone_msgs::ControlCommand_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::drone_msgs::ControlCommand_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::drone_msgs::ControlCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "969640b304f3a446799efdd5c334e9b7";
  }

  static const char* value(const ::drone_msgs::ControlCommand_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x969640b304f3a446ULL;
  static const uint64_t static_value2 = 0x799efdd5c334e9b7ULL;
};

template<class ContainerAllocator>
struct DataType< ::drone_msgs::ControlCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "drone_msgs/ControlCommand";
  }

  static const char* value(const ::drone_msgs::ControlCommand_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::drone_msgs::ControlCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n"
"\n"
"## ID should increased self\n"
"uint32 Command_ID\n"
"\n"
"string source\n"
"\n"
"uint8 Mode\n"
"# enum\n"
"uint8 Idle=0\n"
"uint8 Takeoff=1\n"
"uint8 Hold=2\n"
"uint8 Land=3\n"
"uint8 Move=4\n"
"uint8 Disarm=5\n"
"uint8 Attitude=6\n"
"uint8 AttitudeRate=7\n"
"uint8 Rate=8\n"
"\n"
"## Setpoint Reference\n"
"PositionReference Reference_State\n"
"AttitudeReference Attitude_sp\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: drone_msgs/PositionReference\n"
"std_msgs/Header header\n"
"\n"
"## Setpoint position reference for PX4 Control\n"
"\n"
"## Setpoint Mode\n"
"uint8 Move_mode\n"
"\n"
"uint8 XYZ_POS      = 0  ##0b00\n"
"uint8 XY_POS_Z_VEL = 1  ##0b01\n"
"uint8 XY_VEL_Z_POS = 2  ##0b10\n"
"uint8 XYZ_VEL = 3       ##0b11\n"
"uint8 XYZ_ACC = 4\n"
"uint8 XYZ_POS_VEL   = 5  \n"
"uint8 TRAJECTORY   = 6\n"
"\n"
"## Reference Frame\n"
"uint8 Move_frame\n"
"\n"
"uint8 ENU_FRAME  = 0\n"
"uint8 BODY_FRAME = 1\n"
"\n"
"\n"
"\n"
"## Tracking life\n"
"float32 time_from_start          ## [s]\n"
"\n"
"float32[3] position_ref          ## [m]\n"
"float32[3] velocity_ref          ## [m/s]\n"
"float32[3] acceleration_ref      ## [m/s^2]\n"
"\n"
"bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率\n"
"float32 yaw_ref                  ## [rad]\n"
"float32 yaw_rate_ref             ## [rad/s] \n"
"\n"
"Bspline bspline\n"
"================================================================================\n"
"MSG: drone_msgs/Bspline\n"
"int32 order                 ## \n"
"int64 traj_id               ## id of trajecotry\n"
"float64[] knots             ## knots list\n"
"geometry_msgs/Point[] pts   ## control points list\n"
"time start_time             ## time stamp\n"
"\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: drone_msgs/AttitudeReference\n"
"std_msgs/Header header\n"
"\n"
"## Setpoint Attitude + T\n"
"float32[3] thrust_sp                   ## Single Rotor Thrust setpoint\n"
"float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint\n"
"float32[3] desired_attitude            ## [rad] Eurler angle setpoint\n"
"geometry_msgs/Quaternion desired_att_q ## quat setpoint\n"
"geometry_msgs/Vector3 body_rate  ## [rad/s]\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::drone_msgs::ControlCommand_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::drone_msgs::ControlCommand_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.Command_ID);
      stream.next(m.source);
      stream.next(m.Mode);
      stream.next(m.Reference_State);
      stream.next(m.Attitude_sp);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ControlCommand_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::drone_msgs::ControlCommand_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::drone_msgs::ControlCommand_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "Command_ID: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.Command_ID);
    s << indent << "source: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.source);
    s << indent << "Mode: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.Mode);
    s << indent << "Reference_State: ";
    s << std::endl;
    Printer< ::drone_msgs::PositionReference_<ContainerAllocator> >::stream(s, indent + "  ", v.Reference_State);
    s << indent << "Attitude_sp: ";
    s << std::endl;
    Printer< ::drone_msgs::AttitudeReference_<ContainerAllocator> >::stream(s, indent + "  ", v.Attitude_sp);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DRONE_MSGS_MESSAGE_CONTROLCOMMAND_H
