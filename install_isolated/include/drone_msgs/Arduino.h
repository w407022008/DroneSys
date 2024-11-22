// Generated by gencpp from file drone_msgs/Arduino.msg
// DO NOT EDIT!


#ifndef DRONE_MSGS_MESSAGE_ARDUINO_H
#define DRONE_MSGS_MESSAGE_ARDUINO_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>

namespace drone_msgs
{
template <class ContainerAllocator>
struct Arduino_
{
  typedef Arduino_<ContainerAllocator> Type;

  Arduino_()
    : header()
    , message_type(0)
    , airflow_sensor_num(0)
    , current()
    , voltage()
    , power()
    , pow_diff()
    , diff_volt()
    , quaternion()
    , eular_angle()
    , acc()
    , mag()
    , gyro()
    , baro(0)
    , temp(0.0)  {
      current.assign(0.0);

      voltage.assign(0.0);

      power.assign(0.0);

      pow_diff.assign(0.0);

      diff_volt.assign(0.0);
  }
  Arduino_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , message_type(0)
    , airflow_sensor_num(0)
    , current()
    , voltage()
    , power()
    , pow_diff()
    , diff_volt()
    , quaternion(_alloc)
    , eular_angle(_alloc)
    , acc(_alloc)
    , mag(_alloc)
    , gyro(_alloc)
    , baro(0)
    , temp(0.0)  {
  (void)_alloc;
      current.assign(0.0);

      voltage.assign(0.0);

      power.assign(0.0);

      pow_diff.assign(0.0);

      diff_volt.assign(0.0);
  }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef uint8_t _message_type_type;
  _message_type_type message_type;

   typedef uint8_t _airflow_sensor_num_type;
  _airflow_sensor_num_type airflow_sensor_num;

   typedef boost::array<float, 16>  _current_type;
  _current_type current;

   typedef boost::array<float, 16>  _voltage_type;
  _voltage_type voltage;

   typedef boost::array<float, 16>  _power_type;
  _power_type power;

   typedef boost::array<float, 8>  _pow_diff_type;
  _pow_diff_type pow_diff;

   typedef boost::array<float, 4>  _diff_volt_type;
  _diff_volt_type diff_volt;

   typedef  ::geometry_msgs::Quaternion_<ContainerAllocator>  _quaternion_type;
  _quaternion_type quaternion;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _eular_angle_type;
  _eular_angle_type eular_angle;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _acc_type;
  _acc_type acc;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _mag_type;
  _mag_type mag;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _gyro_type;
  _gyro_type gyro;

   typedef uint32_t _baro_type;
  _baro_type baro;

   typedef float _temp_type;
  _temp_type temp;



// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(AIRFLOW)
  #undef AIRFLOW
#endif
#if defined(_WIN32) && defined(FORCE)
  #undef FORCE
#endif
#if defined(_WIN32) && defined(IMU)
  #undef IMU
#endif

  enum {
    AIRFLOW = 0u,
    FORCE = 1u,
    IMU = 2u,
  };


  typedef boost::shared_ptr< ::drone_msgs::Arduino_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::drone_msgs::Arduino_<ContainerAllocator> const> ConstPtr;

}; // struct Arduino_

typedef ::drone_msgs::Arduino_<std::allocator<void> > Arduino;

typedef boost::shared_ptr< ::drone_msgs::Arduino > ArduinoPtr;
typedef boost::shared_ptr< ::drone_msgs::Arduino const> ArduinoConstPtr;

// constants requiring out of line definition

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::drone_msgs::Arduino_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::drone_msgs::Arduino_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::drone_msgs::Arduino_<ContainerAllocator1> & lhs, const ::drone_msgs::Arduino_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.message_type == rhs.message_type &&
    lhs.airflow_sensor_num == rhs.airflow_sensor_num &&
    lhs.current == rhs.current &&
    lhs.voltage == rhs.voltage &&
    lhs.power == rhs.power &&
    lhs.pow_diff == rhs.pow_diff &&
    lhs.diff_volt == rhs.diff_volt &&
    lhs.quaternion == rhs.quaternion &&
    lhs.eular_angle == rhs.eular_angle &&
    lhs.acc == rhs.acc &&
    lhs.mag == rhs.mag &&
    lhs.gyro == rhs.gyro &&
    lhs.baro == rhs.baro &&
    lhs.temp == rhs.temp;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::drone_msgs::Arduino_<ContainerAllocator1> & lhs, const ::drone_msgs::Arduino_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace drone_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::drone_msgs::Arduino_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::drone_msgs::Arduino_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::drone_msgs::Arduino_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::drone_msgs::Arduino_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::drone_msgs::Arduino_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::drone_msgs::Arduino_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::drone_msgs::Arduino_<ContainerAllocator> >
{
  static const char* value()
  {
    return "75d87b27eabead7e8d84149e18bb1bd0";
  }

  static const char* value(const ::drone_msgs::Arduino_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x75d87b27eabead7eULL;
  static const uint64_t static_value2 = 0x8d84149e18bb1bd0ULL;
};

template<class ContainerAllocator>
struct DataType< ::drone_msgs::Arduino_<ContainerAllocator> >
{
  static const char* value()
  {
    return "drone_msgs/Arduino";
  }

  static const char* value(const ::drone_msgs::Arduino_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::drone_msgs::Arduino_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n"
"\n"
"uint8 message_type\n"
"# enum message_type\n"
"uint8 AIRFLOW = 0  \n"
"uint8 FORCE   = 1  \n"
"uint8 IMU  = 2  \n"
"\n"
"## Airflow Measurement\n"
"uint8 airflow_sensor_num        ## the number of airflow sensor\n"
"float32[16] current             ## airflow sensor current measurement [mA]\n"
"float32[16] voltage             ## airflow sensor voltage measurement [mV]\n"
"float32[16] power               ## airflow sensor power measurement [mW]\n"
"float32[8] pow_diff             ## airflow sensor power measurement difference [mW]\n"
"\n"
"## Force Measurement\n"
"float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]\n"
"\n"
"## IMU Measurement\n"
"geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.\n"
"geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.\n"
"geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]\n"
"geometry_msgs/Vector3 mag                  ## [m/s^2]\n"
"geometry_msgs/Vector3 gyro                 ## [m/s^2]\n"
"uint32 baro                      ## [pascal]\n"
"float32 temp                     ## [degree]\n"
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

  static const char* value(const ::drone_msgs::Arduino_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::drone_msgs::Arduino_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.message_type);
      stream.next(m.airflow_sensor_num);
      stream.next(m.current);
      stream.next(m.voltage);
      stream.next(m.power);
      stream.next(m.pow_diff);
      stream.next(m.diff_volt);
      stream.next(m.quaternion);
      stream.next(m.eular_angle);
      stream.next(m.acc);
      stream.next(m.mag);
      stream.next(m.gyro);
      stream.next(m.baro);
      stream.next(m.temp);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Arduino_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::drone_msgs::Arduino_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::drone_msgs::Arduino_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "message_type: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.message_type);
    s << indent << "airflow_sensor_num: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.airflow_sensor_num);
    s << indent << "current[]" << std::endl;
    for (size_t i = 0; i < v.current.size(); ++i)
    {
      s << indent << "  current[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.current[i]);
    }
    s << indent << "voltage[]" << std::endl;
    for (size_t i = 0; i < v.voltage.size(); ++i)
    {
      s << indent << "  voltage[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.voltage[i]);
    }
    s << indent << "power[]" << std::endl;
    for (size_t i = 0; i < v.power.size(); ++i)
    {
      s << indent << "  power[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.power[i]);
    }
    s << indent << "pow_diff[]" << std::endl;
    for (size_t i = 0; i < v.pow_diff.size(); ++i)
    {
      s << indent << "  pow_diff[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.pow_diff[i]);
    }
    s << indent << "diff_volt[]" << std::endl;
    for (size_t i = 0; i < v.diff_volt.size(); ++i)
    {
      s << indent << "  diff_volt[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.diff_volt[i]);
    }
    s << indent << "quaternion: ";
    s << std::endl;
    Printer< ::geometry_msgs::Quaternion_<ContainerAllocator> >::stream(s, indent + "  ", v.quaternion);
    s << indent << "eular_angle: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.eular_angle);
    s << indent << "acc: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.acc);
    s << indent << "mag: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.mag);
    s << indent << "gyro: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.gyro);
    s << indent << "baro: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.baro);
    s << indent << "temp: ";
    Printer<float>::stream(s, indent + "  ", v.temp);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DRONE_MSGS_MESSAGE_ARDUINO_H
