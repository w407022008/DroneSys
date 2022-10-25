#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <string>
#include <sensor_msgs/Range.h>

using namespace std;

class TFmini
{
public:
  TFmini(std::string _name, int _baudRate);
  ~TFmini(){};
  float getDist();
  void closePort();

  unsigned char dataBuf[7];

private:
  std::string portName_;
  int baudRate_;
  int serial_;

  bool readData(unsigned char *_buf, int _nRead);
};

TFmini::TFmini(std::string _name, int _baudRate) : portName_(_name), baudRate_(_baudRate)
{
  serial_ = open(_name.c_str(), O_RDWR);
  if(serial_ == -1)
  {
    ROS_ERROR_STREAM("Failed to open serial port!");
    exit(0);
  }

  struct termios options;
  if(tcgetattr(serial_, &options) != 0){
    ROS_ERROR_STREAM("Can't get serial port sets!");
    exit(0);
  }
  tcflush(serial_, TCIFLUSH);
  switch(_baudRate)
  {
    case 921600:
      cfsetispeed(&options, B921600);
      cfsetospeed(&options, B921600);
      break;
    case 576000:
      cfsetispeed(&options, B576000);
      cfsetospeed(&options, B576000);
      break;
    case 500000:
      cfsetispeed(&options, B500000);
      cfsetospeed(&options, B500000);
      break;
    case 460800:
      cfsetispeed(&options, B460800);
      cfsetospeed(&options, B460800);
      break;
    case 230400:
      cfsetispeed(&options, B230400);
      cfsetospeed(&options, B230400);
      break;
    case 115200:
      cfsetispeed(&options, B115200);
      cfsetospeed(&options, B115200);
      break;
    case 57600:
      cfsetispeed(&options, B57600);
      cfsetospeed(&options, B57600);
      break;
    case 38400:
      cfsetispeed(&options, B38400);
      cfsetospeed(&options, B38400);
      break;
    case 19200:
      cfsetispeed(&options, B19200);
      cfsetospeed(&options, B19200);
      break;
    case 9600:
      cfsetispeed(&options, B9600);
      cfsetospeed(&options, B9600);
      break;
    case 4800:
      cfsetispeed(&options, B4800);
      cfsetospeed(&options, B4800);
      break;
    default:
      ROS_ERROR_STREAM("Unsupported baud rate!");
      exit(0);
  }
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  options.c_oflag &= ~OPOST;
  options.c_iflag &= ~(IXON | IXOFF | IXANY);
  options.c_iflag &= ~(INLCR | ICRNL | IGNCR);
  options.c_oflag &= ~(ONLCR | OCRNL);
  if(tcsetattr(serial_, TCSANOW, &options) != 0){
    ROS_ERROR_STREAM("Can't set serial port options!");
    exit(0);
  }
}

bool TFmini::readData(unsigned char* _buf, int _nRead)
{
  int total = 0, ret = 0;

  while(total != _nRead)
  {
    ret = read(serial_, _buf + total, (_nRead - total));
    if(ret < 0)
    {
      return false;
    }
    total += ret;
  }
#ifdef DEBUG
  for(int i = 0; i < _nRead; i++)
  {
    printf("%02x ", _buf[i]);
  }
  printf("\n");
#endif // DEBUG

  return true;
}

float TFmini::getDist()
{
  while(true)
  {
    if(readData(dataBuf, 2))
    {
      if(dataBuf[0] == 0x59 && dataBuf[1] == 0x59)
        break;
    }
    else
    {
      return -1.0;
    }
  }

  if(readData(dataBuf, 7))
  {
    int sumCheck = (0x59 + 0x59) % 256;
    for(int i = 0; i < 6; i++)
    {
      sumCheck = (sumCheck + dataBuf[i]) % 256;
    }

    if(sumCheck == dataBuf[6])
    {
      return ((float)(dataBuf[1] << 8 | dataBuf[0]) / 100.0);
    }
    else
    {
      return 0.0;
    }
  }
  else
  {
    return -1.0;
  }
}

void TFmini::closePort()
{
  close(serial_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "TFmini");
  ros::NodeHandle nh("~");
  std::string id = "TFmini";
  std::string portName;
  int baud_rate;
  TFmini *tfmini_obj;

  nh.param("serial_port", portName, std::string("/dev/ttyUSB0"));
  nh.param("baud_rate", baud_rate, 115200);

  tfmini_obj = new TFmini(portName, baud_rate);
  ros::Publisher pub_range = nh.advertise<sensor_msgs::Range>("/sensors/tfmini_range", 1000, true);
  sensor_msgs::Range TFmini_range;
  TFmini_range.radiation_type = sensor_msgs::Range::INFRARED;
  TFmini_range.field_of_view = 0.04;
  TFmini_range.min_range = 0.3;
  TFmini_range.max_range = 12;
  TFmini_range.header.frame_id = id;
  float dist = 0;
  ROS_INFO_STREAM("Start processing ...");

  while(ros::master::check() && ros::ok())
  {
    ros::spinOnce();
    dist = tfmini_obj->getDist();
    if(dist > 0 && dist < TFmini_range.max_range)
    {
      TFmini_range.range = dist;
      TFmini_range.header.stamp = ros::Time::now();
      pub_range.publish(TFmini_range);

      cout << "get" << dist<<endl;
    }
    else if(dist == -1.0)
    {
      ROS_ERROR_STREAM("Failed to read data. TFmini ros node stopped!");
      break;
    }
    else if(dist == 0.0)
    {
      ROS_ERROR_STREAM("Data validation error!");
    }
  }

  tfmini_obj->closePort();
}
