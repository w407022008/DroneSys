#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "cJSON.h"
#include "frame_struct.h"
#include "serial.hh"

extern frame_t *handle_process(std::string s);
extern const uint8_t color_lut_jet[][3];

using namespace std::chrono_literals;
using namespace cv;

#define ser (*pser)
static Serial *pser;
static float uvf_parms[4];
static ros::NodeHandle *localNode;
static ros::Publisher publisher_depth_raw, publisher_depth_rect;
static ros::Publisher publisher_pointcloud;

static bool InitSipeedTOF_MSA010_Publisher(ros::NodeHandle *n) {
  localNode = n;
  std::string s;
  bool via_usb;
  localNode->param<std::string>("device", s, "/dev/ttyUSB0");
  localNode->param<bool>("via_usb", via_usb, false);
  // this->declare_parameter("device", "/dev/ttyUSB0");
  // rclcpp::Parameter device_param = this->get_parameter("device");
  // s = device_param.as_string();
  std::cout << "use device: " << s << std::endl;
  std::cout << "use interface: " << (via_usb?"usb":"uart") << std::endl;
  pser = new Serial(s);

  ser << "AT\r";
  ser >> s;
  if (s.compare("OK\r\n")) {
    // not this serial port
    std::cout << "not this serial port" << std::endl;
    return 1;
  }

  ser << "AT+COEFF?\r";
  ser >> s;
/*  if (s.compare("+COEFF=1\r\nOK\r\n")) {
    // not this serial port
    std::cout << "not this serial port" << std::endl;
    return 1;
  }
  */
  s = s.substr(14, s.length() - 14);
  if (s.length() == 0) {
    ser >> s;
  }
  //std::cout<<s.c_str()<<std::endl;

  cJSON *cparms = cJSON_ParseWithLength((const char *)s.c_str(), s.length());
  uint32_t tmp;
  uvf_parms[0] =
      ((float)((cJSON_GetObjectItem(cparms, "fx")->valueint) / 262144.0f));
  uvf_parms[1] =
      ((float)((cJSON_GetObjectItem(cparms, "fy")->valueint) / 262144.0f));
  uvf_parms[2] =
      ((float)((cJSON_GetObjectItem(cparms, "u0")->valueint) / 262144.0f));
  uvf_parms[3] =
      ((float)((cJSON_GetObjectItem(cparms, "v0")->valueint) / 262144.0f));
  std::cout << "fx: " << uvf_parms[0] << std::endl;
  std::cout << "fy: " << uvf_parms[1] << std::endl;
  std::cout << "u0: " << uvf_parms[2] << std::endl;
  std::cout << "v0: " << uvf_parms[3] << std::endl;

  /* do not delete it. It is waiting */
  ser >> s;

  if(via_usb){
    ser << "AT+DISP=2\r";// 1:only lcd; 2: only usb; 4:only uart;
    ser >> s;
    if (s.compare("OK\r\n")) {
      // not this serial port
      std::cout << "lcd & usb display setting error!" << std::endl;
      return 1;
    }else{
      std::cout << "only usb display" << std::endl;
    }
  }else{
    ser << "AT+DISP=4\r";// 1:only lcd; 2: only usb; 4:only uart;
    ser >> s;
    if (s.compare("OK\r\n")) {
      // not this serial port
      std::cout << "lcd & usb display setting error!" << std::endl;
      return 1;
    }else{
      std::cout << "only uart display" << std::endl;
    }
  
    ser << "AT+BAUD=7\r";
    ros::Duration(0.1).sleep();
    pser->reset(2000000u);
  }  
  
/*
  ser << "AT+FPS=9\r";
  ser >> s;
  std::cout<<s.c_str()<<std::endl;
*/

  std::cout<<"Initialization finished."<<std::endl;
  publisher_depth_raw = n->advertise<sensor_msgs::Image>("depth_raw", 10);
  publisher_depth_rect = n->advertise<sensor_msgs::Image>("depth_rect", 10);
  publisher_pointcloud = n->advertise<sensor_msgs::PointCloud2>("cloud", 10);
  //publisher_pointcloud = n->advertise<pcl::PointCloud<pcl::PointXYZ>>("cloud", 10); 
  return 0;
}

uint8_t* FlyingPointFilter(uint8_t* data,float threshold) {
    uint8_t *buffer=NULL;
    buffer= (uint8_t*)malloc(100*100);
    memcpy(buffer,data,100*100);
    cv::Mat deep_img(100,100,CV_8UC1,data);
    cv::Mat l_img,r_img,u_img,d_img;
    deep_img(Rect(0,0,99,100)).copyTo(l_img);
    deep_img(Rect(1,0,99,100)).copyTo(r_img);
    deep_img(Rect(0,0,100,99)).copyTo(u_img);
    deep_img(Rect(0,1,100,99)).copyTo(d_img);
    cv::Mat tmph,tmpv;
    cv::absdiff(l_img,r_img,tmph);
    cv::absdiff(u_img,d_img,tmpv);
    cv::Mat tmp1(100,100,CV_8UC1,Scalar_<uint8_t>(0));
    cv::Mat tmp2(100,100,CV_8UC1,Scalar_<uint8_t>(0));
    cv::Mat tmp3(100,100,CV_8UC1,Scalar_<uint8_t>(0));
    cv::Mat tmp4(100,100,CV_8UC1,Scalar_<uint8_t>(0));
    tmph.copyTo(tmp1(Rect(0,0,99,100)));
    tmph.copyTo(tmp2(Rect(1,0,99,100)));
    tmpv.copyTo(tmp3(Rect(0,0,100,99)));
    tmpv.copyTo(tmp4(Rect(0,1,100,99)));
    cv::Mat abs_diff(100,100,CV_8UC1,Scalar_<uint8_t>(0));
    cv::max(tmp1,abs_diff,abs_diff);
    cv::max(tmp2,abs_diff,abs_diff);
    cv::max(tmp3,abs_diff,abs_diff);
    cv::max(tmp4,abs_diff,abs_diff);
    cv::Mat absf,deepf;
    deep_img.convertTo(deepf,CV_32FC1);
    abs_diff.convertTo(absf,CV_32FC1);
    absf/=deepf;
    cv::Mat mask=absf<threshold;
    cv::Mat imgout;
    deep_img.copyTo(imgout,mask);
//    std::cout<<mask<<std::endl;
    std::vector<uint8_t>data_out(imgout.begin<uint8_t>(), imgout.end<uint8_t>());
    cv::Mat tmp(100,100,CV_8UC1,data_out.data());
    memcpy(buffer,tmp.datastart,100*100);
    return buffer;
}

uint8_t* FlyingPointFilter(uint8_t* data){
    uint8_t *buffer=NULL;
    buffer= (uint8_t*)malloc(100*100);
    memcpy(buffer,data,100*100);
    cv::Mat image(100,100,CV_8UC1,data);
    cv::Mat image_blur;
    cv::GaussianBlur(image,image_blur,cv::Size(7,7),0);
    
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(image_blur,sobel_x,CV_64F,1,0,5);
    cv::Sobel(image_blur,sobel_y,CV_64F,0,1,5);
    
    cv::Mat gradient_magnitude;
    cv::magnitude(sobel_x,sobel_y,gradient_magnitude);
    
    double upper_bound = 1000;
    double lower_bound = 500;
    
    cv::Mat boundary_mask = cv::Mat::zeros(image.size(),CV_8UC1);
    boundary_mask.setTo(255,gradient_magnitude >= upper_bound);
    
    std::deque<cv::Point> to_check;
    for(int y=0;y<gradient_magnitude.rows;y++){
    	for(int x=0;x<gradient_magnitude.cols;x++){
    		if(boundary_mask.at<uchar>(y,x) == 255){
    			to_check.push_back(cv::Point(x,y));
    		}
    	}
    }
    
    std::vector<cv::Point> neighbors = { {-1,-1}, {-1,0}, {-1,-1},
    					{0,-1},            {0,1},
    					{1,-1},{1,0},{1,1}};
    while(!to_check.empty()){
    	cv::Point p = to_check.front();
    	to_check.pop_front();
    	
    	for(const auto& neighbor : neighbors){
    		cv::Point n(p.x+neighbor.x, p.y+neighbor.y);
    		if(n.x>=0 && n.x<gradient_magnitude.cols && n.y>=0 && n.y<gradient_magnitude.rows){
    			if(lower_bound <= gradient_magnitude.at<double>(n.y,n.x) && 
    			gradient_magnitude.at<double>(n.y,n.x) < upper_bound && 
    			boundary_mask.at<uchar>(n.y,n.x) == 0){
    				boundary_mask.at<uchar>(n.y,n.x) = 255;
    				to_check.push_back(n);
    			}
    		}
    	}
    }
    
    cv::Mat filtered_image = image.clone();
    cv::Mat frame;
    cv::Canny(boundary_mask,frame,30,90);
    
    for(int y=0;y<frame.rows;++y){
    	for(int x=0;x<frame.cols;++x){
    		if(frame.at<uchar>(y,x) == 255){
    			to_check.push_back(cv::Point(x,y));
    		}
    	}
    }
    while(!to_check.empty()){
    	cv::Point p = to_check.front();
    	to_check.pop_front();
    	
    	for(const auto& neighbor : neighbors){
    		cv::Point n(p.x+neighbor.x, p.y+neighbor.y);
    		if(n.x>=0 && n.x<gradient_magnitude.cols && n.y>=0 && n.y<gradient_magnitude.rows){
    			if(boundary_mask.at<uchar>(n.y,n.x) == 255){
    				filtered_image.at<uchar>(n.y,n.x) = filtered_image.at<uchar>(p.y,p.x);
    				boundary_mask.at<uchar>(n.y,n.x) = 0;
    				to_check.push_back(n);
    			}
    		}
    	}
    }
    
    memcpy(buffer,filtered_image.datastart,100*100);
    return buffer;
}

static void timer_callback() {
  std::string s;
  std::stringstream sstream;
  frame_t *f;
  do{
    ser >> s;
    if (s.empty()) {
      return;
    }
    f = handle_process(s);
  }while (!f);
  // cout << f << endl;
  uint8_t rows, cols, *depth;
  rows = f->frame_head.resolution_rows;
  cols = f->frame_head.resolution_cols;
  depth = f->payload;
  //depth = FlyingPointFilter(depth,0.03);
  cv::Mat md(rows, cols, CV_8UC1, depth);

  sstream << md.size();

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "tof";

  sensor_msgs::Image msg_depth =
      *cv_bridge::CvImage(header, "mono8", md).toImageMsg().get();
  ROS_INFO("Publishing: depth:%s", sstream.str().c_str());
  publisher_depth_raw.publish(msg_depth);
  
  depth = FlyingPointFilter(depth);
  md = cv::Mat(rows, cols, CV_8UC1, depth);
  msg_depth =
      *cv_bridge::CvImage(header, "mono8", md).toImageMsg().get();
  publisher_depth_rect.publish(msg_depth);
  
  
  float fox = uvf_parms[0];
  float foy = uvf_parms[1];
  float u0 = uvf_parms[2];
  float v0 = uvf_parms[3];
  //cv::Mat md(rows,cols,CV_8UC1,Scalar_<uint8_t>(0));
  //mat.copyTo(md);
  
  /*
  pcl::PointCloud<pcl::PointXYZ> _pointcloud;
  _pointcloud.clear();
  
  for (int v = 0; v < rows; v++)
  {
    uint8_t *row_ptr;
    row_ptr = md.ptr<uint8_t>(v);
    for (int u = 0; u < cols; u++)
    {
      double depth = md.at<uint8_t>(v,u) * 10 * 0.001;
      row_ptr++;

      
      pcl::PointXYZ newPoint;
      newPoint.x = (u - u0) * depth / fox;
      newPoint.y = (v - v0) * depth / foy;
      newPoint.z = depth;

      _pointcloud.push_back(newPoint);
    }
  }
  _pointcloud.header.stamp = header.stamp.toSec()*1e6;
  _pointcloud.header.frame_id = "tof";
  publisher_pointcloud.publish(_pointcloud);
  */
  
  
  sensor_msgs::PointCloud2 pcmsg;
  pcmsg.header = header;
  pcmsg.height = rows;
  pcmsg.width = cols;
  pcmsg.is_bigendian = false;
  pcmsg.point_step = 16;
  pcmsg.row_step = pcmsg.point_step * rows;
  pcmsg.is_dense = false;
  pcmsg.fields.resize(pcmsg.point_step / 4);
  pcmsg.fields[0].name = "x";
  pcmsg.fields[0].offset = 0;
  pcmsg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  pcmsg.fields[0].count = 1;
  pcmsg.fields[1].name = "y";
  pcmsg.fields[1].offset = 4;
  pcmsg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  pcmsg.fields[1].count = 1;
  pcmsg.fields[2].name = "z";
  pcmsg.fields[2].offset = 8;
  pcmsg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  pcmsg.fields[2].count = 1;
  if(pcmsg.point_step >= 16){
    pcmsg.fields[3].name = "rgb";
    pcmsg.fields[3].offset = 12;
    pcmsg.fields[3].datatype = sensor_msgs::PointField::UINT32;
    pcmsg.fields[3].count = 1;
  }
std::cout<<md.at<uint8_t>(50,50)/ 1000.0f * 6<<std::endl;
  pcmsg.data.resize((pcmsg.height) * (pcmsg.width) * (pcmsg.point_step), 0x00);
  uint8_t *ptr = pcmsg.data.data();
  for (int j = 0; j < pcmsg.height; j++){
    for (int i = 0; i < pcmsg.width; i++) {
      float cx = (((float)i) - u0) / fox;
      float cy = (((float)j) - v0) / foy;
      int val = md.at<uint8_t>(j,i);
      float dst = val / 1000.0f * 6;
      float x = dst * cx;
      float y = dst * cy;
      float z = dst;

      *((float *)(ptr + 0)) = x;
      *((float *)(ptr + 4)) = y;
      *((float *)(ptr + 8)) = z;
      if(pcmsg.point_step >= 16){
        const uint8_t *color = color_lut_jet[val];
        uint32_t color_r = color[0];
        uint32_t color_g = color[1];
        uint32_t color_b = color[2];
        *((uint32_t *)(ptr + 12)) =
          (color_r << 16) | (color_g << 8) | (color_b << 0);
      }
      ptr += pcmsg.point_step;
    }
  }
  publisher_pointcloud.publish(pcmsg);
  

  free(f);
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "sipeed_tof");
  ros::NodeHandle n("~");
  ros::Rate loop_rate(30);
  while(InitSipeedTOF_MSA010_Publisher(&n)){
    ros::Duration(0.1).sleep();
  }
  std::cout<<"Start driving."<<std::endl;
  while (ros::ok()) {
    timer_callback();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

const uint8_t color_lut_jet[][3] = {
    {128, 0, 0},     {132, 0, 0},     {136, 0, 0},     {140, 0, 0},
    {144, 0, 0},     {148, 0, 0},     {152, 0, 0},     {156, 0, 0},
    {160, 0, 0},     {164, 0, 0},     {168, 0, 0},     {172, 0, 0},
    {176, 0, 0},     {180, 0, 0},     {184, 0, 0},     {188, 0, 0},
    {192, 0, 0},     {196, 0, 0},     {200, 0, 0},     {204, 0, 0},
    {208, 0, 0},     {212, 0, 0},     {216, 0, 0},     {220, 0, 0},
    {224, 0, 0},     {228, 0, 0},     {232, 0, 0},     {236, 0, 0},
    {240, 0, 0},     {244, 0, 0},     {248, 0, 0},     {252, 0, 0},
    {255, 0, 0},     {255, 4, 0},     {255, 8, 0},     {255, 12, 0},
    {255, 16, 0},    {255, 20, 0},    {255, 24, 0},    {255, 28, 0},
    {255, 32, 0},    {255, 36, 0},    {255, 40, 0},    {255, 44, 0},
    {255, 48, 0},    {255, 52, 0},    {255, 56, 0},    {255, 60, 0},
    {255, 64, 0},    {255, 68, 0},    {255, 72, 0},    {255, 76, 0},
    {255, 80, 0},    {255, 84, 0},    {255, 88, 0},    {255, 92, 0},
    {255, 96, 0},    {255, 100, 0},   {255, 104, 0},   {255, 108, 0},
    {255, 112, 0},   {255, 116, 0},   {255, 120, 0},   {255, 124, 0},
    {255, 128, 0},   {255, 132, 0},   {255, 136, 0},   {255, 140, 0},
    {255, 144, 0},   {255, 148, 0},   {255, 152, 0},   {255, 156, 0},
    {255, 160, 0},   {255, 164, 0},   {255, 168, 0},   {255, 172, 0},
    {255, 176, 0},   {255, 180, 0},   {255, 184, 0},   {255, 188, 0},
    {255, 192, 0},   {255, 196, 0},   {255, 200, 0},   {255, 204, 0},
    {255, 208, 0},   {255, 212, 0},   {255, 216, 0},   {255, 220, 0},
    {255, 224, 0},   {255, 228, 0},   {255, 232, 0},   {255, 236, 0},
    {255, 240, 0},   {255, 244, 0},   {255, 248, 0},   {255, 252, 0},
    {254, 255, 1},   {250, 255, 6},   {246, 255, 10},  {242, 255, 14},
    {238, 255, 18},  {234, 255, 22},  {230, 255, 26},  {226, 255, 30},
    {222, 255, 34},  {218, 255, 38},  {214, 255, 42},  {210, 255, 46},
    {206, 255, 50},  {202, 255, 54},  {198, 255, 58},  {194, 255, 62},
    {190, 255, 66},  {186, 255, 70},  {182, 255, 74},  {178, 255, 78},
    {174, 255, 82},  {170, 255, 86},  {166, 255, 90},  {162, 255, 94},
    {158, 255, 98},  {154, 255, 102}, {150, 255, 106}, {146, 255, 110},
    {142, 255, 114}, {138, 255, 118}, {134, 255, 122}, {130, 255, 126},
    {126, 255, 130}, {122, 255, 134}, {118, 255, 138}, {114, 255, 142},
    {110, 255, 146}, {106, 255, 150}, {102, 255, 154}, {98, 255, 158},
    {94, 255, 162},  {90, 255, 166},  {86, 255, 170},  {82, 255, 174},
    {78, 255, 178},  {74, 255, 182},  {70, 255, 186},  {66, 255, 190},
    {62, 255, 194},  {58, 255, 198},  {54, 255, 202},  {50, 255, 206},
    {46, 255, 210},  {42, 255, 214},  {38, 255, 218},  {34, 255, 222},
    {30, 255, 226},  {26, 255, 230},  {22, 255, 234},  {18, 255, 238},
    {14, 255, 242},  {10, 255, 246},  {6, 255, 250},   {2, 255, 254},
    {0, 252, 255},   {0, 248, 255},   {0, 244, 255},   {0, 240, 255},
    {0, 236, 255},   {0, 232, 255},   {0, 228, 255},   {0, 224, 255},
    {0, 220, 255},   {0, 216, 255},   {0, 212, 255},   {0, 208, 255},
    {0, 204, 255},   {0, 200, 255},   {0, 196, 255},   {0, 192, 255},
    {0, 188, 255},   {0, 184, 255},   {0, 180, 255},   {0, 176, 255},
    {0, 172, 255},   {0, 168, 255},   {0, 164, 255},   {0, 160, 255},
    {0, 156, 255},   {0, 152, 255},   {0, 148, 255},   {0, 144, 255},
    {0, 140, 255},   {0, 136, 255},   {0, 132, 255},   {0, 128, 255},
    {0, 124, 255},   {0, 120, 255},   {0, 116, 255},   {0, 112, 255},
    {0, 108, 255},   {0, 104, 255},   {0, 100, 255},   {0, 96, 255},
    {0, 92, 255},    {0, 88, 255},    {0, 84, 255},    {0, 80, 255},
    {0, 76, 255},    {0, 72, 255},    {0, 68, 255},    {0, 64, 255},
    {0, 60, 255},    {0, 56, 255},    {0, 52, 255},    {0, 48, 255},
    {0, 44, 255},    {0, 40, 255},    {0, 36, 255},    {0, 32, 255},
    {0, 28, 255},    {0, 24, 255},    {0, 20, 255},    {0, 16, 255},
    {0, 12, 255},    {0, 8, 255},     {0, 4, 255},     {0, 0, 255},
    {0, 0, 252},     {0, 0, 248},     {0, 0, 244},     {0, 0, 240},
    {0, 0, 236},     {0, 0, 232},     {0, 0, 228},     {0, 0, 224},
    {0, 0, 220},     {0, 0, 216},     {0, 0, 212},     {0, 0, 208},
    {0, 0, 204},     {0, 0, 200},     {0, 0, 196},     {0, 0, 192},
    {0, 0, 188},     {0, 0, 184},     {0, 0, 180},     {0, 0, 176},
    {0, 0, 172},     {0, 0, 168},     {0, 0, 164},     {0, 0, 160},
    {0, 0, 156},     {0, 0, 152},     {0, 0, 148},     {0, 0, 144},
    {0, 0, 140},     {0, 0, 136},     {0, 0, 132},     {0, 0, 128}};
