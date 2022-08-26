#ifndef _OCCUPY_MAP_H
#define _OCCUPY_MAP_H

#include <iostream>
#include <algorithm>

#include <ros/ros.h>
#include <Eigen/Eigen>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <tf/tf.h>

#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/LaserScan.h>

#include "tools.h"
#include "message_utils.h"

#define NODE_NAME "Global_Planner [map]"

namespace Global_Planning
{

extern ros::Publisher message_pub;

class Occupy_map
{
    public:
        Occupy_map(){}

        // 全局点云指针
        sensor_msgs::PointCloud2 global_env_;
    	// 点云获取
		bool flag_pcl_ground_removal, flag_pcl_downsampling;
		double max_ground_height, size_of_voxel_grid;
		int timeSteps_fusingSamples;
		pcl::PointCloud<pcl::PointXYZ> local_point_cloud;

        // 地图是否占据容器， 从编程角度来讲，这就是地图变为单一序列化后的索引
        std::vector<int> occupancy_buffer_;  // 0 is free, 1 is occupied
        // 地图分辨率
        double resolution_, inv_resolution_;
        // 膨胀参数
        double inflate_;
        //是否2D规划
        bool is_2D, is_rgbd, is_lidar;
        double fly_height_2D;
        bool debug_mode;
        // 地图原点,地图尺寸
        Eigen::Vector3d origin_, map_size_3d_, min_range_, max_range_;
        // 占据图尺寸 = 地图尺寸 / 分辨率
        Eigen::Vector3i grid_size_;

        bool has_global_point;
           
        // 显示相关
        void show_gpcl_marker(visualization_msgs::Marker &m, int id, Eigen::Vector4d color);

        // 发布点云用于rviz显示
        ros::Publisher global_pcl_pub, inflate_pcl_pub;
        
        tf::TransformListener tfListener;

        //初始化
        void init(ros::NodeHandle& nh);
        // 地图更新函数 - 输入：全局点云
        void map_update_gpcl(const sensor_msgs::PointCloud2ConstPtr & global_point);
        // 地图更新函数 - 输入：局部点云
        void map_update_lpcl(const sensor_msgs::PointCloud2ConstPtr & local_point);
        // 地图更新函数 - 输入：二维激光雷达
        void map_update_laser(const sensor_msgs::LaserScanConstPtr & local_point);
        // 地图膨胀
        void inflate_point_cloud(void);
        // 判断当前点是否在地图内
        bool isInMap(Eigen::Vector3d pos);
        // 设置占据
        void setOccupancy(Eigen::Vector3d pos, int occ);
        // 由位置计算索引
        void posToIndex(Eigen::Vector3d pos, Eigen::Vector3i &id);
        // 由索引计算位置
        void indexToPos(Eigen::Vector3i id, Eigen::Vector3d &pos);
        // 根据位置返回占据状态
        int getOccupancy(Eigen::Vector3d pos);
        // 根据索引返回占据状态
        int getOccupancy(Eigen::Vector3i id);
        // 检查安全
        bool check_safety(Eigen::Vector3d& pos, double check_distance/*, Eigen::Vector3d& map_point*/);
        
        //旋转矩阵：机体系到惯性系
		Eigen::Matrix3f get_rotation_matrix(float phi, float theta, float psi)
		{
			Eigen::Matrix3f Rota_Mat;

			float r11 = cos(theta)*cos(psi);
			float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
			float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
			float r21 = cos(theta)*sin(psi);
			float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
			float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
			float r31 = - sin(theta);
			float r32 = sin(phi)*cos(theta);
			float r33 = cos(phi)*cos(theta); 
			Rota_Mat << r11,r12,r13,r21,r22,r23,r31,r32,r33;

			return Rota_Mat;
		}
        
        
        // 定义该类的指针
        typedef std::shared_ptr<Occupy_map> Ptr;
};

}



#endif
