#include <occupy_map.h>
#include <time.h>

namespace Global_Planning
{
void Occupy_map::init(ros::NodeHandle& nh)
{
    nh.param("global_planner/is_2D", is_2D, false); 
    nh.param("global_planner/is_rgbd", is_rgbd, false); 
    nh.param("global_planner/is_lidar", is_lidar, false); 
    if (is_2D) {
    	is_rgbd = false; is_lidar = false;
    }
    nh.param("global_planner/ground_removal", flag_pcl_ground_removal, false);
    nh.param("global_planner/max_ground_height", max_ground_height, 0.1);
    nh.param("global_planner/downsampling", flag_pcl_downsampling, false);
    nh.param("global_planner/timeSteps_fusingSamples", timeSteps_fusingSamples, 4);

    nh.param("global_planner/fly_height_2D", fly_height_2D, 1.0);

    nh.param("map/origin_x", origin_(0), -5.0);
    nh.param("map/origin_y", origin_(1), -5.0);
    nh.param("map/origin_z", origin_(2), 0.0);

    nh.param("map/map_size_x", map_size_3d_(0), 10.0);
    nh.param("map/map_size_y", map_size_3d_(1), 10.0);
    nh.param("map/map_size_z", map_size_3d_(2), 5.0);

    nh.param("map/resolution", resolution_,  0.2);

    nh.param("map/inflate", inflate_,  0.3);

    global_pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/drone_msg/planning/global_pcl",  10); 

    inflate_pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/drone_msg/planning/global_inflate_pcl", 1);
 
    this->inv_resolution_ = 1.0 / resolution_;
    for (int i = 0; i < 3; ++i)
    {
        grid_size_(i) = ceil(map_size_3d_(i) / resolution_);
    }
        
    occupancy_buffer_.resize(grid_size_(0) * grid_size_(1) * grid_size_(2));
    fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), 0.0);

    min_range_ = origin_;
    max_range_ = origin_ + map_size_3d_;   

    if(is_2D == true)
    {
        min_range_(2) = fly_height_2D - resolution_;
        max_range_(2) = fly_height_2D + resolution_;
    }
}

void Occupy_map::map_update_gpcl(const sensor_msgs::PointCloud2ConstPtr & global_point)
{
    has_global_point = true;
    global_env_ = *global_point;
}

void Occupy_map::map_update_lpcl(const sensor_msgs::PointCloud2ConstPtr & local_point)
{
    if(!is_rgbd && !is_lidar)
        return;
    
	tf::StampedTransform transform;
	if (is_rgbd)
		try{
			tfListener.waitForTransform("/map","/realsense_camera_link",local_point->header.stamp,ros::Duration(4.0));
			tfListener.lookupTransform("/map", "/realsense_camera_link", local_point->header.stamp, transform);
		}
			catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}

	if (is_lidar)
		try{
			tfListener.waitForTransform("/map","/3Dlidar_link",local_point->header.stamp,ros::Duration(4.0));
			tfListener.lookupTransform("/map", "/3Dlidar_link", local_point->header.stamp, transform);
		}
			catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}
	
	
	tf::Quaternion q = transform.getRotation();
	tf::Vector3 Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());

    double roll,pitch,yaw;
    tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
    Eigen::Matrix3f Rotation = get_rotation_matrix(roll, pitch, yaw);


	pcl::PointCloud<pcl::PointXYZ> latest_local_pcl_;
	pcl::fromROSMsg(*local_point, latest_local_pcl_);
	
    pcl::PointCloud<pcl::PointXYZ> _pointcloud;

    _pointcloud.clear();
    pcl::PointXYZ newPoint;
    Eigen::Vector3f _laser_point_body_body_frame,_laser_point_body_ENU_frame;
    
    for (int i = 0; i < (int)latest_local_pcl_.points.size(); i++)
    {
		if(is_rgbd && latest_local_pcl_.points[i].z == 1) continue;
        _laser_point_body_body_frame[0] = latest_local_pcl_.points[i].x;
        _laser_point_body_body_frame[1] = latest_local_pcl_.points[i].y;
        _laser_point_body_body_frame[2] = latest_local_pcl_.points[i].z;
        _laser_point_body_ENU_frame = Rotation * _laser_point_body_body_frame;
        newPoint.x = Origin.getX() + _laser_point_body_ENU_frame[0];
        newPoint.y = Origin.getY() + _laser_point_body_ENU_frame[1];
        newPoint.z = Origin.getZ() + _laser_point_body_ENU_frame[2];

        _pointcloud.push_back(newPoint);
    }
		
	pcl::PassThrough<pcl::PointXYZ> ground_removal;
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("z");
	ground_removal.setFilterLimits (origin_(2), map_size_3d_(2));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("y");
	ground_removal.setFilterLimits (origin_(1), map_size_3d_(1));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("x");
	ground_removal.setFilterLimits (origin_(0), map_size_3d_(0));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	
	if(flag_pcl_ground_removal){
		pcl::PassThrough<pcl::PointXYZ> ground_removal;
		ground_removal.setInputCloud (_pointcloud.makeShared());
		ground_removal.setFilterFieldName ("z");
		ground_removal.setFilterLimits (-1.0, max_ground_height);
		ground_removal.setFilterLimitsNegative (true);
		ground_removal.filter (_pointcloud);
	}
	
	if (flag_pcl_downsampling){
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(_pointcloud.makeShared());
		sor.setLeafSize(resolution_, resolution_, resolution_);
		sor.filter(_pointcloud);
	}
	
	local_point_cloud += _pointcloud;

	if (local_point_cloud.points.size()>5000){
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(local_point_cloud.makeShared());
		sor.setLeafSize(2*resolution_, 2*resolution_, 2*resolution_);
		sor.filter(local_point_cloud);
		cout << "point cloud resize: " << (int)local_point_cloud.points.size() << endl;
	}
	
	local_point_cloud.header.seq++;
	local_point_cloud.header.stamp = (local_point->header.stamp).toNSec()/1e3;
	local_point_cloud.header.frame_id = "/map";

	pcl::toROSMsg(local_point_cloud, global_env_);
    has_global_point = true;

}

void Occupy_map::map_update_laser(const sensor_msgs::LaserScanConstPtr & local_point)
{
	tf::StampedTransform transform;
	if (is_2D)
		try{
			tfListener.waitForTransform("/map","/lidar_link",local_point->header.stamp,ros::Duration(4.0));
			tfListener.lookupTransform("/map", "/lidar_link", local_point->header.stamp, transform);
		}
			catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}
	
	tf::Quaternion q = transform.getRotation();
	tf::Vector3 Origin = tf::Vector3(transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ());

    double roll,pitch,yaw;
    tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
    Eigen::Matrix3f R_Body_to_ENU = get_rotation_matrix(roll, pitch, yaw);

    sensor_msgs::LaserScan::ConstPtr _laser_scan;

    _laser_scan = local_point;

    pcl::PointCloud<pcl::PointXYZ> _pointcloud;

    _pointcloud.clear();
    pcl::PointXYZ newPoint;
    Eigen::Vector3f _laser_point_body_body_frame,_laser_point_body_ENU_frame;
    double newPointAngle;

    int beamNum = _laser_scan->ranges.size();
    for (int i = 0; i < beamNum; i++)
    {
    	if(_laser_scan->ranges[i] < inflate_) continue;
        newPointAngle = _laser_scan->angle_min + _laser_scan->angle_increment * i;
        _laser_point_body_body_frame[0] = _laser_scan->ranges[i] * cos(newPointAngle);
        _laser_point_body_body_frame[1] = _laser_scan->ranges[i] * sin(newPointAngle);
        _laser_point_body_body_frame[2] = 0.0;
        _laser_point_body_ENU_frame = R_Body_to_ENU * _laser_point_body_body_frame;
        newPoint.x = Origin.getX() + _laser_point_body_ENU_frame[0];
        newPoint.y = Origin.getY() + _laser_point_body_ENU_frame[1];
        newPoint.z = Origin.getZ() + _laser_point_body_ENU_frame[2];
        
        _pointcloud.push_back(newPoint);
    }	
		
	pcl::PassThrough<pcl::PointXYZ> ground_removal;
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("z");
	ground_removal.setFilterLimits (origin_(2), map_size_3d_(2));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("y");
	ground_removal.setFilterLimits (origin_(1), map_size_3d_(1));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	ground_removal.setInputCloud (_pointcloud.makeShared());
	ground_removal.setFilterFieldName ("x");
	ground_removal.setFilterLimits (origin_(0), map_size_3d_(0));
	ground_removal.setFilterLimitsNegative (false);
	ground_removal.filter (_pointcloud);
	
	if(flag_pcl_ground_removal){
		
		pcl::PassThrough<pcl::PointXYZ> ground_removal;
		ground_removal.setInputCloud (_pointcloud.makeShared());
		ground_removal.setFilterFieldName ("z");
		ground_removal.setFilterLimits (-1.0, max_ground_height);
		ground_removal.setFilterLimitsNegative (true);
		ground_removal.filter (_pointcloud);
	}
	
	if (flag_pcl_downsampling){
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(_pointcloud.makeShared());
		sor.setLeafSize(resolution_, resolution_, resolution_);
		sor.filter(_pointcloud);
	}

	
	local_point_cloud += _pointcloud;

	if (local_point_cloud.points.size()>5000){
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(local_point_cloud.makeShared());
		sor.setLeafSize(2*resolution_, 2*resolution_, 2*resolution_);
		sor.filter(local_point_cloud);
	}
	
	local_point_cloud.header.seq++;
	local_point_cloud.header.stamp = (ros::Time::now ()).toNSec()/1e3;
	local_point_cloud.header.frame_id = "/map";

	pcl::toROSMsg(local_point_cloud, global_env_);
	
    has_global_point = true;

}

void Occupy_map::inflate_point_cloud(void)
{
    if(!has_global_point)
    { 
        ROS_ERROR("No pcl input.");
        return;
    }

    global_pcl_pub.publish(global_env_);

    ros::Time time_start = ros::Time::now();

    pcl::PointCloud<pcl::PointXYZ> latest_global_cloud_;
    pcl::fromROSMsg(global_env_, latest_global_cloud_);

    if ((int)latest_global_cloud_.points.size() == 0)  return;

    pcl::PointCloud<pcl::PointXYZ> cloud_inflate_vis_;
    cloud_inflate_vis_.clear();

    const int ifn = ceil(inflate_ * inv_resolution_);

    pcl::PointXYZ pt_inf;
    Eigen::Vector3d p3d, p3d_inf;

    for (size_t i = 0; i < latest_global_cloud_.points.size(); ++i) 
    {
        p3d(0) = latest_global_cloud_.points[i].x;
        p3d(1) = latest_global_cloud_.points[i].y;
        p3d(2) = latest_global_cloud_.points[i].z;

        if(!isInMap(p3d))
        {
            continue;
        }

        for (int x = -ifn; x <= ifn; ++x)
            for (int y = -ifn; y <= ifn; ++y)
                for (int z = -ifn; z <= ifn; ++z) 
                {
                    p3d_inf(0) = pt_inf.x = p3d(0) + x * resolution_;
                    p3d_inf(1) = pt_inf.y = p3d(1) + y * resolution_;
                    p3d_inf(2) = pt_inf.z = p3d(2) + z * resolution_;

                    if(!isInMap(p3d_inf))
                    {
                        continue;
                    }

                    cloud_inflate_vis_.push_back(pt_inf);

                    this->setOccupancy(p3d_inf, 1);
                }
    }

    cloud_inflate_vis_.header.frame_id = "world";

    sensor_msgs::PointCloud2 map_inflate_vis;
    pcl::toROSMsg(cloud_inflate_vis_, map_inflate_vis);

    inflate_pcl_pub.publish(map_inflate_vis);

    static int exec_num=0;
    exec_num++;

    if(exec_num == 20)
    {
        printf("inflate global point take %f [s].\n",   (ros::Time::now()-time_start).toSec());
        exec_num=0;
    }  

}

void Occupy_map::setOccupancy(Eigen::Vector3d pos, int occ) 
{
    if (occ != 1 && occ != 0) 
        return;

    if (!isInMap(pos))
        return;

    Eigen::Vector3i id;
    posToIndex(pos, id);

    occupancy_buffer_[id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2)] = occ;
}

bool Occupy_map::isInMap(Eigen::Vector3d pos) 
{
    if (pos(0) < min_range_(0) + 1e-4 || pos(1) < min_range_(1) + 1e-4 || pos(2) < min_range_(2) + 1e-4) 
    {
        return false;
    }

    if (pos(0) > max_range_(0) - 1e-4 || pos(1) > max_range_(1) - 1e-4 || pos(2) > max_range_(2) - 1e-4) 
    {
        return false;
    }

    return true;
}

bool Occupy_map::isInMap(Eigen::Vector3i id) 
{
    if (id(0) < 0 || id(0) >= grid_size_(0) || id(1) < 0 || id(1) >= grid_size_(1) || id(2) < 0 ||
        id(2) >= grid_size_(2))
    {
        return false;
    }
    return true;
}

bool Occupy_map::check_safety(Eigen::Vector3d& pos, double check_distance)
{
    if(!isInMap(pos))
        return 0;
    
    Eigen::Vector3i id;
    posToIndex(pos, id);
    Eigen::Vector3i id_occ;
    Eigen::Vector3d pos_occ;

    int check_dist = int(check_distance/resolution_);
    // int cnt=0;
    for(int ix=-check_dist; ix<=check_dist; ix++)
        for(int iy=-check_dist; iy<=check_dist; iy++)
            for(int iz=-check_dist; iz<=check_dist; iz++){
                id_occ(0) = id(0)+ix;
                id_occ(1) = id(1)+iy;
                id_occ(2) = id(2)+iz;
                indexToPos(id_occ, pos_occ);
                if(!isInMap(pos_occ) || getOccupancy(id_occ)){
                    return 0;
                    // cnt++;             
                }
            }
        
    // if(cnt>5)
    //     return 0;
    
    return 1;

}

bool Occupy_map::check_safety(Eigen::Vector3i& id, double check_distance)
{
    if(!isInMap(id))
        return 0;
    
    Eigen::Vector3i id_occ;
    Eigen::Vector3d pos_occ;

    int check_dist = int(check_distance/resolution_);
    // int cnt=0;
    for(int ix=-check_dist; ix<=check_dist; ix++)
        for(int iy=-check_dist; iy<=check_dist; iy++)
            for(int iz=-check_dist; iz<=check_dist; iz++){
                id_occ(0) = id(0)+ix;
                id_occ(1) = id(1)+iy;
                id_occ(2) = id(2)+iz;
                indexToPos(id_occ, pos_occ);
                if(!isInMap(pos_occ) || getOccupancy(id_occ)){
                    return 0;
                    // cnt++;             
                }
            }
        
    // if(cnt>5)
    //     return 0;
    
    return 1;

}

void Occupy_map::posToIndex(Eigen::Vector3d pos, Eigen::Vector3i &id) 
{
    for (int i = 0; i < 3; ++i)
    {
        id(i) = floor((pos(i) - origin_(i)) * inv_resolution_);
    }
       
}

void Occupy_map::indexToPos(Eigen::Vector3i id, Eigen::Vector3d &pos) 
{
    for (int i = 0; i < 3; ++i)
    {
        pos(i) = (id(i) + 0.5) * resolution_ + origin_(i);
    }
}

int Occupy_map::getOccupancy(Eigen::Vector3d pos) 
{
    if (!isInMap(pos))
    {
        return -1;
    }
        
    Eigen::Vector3i id;
    posToIndex(pos, id);

    return occupancy_buffer_[id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2)];
}

int Occupy_map::getOccupancy(Eigen::Vector3i id) 
{
    if(!isInMap(id))
        return -1;
    return occupancy_buffer_[id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2)];
}
}
