/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_NX, ACC_NY, ACC_NZ, ACC_W, ACC_WX, ACC_WY,ACC_WZ;
double GYR_N, GYR_NX, GYR_NY, GYR_NZ, GYR_W, GYR_WX, GYR_WY, GYR_WZ;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int INPUT_RATE;
int CUT_RATE;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int CORNER_DETECTOR;
int EQUALIZE_METHOD;
int FLOW_BACK;
int TF_PUB;
int PRINT;


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    INPUT_RATE = fsSettings["input_rate"];
    CUT_RATE = fsSettings["cut_rate"];
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    CORNER_DETECTOR = fsSettings["corner_detector"];
    EQUALIZE_METHOD = fsSettings["equalize_method"];
    FLOW_BACK = fsSettings["flow_back"];
    TF_PUB = fsSettings["tf_pub"];
    PRINT = fsSettings["print_info"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_NX = fsSettings["acc_nx"];
        ACC_NY = fsSettings["acc_ny"];
        ACC_NZ = fsSettings["acc_nz"];
        if(ACC_NX==0 && ACC_NY==0 && ACC_NZ==0){
            ACC_NX = ACC_N;
            ACC_NY = ACC_N;
            ACC_NZ = ACC_N;
        }
        ACC_W = fsSettings["acc_w"];
        ACC_WX = fsSettings["acc_wx"];
        ACC_WY = fsSettings["acc_wy"];
        ACC_WZ = fsSettings["acc_wz"];
        if(ACC_WX==0 && ACC_WY==0 && ACC_WZ==0){
            ACC_WX = ACC_W;
            ACC_WY = ACC_W;
            ACC_WZ = ACC_W;
        }
        GYR_N = fsSettings["gyr_n"];
        GYR_NX = fsSettings["gyr_nx"];
        GYR_NY = fsSettings["gyr_ny"];
        GYR_NZ = fsSettings["gyr_nz"];
        if(GYR_NX==0 && GYR_NY==0 && GYR_NZ==0){
            GYR_NX = GYR_N;
            GYR_NY = GYR_N;
            GYR_NZ = GYR_N;
        }
        GYR_W = fsSettings["gyr_w"];
        GYR_WX = fsSettings["gyr_wx"];
        GYR_WY = fsSettings["gyr_wy"];
        GYR_WZ = fsSettings["gyr_wz"];
        if(GYR_WX==0 && GYR_WY==0 && GYR_WZ==0){
            GYR_WX = GYR_W;
            GYR_WY = GYR_W;
            GYR_WZ = GYR_W;
        }
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    NUM_OF_CAM = fsSettings["num_of_cam"];
    STEREO = fsSettings["use_stereo"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("system does not support multi-camera now!\n");
        assert(0);
    }else if(NUM_OF_CAM != 2)
        STEREO = 0;


    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];

    for(int id=0; id<NUM_OF_CAM; ++id)
    {
        int pn = config_file.find_last_of('/');
        std::string configPath = config_file.substr(0, pn);
        
        std::string camCalib;
        fsSettings["cam"+std::to_string(id)+"_calib"] >> camCalib;
        std::string camPath = configPath + "/" + camCalib;
        CAM_NAMES.push_back(camPath);

        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }
    if (ESTIMATE_EXTRINSIC == 2) // TODO: Currently Only estimate cam0 extrinsic
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC[0] = Eigen::Matrix3d::Identity();
        TIC[0] = Eigen::Vector3d::Zero();
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    // if(NUM_OF_CAM == 2)
    // {
    //     STEREO = 1;
    //     std::string cam1Calib;
    //     fsSettings["cam1_calib"] >> cam1Calib;
    //     std::string cam1Path = configPath + "/" + cam1Calib; 
    //     //printf("%s cam1 path\n", cam1Path.c_str() );
    //     CAM_NAMES.push_back(cam1Path);
        
    //     cv::Mat cv_T;
    //     fsSettings["body_T_cam1"] >> cv_T;
    //     Eigen::Matrix4d T;
    //     cv::cv2eigen(cv_T, T);
    //     RIC.push_back(T.block<3, 3>(0, 0));
    //     TIC.push_back(T.block<3, 1>(0, 3));
    // }
    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
