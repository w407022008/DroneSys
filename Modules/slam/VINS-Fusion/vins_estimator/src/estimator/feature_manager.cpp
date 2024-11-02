/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(std::vector<Eigen::Matrix3d> _ric)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &feature_point : feature_list)
    {
        feature_point.used_num = feature_point.feature_per_frame.size();
        if (feature_point.used_num < 4)
            continue;

        feature_point.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", feature_point->feature_id, feature_point-> start_frame, feature_point->estimated_depth);
        if (feature_point.estimated_depth < 0)
        {
            feature_point.solve_flag = 2;
        }
        else
            feature_point.solve_flag = 1;
    }
}

void FeatureManager::clearDepth()
{
    for (auto &feature_point : feature_list)
        feature_point.estimated_depth = -1;
}

void FeatureManager::clearState()
{
    feature_list.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature_list)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    return cnt;
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature_list)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &feature_point : feature_list)
    {
        feature_point.used_num = feature_point.feature_per_frame.size();
        if (feature_point.used_num < 4)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / feature_point.estimated_depth;
#else
        dep_vec(++feature_index) = feature_point->estimated_depth;
#endif
    }
    return dep_vec;
}

// Add new feature into feature_list and check if it is a keyframe
// features_in_new_image: feature_id(global unique), camera_id(left0/right1), feature(point,u,v,velocity, cur_td, is_stereo)
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &features_in_new_image, double td)
{
    ROS_DEBUG("old recorded feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    for (auto &feature_point : features_in_new_image)
    {
        FeaturePerFrame f_per_fra(feature_point.second[0].second, td);
        assert(feature_point.second[0].first == 0);
        if(feature_point.second.size() == 2) // feature in stereo images
        {
            // add stereo pair feature information
            f_per_fra.rightObservation(feature_point.second[1].second);
            assert(feature_point.second[1].first == 1);
        }

        int feature_id = feature_point.first; // global unique id
        auto it = find_if(feature_list.begin(), feature_list.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature_list.end())
        {
            // Cannot find it in the feature list
            // Add a new feature untracked
            feature_list.push_back(FeaturePerId(feature_id, frame_count)); // frame_cout: current frame id in the sliding window
            feature_list.back().feature_per_frame.push_back(f_per_fra); // the feature is first time to track from current frame id, how many size of vector how many times tracked
            new_feature_num++;
        }
        else if (it->feature_id == feature_id)
        {
            // Find it, it is a old feature
            // Keep the old feature tracked
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( it-> feature_per_frame.size() >= 4) // the feature tracked over 4 times in the sliding window
                long_track_num++;
        }
    }

    // std::cout<<frame_count<<" "<<last_track_num<<" "<<long_track_num<<" "<<new_feature_num<<std::endl;
    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    for (auto &feature_point : feature_list)
    {
        if (feature_point.start_frame <= frame_count - 2 &&
            feature_point.start_frame + int(feature_point.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(feature_point, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &feature_point, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = feature_point.feature_per_frame[frame_count - 2 - feature_point.start_frame];
    const FeaturePerFrame &frame_j = feature_point.feature_per_frame[frame_count - 1 - feature_point.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    return sqrt(du*du + dv*dv);

    Vector3d p_i_comp = p_i;
    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
    
    return ans;
}

// initialize pose if with stereo
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], std::vector<Vector3d> tic, std::vector<Matrix3d> ric)
{

    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &feature_point : feature_list)
        {
            if (feature_point.estimated_depth > 0)
            {
                int index = frameCnt - feature_point.start_frame;
                if((int)feature_point.feature_per_frame.size() >= index + 1)
                {
                    Vector3d ptsInCam = ric[0] * (feature_point.feature_per_frame[0].point * feature_point.estimated_depth) + tic[0];
                    Vector3d ptsInWorld = Rs[feature_point.start_frame] * ptsInCam + Ps[feature_point.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    cv::Point2f point2d(feature_point.feature_per_frame[index].point.x(), feature_point.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose(); 
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

// calculate depth if it is invalid (new or fail)
void FeatureManager::triangulate(Vector3d Ps[], Matrix3d Rs[], std::vector<Vector3d> tic, std::vector<Matrix3d> ric)
{
    for (auto &feature_point : feature_list)
    {
        if (feature_point.estimated_depth > 0) // old calculated feature
            continue;

        // Step 1: re-initialize the depth with triangulation from first frame
        if(STEREO && feature_point.feature_per_frame[0].is_stereo)
        {
            int keyframe_id = feature_point.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[keyframe_id] + Rs[keyframe_id] * tic[0];
            Eigen::Matrix3d R0 = Rs[keyframe_id] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            //cout << "left pose " << leftPose << endl;

            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[keyframe_id] + Rs[keyframe_id] * tic[1];
            Eigen::Matrix3d R1 = Rs[keyframe_id] * ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = feature_point.feature_per_frame[0].point.head(2);
            point1 = feature_point.feature_per_frame[0].pointRight.head(2);
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;

            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                feature_point.estimated_depth = depth;
            else
                feature_point.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[feature_point.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",feature_point.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        else if(feature_point.feature_per_frame.size() > 1)
        {
            int keyframe_id = feature_point.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[keyframe_id] + Rs[keyframe_id] * tic[0];
            Eigen::Matrix3d R0 = Rs[keyframe_id] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            keyframe_id++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[keyframe_id] + Rs[keyframe_id] * tic[0];
            Eigen::Matrix3d R1 = Rs[keyframe_id] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = feature_point.feature_per_frame[0].point.head(2);
            point1 = feature_point.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                feature_point.estimated_depth = depth;
            else
                feature_point.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[feature_point.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",feature_point.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        
        feature_point.used_num = feature_point.feature_per_frame.size();
        if (feature_point.used_num < 4)
            continue;

        // Step 2: for those long track features
        // update depth by combining all tracked feature infomation
        int keyframe_id = feature_point.start_frame;
        int keyframe_jd = keyframe_id - 1;

        Eigen::MatrixXd svd_A(2 * feature_point.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[keyframe_id] + Rs[keyframe_id] * tic[0];
        Eigen::Matrix3d R0 = Rs[keyframe_id] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : feature_point.feature_per_frame)
        {
            keyframe_jd++;

            Eigen::Vector3d t1 = Ps[keyframe_jd] + Rs[keyframe_jd] * tic[0];
            Eigen::Matrix3d R1 = Rs[keyframe_jd] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (keyframe_id == keyframe_jd)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //feature_point->estimated_depth = -b / A;

        feature_point.estimated_depth = svd_method;

        if (feature_point.estimated_depth < 0.1)
        {
            feature_point.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


void FeatureManager::removeFailures()
{
    for (auto it = feature_list.begin(), it_next = feature_list.begin();
         it != feature_list.end(); it = it_next)
    {
        it_next++;

        if (it->solve_flag == 2) // negative depth
            feature_list.erase(it);
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature_list.begin(), it_next = feature_list.begin();
         it != feature_list.end(); it = it_next)
    {
        it_next++;
        
        itSet = outlierIndex.find(it->feature_id);
        if(itSet != outlierIndex.end())
        {
            feature_list.erase(it);
            //printf("remove outlier %d \n", it->feature_id);
        }
    }
}

void FeatureManager::removeOldestInWindow_andUpdateItsDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature_list.begin(), it_next = feature_list.begin();
         it != feature_list.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--; // slide all frame
        else // for the oldest frame
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;// inv_depth: point[2] must be 1
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature_list.erase(it); // remove only those tracking lost for long time
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::removeOldestInWindow()
{
    for (auto it = feature_list.begin(), it_next = feature_list.begin();
         it != feature_list.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--; // slide all frame
        else // for the oldest frame
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // remove the first track, since the frame slide outside window
            if (it->feature_per_frame.size() == 0)
                feature_list.erase(it); // delete if null
        }
    }
}

void FeatureManager::removeNewestInWindow(int frame_count)
{
    for (auto it = feature_list.begin(), it_next = feature_list.begin(); it != feature_list.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--; // last frame is not key frame, should be replaced, so the last frame belong to the last second one
        }
        else
        {
            if (it->endFrame() < frame_count - 1){
                continue;
            }
            int j = WINDOW_SIZE - 1 - it->start_frame;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // remove the last tracking, since it is not keyframe
            if (it->feature_per_frame.size() == 0) // i.e WINDOW_SIZE - 1 == it->start_frame
                feature_list.erase(it); // erase feature if it is fresh, since this frame is not keyframe
        }
    }
}