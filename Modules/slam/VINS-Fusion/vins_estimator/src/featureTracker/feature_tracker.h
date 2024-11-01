/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file, const bool use_stereo);
    void showUndistortion(const string &name);
    void robustTestWithRANSACFundamental();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);

    int row, col;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
    
    cv::Mat imTrack;
    vector<camodocal::CameraPtr> m_camera;
    cv::Mat mask, fisheye_mask;
    vector<cv::Point2f> predict_pts, predict_pts_debug;
    vector<int> track_cnt;

    double prev_time;
    cv::Mat prev_img;
    vector<cv::Point2f> prev_pts;
    vector<cv::Point2f> prev_un_pts;
    map<int, cv::Point2f> prev_un_pts_map;
    map<int, cv::Point2f> prev_un_right_pts_map;

    double cur_time;
    cv::Mat cur_img;
    vector<cv::Point2f> cur_pts, cur_right_pts;
    vector<cv::Point2f> cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
};

static void perform_griding(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::Point2f> &cur_pts, 
                            std::vector<cv::Point2f> &pts, int num_features, 
                            int grid_x=10, int grid_y=10, int threshold=10, bool nonmaxSuppression=true) {

    // Calculate the size our extraction boxes should be
    int size_x = img.cols / grid_x;
    int size_y = img.rows / grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1); // num of features detected in this grid

    auto it0 = cur_pts.begin();
    while (it0 != cur_pts.end()) {
        cv::Point2f pt = *it0;
        int x_grid = std::floor(pt.x / size_x);
        int y_grid = std::floor(pt.y / size_y);
        
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        it0++;
    }

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    int num_featsneeded = num_features - (int)cur_pts.size();
    if (num_featsneeded < std::min((int)(0.2*num_features), num_features))
        return;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_LINEAR);

    // Create grids we need to extract from
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, num_features_grid);
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid.cols; x++) {
        for (int y = 0; y < grid_2d_grid.rows; y++) {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) > 0) {
                valid_locs.emplace_back(x, y);
            }
        }
    }
    // Return if there is nothing to extract
    if (valid_locs.empty())
      return;

    // We want to have equally distributed features
    // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
    // NOTE: Thus if we extract 1 point per grid we have
    // NOTE:    -> 1 = num_features / (grid_x * grid_y)
    // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
    // NOTE:    -> grid_y = sqrt(num_features / ratio)
    if (num_features < grid_x * grid_y) {
      double ratio = (double)grid_x / (double)grid_y;
      grid_y = std::ceil(std::sqrt(num_features / ratio));
      grid_x = std::ceil(grid_y * ratio);
    }
    assert(grid_x > 0);
    assert(grid_y > 0);
    assert(num_features_grid > 0);

    // Make sure our sizes are not zero
    assert(size_x > 0);
    assert(size_y > 0);

    // Parallelize our 2d grid extraction!!
    std::vector<std::vector<cv::Point2f>> collection(valid_locs.size());
    parallel_for_(cv::Range(0, (int)valid_locs.size()), [&](const cv::Range &range) {
        for (int r = range.start; r < range.end; r++) {

            // Calculate what cell xy value we are in
            auto grid = valid_locs.at(r);
            int x = grid.first * size_x;
            int y = grid.second * size_y;

            // Skip if we are out of bounds
            if (x + size_x > img.cols || y + size_y > img.rows)
            continue;

            // Calculate where we should be extracting from
            cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

            // Extract FAST features for this part of the image
            std::vector<cv::KeyPoint> pts_new;
            cv::FAST(img(img_roi), pts_new, threshold, nonmaxSuppression);
            
            if(false){
                // - select by cornerMinEigenVal quality
                vector<pair<cv::KeyPoint, double>> qualities;
                for(const auto& kp:pts_new){
                    int x = static_cast<int>(kp.pt.x);
                    int y = static_cast<int>(kp.pt.y);
                    int window_size = 3;
                    int x1 = max(0,x-window_size);
                    int y1 = max(0,y-window_size);
                    int x2 = min(img.cols, x+window_size+1);
                    int y2 = min(img.cols, y+window_size+1);
                    cv::Mat window = img(cv::Rect(x1,y1,x2-x1,y2-y1));
                    if(window.rows>1 && window.cols>1){
                        cv::Mat eigenvalues;
                        cv::cornerMinEigenVal(window,eigenvalues,3);
                        double response;
                        cv::minMaxLoc(eigenvalues,&response,nullptr);
                        qualities.emplace_back(kp,response);
                    }
                }
                sort(qualities.begin(), qualities.end(),[](const pair<cv::KeyPoint,double>& a, const pair<cv::KeyPoint,double>& b)
                    {return a.second > b.second;});
                for(int i=0;i<std::min(num_features_grid,static_cast<int>(pts_new.size()));++i)
                    pts_new.at(i) = qualities[i].first;
            }else{
                // - get the top number from this
                std::sort(pts_new.begin(), pts_new.end(), 
                        [](const cv::KeyPoint& a, const cv::KeyPoint& b){return a.response > b.response;});
            }
            // Append the "best" ones to our vector
            // Note that we need to "correct" the point u,v since we extracted it in a ROI
            // So we should append the location of that ROI in the image
            for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++) {
                // Create keypoint
                cv::KeyPoint pt_cor = pts_new.at(i);
                pt_cor.pt.x += (float)x;
                pt_cor.pt.y += (float)y;

                // Check if it is in the mask region
                // NOTE: mask has max value of 0 (black) if it should be removed
                if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) < 127)
                    continue;
                collection.at(r).push_back(pt_cor.pt);
            }
        }
    });

    // Combine all the collections into our single vector
    for (size_t r = 0; r < collection.size(); r++) {
      pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
    }

    // Return if no points
    if (pts.empty())
      return;

        // Sub-pixel refinement parameters
        cv::Size win_size = cv::Size(5, 5);
        cv::Size zero_zone = cv::Size(-1, -1);
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

        // Get vector of points
        std::vector<cv::Point2f> pts_refined;
        for (size_t i = 0; i < pts.size(); i++) {
            pts_refined.push_back(pts.at(i));
        }

        // Finally get sub-pixel for all extracted features
        cv::cornerSubPix(img, pts_refined, win_size, zero_zone, term_crit);

        // Save the refined points!
        for (size_t i = 0; i < pts.size(); i++) {
            pts.at(i) = pts_refined.at(i);
        }
  }