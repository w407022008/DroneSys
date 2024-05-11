/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
  // pair of images and source points
  vector<Point2f> prevPts, nextPts;
  Mat prevImg = imread("RubberWhale1.png", cv::IMREAD_GRAYSCALE);
  Mat nextImg = imread("RubberWhale2.png", cv::IMREAD_GRAYSCALE);
  goodFeaturesToTrack(prevImg, prevPts, 100, 0.01, 2.0);

  // pyramids
  vector<Mat> prevPyr, nextPyr;
  Size winSize(21,21);
  int maxLevel = 3;
  buildOpticalFlowPyramid(prevImg, prevPyr, winSize, maxLevel, true);
  buildOpticalFlowPyramid(nextImg, nextPyr, winSize, maxLevel, true);

  // clone pyramids (no ROIs)
  vector<Mat> p1, p2;
  for (int i=0; i<prevPyr.size(); ++i) {
    p1.push_back(prevPyr[i]);
    p2.push_back(nextPyr[i]);
  }
  
  // compute sparse flow
  Mat status;
  calcOpticalFlowPyrLK(p1, p2, prevPts, nextPts, status, noArray(), winSize, maxLevel);
  cout << nextPts.size() << " points" << endl;

  return 0;
}
