#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <bspline.h>

using std::vector;
namespace Histo_Planning
{
class PlanningVisualization
{
private:
  enum DRAW_ID
  {
    GOAL = 1,
    PATH_CTRL_PT = 100,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    PREDICTION = 500
  };

  /* data */
  ros::Publisher init_traj_pub, opt_traj_pub;

  void displaySphereList(ros::Publisher &pub, vector<Eigen::Vector3d> list, double resolution, Eigen::Vector4d color, int id);

public:
  PlanningVisualization(/* args */)
  {
  }
  ~PlanningVisualization()
  {
  }

  PlanningVisualization(ros::NodeHandle& nh);

  void drawBspline(Bspline bspline, double size, Eigen::Vector4d color, bool show_ctrl_pts = false,
                   double size2 = 0.1, Eigen::Vector4d color2 = Eigen::Vector4d(1, 1, 0, 1), int id1 = 0, int id2 = 0, bool optimized = false);

  typedef std::shared_ptr<PlanningVisualization> Ptr;
};
}
#endif
