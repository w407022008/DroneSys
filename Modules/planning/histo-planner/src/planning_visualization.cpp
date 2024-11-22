#include <planning_visualization.h>

using std::cout;
using std::endl;
namespace Histo_Planning
{
PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh)
{
  init_traj_pub = nh.advertise<visualization_msgs::Marker>("/histo_planner/initial_trajectory", 10);
  opt_traj_pub = nh.advertise<visualization_msgs::Marker>("/histo_planner/optimized_trajectory", 10);
}

void PlanningVisualization::displaySphereList(ros::Publisher &pub, vector<Eigen::Vector3d> list, double resolution, Eigen::Vector4d color,
                                              int id)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = id;
  mk.pose.orientation.x = 0.0, mk.pose.orientation.y = 0.0, mk.pose.orientation.z = 0.0, mk.pose.orientation.w = 1.0;
  mk.color.r = color(0), mk.color.g = color(1), mk.color.b = color(2), mk.color.a = color(3);
  mk.scale.x = resolution, mk.scale.y = resolution, mk.scale.z = resolution;
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++)
  {
    pt.x = list[i](0), pt.y = list[i](1), pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pub.publish(mk);
}

void PlanningVisualization::drawBspline(Bspline bspline, double size, Eigen::Vector4d color,
                                        bool show_ctrl_pts, double size2, Eigen::Vector4d color2, int id1, int id2, bool optimized)
{
  vector<Eigen::Vector3d> traj_pts;
  double tm, tmp;
  bspline.getTimeSpan(tm, tmp);
  for (double t = tm; t <= tmp; t += 0.02)
  {
    Eigen::Vector3d pt = bspline.evaluateDeBoor(t);
    traj_pts.push_back(pt);
  }

  if(optimized) 
  	displaySphereList(opt_traj_pub,traj_pts, size, color, BSPLINE + id1 % 100);
  else
  	displaySphereList(init_traj_pub, traj_pts, size, color, PATH + id1 % 100);

  // draw the control point
  if (!show_ctrl_pts)
    return;

  Eigen::MatrixXd ctrl_pts = bspline.getControlPoint();

  vector<Eigen::Vector3d> ctp;
  for (int i = 0; i < int(ctrl_pts.rows()); ++i)
  {
    Eigen::Vector3d pt = ctrl_pts.row(i).transpose();
    ctp.push_back(pt);
  }
  if(optimized) 
  	displaySphereList(opt_traj_pub, ctp, size2, color2, BSPLINE_CTRL_PT + id2 % 100);
  else
  	displaySphereList(init_traj_pub, ctp, size2, color2, PATH_CTRL_PT + id2 % 100);
}

}
