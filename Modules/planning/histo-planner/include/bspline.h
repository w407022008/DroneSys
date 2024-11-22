#ifndef _BSPLINE_H_
#define _BSPLINE_H_

#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>

using namespace std;

namespace Histo_Planning
{
class Bspline
{
private:
  /* bspline */
  int p_, n_, m_;
  Eigen::MatrixXd control_points_;
  Eigen::VectorXd u_, u;  // knots vector
  double interval_;    // init interval

  Eigen::Vector3d x0_, v0_, a0_;

  Eigen::MatrixXd getDerivativeControlPoints();

public:
  static double limit_vel_, limit_acc_, limit_omega_;
  static bool yaw_track;

  Bspline()
  {
  }
  Bspline(Eigen::MatrixXd points, int order, double interval, bool zero = true);
  ~Bspline();
  void setKnot(Eigen::VectorXd knot);
  Eigen::VectorXd getKnot();

  Eigen::MatrixXd getControlPoint()
  {
    return control_points_;
  }

  void getTimeSpan(double& um, double& um_p);

  Eigen::Vector3d evaluateDeBoor(double t);
  Eigen::Vector3d getLocation(double t_cur, double dist_forward);

  Bspline getDerivative();

  static void cubicSamplePts_to_BsplineCtlPts(vector<Eigen::Vector3d> samples, double ts, Eigen::MatrixXd& control_pts);
  static void BsplineParameterize(const double& ts, const vector<Eigen::Vector3d>& point_set,
                                  const vector<Eigen::Vector3d>& start_end_derivative, Eigen::MatrixXd& ctrl_pts);

  /* check feasibility, reallocate time and recompute first 3 ctrl pts */
  bool checkFeasibility(bool show = false, double ts = 1.0);
  bool reallocateTime(bool show = false, double ts = 1.0);
  void recomputeInit();

  /* for evaluation */
  double getTimeSum();
  double getLength();
  double getJerk();

  void getMeanMinMax(double& mean_, double& min_, double& max_, double t_start_=-1.0, double t_end_=-1.0, int sample_num=0);
  void getVelMeanMinMax(double& mean_v, double& min_v, double& max_v, double t_start_=-1.0, double t_end_=-1.0, int sample_num=0);
  void getAccMeanMinMax(double& mean_a, double& min_a, double& max_a, double t_start_=-1.0, double t_end_=-1.0, int sample_num=0);

  // typedef std::shared_ptr<Bspline> Ptr;
};
}
#endif
