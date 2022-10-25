#ifndef _BSPLINE_OPTIMIZER_H_
#define _BSPLINE_OPTIMIZER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <histogram.h>
#include "math.h"

// Gradient and elastic band optimization

// Input: a signed distance field and a sequence of points
// Output: the optimized sequence of points
// The format of points: N x 3 matrix, each row is a point
namespace Histo_Planning
{
class BsplineOptimizer
{
private:
  histo_planning_alg::Ptr hist_;
  Eigen::MatrixXd control_points_;  // nx3
  Eigen::Vector3d start_pt_, end_pt_;
  vector<tuple<int, int, Eigen::Vector3d>> ranges_;
  int yaw_tracking_mode;
  double forbidden_plus_safe_distance,safe_distance,forbidden_range;

  /* optimization parameters */
  double lamda_smooth;                       // curvature weight
  double lamda_obs;                          // distance weight
  double lamda_feas;                         // feasibility weight
  double lamda_end;                          // end point weight
  double lamda_tensile;                      // guide cost weight
  double ratio;                              // bend to tensile strength, the smaller the softer 
  double ratio_limit;
  double max_vel_, max_acc_, max_yaw_rate_;  // constrains parameters
  int variable_num_;
  int algorithm_;
  int max_iteration_num_, iter_num_;
  double min_iter_err, max_iter_time;
  std::vector<double> best_variable_;
  double min_cost_;
  int start_id_, end_id_;

  /* bspline */
  double ts;  // ts
  double ts2;  // ts
  double ts_inv;  // ts_inv
  double ts_inv2;  // ts_inv2
  double ts_inv4;  // ts_inv4
  int order_;                // bspline order
  Eigen::Vector3d start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_;

  int frontend_constrain_, backtend_constrain_;
  bool dynamic_,resetOptRange = false;
  double time_traj_start_;

  int collision_type_;

public:
  BsplineOptimizer()
  {
  }
  ~BsplineOptimizer()
  {
  }

  enum END_CONSTRAINT
  {
    NONE = 0,
    POS = 1,
    VEL = 2,
    ACC = 4,
    POS_VEL = 3,
    POS_ACC = 5,
    VEL_ACC = 6,
    ALL = 7,
    HARD_CONSTRAINT = 11,
    SOFT_CONSTRAINT = 12
  };

  /* main API */
  void setControlPoints(Eigen::MatrixXd points);
  void setBSplineInterval(double interval_);
  void setEnvironment(shared_ptr<histo_planning_alg> histogram);

  void setParam(ros::NodeHandle& nh);
  void resetOptimizationRange(int start, int end);
  
  void setBoundaryCondition(Eigen::Vector3d start_pos, Eigen::Vector3d end_pos, Eigen::Vector3d start_vel=Eigen::Vector3d(0.0,0.0,0.0), Eigen::Vector3d end_vel=Eigen::Vector3d(0.0,0.0,0.0), Eigen::Vector3d start_acc=Eigen::Vector3d(0.0,0.0,0.0), Eigen::Vector3d end_acc=Eigen::Vector3d(0.0,0.0,0.0));

  void optimize(int frontend_constrain, int backtend_constrain, bool dynamic=false, double time_start = -1.0);
  Eigen::MatrixXd getControlPoints();

private:
  /* NLopt cost */
  static double costFunction(const std::vector<double>& x, std::vector<double>& grad, void* func_data);
  /* helper function */
  void getDistanceAndGradient(Eigen::Vector3d& pos, double& dist, Eigen::Vector3d& grad);

  /* calculate each part of cost function with control points q */
  void combineCost(const std::vector<double>& x, vector<double>& grad, double& cost);

  void calcTensileCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient);
  void calcSmoothnessCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient);
  void calcAvoidanceCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient);
  void calcFeasibilityCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient);
  void calcEndpointCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient);

public:
  /* for evaluation */
  vector<double> vec_cost_;
  vector<double> vec_time_;
  ros::Time time_start_;

  void getCostCurve(vector<double>& cost, vector<double>& time)
  {
    cost = vec_cost_;
    time = vec_time_;
  }

  typedef shared_ptr<BsplineOptimizer> Ptr;
};
}
#endif
