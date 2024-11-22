#include "bspline_optimizer.h"
#include <nlopt.hpp>
#define debug false
using namespace std;

namespace Histo_Planning
{
void BsplineOptimizer::setParam(ros::NodeHandle& nh)
{
  nh.param("optimization/lamda_smooth", lamda_smooth, -1.0);
  nh.param("optimization/lamda_obs", lamda_obs, -1.0);
  nh.param("optimization/lamda_feas", lamda_feas, -1.0);
  nh.param("optimization/lamda_end", lamda_end, -1.0);
  nh.param("optimization/lamda_tensile", lamda_tensile, -1.0);
  nh.param("optimization/ratio", ratio, 1.0);
  nh.param("optimization/ratio_limit", ratio_limit, 2.0);
  nh.param("histo_planner/yaw_rate", max_yaw_rate_, -1.0);
  nh.param("optimization/max_vel", max_vel_, -1.0);
  nh.param("optimization/max_acc", max_acc_, -1.0);
  nh.param("optimization/max_iteration_num", max_iteration_num_, -1);
  nh.param("optimization/min_iter_err", min_iter_err, 1e-3);
  nh.param("optimization/max_iter_time", max_iter_time, 2e-3);
  nh.param("optimization/algorithm", algorithm_, -1);
  nh.param("optimization/order", order_, -1);
  nh.param("histo_planner/yaw_tracking_mode", yaw_tracking_mode, 0); 
  nh.param("histo_planner/time_traj_interval", ts, 0.1);   // time interval
  nh.param("histo_planner/forbidden_range", forbidden_range, 0.20);
  nh.param("histo_planner/max_tracking_error", safe_distance, 0.2);
  forbidden_plus_safe_distance = safe_distance + forbidden_range;
  
  std::cout << "[Optimizer]: bspline interval:\t" << ts << std::endl;
  std::cout << "[Optimizer]: weight for smoothness:\t" << lamda_smooth << std::endl;
  std::cout << "[Optimizer]: weight for distance:\t" << lamda_obs << std::endl;
  std::cout << "[Optimizer]: weight for feasibility:\t" << lamda_feas << std::endl;
  //std::cout << "[Optimizer]: weight for end point:\t" << lamda_end << std::endl;
  std::cout << "[Optimizer]: weight for tensile:\t" << lamda_tensile << std::endl;
}

void BsplineOptimizer::setEnvironment(shared_ptr<histo_planning_alg> histogram)
{
  this->hist_ = histogram;
}

/* control points [Nx3] includes order_ start_point and end_point */
void BsplineOptimizer::setControlPoints(Eigen::MatrixXd points)
{
  this->control_points_ = points;
  this->start_id_ = order_; // the first one control point to be optimized
  this->end_id_ = this->control_points_.rows() - order_; // the last one
  if(debug) cout << "[Optimization]: get " << control_points_.rows() << " control points" << endl;
}

void BsplineOptimizer::resetOptimizationRange(int start, int end)
{
  this->start_id_ = min(max(start, order_), int(control_points_.rows() - order_));
  this->end_id_ = min(max(end, order_), int(control_points_.rows() - order_));
  if(debug) cout << "[Optimization]: opt range:" << this->start_id_ << ", " << this->end_id_ << endl;
  resetOptRange = true;
}

void BsplineOptimizer::setBSplineInterval(double interval_)
{
  this->ts = interval_;
  this->ts2 = ts2*ts2;
  this->ts_inv = 1/ts;
  this->ts_inv2 = ts_inv*ts_inv;
  this->ts_inv4 = ts_inv2*ts_inv2;
}

void BsplineOptimizer::setBoundaryCondition(Eigen::Vector3d start_pos, Eigen::Vector3d end_pos, Eigen::Vector3d start_vel, Eigen::Vector3d end_vel, Eigen::Vector3d start_acc, Eigen::Vector3d end_acc)
{
  start_pos_ = start_pos;
  start_vel_ = start_vel;
  start_acc_ = start_acc;
  end_pos_ = end_pos;
  end_vel_ = end_vel;
  end_acc_ = end_acc;
}

Eigen::MatrixXd BsplineOptimizer::getControlPoints()
{
  return this->control_points_;
}

/* best algorithm_ is 40: SLSQP(constrained), 11 LBFGS(unconstrained barrier method */
void BsplineOptimizer::optimize(int frontend_constrain, int backtend_constrain, bool dynamic, double time_start)
{
  frontend_constrain_ = frontend_constrain;
  backtend_constrain_ = backtend_constrain;
  /* ---------- initialize solver ---------- */
  //  end_constrain_ = end_cons;
  time_traj_start_ = time_start;
  iter_num_ = 0;

  variable_num_ = order_ * (end_id_ - start_id_);
  if (resetOptRange){
    frontend_constrain = END_CONSTRAINT::ALL;
    backtend_constrain = END_CONSTRAINT::ALL;
  }else{
    switch(frontend_constrain){
      case END_CONSTRAINT::ALL :
        {
          if(debug) std::cout << "set start pos + vel + acc constrain!" << std::endl;
          start_id_ = start_id_;
          break;
        }
			
      case END_CONSTRAINT::POS_VEL :
        {
          if(debug) std::cout << "set start pos + vel constrain!" << std::endl;
          variable_num_ += order_;
          start_id_ = start_id_-1;
          break;
        }
			
      case END_CONSTRAINT::POS :
        if(debug) std::cout << "set start pos constrain!" << std::endl;
        variable_num_ += 2*order_;
        start_id_ = start_id_-2;
      break;
			
      default :
        std::cout << "unknown start constrain!" << std::endl;
    }
	
    switch(backtend_constrain_){
      case END_CONSTRAINT::ALL :
        {
          if(debug) std::cout << "set end pos + vel + acc constrain!" << std::endl;
          end_id_ = end_id_;
          break;
        }
        
      case END_CONSTRAINT::POS_VEL :
        {
          if(debug) std::cout << "set end pos + vel constrain!" << std::endl;
          variable_num_ += order_;
          end_id_ = end_id_+1;
          break;
        }
        
      case END_CONSTRAINT::POS :
        if(debug) std::cout << "set end pos constrain!" << std::endl;
        variable_num_ += 2*order_;
        end_id_ = end_id_+2;
        break;
        
      default :
        std::cout << "unknown end constrain!" << std::endl;
    }
    
  }

  min_cost_ = std::numeric_limits<double>::max();

  nlopt::opt opt(nlopt::algorithm(algorithm_), variable_num_);

  opt.set_min_objective(BsplineOptimizer::costFunction, this);
  opt.set_maxeval(max_iteration_num_);
  opt.set_xtol_rel(min_iter_err);
  opt.set_maxtime(max_iter_time);

  /* ---------- init variables ---------- */
  vector<double> x(variable_num_);
  double final_cost;
  for (int i = 0; i < int(control_points_.rows()); ++i)
  {
    if (i < start_id_ || i >= end_id_)
      continue;
    
    for (int j = 0; j < order_; j++)
      x[order_ * (i - start_id_) + j] = control_points_(i, j); 
  }
  //cout << "control_points_ num:" << control_points_.rows() << endl;
//  start_pt_ = (1 / 6.0) * (control_points_.row(0) + 4 * control_points_.row(1) + control_points_.row(2));// start point of trajectory
//  end_pt_ = (1 / 6.0) * (control_points_.row(control_points_.rows() - order_) + 4 * control_points_.row(control_points_.rows() - order_+1) +
//           control_points_.row(control_points_.rows() - order_+2));// end point of trajectory which is uniquely defined by last _order control points

  try
  {
    /* ---------- optimization ---------- */
    cout << fixed << setprecision(7);
    vec_time_.clear();
    vec_cost_.clear();
    // time_start_ = ros::Time::now();

    nlopt::result result = opt.optimize(x, final_cost);

    /* ---------- get results ---------- */
    std::cout << "[Optimization]: iter num: " << iter_num_ << std::endl;
    // cout << "Min cost:" << min_cost_ << endl;
    
    /* head control points */
    switch(frontend_constrain_){
      case END_CONSTRAINT::ALL :
        {
          Eigen::Vector3d q_0 = start_pos_ - start_vel_*ts + start_acc_*ts2/3;
          Eigen::Vector3d q_1 = start_pos_ - start_acc_*ts2/6;
          Eigen::Vector3d q_2 = start_pos_ + start_vel_*ts + start_acc_*ts2/3;
          control_points_.row(0) = q_0.transpose();
          control_points_.row(1) = q_1.transpose();
          control_points_.row(2) = q_2.transpose();
          break;
        }
      case END_CONSTRAINT::POS_VEL :
        {
          Eigen::Vector3d q_2(best_variable_[0], best_variable_[1], best_variable_[2]);
          Eigen::Vector3d q_0 = q_2 - 2 * ts * start_vel_;
          control_points_.row(0) = q_0.transpose();
          Eigen::Vector3d q_1 = (6*start_pos_ - q_0 - q_2)/4;
          control_points_.row(1) = q_1.transpose();
          break;
        }
      case END_CONSTRAINT::POS :
        Eigen::Vector3d q_1(best_variable_[0], best_variable_[1], best_variable_[2]);
        Eigen::Vector3d q_2(best_variable_[3], best_variable_[4], best_variable_[5]);
        Eigen::Vector3d q_0 = 6*start_pos_ - 4*q_1 - q_2;
        control_points_.row(0) = q_0.transpose();
    
    }
    
    //cout << "best_variable_ num:" << best_variable_.size()/3 << endl;
    /* opt control points */
    for (int i = start_id_; i < end_id_; ++i)
    {
      for (int j = 0; j < order_; j++)
        control_points_(i, j) = best_variable_[order_ * (i - start_id_) + j];
    }
    //cout << "opt pt num:" << end_id_ - start_id_ << endl;

    /* tail control points */
    switch(backtend_constrain_){
      case END_CONSTRAINT::ALL :
        {
          Eigen::Vector3d q_2 = end_pos_ - end_vel_*ts + end_acc_*ts2/3;
          Eigen::Vector3d q_1 = end_pos_ - end_acc_*ts2/6;
          Eigen::Vector3d q_0 = end_pos_ + end_vel_*ts + end_acc_*ts2/3;
          control_points_.row(end_id_) = q_2.transpose();
          control_points_.row(end_id_+1) = q_1.transpose();
          control_points_.row(end_id_+2) = q_0.transpose();
          break;
        }
      case END_CONSTRAINT::POS_VEL :
        {
          Eigen::Vector3d q_2(x[variable_num_-3], x[variable_num_-2], x[variable_num_-1]);
          Eigen::Vector3d q_0 = q_2 + 2 * ts * end_vel_;
          Eigen::Vector3d q_1 = (6*end_pos_ - q_0 - q_2)/4;
          control_points_.row(end_id_) = q_1.transpose();
          control_points_.row(end_id_+1) = q_0.transpose();
          break;
        }
      case END_CONSTRAINT::POS :
        Eigen::Vector3d q_2(x[variable_num_-6], x[variable_num_-5], x[variable_num_-4]);
        Eigen::Vector3d q_1(x[variable_num_-3], x[variable_num_-2], x[variable_num_-1]);
        Eigen::Vector3d q_0 = 6*end_pos_ - 4*q_1 - q_2;
        control_points_.row(end_id_) = q_0.transpose();
    }
  }
  catch (std::exception& e)
  {
    cout << "[Optimization]: nlopt exception: " << e.what() << endl;
  }
}

void BsplineOptimizer::calcTensileCost(const vector<Eigen::Vector3d>& q, double& cost,
                                          vector<Eigen::Vector3d>& gradient)
{
  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));
  
  Eigen::Vector3d length;
  Eigen::Vector3d vel;
  Eigen::Vector3d vel_unit;
  Eigen::Vector3d accel;
  Eigen::Vector3d accel_nor;
  for (uint i = 0; i < (q.size()-2); i++) // all control points
  {
    /* evaluate tensile */
    length = (q[i] - 2*q[i + 1] + q[i+2]);
    cost += length.squaredNorm();
    gradient[i + 1] += length * (-4.0);
//    length = (q[i] - q[i + 1]);
//    cost += length.squaredNorm();
//    gradient[i    ] += length * (2.0);
//    gradient[i + 1] += length * (-2.0);
//    length = (q[i+1] - q[i + 2]);
//    cost += length.squaredNorm();
//    gradient[i + 1] += length * (2.0);
//    gradient[i + 2] += length * (-2.0);

    /* evaluate vel */
    vel = (q[i] - q[i + 2]);
    vel_unit = vel.normalized();
    
    /* evaluate accel */
//    accel = ratio*(q[i + 2] - 2 * q[i + 1] + q[i]);
//    cost += 2*accel.squaredNorm();
//    gradient[i    ] += accel * (4.0);
//    gradient[i + 1] += accel * (-8.0);
//    gradient[i + 2] += accel * (4.0);
    
    /* evaluate accel_nor */
    accel_nor = q[i + 2] - 2 * q[i + 1] + q[i];
    accel_nor = ratio*vel_unit.cross(accel_nor.cross(vel_unit));
    float delta = accel_nor.squaredNorm();
    cost += delta;
    gradient[i    ] += accel_nor * (2.0);
    gradient[i + 1] += accel_nor * (-4.0);
    gradient[i + 2] += accel_nor * (2.0);
    
  }  
}

void BsplineOptimizer::calcSmoothnessCost(const vector<Eigen::Vector3d>& q, double& cost,
                                          vector<Eigen::Vector3d>& gradient)
{
  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));
  
  Eigen::Vector3d jerk;
  for (uint i = 0; i < (q.size()-3); i++) // all control points
  {
    /* evaluate jerk */
    jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i];
    cost += jerk.squaredNorm();
    gradient[i    ] += jerk * (-2.0);
    gradient[i + 1] += jerk * (6.0);
    gradient[i + 2] += jerk * (-6.0);
    gradient[i + 3] += jerk * (2.0);
  }
  
}

void BsplineOptimizer::calcAvoidanceCost(const vector<Eigen::Vector3d>& q, double& cost,
                                        vector<Eigen::Vector3d>& gradient)
{
  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

  double dist;
  Eigen::Vector3d dist_grad, g_zero(0, 0, 0);

  for (uint i = 1; i < q.size() - 1; i++) // all control points
  {
    Eigen::Vector3d pos_ = (q[i-1] + 4*q[i] + q[i+1])/6;

    dist = hist_->getDistWithGrad(pos_, dist_grad);

    if (dist<0.0) {
      continue; // is not in map.
    }

    // Option 0: Cosine gradient
    double dist0_ = forbidden_range;
    double dist1_ = forbidden_range+2*safe_distance;
    
    if (dist < dist0_){
      cost += 5*(dist1_+dist0_) - 10*dist;
      gradient[i-1] -= 16.6666 * dist_grad;
      gradient[i] -= 66.6666 * dist_grad;
      gradient[i+1] -= 16.6666 * dist_grad;
    }else if(dist < dist1_){
      double var = (dist-dist0_);
      double gain = M_PI/(dist1_-dist0_);
      cost += 5*(dist1_+dist0_) - 5*((dist+dist0_) + sin(var*gain)/gain);
      gradient[i-1] -= 5*(cos(var*gain)+1) * 0.16666 * dist_grad;
      gradient[i] -= 5*(cos(var*gain)+1) * 0.66666 * dist_grad;
      gradient[i+1] -= 5*(cos(var*gain)+1) * 0.16666 * dist_grad;
    }
    // Option 1: Polynomial gradient
//    double dist0_ = forbidden_plus_safe_distance+safe_distance;
//    double dist1_ = forbidden_plus_safe_distance+3*safe_distance;
    
//    if (dist < dist1_){
//      dist_grad = dist_grad.normalized();
//      if (dist < dist0_){
//        double ratio_ = 1/max(1e-6,dist-forbidden_range) - 1/safe_distance;
//        ratio_ = ratio_>ratio_limit ? ratio_limit : ratio_;
//        cost += pow(dist - dist1_, 2) * ratio_;
//        gradient[i-1] += ratio_ * (dist - dist1_) * 0.3333 * dist_grad;
//        gradient[i] += ratio_ * (dist - dist1_) * 1.3333 * dist_grad;
//        gradient[i+1] += ratio_ * (dist - dist1_) * 0.3333 * dist_grad;
			
//      } else {
//        cost += pow(dist - dist1_, 2);
//        gradient[i-1] += (dist - dist1_) * 0.3333 * dist_grad;
//        gradient[i] += (dist - dist1_) * 1.3333 * dist_grad;
//        gradient[i+1] += (dist - dist1_) * 0.3333 * dist_grad;
//      }
//    }
    // Option 2: Inverted gradient
//    if (dist < dist1_){
//        if (dist < dist0_){
//            double ratio_ = pow(dist0_ / dist,2);
//            ratio_ = ratio_>ratio_limit ? ratio_limit : ratio_;
//            cost += 1/(max(forbidden_plus_safe_distance+1e-3, dist) - forbidden_plus_safe_distance) - 1/(dist1_-forbidden_plus_safe_distance) * ratio_;
//            gradient[i-1] += -1/pow(max(1e-3,dist - forbidden_plus_safe_distance),2) * 0.16667 * dist_grad;
//            gradient[i] += -1/pow(max(1e-3,dist - forbidden_plus_safe_distance),2) * 0.66667 * dist_grad;
//            gradient[i+1] += -1/pow(max(1e-3,dist - forbidden_plus_safe_distance),2) * 0.16667 * dist_grad;
//			
//          } else {
//            cost += 1/(dist - forbidden_plus_safe_distance) - 1/(dist1_-forbidden_plus_safe_distance);
//            gradient[i-1] += -1/pow(dist - forbidden_plus_safe_distance,2) * 0.16667 * dist_grad;
//            gradient[i] += -1/pow(dist - forbidden_plus_safe_distance,2) * 0.66667 * dist_grad;
//            gradient[i+1] += -1/pow(dist - forbidden_plus_safe_distance,2) * 0.16667 * dist_grad;
//           }
//    }
  }
}

void BsplineOptimizer::calcFeasibilityCost(const vector<Eigen::Vector3d>& q, double& cost,
                                           vector<Eigen::Vector3d>& gradient)
{
  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

  /* ---------- abbreviation ---------- */
  double vm2, am2, vel_norm2;
  vm2 = max_vel_ * max_vel_;
  am2 = max_acc_ * max_acc_;

  Eigen::Vector3d vel, accel, accel_nor;
  for (uint i = 0; i < q.size() - 2; i++)
  {
    /* ---------- velocity feasibility ---------- */
    vel = (q[i + 2] - q[i])/2; // ?? q[i + 2] - q[i]
    vel_norm2 = pow(vel.norm(),2);
    double vd = vel_norm2 * ts_inv2 - vm2;
    if(vd > 0.0){
      cost += pow(vd, 2);
      gradient[i    ] += vd * ts_inv2 * (-2.0) * vel;
      gradient[i + 2] += vd * ts_inv2 * (2.0) * vel;
    }
    
    /* ---------- acceleration feasibility ---------- */
    accel = q[i + 2] - 2 * q[i + 1] + q[i];
    double ad = accel.squaredNorm() * ts_inv4 - am2;
    if(ad > 0.0){
      cost += pow(ad, 2);
      gradient[i    ] += ad * ts_inv4 * (4.0) * accel;
      gradient[i + 1] += ad * ts_inv4 * (-8.0) * accel;
      gradient[i + 2] += ad * ts_inv4 * (4.0) * accel;
    }
    
    /* ---------- yaw rate feasibility ---------- */
    if(yaw_tracking_mode>0 && (i>0 && i<q.size()-3)){
      accel_nor = (accel.cross(vel))/vel_norm2 * ts_inv;
      float delta = accel_nor.norm();
      double omegad = delta - 0.8*max_yaw_rate_;
      if(omegad > 0.0){
        cost += omegad;
        gradient[i    ] += (q[i+2] - q[i])/2*ts_inv;
        gradient[i + 1] += (2 * q[i + 1] - q[i+2] - q[i])/ts_inv2;
        gradient[i + 2] += (q[i] - q[i+2])/2*ts_inv;
      }
    }
  }

}

void BsplineOptimizer::calcEndpointCost(const vector<Eigen::Vector3d>& q, double& cost,
                                        vector<Eigen::Vector3d>& gradient)
{
  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

  switch(frontend_constrain_){
    case END_CONSTRAINT::POS_VEL :
    {
      Eigen::Vector3d q_2, q_1, q_0, qa;
      q_2 = q[2];
      q_1 = q[1];
      q_0 = q[0];

      qa = (q_0 - 2*q_1 + q_2) * ts_inv2 - end_acc_;
      cost += qa.squaredNorm();

      gradient[0] += 2 * qa * (ts_inv2);
      //gradient[1] += - 4 * qa * (ts_inv2);
      //gradient[2] += 2 * qa * (ts_inv2);
    break;
    }
    case END_CONSTRAINT::POS :
    {
      Eigen::Vector3d q_2, q_1, q_0, qv, qa;
      q_2 = q[2];
      q_1 = q[1];
      q_0 = q[0];

      qv = (q_2 - q_0) / 2 * ts_inv - end_vel_;
      qa = (q_0 - 2*q_1 + q_2) * ts_inv2 - end_acc_;
      cost += qv.squaredNorm() + qa.squaredNorm();

      gradient[0] += - qv * ts_inv + 2 * qa * (ts_inv2);
      gradient[1] +=               - 4 * qa * (ts_inv2);
      //gradient[2] +=   qv * ts_inv + 2 * qa * (ts_inv2);
    break;
    }
  }

  uint q_size = q.size();
  switch(backtend_constrain_){
    case END_CONSTRAINT::POS_VEL :
    {
      Eigen::Vector3d q_2, q_1, q_0, qa;
      q_0 = q[q_size - 3];
      q_1 = q[q_size - 2];
      q_2 = q[q_size - 1];

      qa = (q_0 - 2*q_1 + q_2) * ts_inv2 - end_acc_;
      cost += qa.squaredNorm();

      gradient[q_size - 3] += 2 * qa * (ts_inv2);
      //gradient[q_size - 2] += - 4 * qa * (ts_inv2);
      //gradient[q_size - 1] += 2 * qa * (ts_inv2);
    break;
    }
    case END_CONSTRAINT::POS :
    {
      Eigen::Vector3d q_2, q_1, q_0, qv, qa;
      q_0 = q[q_size - 3];
      q_1 = q[q_size - 2];
      q_2 = q[q_size - 1];

      qv = (q_2 - q_0) / 2 * ts_inv - end_vel_;
      qa = (q_0 - 2*q_1 + q_2) * ts_inv2 - end_acc_;
      cost += qv.squaredNorm() + qa.squaredNorm();

      gradient[q_size - 3] += - qv * (ts_inv) + 2 * qa * (ts_inv2);
      gradient[q_size - 2] +=                 - 4 * qa * (ts_inv2);
      //gradient[q_size - 1] +=   qv * (ts_inv) + 2 * qa * (ts_inv2);
    break;
    }
  }
}

void BsplineOptimizer::combineCost(const std::vector<double>& x, std::vector<double>& grad, double& f_combine)
{
  /* ---------- convert to control point vector ---------- */
  vector<Eigen::Vector3d> q;
  /* head control points */
  switch(frontend_constrain_){
    case END_CONSTRAINT::ALL :
      {
        Eigen::Vector3d q_0 = start_pos_ - start_vel_*ts + start_acc_*ts2/3;
        Eigen::Vector3d q_1 = start_pos_ - start_acc_*ts2/6;
        Eigen::Vector3d q_2 = start_pos_ + start_vel_*ts + start_acc_*ts2/3;
        q.push_back(q_0);
        q.push_back(q_1);
        q.push_back(q_2);
        break;
      }
    case END_CONSTRAINT::POS_VEL :
      {
        Eigen::Vector3d q_2(x[0], x[1], x[2]);
        Eigen::Vector3d q_0 = q_2 - 2 * ts * start_vel_;
        q.push_back(q_0);
        Eigen::Vector3d q_1 = (6*start_pos_ - q_0 - q_2)/4;
        q.push_back(q_1);
        break;
      }
    case END_CONSTRAINT::POS :
      Eigen::Vector3d q_1(x[0], x[1], x[2]);
      Eigen::Vector3d q_2(x[3], x[4], x[5]);
      Eigen::Vector3d q_0 = 6*start_pos_ - 4*q_1 - q_2;
      q.push_back(q_0);
  }

  /* optimized control points */
  for (int i = 0; i < variable_num_ / order_; i++)
  {
    Eigen::Vector3d qi(x[order_ * i], x[order_ * i + 1], x[order_ * i + 2]);
    q.push_back(qi);
  }

  /* tail control points */
  switch(backtend_constrain_){
    case END_CONSTRAINT::ALL :
      {
        Eigen::Vector3d q_2 = end_pos_ - end_vel_*ts + end_acc_*ts2/3;
        Eigen::Vector3d q_1 = end_pos_ - end_acc_*ts2/6;
        Eigen::Vector3d q_0 = end_pos_ + end_vel_*ts + end_acc_*ts2/3;
        q.push_back(q_2);
        q.push_back(q_1);
        q.push_back(q_0);
        break;
      }
    case END_CONSTRAINT::POS_VEL :
      {
        Eigen::Vector3d q_2(x[variable_num_-3], x[variable_num_-2], x[variable_num_-1]);
        Eigen::Vector3d q_0 = q_2 + 2 * ts * end_vel_;
        Eigen::Vector3d q_1 = (6*end_pos_ - q_0 - q_2)/4;
        q.push_back(q_1);
        q.push_back(q_0);
        break;
      }
    case END_CONSTRAINT::POS :
      Eigen::Vector3d q_2(x[variable_num_-6], x[variable_num_-5], x[variable_num_-4]);
      Eigen::Vector3d q_1(x[variable_num_-3], x[variable_num_-2], x[variable_num_-1]);
      Eigen::Vector3d q_0 = 6*end_pos_ - 4*q_1 - q_2;
      q.push_back(q_0);
  }

  /* ---------- evaluate cost and gradient ---------- */
  double f_tensile, f_smoothness, f_avoidance, f_feasibility;
  vector<Eigen::Vector3d> g_tensile, g_smoothness, g_avoidance, g_feasibility;
  g_tensile.resize(control_points_.rows());
  calcTensileCost(q,f_tensile,g_tensile);
  
  g_smoothness.resize(control_points_.rows());
  calcSmoothnessCost(q, f_smoothness, g_smoothness);
  
  g_avoidance.resize(control_points_.rows());
  calcAvoidanceCost(q, f_avoidance, g_avoidance);
  
  g_feasibility.resize(control_points_.rows());
  calcFeasibilityCost(q, f_feasibility, g_feasibility);
  
  //g_endpoint.resize(control_points_.rows());
  //calcEndpointCost(q, f_endpoint, g_endpoint);

  /* ---------- convert to NLopt format...---------- */
  grad.resize(variable_num_);

  f_combine = lamda_smooth * f_smoothness + lamda_obs * f_avoidance + lamda_feas * f_feasibility + lamda_tensile * f_tensile;
  //if(lamda_end>0.0) f_combine += lamda_end * f_endpoint;

  for (int i = 0; i < variable_num_ / order_; i++)
    for (int j = 0; j < 3; j++)
    {
      /* the first start_id_ points is static here */
      grad[order_ * i + j] = lamda_smooth * g_smoothness[i + start_id_](j) + lamda_obs * g_avoidance[i + start_id_](j)
                         + lamda_feas * g_feasibility[i + start_id_](j) + lamda_tensile * g_tensile[i + start_id_](j);
      //if(lamda_end>0.0) grad[order_ * i + j] += lamda_end * g_endpoint[i + start_id_](j);
    }

  /* ---------- print cost ---------- */
  iter_num_ += 1;

  if (false && iter_num_ % 10 == 0){
    cout << "[Optimization]: iter_num_: " << iter_num_ 
         << " smooth: " << lamda_smooth * f_smoothness << " , dist: " << lamda_obs * f_avoidance << ", fea: " << lamda_feas * f_feasibility
         <<  ", tensile: " << lamda_tensile * f_tensile <<  ", total: " << f_combine
         << endl;
  }

}

double BsplineOptimizer::costFunction(const std::vector<double>& x, std::vector<double>& grad, void* func_data)
{
  BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(func_data);

  double cost;
  opt->combineCost(x, grad, cost);

  /* save the min cost result */
  if (cost < opt->min_cost_)
  {
    opt->min_cost_ = cost;
    opt->best_variable_ = x;
  }

  return cost;

  // /* ---------- evaluation ---------- */

  // ros::Time te1 = ros::Time::now();
  // double time_now = (te1 - opt->time_start_).toSec();
  // opt->vec_time_.push_back(time_now);
  // if (opt->vec_cost_.size() == 0)
  // {
  //   opt->vec_cost_.push_back(f_combine);
  // }
  // else if (opt->vec_cost_.back() > f_combine)
  // {
  //   opt->vec_cost_.push_back(f_combine);
  // }
  // else
  // {
  //   opt->vec_cost_.push_back(opt->vec_cost_.back());
  // }
}

}  // namespace histo_planner
