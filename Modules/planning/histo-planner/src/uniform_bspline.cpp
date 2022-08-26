#include "uniform_bspline.h"
#include <ros/ros.h>

namespace Histo_Planning
{
double UniformBspline::limit_vel_;
double UniformBspline::limit_acc_;
double UniformBspline::limit_omega_;
bool UniformBspline::yaw_track;

// control points is a (n+1)x3 matrix
UniformBspline::UniformBspline(Eigen::MatrixXd points, int order, double interval_, bool zero)
{
  this->p_ = order;

  control_points_ = points;
  this->n_ = points.rows() - 1;

  this->m_ = this->n_ + this->p_ + 1;

  // calculate knots vector
  this->interval_ = interval_;
  this->u_ = Eigen::VectorXd::Zero(this->m_ + 1);
  for (int i = 0; i <= this->m_; ++i)
  {
    if (i <= this->p_)
      this->u_(i) = double(-this->p_ + i) * this->interval_;

    else if (i > this->p_ && i <= this->m_ - this->p_)
    {
      this->u_(i) = this->u_(i - 1) + this->interval_;
    }
    else if (i > this->m_ - this->p_)
    {
      this->u_(i) = this->u_(i - 1) + this->interval_;
    }
  }


this->u = this->u_; // base on uniform b spline with time remapping.


  // show the result
  // cout << "p_: " << p_ << "  n: " << n << "  m: " << m << endl;
  // cout << "control pts:\n" << control_points_ << "\nknots:\n" <<
  // this->u_.transpose() << endl; cout << "M3:\n" << M[0] << "\nM4:\n" << M[1]
  // << "\nM5:\n" << M[2] << endl;

  if (zero) // if the start point is not determined
  {
    x0_ = (1 / 6.0) * (control_points_.row(0) + 4 * control_points_.row(1) + control_points_.row(2));
    v0_ = (1 / 2.0 / interval_) * (control_points_.row(2) - control_points_.row(0));
    a0_ = (1 / interval_ / interval_) * (control_points_.row(0) - 2 * control_points_.row(1) + control_points_.row(2));

    // cout << "initial state: " << x0_.transpose() << "\n"
    //      << v0_.transpose() << "\n"
    //      << a0_.transpose() << endl;
  }
}

UniformBspline::~UniformBspline()
{
}

void UniformBspline::setKnot(Eigen::VectorXd knot)
{
  this->u_ = knot;
}

Eigen::VectorXd UniformBspline::getKnot()
{
  return this->u_;
}

void UniformBspline::getTimeSpan(double& um, double& um_p)
{
  um = this->u_(this->p_);
  um_p = this->u_(this->m_ - this->p_);
}

Eigen::Vector3d UniformBspline::evaluateDeBoor(double t)
{
  if (t < this->u_(this->p_) || t > this->u_(this->m_ - this->p_))
  {
    cout << "Out of trajectory range. u_0 = " << this->u_(this->p_) << ", t = " << t << ", u_n = " << this->u_(this->m_ - this->p_) << endl;
    // return Eigen::Vector3d::Zero(3);
    if (t < this->u_(this->p_))
      t = this->u_(this->p_);
    if (t > this->u_(this->m_ - this->p_))
      t = this->u_(this->m_ - this->p_);
  }

  // determine which [ui,ui+1] lay in
  int k = this->p_;
  while (true)
  {
    if (this->u_(k + 1) >= t)
      break;
    ++k;
  }

  double new_t = u(k) + (t-u_(k))/(u_(k+1)-u_(k)) * (u(k+1)-u(k));  // Time remapping.

  /* deBoor's alg */
  vector<Eigen::Vector3d> d;
  for (int i = 0; i <= p_; ++i)
  {
    d.push_back(control_points_.row(k - p_ + i));
    // cout << d[i].transpose() << endl;
  }

  for (int r = 1; r <= p_; ++r)
  {
    for (int i = p_; i >= r; --i)
    {
      double alpha = (new_t - u[i + k - p_]) / (u[i + 1 + k - r] - u[i + k - p_]);// HERE!! the curve will be change if the knots has been changed. 
      // cout << "alpha: " << alpha << endl;
      d[i] = (1 - alpha) * d[i - 1] + alpha * d[i];
    }
  }

  return d[p_];
}

Eigen::Vector3d UniformBspline::getLocation(double t_cur, double dist_forward)
{
  double length = 0.0;
  double t_end = this->u_(this->m_ - this->p_);

  Eigen::Vector3d p_a = evaluateDeBoor(t_cur);
  Eigen::Vector3d p_b = p_a;
  for (double t = t_cur + 0.01; t <= t_end; t += 0.01)
  {
    p_b = evaluateDeBoor(t);
    length += (p_b - p_a).norm();
    p_a = p_b;
    if(length>=dist_forward)
      break;
  }
  return p_b;
}

Eigen::MatrixXd UniformBspline::getDerivativeControlPoints()
{
  // The derivative of a b-spline is also a b-spline, its order become p_-1
  // control point Qi = p_*(Pi+1-Pi)/(ui+p_+1-ui+1)
  Eigen::MatrixXd ctp = Eigen::MatrixXd::Zero(control_points_.rows() - 1, 3);
  for (int i = 0; i < ctp.rows(); ++i)
  {
    ctp.row(i) = p_ * (control_points_.row(i + 1) - control_points_.row(i)) / (u_(i + p_ + 1) - u_(i + 1));
  }
  return ctp;
}

UniformBspline UniformBspline::getDerivative()
{
  Eigen::MatrixXd ctp = this->getDerivativeControlPoints();
  UniformBspline derivative = UniformBspline(ctp, p_ - 1, this->interval_, false);

  /* cut the first and last knot */
  Eigen::VectorXd knot(this->u_.rows() - 2);
  knot = this->u_.segment(1, this->u_.rows() - 2);
  derivative.setKnot(knot);

  return derivative;
}

bool UniformBspline::checkFeasibility(bool show, double ts)
{
  // SETY << "[Bspline]: total points size: " << control_points_.rows() << endl;

// Option 1: Continuous sampling
  bool fea = true;
  UniformBspline vel = getDerivative();
  UniformBspline acc = getDerivative().getDerivative();

  double max_vel, min_vel, mean_vel;
  double max_acc, min_acc, mean_acc;
  for(int i=0; i<this->m_-2*this->p_; i++)
  {
    vel.getMeanMinMax(mean_vel, min_vel, max_vel, this->u_(this->p_+i), this->u_(this->p_+i+1), this->interval_/4);
    acc.getMeanMinMax(mean_acc, min_acc, max_acc, this->u_(this->p_+i), this->u_(this->p_+i+1), this->interval_/4);
    cout << "[checkFea]: spline segment N" << i << ",max_vel:=" << max_vel << ",max_acc:=" << max_acc << endl;
	double ratio_vel = max_vel / limit_vel_;
	double ratio_acc = max_acc / limit_acc_;
	if(ratio_vel>1.0 || ratio_acc>1.0)
	{
	  fea = false;
	  if (show)
        cout << "[checkFea]: Infeasible: vel_max=" << max_vel << "[" << limit_vel_ << "]" << " vel_mean=" << mean_vel << " vel_min=" << min_vel << ", acc_max=" << max_acc << "[" << limit_acc_ << "]" << " acc_mean=" << mean_acc << " acc_min=" << min_acc << ", ratio_max=" << max(max_vel / limit_vel_, max_acc / limit_acc_) << endl;
	}
  }


// Option 2: Midpoint sampling
//  bool fea, fea_vel, fea_acc = true;
//  Eigen::MatrixXd P = control_points_;
//  /* vel +acc feasibility */
//  double max_vel, max_acc = 0.0;
//  for (int i = 0; i < P.rows() - 2; ++i)
//  {
//    Eigen::Vector3d vel = p_ * (P.row(i + 2) - P.row(i+1)) / 2 / (u_(i + p_ + 2) - u_(i + 2)) +
//                          p_ * (P.row(i + 1) - P.row(i)) / 2 / (u_(i + p_ + 1) - u_(i + 1));
//                          
//    Eigen::Vector3d acc = p_ * (p_ - 1) *
//                          ((P.row(i + 2) - P.row(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) -
//                           (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
//                          (u_(i + p_ + 1) - u_(i + 2));

//    if (vel.norm() > limit_vel_ + 1e-4){
//      fea_vel = false;
//      fea = false;
//      max_vel = max(max_vel, vel.norm());
//    }

//    if (acc.norm() > limit_acc_ + 1e-4){
//      fea_acc = false;
//      fea = false;
//      max_acc = max(max_acc, acc.norm());
//    }
//    
//    if (show && !(fea_vel && fea_acc))
//      cout << "[checkFea]: Infeasible N." << i << ": vel=" << vel.norm() << ": vel_max=" << limit_vel_ << ", acc=" << acc.norm() << ", acc_max=" << limit_acc_ << ", ratio_max=" << max(max_vel / limit_vel_, max_acc / limit_acc_) << endl;
//      
//    fea_vel = true; fea_acc = true;
//    max_vel = 0.0; max_acc = 0.0;
//  }

  return fea;
}

bool UniformBspline::reallocateTime(bool show, double ts)
{
  // SETY << "[Bspline]: total points size: " << control_points_.rows() << endl;
  // cout << "origin knots:\n" << u_.transpose() << endl;
  bool fea = true;

  UniformBspline vel = getDerivative();
  UniformBspline acc = vel.getDerivative();

  double max_vel, min_vel, mean_vel;
  double max_acc, min_acc, mean_acc;
  //double ratio_list[this->m_-2*this->p_];
  for(int i=0; i<this->m_-2*this->p_; i++)
  {
    vel.getMeanMinMax(mean_vel, min_vel, max_vel, this->u_(this->p_+i), this->u_(this->p_+i+1), 4);
    acc.getMeanMinMax(mean_acc, min_acc, max_acc, this->u_(this->p_+i), this->u_(this->p_+i+1), 4);
    double ratio_vel, ratio_acc;
    ratio_vel = (0.2*mean_vel+0.8*max_vel) / limit_vel_;
    if(yaw_track){
      ratio_acc = max((0.2*mean_acc+0.8*max_acc) / limit_acc_, (mean_acc/mean_vel) / limit_omega_);
    }else{
      ratio_acc = (0.2*mean_acc+0.8*max_acc) / limit_acc_;
    }
	
    if(show)
      cout << "[reaallocateTime]: spline segment(0.1.2..) N" << i << " [timeSpan:" << this->u_(this->p_+i) << "~" << this->u_(this->p_+i+1) << "],mean_vel:=" << mean_vel << ",min_vel:=" << min_vel << ",max_vel:=" << max_vel << "[" << limit_vel_ << "],mean_acc:=" << mean_acc << ",min_acc:=" << min_acc << ",max_acc:=" << max_acc << "[" << limit_acc_ << "],ratio_vel:=" << ratio_vel << ",ratio_acc:=" << ratio_acc << endl;
      
    if(ratio_vel>1.1 && ratio_vel>ratio_acc)
    {
      fea = false;
      double time_ori = this->u_(this->p_+i+1) - this->u_(i+1);
      double time_new = (pow(ratio_vel,0.3)) * time_ori;
      double delta_t = (time_new - time_ori);
      double t_inc = delta_t / double(this->p_);
    
      for(int j=1; j<this->p_; j++)
        this->u_(i+1 + j) += t_inc * j;
      
      for(int j=this->p_+i+1; j<this->u_.rows(); j++)
        this->u_(j) += delta_t;
        
      if(i == this->m_-2*this->p_-1)
      	for(int j=this->m_-this->p_; j<this->u_.rows(); j++)
          this->u_(j) += 1*t_inc;
    }
    else if(ratio_acc>1.1 && ratio_vel<=ratio_acc)
    {
      fea = false;
      double time_ori = this->u_(this->p_+i+1) - this->u_(i+2);
      double time_new = (pow(ratio_acc,0.5)) * time_ori;
      double delta_t = (time_new - time_ori)*0.8;
      double t_inc = delta_t / double(this->p_-1);
    
      for(int j=1; j<this->p_-1; j++)
        this->u_(i+2 + j) += j * t_inc;
      
      for(int j=this->p_+i+1; j<this->u_.rows(); j++)
        this->u_(j) += delta_t;
        
      if(i == this->m_-2*this->p_-1){
      	for(int j=this->m_-this->p_; j<this->u_.rows(); j++)
          this->u_(j) += 2*delta_t;
      }
    }
    if(!fea)
    {
      vel.setKnot(this->u_.segment(1, this->u_.rows() - 2));
      acc.setKnot(this->u_.segment(2, this->u_.rows() - 3));
    }
  }

  return fea;
}

void UniformBspline::recomputeInit()
{
  double t1 = u_(1), t2 = u_(2), t3 = u_(3), t4 = u_(4), t5 = u_(5);

  /* write the A matrix */
  Eigen::Matrix3d A;

  /* position */
  A(0, 0) = ((t2 - t5) * (t3 - t4) * (t3 - t4)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(0, 1) = ((t2 - t5) * (t3 - t4) * (t1 - t3)) / ((t1 - t4) * (t2 - t4) * (t2 - t5)) +
            ((t1 - t4) * (t2 - t3) * (t3 - t5)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(0, 2) = ((t1 - t4) * (t2 - t3) * (t2 - t3)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));

  /* velocity */
  A(1, 0) = 3.0 * ((t2 - t5) * (t3 - t4)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(1, 1) = 3.0 * ((t1 - t4) * (t2 - t3) - (t2 - t5) * (t3 - t4)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(1, 2) = -3.0 * ((t1 - t4) * (t2 - t3)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));

  /* acceleration */
  A(2, 0) = 6.0 * (t2 - t5) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(2, 1) = -6.0 * ((t1 - t4) + (t2 - t5)) / ((t1 - t4) * (t2 - t4) * (t2 - t5));
  A(2, 2) = 6.0 * (t1 - t4) / ((t1 - t4) * (t2 - t4) * (t2 - t5));

  /* write B = (bx, by, bz) */
  Eigen::Matrix3d B;
  Eigen::Vector3d bx, by, bz;
  B.row(0) = x0_;
  B.row(1) = v0_;
  B.row(2) = a0_;
  // cout << "B:\n" << B << endl;

  bx = B.col(0);
  by = B.col(1);
  bz = B.col(2);

  /* solve */
  Eigen::Vector3d px = A.colPivHouseholderQr().solve(bx);
  Eigen::Vector3d py = A.colPivHouseholderQr().solve(by);
  Eigen::Vector3d pz = A.colPivHouseholderQr().solve(bz);

  Eigen::Matrix3d P;
  P.col(0) = px;
  P.col(1) = py;
  P.col(2) = pz;

  control_points_.row(0) = P.row(0);
  control_points_.row(1) = P.row(1);
  control_points_.row(2) = P.row(2);

  B = A * P;
  // cout << "B:\n" << B << endl;
}

// input :
//      sample : 3 x (K+2) (for 3 order) for x, y, z sample
//      ts
// output:
//      control_pts (K+6)x3 two repeated points at start and end
void UniformBspline::cubicSamplePts_to_BsplineCtlPts(vector<Eigen::Vector3d> samples, double ts, Eigen::MatrixXd& control_pts)
{
  int K = samples.size() - 4 - 2; // smaples_[3xN]

  // write A
  Eigen::VectorXd prow(3), vrow(3), arow(3);
  prow << 1, 4, 1;
  vrow << -1, 0, 1;
  arow << 1, -2, 1;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 6, K + 4);

  for (int i = 0; i < K + 2; ++i)
    A.block(i, i, 1, 3) = (1 / 6.0) * prow.transpose();

  A.block(K + 2, 0, 1, 3) = A.block(K + 3, K + 1, 1, 3) = (1 / 2.0 / ts) * vrow.transpose();
  A.block(K + 4, 0, 1, 3) = A.block(K + 5, 0, 1, 3) = (1 / ts / ts) * arow.transpose();

  // write b
  Eigen::VectorXd bx(K + 6), by(K + 6), bz(K + 6);
  for (int i = 0; i < K + 6; ++i)
  {
    bx(i) = samples[i][0];
    by(i) = samples[i][1];
    bz(i) = samples[i][2];
  }

  // solve Ax = b
  Eigen::VectorXd px = A.colPivHouseholderQr().solve(bx);
  Eigen::VectorXd py = A.colPivHouseholderQr().solve(by);
  Eigen::VectorXd pz = A.colPivHouseholderQr().solve(bz);

  // convert to control pts
  control_pts.resize(K + 4, 3);
  control_pts.col(0) = px;
  control_pts.col(1) = py;
  control_pts.col(2) = pz;
}

void UniformBspline::BsplineParameterize(const double& ts, const vector<Eigen::Vector3d>& point_set,
                                            const vector<Eigen::Vector3d>& start_end_derivative,
                                            Eigen::MatrixXd& ctrl_pts)
{
  if (ts <= 0)
  {
    cout << "[B-spline]:time step error." << endl;
    return;
  }

  if (point_set.size() <= 3)
  {
    cout << "[B-spline]:point set have only " << point_set.size() << " points." << endl;
    return;
  }

  if (start_end_derivative.size() != 4)
  {
    cout << "[B-spline]:derivatives error." << endl;
  }

  int K = point_set.size() - 2;

  // write A
  Eigen::VectorXd prow(3), vrow(3), arow(3);
  prow << 1, 4, 1;
  vrow << -1, 0, 1;
  arow << 1, -2, 1;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 6, K + 4);

  for (int i = 0; i < K + 2; ++i)
    A.block(i, i, 1, 3) = prow.transpose();
  A.block(K + 2, 0, 1, 3) = A.block(K + 3, K + 1, 1, 3) = vrow.transpose();
  A.block(K + 4, 0, 1, 3) = A.block(K + 5, K + 1, 1, 3) = arow.transpose();

  // cout << "A:\n" << A << endl;
  A.block(0, 0, K + 2, K + 4) = (1 / 6.0) * A.block(0, 0, K + 2, K + 4);
  A.block(K + 2, 0, 2, K + 4) = (1 / 2.0 / ts) * A.block(K + 2, 0, 2, K + 4);
  A.row(K + 4) = (1 / ts / ts) * A.row(K + 4);
  A.row(K + 5) = (1 / ts / ts) * A.row(K + 5);

  // write b
  Eigen::VectorXd bx(K + 6), by(K + 6), bz(K + 6);
  for (int i = 0; i < K + 2; ++i)
  {
    bx(i) = point_set[i](0), by(i) = point_set[i](1), bz(i) = point_set[i](2);
  }

  for (int i = 0; i < 4; ++i)
  {
    bx(K + 2 + i) = start_end_derivative[i](0);
    by(K + 2 + i) = start_end_derivative[i](1);
    bz(K + 2 + i) = start_end_derivative[i](2);
  }

  // solve Ax = b
  Eigen::VectorXd px = A.colPivHouseholderQr().solve(bx);
  Eigen::VectorXd py = A.colPivHouseholderQr().solve(by);
  Eigen::VectorXd pz = A.colPivHouseholderQr().solve(bz);

  // convert to control pts
  ctrl_pts.resize(K + 4, 3);
  ctrl_pts.col(0) = px;
  ctrl_pts.col(1) = py;
  ctrl_pts.col(2) = pz;

  cout << "[B-spline]: parameterization ok." << endl;
}

double UniformBspline::getTimeSum()
{
  double tm, tmp;
  getTimeSpan(tm, tmp);
  return tmp - tm;
}

double UniformBspline::getLength()
{
  double length = 0.0;

  double tm, tmp;
  getTimeSpan(tm, tmp);
  Eigen::Vector3d p_l = evaluateDeBoor(tm), p_n;
  for (double t = tm + 0.01; t <= tmp; t += 0.01)
  {
    p_n = evaluateDeBoor(t);
    length += (p_n - p_l).norm();
    p_n = p_l;
  }

  return length;
}

double UniformBspline::getJerk()
{
  UniformBspline jerk_traj = getDerivative().getDerivative().getDerivative();
  Eigen::VectorXd times = jerk_traj.getKnot();
  Eigen::MatrixXd ctrl_pts = jerk_traj.getControlPoint();

  cout << "num knot:" << times.rows() << endl;
  cout << "num ctrl pts:" << ctrl_pts.rows() << endl;

  double jerk = 0.0;
  for (int i = 0; i < ctrl_pts.rows(); ++i)
  {
    jerk += (times(i + 1) - times(i)) * ctrl_pts(i, 0) * ctrl_pts(i, 0);
    jerk += (times(i + 1) - times(i)) * ctrl_pts(i, 1) * ctrl_pts(i, 1);
    jerk += (times(i + 1) - times(i)) * ctrl_pts(i, 2) * ctrl_pts(i, 2);
  }

  return jerk;
}

void UniformBspline::getMeanMinMax(double& mean_, double& min_, double& max_, double t_start_, double t_end_, int sample_num)
{
  double tm, tmp, t_interval;
  if(t_start_ == -1.0 && t_end_ == -1.0){
    this->getTimeSpan(tm, tmp);
  }else{
    tm = t_start_;
    tmp = t_end_;
  }
  if(sample_num != 0)
    t_interval = (tmp-tm)/(sample_num-1);
  else
    t_interval = 0.01;

  double max_value = -1.0, mean_value = 0.0, min_value = std::numeric_limits<double>::max();
  int num = 0;
  for (double t = tm; t <= tmp; t += t_interval)
  {
    Eigen::Vector3d v3d = this->evaluateDeBoor(t);
    double value = v3d.norm();
//cout << value << endl;
    mean_value += value;
    ++num;
    if (value < min_value)
    {
      min_value = value;
    }
    if (value > max_value)
    {
      max_value = value;
    }
  }
//cout << endl;
  mean_ = mean_value / double(num);
  min_ = min_value;
  max_ = max_value;
}

void UniformBspline::getVelMeanMinMax(double& mean_v, double& min_v, double& max_v, double t_start_, double t_end_, int sample_num)
{
  UniformBspline vel = getDerivative();
  double tm, tmp, t_interval;
  if(t_start_ == -1.0 && t_end_ == -1.0){
    vel.getTimeSpan(tm, tmp);
  }else{
    tm = t_start_;
    tmp = t_end_;
  } 
  if(sample_num != 0)
    t_interval = (tmp-tm)/(sample_num-1);
  else
    t_interval = 0.01;

  double max_vel = -1.0, mean_vel = 0.0, min_vel = std::numeric_limits<double>::max();
  int num = 0;
  for (double t = tm; t <= tmp; t += t_interval)
  {
    Eigen::Vector3d v3d = vel.evaluateDeBoor(t);
    double vn = v3d.norm();

    mean_vel += vn;
    ++num;
    if (vn < min_vel)
    {
      min_vel = vn;
    }
    if (vn > max_vel)
    {
      max_vel = vn;
    }
  }

  mean_v = mean_vel / double(num);
  min_v = min_vel;
  max_v = max_vel;
}

void UniformBspline::getAccMeanMinMax(double& mean_a, double& min_a, double& max_a, double t_start_, double t_end_, int sample_num)
{
  UniformBspline acc = getDerivative().getDerivative();
  double tm, tmp, t_interval;
  if(t_start_ == -1.0 && t_end_ == -1.0){
    acc.getTimeSpan(tm, tmp);
  }else{
    tm = t_start_;
    tmp = t_end_;
  } 
  if(sample_num != 0)
    t_interval = (tmp-tm)/(sample_num-1);
  else
    t_interval = 0.01;

  double max_acc = -1.0, mean_acc = 0.0, min_acc = std::numeric_limits<double>::max();
  int num = 0;
  for (double t = tm; t <= tmp; t += t_interval)
  {
    Eigen::Vector3d a3d = acc.evaluateDeBoor(t);
    double an = a3d.norm();

    mean_acc += an;
    ++num;
    if (an < min_acc)
    {
      min_acc = an;
    }
    if (an > max_acc)
    {
      max_acc = an;
    }
  }

  mean_a = mean_acc / double(num);
  min_a = min_acc;
  max_a = max_acc;
}
}
