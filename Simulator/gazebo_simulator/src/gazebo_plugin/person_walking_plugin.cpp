#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>
#include <random>

namespace gazebo
{
  class PersonWalkingPlugin : public ModelPlugin
  {
    public: PersonWalkingPlugin() : ModelPlugin()
    {
      // 初始化随机数生成器
      this->random_gen = std::default_random_engine(this->seed());
      // 随机方向变化范围 (-0.5 到 0.5)
      this->direction_distribution = std::uniform_real_distribution<double>(-0.5, 0.5);
      // 随机速度范围 (0.5 到 1.5 m/s)
      this->speed_distribution = std::uniform_real_distribution<double>(0.5, 1.5);
      // 随机转向角度范围 (-pi/2 到 pi/2 弧度)
      this->turn_distribution = std::uniform_real_distribution<double>(-M_PI * 2, M_PI * 2);
    }

    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
      // 存储模型指针
      this->model = _parent;

      // 创建连接，使OnUpdate在每次世界更新时被调用
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&PersonWalkingPlugin::OnUpdate, this));

      // 初始化时间
      this->last_turn_time = this->model->GetWorld()->SimTime();
      this->last_update_time = this->model->GetWorld()->SimTime();

      // 初始化运动参数
      this->linear_velocity = ignition::math::Vector3d(1.0, 0.0, 0.0);
      this->position = this->model->WorldPose().Pos();
      
      // 随机初始化朝向
      double yaw = this->direction_distribution(this->random_gen) * M_PI * 2;
      this->model->SetWorldPose(ignition::math::Pose3d(
          this->position, 
          ignition::math::Quaterniond(0, 0, yaw)));
    }

    public: void OnUpdate()
    {
      // 获取当前时间
      common::Time current_time = this->model->GetWorld()->SimTime();
      
      // 每隔5秒改变一次方向
      double time_diff = (current_time - this->last_update_time).Double();
      this->last_update_time = current_time;
      if ((current_time - this->last_turn_time).Double() > 5.0) {
        this->ChangeDirection();
        this->last_turn_time = current_time;
      }

      // 更新位置
      this->UpdatePosition(time_diff);
    }

    private: void ChangeDirection()
    {
      // 随机改变速度
      double speed = this->speed_distribution(this->random_gen);
      
      // 随机改变转向角度
      double turn = this->turn_distribution(this->random_gen);
      
      // 获取当前朝向
      ignition::math::Pose3d pose = this->model->WorldPose();
      ignition::math::Quaterniond rotation = pose.Rot();
      
      // 添加转向角
      ignition::math::Quaterniond yaw_rotation(0, 0, turn);
      rotation = yaw_rotation * rotation;
      
      // 设置新的朝向
      this->model->SetWorldPose(ignition::math::Pose3d(pose.Pos(), rotation));
      
      // 根据新朝向设置速度
      ignition::math::Vector3d forward(cos(rotation.Yaw()), sin(rotation.Yaw()), 0);
      this->linear_velocity = forward * speed;
    }

    private: void UpdatePosition(double _dt)
    {
      // 计算新位置
      ignition::math::Vector3d new_position = this->position + this->linear_velocity * _dt;
      
      // 保持在地面高度 (假设地面为z=0)
      new_position.Z() = 0.0;
      
      // 更新位置
      this->position = new_position;
      
      // 应用新位置到模型
      ignition::math::Pose3d current_pose = this->model->WorldPose();
      this->model->SetWorldPose(ignition::math::Pose3d(
          this->position, 
          current_pose.Rot()));
    }

    private: unsigned int seed()
    {
      // 生成随机种子
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    // 指向模型的指针
    private: physics::ModelPtr model;

    // 指向更新事件的连接指针
    private: event::ConnectionPtr updateConnection;

    // 随机数生成器
    private: std::default_random_engine random_gen;
    private: std::uniform_real_distribution<double> direction_distribution;
    private: std::uniform_real_distribution<double> speed_distribution;
    private: std::uniform_real_distribution<double> turn_distribution;

    // 时间记录
    private: common::Time last_update_time, last_turn_time;

    // 运动参数
    private: ignition::math::Vector3d linear_velocity;
    private: ignition::math::Vector3d position;
  };

  // 注册插件
  GZ_REGISTER_MODEL_PLUGIN(PersonWalkingPlugin)
}