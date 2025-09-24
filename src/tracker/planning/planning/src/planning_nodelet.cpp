#include <geometry_msgs/PoseStamped.h>
#include <mapping/mapping.h>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/ReplanState.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <traj_opt/traj_opt.h>

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <env/env.hpp>
#include <prediction/prediction.hpp>
#include <thread>
#include <visualization/visualization.hpp>
#include <wr_msg/wr_msg.hpp>

namespace planning {

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

class Nodelet : public nodelet::Nodelet {
 private:
  std::thread initThread_;
  ros::Subscriber gridmap_sub_, odom_sub_, target_sub_, triger_sub_, land_triger_sub_;
  ros::Timer plan_timer_;

  ros::Publisher traj_pub_, heartbeat_pub_, replanState_pub_;

  std::shared_ptr<mapping::OccGridMap> gridmapPtr_;
  std::shared_ptr<env::Env> envPtr_;
  std::shared_ptr<visualization::Visualization> visPtr_;
  std::shared_ptr<traj_opt::TrajOpt> trajOptPtr_;
  std::shared_ptr<prediction::Predict> prePtr_;

  enum class Mode { WAYPOINT, TRACKING };

  struct TargetSnapshot {
    bool received = false;
    bool fresh = false;
    double age = std::numeric_limits<double>::infinity();
    double speed_sq = 0.0;
    nav_msgs::Odometry msg;
  };

  struct PlanningSnapshot {
    ros::Time stamp;
    nav_msgs::Odometry odom;
    quadrotor_msgs::OccMap3d map;
    TargetSnapshot target;
    bool has_goal = false;
    bool land_trigger_active = false;
  };

  class ModeManager {
   public:
    ModeManager() = default;

    Mode decide(bool target_fresh, bool allow_tracking, bool land_triggered,
                Mode current, bool& changed) const {
      Mode desired = Mode::WAYPOINT;
      if (land_triggered) {
        desired = Mode::TRACKING;
      } else if (target_fresh && allow_tracking) {
        desired = Mode::TRACKING;
      }
      changed = desired != current;
      return desired;
    }
  };

  class BehaviorStrategy {
   public:
    explicit BehaviorStrategy(Nodelet& node) : node_(node) {}
    virtual ~BehaviorStrategy() = default;

    virtual bool ready(const PlanningSnapshot& snapshot) const = 0;
    virtual void run(const PlanningSnapshot& snapshot) = 0;

   protected:
    Nodelet& node_;
  };

  class TrackingStrategy : public BehaviorStrategy {
   public:
    explicit TrackingStrategy(Nodelet& node) : BehaviorStrategy(node) {}

    bool ready(const PlanningSnapshot& snapshot) const override;
    void run(const PlanningSnapshot& snapshot) override;
  };

  class WaypointStrategy : public BehaviorStrategy {
   public:
    explicit WaypointStrategy(Nodelet& node) : BehaviorStrategy(node) {}

    bool ready(const PlanningSnapshot& snapshot) const override;
    void run(const PlanningSnapshot& snapshot) override;
  };

  Mode mode_ = Mode::WAYPOINT;
  ModeManager mode_manager_;
  double target_timeout_ = 2.0;
  double target_speed_threshold_ = 0.2;
  double target_speed_threshold_sq_ = 0.04;
  double tracking_height_offset_ = 1.0;
  std::atomic<double> last_target_time_ = ATOMIC_VAR_INIT(0.0);
  std::atomic<double> last_target_speed_sq_ = ATOMIC_VAR_INIT(0.0);
  TrackingStrategy tracking_strategy_{*this};
  WaypointStrategy waypoint_strategy_{*this};
  Eigen::Vector3d goal_;
  Eigen::Vector3d land_p_;
  Eigen::Quaterniond land_q_;

  // NOTE just for debug
  bool debug_ = false;
  quadrotor_msgs::ReplanState replanStateMsg_;
  ros::Publisher gridmap_pub_, inflate_gridmap_pub_;
  quadrotor_msgs::OccMap3d occmap_msg_;

  double tracking_dur_, tracking_dist_, tolerance_d_;
  double waypoint_stop_dist_ = 0.0;

  Trajectory traj_poly_;
  ros::Time replan_stamp_;
  int traj_id_ = 0;
  bool wait_hover_ = true;
  bool force_hover_ = true;

  nav_msgs::Odometry odom_msg_, target_msg_;
  quadrotor_msgs::OccMap3d map_msg_;
  std::atomic_flag odom_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag target_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag gridmap_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_bool odom_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool map_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool triger_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool target_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool land_triger_received_ = ATOMIC_VAR_INIT(false);

  void pub_hover_p(const Eigen::Vector3d& hover_p, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = true;
    traj_msg.hover_p.resize(3);
    for (int i = 0; i < 3; ++i) {
      traj_msg.hover_p[i] = hover_p[i];
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    traj_pub_.publish(traj_msg);
  }
  void pub_traj(const Trajectory& traj, const double& yaw, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = false;
    traj_msg.order = 5;
    Eigen::VectorXd durs = traj.getDurations();
    int piece_num = traj.getPieceNum();
    traj_msg.duration.resize(piece_num);
    traj_msg.coef_x.resize(6 * piece_num);
    traj_msg.coef_y.resize(6 * piece_num);
    traj_msg.coef_z.resize(6 * piece_num);
    for (int i = 0; i < piece_num; ++i) {
      traj_msg.duration[i] = durs(i);
      CoefficientMat cMat = traj[i].getCoeffMat();
      int i6 = i * 6;
      for (int j = 0; j < 6; j++) {
        traj_msg.coef_x[i6 + j] = cMat(0, j);
        traj_msg.coef_y[i6 + j] = cMat(1, j);
        traj_msg.coef_z[i6 + j] = cMat(2, j);
      }
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    // NOTE yaw
    traj_msg.yaw = yaw;
    traj_pub_.publish(traj_msg);
  }

  //bool captureSnapshot(PlanningSnapshot& snapshot);
  //void updateMode(const PlanningSnapshot& snapshot);
  //void executeTracking(const PlanningSnapshot& snapshot);
  //void executeWaypoint(const PlanningSnapshot& snapshot);
  //Eigen::MatrixXd buildInitialState(const Eigen::Vector3d& odom_p,
                                    // const Eigen::Vector3d& odom_v,
                                    // const ros::Time& stamp, ros::Time& replan_stamp);

  void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    goal_ << msgPtr->pose.position.x, msgPtr->pose.position.y, msgPtr->pose.position.z;
    triger_received_ = true;
  }

  void land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    land_p_.x() = msgPtr->pose.position.x;
    land_p_.y() = msgPtr->pose.position.y;
    land_p_.z() = msgPtr->pose.position.z;
    land_q_.w() = msgPtr->pose.orientation.w;
    land_q_.x() = msgPtr->pose.orientation.x;
    land_q_.y() = msgPtr->pose.orientation.y;
    land_q_.z() = msgPtr->pose.orientation.z;
    land_triger_received_ = true;
  }

  void odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (odom_lock_.test_and_set())
      ;
    odom_msg_ = *msgPtr;
    odom_received_ = true;
    odom_lock_.clear();
  }

  void target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (target_lock_.test_and_set())
      ;
    target_msg_ = *msgPtr;
    target_received_ = true;
    last_target_time_.store(ros::Time::now().toSec());
    const Eigen::Vector3d target_v(msgPtr->twist.twist.linear.x,
                                   msgPtr->twist.twist.linear.y,
                                   msgPtr->twist.twist.linear.z);
    last_target_speed_sq_.store(target_v.squaredNorm());
    target_lock_.clear();
  }

  void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
    while (gridmap_lock_.test_and_set())
      ;
    map_msg_ = *msgPtr;
    map_received_ = true;
    gridmap_lock_.clear();
  }

  bool captureSnapshot(PlanningSnapshot& snapshot) {
    if (!odom_received_ || !map_received_) {
      return false;
    }

    snapshot.stamp = ros::Time::now();

    while (odom_lock_.test_and_set())
      ;
    snapshot.odom = odom_msg_;
    odom_lock_.clear();

    while (gridmap_lock_.test_and_set())
      ;
    snapshot.map = map_msg_;
    gridmap_lock_.clear();

    snapshot.has_goal = triger_received_.load();
    snapshot.land_trigger_active = land_triger_received_.load();

    snapshot.target.received = target_received_.load();
    double last_stamp = last_target_time_.load();
    if (snapshot.target.received && last_stamp > 0.0) {
      snapshot.target.age = snapshot.stamp.toSec() - last_stamp;
      snapshot.target.speed_sq = last_target_speed_sq_.load();
      if (snapshot.target.age <= target_timeout_) {
        snapshot.target.fresh = true;
        while (target_lock_.test_and_set())
          ;
        snapshot.target.msg = target_msg_;
        target_lock_.clear();
      }
    }

    return true;
  }

  void updateMode(const PlanningSnapshot& snapshot) {
    bool allow_tracking = snapshot.land_trigger_active ||
                          snapshot.target.speed_sq >= target_speed_threshold_sq_;
    bool changed = false;
    Mode desired_mode = mode_manager_.decide(snapshot.target.fresh, allow_tracking,
                                             snapshot.land_trigger_active, mode_, changed);
    if (changed) {
      mode_ = desired_mode;
      force_hover_ = true;
      wait_hover_ = true;
      if (mode_ == Mode::TRACKING) {
        ROS_INFO("[planner] switch to tracking mode");
      } else {
        ROS_INFO("[planner] switch to waypoint mode");
      }
    }

    if (!snapshot.target.fresh) {
      target_received_ = false;
      last_target_speed_sq_.store(0.0);
    }
  }

  void main_timer_callback(const ros::TimerEvent& event) {
    heartbeat_pub_.publish(std_msgs::Empty());

    PlanningSnapshot snapshot;
    if (!captureSnapshot(snapshot)) {
      return;
    }

    updateMode(snapshot);

    BehaviorStrategy* strategy =
        mode_ == Mode::TRACKING ? static_cast<BehaviorStrategy*>(&tracking_strategy_)
                                 : static_cast<BehaviorStrategy*>(&waypoint_strategy_);

    if (!strategy->ready(snapshot)) {
      return;
    }

    strategy->run(snapshot);
  }

  Eigen::MatrixXd buildInitialState(const Eigen::Vector3d& odom_p,
                                    const Eigen::Vector3d& odom_v,
                                    const ros::Time& stamp, ros::Time& replan_stamp) {
    Eigen::MatrixXd iniState;
    iniState.setZero(3, 3);
    replan_stamp = stamp + ros::Duration(0.03);
    double replan_t = (replan_stamp - replan_stamp_).toSec();
    if (force_hover_ || replan_t > traj_poly_.getTotalDuration()) {
      iniState.col(0) = odom_p;
      iniState.col(1) = odom_v;
    } else {
      iniState.col(0) = traj_poly_.getPos(replan_t);
      iniState.col(1) = traj_poly_.getVel(replan_t);
      iniState.col(2) = traj_poly_.getAcc(replan_t);
    }
    replanStateMsg_.header.stamp = stamp;
    replanStateMsg_.iniState.resize(9);
    Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3) = iniState;
    return iniState;
  }

  void executeTracking(const PlanningSnapshot& snapshot) {
    if (!snapshot.target.fresh) {
      return;
    }

    const auto& odom_msg = snapshot.odom;
    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                           odom_msg.pose.pose.position.y,
                           odom_msg.pose.pose.position.z);
    Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                           odom_msg.twist.twist.linear.y,
                           odom_msg.twist.twist.linear.z);
    Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,
                              odom_msg.pose.pose.orientation.x,
                              odom_msg.pose.pose.orientation.y,
                              odom_msg.pose.pose.orientation.z);

    if (force_hover_ && odom_v.norm() > 0.1) {
      return;
    }

    replanStateMsg_.target = snapshot.target.msg;

    Eigen::Vector3d target_p(snapshot.target.msg.pose.pose.position.x,
                             snapshot.target.msg.pose.pose.position.y,
                             snapshot.target.msg.pose.pose.position.z);
    Eigen::Vector3d target_v(snapshot.target.msg.twist.twist.linear.x,
                             snapshot.target.msg.twist.twist.linear.y,
                             snapshot.target.msg.twist.twist.linear.z);
    Eigen::Quaterniond target_q(snapshot.target.msg.pose.pose.orientation.w,
                                snapshot.target.msg.pose.pose.orientation.x,
                                snapshot.target.msg.pose.pose.orientation.y,
                                snapshot.target.msg.pose.pose.orientation.z);

    if (snapshot.land_trigger_active) {
      if ((target_p - odom_p).norm() < 0.1 && odom_v.norm() < 0.1 && target_v.norm() < 0.2) {
        if (!wait_hover_) {
          pub_hover_p(odom_p, snapshot.stamp);
          wait_hover_ = true;
        }
        ROS_WARN("[planner] HOVERING...");
        return;
      }
      target_p = target_p + target_q * land_p_;
      wait_hover_ = false;
    } else {
      target_p.z() += tracking_height_offset_;
      Eigen::Vector3d dp = target_p - odom_p;
      double desired_yaw = std::atan2(dp.y(), dp.x());
      Eigen::Vector3d project_yaw = odom_q.toRotationMatrix().col(0);
      double now_yaw = std::atan2(project_yaw.y(), project_yaw.x());
      if (std::fabs(dp.norm() - tracking_dist_) < tolerance_d_ && odom_v.norm() < 0.1 &&
          target_v.norm() < 0.2 && std::fabs(desired_yaw - now_yaw) < 0.5) {
        if (!wait_hover_) {
          pub_hover_p(odom_p, snapshot.stamp);
          wait_hover_ = true;
        }
        ROS_WARN("[planner] HOVERING...");
        replanStateMsg_.state = -1;
        replanState_pub_.publish(replanStateMsg_);
        return;
      }
      wait_hover_ = false;
    }

    gridmapPtr_->from_msg(snapshot.map);
    replanStateMsg_.occmap = snapshot.map;
    prePtr_->setMap(*gridmapPtr_);

    if (envPtr_->checkRayValid(odom_p, target_p)) {
      visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::yellow);
    } else {
      visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::red);
    }

    std::vector<Eigen::Vector3d> target_predcit;
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit);
    if (generate_new_traj_success) {
      Eigen::Vector3d observable_p = target_predcit.back();
      visPtr_->visualize_path(target_predcit, "car_predict");
      std::vector<Eigen::Vector3d> observable_margin;
      for (double theta = 0; theta <= 2 * M_PI; theta += 0.01) {
        observable_margin.emplace_back(observable_p +
                                       tracking_dist_ * Eigen::Vector3d(cos(theta), sin(theta), 0));
      }
      visPtr_->visualize_path(observable_margin, "observable_margin");
    }

    ros::Time replan_stamp;
    Eigen::MatrixXd iniState = buildInitialState(odom_p, odom_v, snapshot.stamp, replan_stamp);

    Eigen::Vector3d p_start = iniState.col(0);
    std::vector<Eigen::Vector3d> path, way_pts;

    if (generate_new_traj_success) {
      if (snapshot.land_trigger_active) {
        generate_new_traj_success = envPtr_->short_astar(p_start, target_p, path);
      } else {
        generate_new_traj_success = envPtr_->findVisiblePath(p_start, target_predcit, way_pts, path);
      }
    }

    std::vector<Eigen::Vector3d> visible_ps;
    std::vector<double> thetas;
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar");
      if (snapshot.land_trigger_active) {
        for (const auto& p : target_predcit) {
          path.push_back(p);
        }
      } else {
        target_predcit.pop_back();
        way_pts.pop_back();
        envPtr_->generate_visible_regions(target_predcit, way_pts, visible_ps, thetas);
        visPtr_->visualize_pointcloud(visible_ps, "visible_ps");
        visPtr_->visualize_fan_shape_meshes(target_predcit, visible_ps, thetas, "visible_region");
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
        for (int i = 0; i < static_cast<int>(way_pts.size()); ++i) {
          rays.emplace_back(target_predcit[i], way_pts[i]);
        }
        visPtr_->visualize_pointcloud(way_pts, "way_pts");
        way_pts.insert(way_pts.begin(), p_start);
        envPtr_->pts2path(way_pts, path);
      }

      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      Eigen::MatrixXd finState;
      finState.setZero(3, 3);
      finState.col(0) = path.back();
      finState.col(1) = target_v;
      if (snapshot.land_trigger_active) {
        finState.col(0) = target_predcit.back();
        generate_new_traj_success =
            trajOptPtr_->generate_traj(iniState, finState, target_predcit, hPolys, traj);
      } else {
        generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState, target_predcit,
                                                               visible_ps, thetas, hPolys, traj);
      }
      visPtr_->visualize_traj(traj, "traj");
    }

    bool valid = false;
    if (generate_new_traj_success) {
      valid = validcheck(traj, replan_stamp);
    } else {
      replanStateMsg_.state = -2;
      replanState_pub_.publish(replanStateMsg_);
    }

    if (valid) {
      force_hover_ = false;
      ROS_WARN("[planner] REPLAN SUCCESS");
      replanStateMsg_.state = 0;
      replanState_pub_.publish(replanStateMsg_);
      Eigen::Vector3d dp = target_p + target_v * 0.03 - iniState.col(0);
      double yaw = std::atan2(dp.y(), dp.x());
      if (snapshot.land_trigger_active) {
        yaw = 2 * std::atan2(target_q.z(), target_q.w());
      }
      pub_traj(traj, yaw, replan_stamp);
      traj_poly_ = traj;
      replan_stamp_ = replan_stamp;
    } else if (force_hover_) {
      ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
      replanStateMsg_.state = 1;
      replanState_pub_.publish(replanStateMsg_);
    } else if (validcheck(traj_poly_, replan_stamp_)) {
      force_hover_ = true;
      ROS_FATAL("[planner] EMERGENCY STOP!!!");
      replanStateMsg_.state = 2;
      replanState_pub_.publish(replanStateMsg_);
      pub_hover_p(iniState.col(0), replan_stamp);
    } else {
      ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ...");
      replanStateMsg_.state = 3;
      replanState_pub_.publish(replanStateMsg_);
    }
  }

  void executeWaypoint(const PlanningSnapshot& snapshot) {
    if (!snapshot.has_goal) {
      return;
    }

    const auto& odom_msg = snapshot.odom;
    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                           odom_msg.pose.pose.position.y,
                           odom_msg.pose.pose.position.z);
    Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                           odom_msg.twist.twist.linear.y,
                           odom_msg.twist.twist.linear.z);

    if (force_hover_ && odom_v.norm() > 0.1) {
      return;
    }

    Eigen::Vector3d local_goal;
    Eigen::Vector3d delta = goal_ - odom_p;
    if (delta.norm() < 15.0) {
      local_goal = goal_;
    } else {
      local_goal = delta.normalized() * 15.0 + odom_p;
    }

    gridmapPtr_->from_msg(snapshot.map);
    replanStateMsg_.occmap = snapshot.map;

    double stop_radius = std::max(waypoint_stop_dist_, 0.0);
    bool no_need_replan = false;
    if (!force_hover_ && !wait_hover_) {
      double last_traj_t_rest =
          traj_poly_.getTotalDuration() - (snapshot.stamp - replan_stamp_).toSec();
      bool new_goal =
          (local_goal - traj_poly_.getPos(traj_poly_.getTotalDuration())).norm() > stop_radius;
      if (!new_goal) {
        if (last_traj_t_rest < 1.0) {
          ROS_WARN("[planner] NEAR GOAL...");
          no_need_replan = true;
        } else if (validcheck(traj_poly_, replan_stamp_, last_traj_t_rest)) {
          ROS_WARN("[planner] NO NEED REPLAN...");
          double t_delta = traj_poly_.getTotalDuration() < 1.0 ? traj_poly_.getTotalDuration() : 1.0;
          double t_yaw = (snapshot.stamp - replan_stamp_).toSec() + t_delta;
          Eigen::Vector3d un_known_p = traj_poly_.getPos(t_yaw);
          Eigen::Vector3d dp = un_known_p - odom_p;
          double yaw = std::atan2(dp.y(), dp.x());
          pub_traj(traj_poly_, yaw, replan_stamp_);
          no_need_replan = true;
        }
      }
    }

    if ((goal_ - odom_p).norm() < stop_radius && odom_v.norm() < 0.2) {
      if (!wait_hover_) {
        pub_hover_p(odom_p, snapshot.stamp);
        wait_hover_ = true;
      }
      ROS_WARN("[planner] HOVERING...");
      replanStateMsg_.state = -1;
      replanState_pub_.publish(replanStateMsg_);
      return;
    }

    wait_hover_ = false;

    if (no_need_replan) {
      return;
    }

    ros::Time replan_stamp;
    Eigen::MatrixXd iniState = buildInitialState(odom_p, odom_v, snapshot.stamp, replan_stamp);

    Eigen::Vector3d p_start = iniState.col(0);
    bool need_extra_corridor = iniState.col(1).norm() > 1.0;
    Eigen::MatrixXd hPoly;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> line;
    if (need_extra_corridor) {
      Eigen::Vector3d v_norm = iniState.col(1).normalized();
      line.first = p_start;
      double step = 0.1;
      for (double dx = step; dx < 1.0; dx += step) {
        p_start += step * v_norm;
        if (gridmapPtr_->isOccupied(p_start)) {
          p_start -= step * v_norm;
          break;
        }
      }
      line.second = p_start;
      envPtr_->generateOneCorridor(line, 2.0, hPoly);
    }

    std::vector<Eigen::Vector3d> path;
    bool generate_new_traj_success = envPtr_->astar_search(p_start, local_goal, path);
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar");
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      if (need_extra_corridor) {
        hPolys.insert(hPolys.begin(), hPoly);
        keyPts.insert(keyPts.begin(), line);
      }
      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      Eigen::MatrixXd finState;
      finState.setZero(3, 3);
      finState.col(0) = path.back();
      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState, hPolys, traj);
      visPtr_->visualize_traj(traj, "traj");
    }

    bool valid = false;
    if (generate_new_traj_success) {
      valid = validcheck(traj, replan_stamp);
    } else {
      replanStateMsg_.state = -2;
      replanState_pub_.publish(replanStateMsg_);
    }

    if (valid) {
      force_hover_ = false;
      ROS_WARN("[planner] REPLAN SUCCESS");
      replanStateMsg_.state = 0;
      replanState_pub_.publish(replanStateMsg_);
      Eigen::Vector3d un_known_p =
          traj.getPos(traj.getTotalDuration() < 1.0 ? traj.getTotalDuration() : 1.0);
      Eigen::Vector3d dp = un_known_p - odom_p;
      double yaw = std::atan2(dp.y(), dp.x());
      pub_traj(traj, yaw, replan_stamp);
      traj_poly_ = traj;
      replan_stamp_ = replan_stamp;
    } else if (force_hover_) {
      ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
      replanStateMsg_.state = 1;
      replanState_pub_.publish(replanStateMsg_);
    } else if (!validcheck(traj_poly_, replan_stamp_)) {
      force_hover_ = true;
      ROS_FATAL("[planner] EMERGENCY STOP!!!");
      replanStateMsg_.state = 2;
      replanState_pub_.publish(replanStateMsg_);
      pub_hover_p(iniState.col(0), replan_stamp);
    } else {
      ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ...");
      replanStateMsg_.state = 3;
      replanState_pub_.publish(replanStateMsg_);
    }
  }









  void debug_timer_callback(const ros::TimerEvent& event) {
    inflate_gridmap_pub_.publish(replanStateMsg_.occmap);
    Eigen::MatrixXd iniState;
    iniState.setZero(3, 3);
    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);

    iniState = Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3);
    Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                             replanStateMsg_.target.pose.pose.position.y,
                             replanStateMsg_.target.pose.pose.position.z);
    Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                             replanStateMsg_.target.twist.twist.linear.y,
                             replanStateMsg_.target.twist.twist.linear.z);
    // std::cout << "target_p: " << target_p.transpose() << std::endl;
    // std::cout << "target_v: " << target_v.transpose() << std::endl;

    // visualize the target and the drone velocity
    visPtr_->visualize_arrow(iniState.col(0), iniState.col(0) + iniState.col(1), "drone_vel");
    visPtr_->visualize_arrow(target_p, target_p + target_v, "target_vel");

    // visualize the ray from drone to target
    if (envPtr_->checkRayValid(iniState.col(0), target_p)) {
      visPtr_->visualize_arrow(iniState.col(0), target_p, "ray", visualization::yellow);
    } else {
      visPtr_->visualize_arrow(iniState.col(0), target_p, "ray", visualization::red);
    }

    // NOTE prediction
    std::vector<Eigen::Vector3d> target_predcit;
    if (gridmapPtr_->isOccupied(target_p)) {
      std::cout << "target is invalid!" << std::endl;
      assert(false);
    }
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit);

    if (generate_new_traj_success) {
      Eigen::Vector3d observable_p = target_predcit.back();
      visPtr_->visualize_path(target_predcit, "car_predict");
      std::vector<Eigen::Vector3d> observable_margin;
      for (double theta = 0; theta <= 2 * M_PI; theta += 0.01) {
        observable_margin.emplace_back(observable_p + tracking_dist_ * Eigen::Vector3d(cos(theta), sin(theta), 0));
      }
      visPtr_->visualize_path(observable_margin, "observable_margin");
    }

    // NOTE path searching
    Eigen::Vector3d p_start = iniState.col(0);
    std::vector<Eigen::Vector3d> path, way_pts;
    if (generate_new_traj_success) {
      generate_new_traj_success = envPtr_->findVisiblePath(p_start, target_predcit, way_pts, path);
    }

    std::vector<Eigen::Vector3d> visible_ps;
    std::vector<double> thetas;
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar");
      // NOTE generate visible regions
      target_predcit.pop_back();
      way_pts.pop_back();
      envPtr_->generate_visible_regions(target_predcit, way_pts,
                                        visible_ps, thetas);
      visPtr_->visualize_pointcloud(visible_ps, "visible_ps");
      visPtr_->visualize_fan_shape_meshes(target_predcit, visible_ps, thetas, "visible_region");
      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      // TODO change the final state
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
      for (int i = 0; i < (int)way_pts.size(); ++i) {
        rays.emplace_back(target_predcit[i], way_pts[i]);
      }
      visPtr_->visualize_pointcloud(way_pts, "way_pts");
      way_pts.insert(way_pts.begin(), p_start);
      envPtr_->pts2path(way_pts, path);
      visPtr_->visualize_path(path, "corridor_path");
      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      // NOTE trajectory optimization
      Eigen::MatrixXd finState;
      finState.setZero(3, 3);
      finState.col(0) = path.back();
      finState.col(1) = target_v;

      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState,
                                                             target_predcit, visible_ps, thetas,
                                                             hPolys, traj);
      visPtr_->visualize_traj(traj, "traj");
    }
    if (!generate_new_traj_success) {
      return;
      // assert(false);
    }
    // check
    bool valid = true;
    std::vector<Eigen::Vector3d> check_pts, invalid_pts;
    double t0 = (ros::Time::now() - replan_stamp).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double check_dur = 1.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration();
    for (double t = t0; t < t0 + delta_t; t += 0.1) {
      Eigen::Vector3d p = traj.getPos(t);
      check_pts.push_back(p);
      if (gridmapPtr_->isOccupied(p)) {
        invalid_pts.push_back(p);
      }
    }
    visPtr_->visualize_path(invalid_pts, "invalid_pts");
    visPtr_->visualize_path(check_pts, "check_pts");
    valid = validcheck(traj, replan_stamp);
    if (!valid) {
      std::cout << "invalid!" << std::endl;
    }
  }

  bool validcheck(const Trajectory& traj, const ros::Time& t_start, const double& check_dur = 1.0) {
    double t0 = (ros::Time::now() - t_start).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration();
    for (double t = t0; t < t0 + delta_t; t += 0.01) {
      Eigen::Vector3d p = traj.getPos(t);
      if (gridmapPtr_->isOccupied(p)) {
        return false;
      }
    }
    return true;
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of planning
    int plan_hz = 10;
    nh.getParam("plan_hz", plan_hz);
    nh.getParam("tracking_dur", tracking_dur_);
    nh.getParam("tracking_dist", tracking_dist_);
    nh.getParam("tolerance_d", tolerance_d_);
    if (!nh.getParam("waypoint_stop_dist", waypoint_stop_dist_)) {
      waypoint_stop_dist_ = tracking_dist_ + tolerance_d_;
    }
    nh.getParam("debug", debug_);
    nh.getParam("target_timeout", target_timeout_);
    nh.getParam("target_speed_threshold", target_speed_threshold_);
    nh.getParam("tracking_height_offset", tracking_height_offset_);
    target_speed_threshold_sq_ = target_speed_threshold_ * target_speed_threshold_;

    gridmapPtr_ = std::make_shared<mapping::OccGridMap>();
    envPtr_ = std::make_shared<env::Env>(nh, gridmapPtr_);
    visPtr_ = std::make_shared<visualization::Visualization>(nh);
    trajOptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh);
    prePtr_ = std::make_shared<prediction::Predict>(nh);

    heartbeat_pub_ = nh.advertise<std_msgs::Empty>("heartbeat", 10);
    traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("trajectory", 1);
    replanState_pub_ = nh.advertise<quadrotor_msgs::ReplanState>("replanState", 1);

    if (debug_) {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::debug_timer_callback, this);
      // TODO read debug data from files
      wr_msg::readMsg(replanStateMsg_, ros::package::getPath("planning") + "/../../../debug/replan_state.bin");
      inflate_gridmap_pub_ = nh.advertise<quadrotor_msgs::OccMap3d>("gridmap_inflate", 10);
      gridmapPtr_->from_msg(replanStateMsg_.occmap);
      prePtr_->setMap(*gridmapPtr_);
      std::cout << "plan state: " << replanStateMsg_.state << std::endl;
    } else {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::main_timer_callback, this);
    }
    gridmap_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1, &Nodelet::gridmap_callback, this, ros::TransportHints().tcpNoDelay());
    odom_sub_ = nh.subscribe<nav_msgs::Odometry>("odom", 10, &Nodelet::odom_callback, this, ros::TransportHints().tcpNoDelay());
    target_sub_ = nh.subscribe<nav_msgs::Odometry>("target", 10, &Nodelet::target_callback, this, ros::TransportHints().tcpNoDelay());
    triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, &Nodelet::triger_callback, this, ros::TransportHints().tcpNoDelay());
    land_triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("land_triger", 10, &Nodelet::land_triger_callback, this, ros::TransportHints().tcpNoDelay());
    ROS_WARN("Planning node initialized!");
  }

 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&Nodelet::init, this, nh));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


bool planning::Nodelet::TrackingStrategy::ready(const PlanningSnapshot& snapshot) const {
    if (!snapshot.target.fresh) {
        return false;
    }
    if (snapshot.land_trigger_active) {
        return true;
    }
    return snapshot.target.speed_sq >= node_.target_speed_threshold_sq_;
}

// 2. 实现 TrackingStrategy 的 run 函数
void planning::Nodelet::TrackingStrategy::run(const PlanningSnapshot& snapshot) {
    node_.executeTracking(snapshot);
}

// 3. 实现 WaypointStrategy 的 ready 函数
bool planning::Nodelet::WaypointStrategy::ready(const PlanningSnapshot& snapshot) const {
    return snapshot.has_goal;
}

// 4. 实现 WaypointStrategy 的 run 函数
void planning::Nodelet::WaypointStrategy::run(const PlanningSnapshot& snapshot) {
    node_.executeWaypoint(snapshot);
}


}  // namespace planning

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(planning::Nodelet, nodelet::Nodelet);
