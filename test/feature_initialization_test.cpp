/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <gtest/gtest.h>
#include <random_numbers/random_numbers.h>

#include <msckf_vio/cam_state.h>
#include <msckf_vio/feature.hpp>


using namespace std;
using namespace Eigen;
using namespace msckf_vio;

// Static member variables in CAMState class
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class
Feature::OptimizationConfig Feature::optimization_config;

TEST(FeatureInitializeTest, sphereDistribution) {
  // Set the real feature at the origin of the world frame.
  Vector3d feature(0.5, 0.0, 0.0);

  // Add 6 camera poses, all of which are able to see the
  // feature at the origin. For simplicity, the six camera
  // view are located at the six intersections between a
  // unit sphere and the coordinate system. And the z axes
  // of the camera frames are facing the origin.
  vector<Isometry3d> cam_poses(6);
  // Positive x axis.
  cam_poses[0].linear() << 0.0,  0.0, -1.0,
    1.0,  0.0,  0.0, 0.0, -1.0,  0.0;
  cam_poses[0].translation() << 1.0,  0.0,  0.0;
  // Positive y axis.
  cam_poses[1].linear() << -1.0,  0.0,  0.0,
     0.0,  0.0, -1.0, 0.0, -1.0,  0.0;
  cam_poses[1].translation() << 0.0,  1.0,  0.0;
  // Negative x axis.
  cam_poses[2].linear() << 0.0,  0.0,  1.0,
    -1.0,  0.0,  0.0, 0.0, -1.0,  0.0;
  cam_poses[2].translation() << -1.0,  0.0,  0.0;
  // Negative y axis.
  cam_poses[3].linear() << 1.0,  0.0,  0.0,
     0.0,  0.0,  1.0, 0.0, -1.0,  0.0;
  cam_poses[3].translation() << 0.0, -1.0,  0.0;
  // Positive z axis.
  cam_poses[4].linear() << 0.0, -1.0,  0.0,
    -1.0,  0.0,  0.0, 0.0, 0.0, -1.0;
  cam_poses[4].translation() << 0.0,  0.0,  1.0;
  // Negative z axis.
  cam_poses[5].linear() << 1.0,  0.0,  0.0,
     0.0,  1.0,  0.0, 0.0,  0.0,  1.0;
  cam_poses[5].translation() << 0.0,  0.0, -1.0;

  // Set the camera states
  CamStateServer cam_states;
  for (int i = 0; i < 6; ++i) {
    CAMState new_cam_state;
    new_cam_state.id = i;
    new_cam_state.time = static_cast<double>(i);
    new_cam_state.orientation = rotationToQuaternion(
        Matrix3d(cam_poses[i].linear().transpose()));
    new_cam_state.position = cam_poses[i].translation();
    cam_states[new_cam_state.id] = new_cam_state;
  }

  // Compute measurements.
  random_numbers::RandomNumberGenerator noise_generator;
  vector<Vector4d, aligned_allocator<Vector4d> > measurements(6);
  for (int i = 0; i < 6; ++i) {
    Isometry3d cam_pose_inv = cam_poses[i].inverse();
    Vector3d p = cam_pose_inv.linear()*feature + cam_pose_inv.translation();
    double u = p(0) / p(2) + noise_generator.gaussian(0.0, 0.01);
    double v = p(1) / p(2) + noise_generator.gaussian(0.0, 0.01);
    //double u = p(0) / p(2);
    //double v = p(1) / p(2);
    measurements[i] = Vector4d(u, v, u, v);
  }

  for (int i = 0; i < 6; ++i) {
    cout << "pose " << i << ":" << endl;
    cout << "orientation: " << endl;
    cout << cam_poses[i].linear() << endl;
    cout << "translation: "  << endl;
    cout << cam_poses[i].translation().transpose() << endl;
    cout << "measurement: " << endl;
    cout << measurements[i].transpose() << endl;
    cout << endl;
  }

  // Initialize a feature object.
  Feature feature_object;
  for (int i = 0; i < 6; ++i)
    feature_object.observations[i] = measurements[i];

  // Compute the 3d position of the feature.
  feature_object.initializePosition(cam_states);

  // Check the difference between the computed 3d
  // feature position and the groud truth.
  cout << "ground truth position: " << feature.transpose() << endl;
  cout << "estimated position: " << feature_object.position.transpose() << endl;
  Eigen::Vector3d error = feature_object.position - feature;
  EXPECT_NEAR(error.norm(), 0, 0.05);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

