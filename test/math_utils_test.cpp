/*
 *This file is part of msckf_vio
 *
 *    msckf_vio is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    msckf_vio is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with msckf_vio.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <msckf_vio/math_utils.hpp>

using namespace std;
using namespace Eigen;
using namespace msckf_vio;

TEST(MathUtilsTest, skewSymmetric) {
  Vector3d w(1.0, 2.0, 3.0);
  Matrix3d w_hat = skewSymmetric(w);
  Vector3d zero_vector = w_hat * w;

  FullPivLU<Matrix3d> lu_helper(w_hat);
  EXPECT_EQ(lu_helper.rank(), 2);
  EXPECT_DOUBLE_EQ(zero_vector.norm(), 0.0);
  return;
}

TEST(MathUtilsTest, quaternionNormalize) {
  Vector4d q(1.0, 1.0, 1.0, 1.0);
  quaternionNormalize(q);

  EXPECT_DOUBLE_EQ(q.norm(), 1.0);
  return;
}

TEST(MathUtilsTest, quaternionToRotation) {
  Vector4d q(0.0, 0.0, 0.0, 1.0);
  Matrix3d R = quaternionToRotation(q);
  Matrix3d zero_matrix = R - Matrix3d::Identity();

  FullPivLU<Matrix3d> lu_helper(zero_matrix);
  EXPECT_EQ(lu_helper.rank(), 0);
  return;
}

TEST(MathUtilsTest, rotationToQuaternion) {
  Vector4d q1(0.0, 0.0, 0.0, 1.0);
  Matrix3d I = Matrix3d::Identity();
  Vector4d q2 = rotationToQuaternion(I);
  Vector4d zero_vector = q1 - q2;

  EXPECT_DOUBLE_EQ(zero_vector.norm(), 0.0);
  return;
}

TEST(MathUtilsTest, quaternionMultiplication) {
  Vector4d q1(2.0, 2.0, 1.0, 1.0);
  Vector4d q2(1.0, 2.0, 3.0, 1.0);
  q1 = q1 / q1.norm();
  q2 = q2 / q2.norm();
  Vector4d q_prod = quaternionMultiplication(q1, q2);

  Matrix3d R1 = quaternionToRotation(q1);
  Matrix3d R2 = quaternionToRotation(q2);
  Matrix3d R_prod = R1 * R2;
  Matrix3d R_prod_cp = quaternionToRotation(q_prod);

  Matrix3d zero_matrix = R_prod - R_prod_cp;

  EXPECT_NEAR(zero_matrix.sum(), 0.0, 1e-10);
  return;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
