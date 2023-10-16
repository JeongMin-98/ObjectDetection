// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file LaneKalmanFilter.hpp
 * @author 김대완 (https://github.com/dawan0111)
 * @brief LaneKalmanFilter Class header file
 * @date 2023-06-26
 */
#ifndef LANE_KALMANFILTER_HPP_
#define LANE_KALMANFILTER_HPP_

#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <memory>

namespace Xycar {
/**
 * @brief LaneKalmanFilter Controller Class
 * @tparam PREC Precision of data
 */

template <typename PREC>
class LaneKalmanFilter
{
public:
    using Ptr = std::unique_ptr<LaneKalmanFilter>; ///< Pointer type of this class

    LaneKalmanFilter(const Eigen::Vector2d& x, const Eigen::Matrix2d& P, const Eigen::Matrix2d& F, const Eigen::Matrix2d& H, const Eigen::Matrix2d& Q, const Eigen::Matrix2d& R,
                     const Eigen::Matrix2d& B)
        : mX(x), mP(P), mF(F), mH(H), mQ(Q), mR(R), mB(B){};
    void predict(const Eigen::Vector2d& u);
    void update(const Eigen::Vector2d& z);

    void set(const Eigen::Vector2d& x) { mX = x; };

    Eigen::Vector2d getState() { return mX; }

private:
    Eigen::Vector2d mX; // 상태 벡터
    Eigen::Matrix2d mB; // inpupt space trnasform matrix
    Eigen::Matrix2d mP; // 공분산 행렬
    Eigen::Matrix2d mF; // 상태 전이 모델
    Eigen::Matrix2d mH; // 측정 모델
    Eigen::Matrix2d mQ; // 공정 잡음의 공분산 행렬
    Eigen::Matrix2d mR; // 측정 잡음의 공분산 행렬
};
} // namespace Xycar
#endif // LANE_KALMANFILTER_HPP_
