#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "helper.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd::Identity(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1;//3; //1;//30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5; //0.6;//0.2;//2*M_PI/400;//30;


    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values 
     */

    /**
     * TODO: Complete the initialization. See ukf.h for other member properties.
     * Hint: one or more values initialized above might be wildly off...
     */
    initialize();
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(const MeasurementPackage & meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */
    void (UKF::*fns[])(const MeasurementPackage &) = {&UKF::UpdateLidar, &UKF::UpdateRadar};
    bool types[] {
        meas_package.sensor_type_ == MeasurementPackage::LASER,
            meas_package.sensor_type_ == MeasurementPackage::RADAR};

    for (int i = 0; i < 2; i ++) {
        if (types[i]) {
            (this->*(fns[i]))(meas_package);
        }
    }
}

void UKF::Prediction(const double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location. 
     * Modify the state vector, x_. Predict sigma points, the state, 
     * and the state covariance matrix.
     */
    UpdatePredictedState(delta_t);
}

/**
 * pre-condition: the augmented matrix is already built.
 * post-condition: the state and covariance are updated to the time 
 * listed in time_us_
 */
void UKF::UpdatePredictedState(const double delta_t) {
    std::cout << "UpdatePredictedState(" << delta_t << ")" << std::endl;
    std::cout << "x_ = " << std::endl << x_ << std::endl;
    std::cout << "P_ = " << std::endl << P_ << std::endl;
    Eigen::MatrixXd Xaug(n_aug_, 2 * n_aug_ + 1); 
    helper::init(Xaug);
    Xaug.fill(0);
    //std::cout << "building augmented state\n";
    BuildAugmentedSigmaPoints(&Xaug);
    //std::cout << "predicting sigma points\n";
    PredictSigmaPoints(Xaug, delta_t);
    //std::cout << "predicting mean and covariance\n";
    PredictMeanAndCovariance();
}

void UKF::PredictMeanAndCovariance() {
    Eigen::VectorXd x(n_x_);
    Eigen::MatrixXd P(n_x_, n_x_);
    x.fill(0);
    P.fill(0);
    //std::cout << "weights = \n" << weights_ << std::endl;
    Eigen::MatrixXd diff = Xsig_pred_;
    //std::cout << "Xsig_pred_ = \n" << Xsig_pred_ << std::endl;
    x = Xsig_pred_ * weights_;
    for (int i = 0; i < 2 * n_aug_ + 1; i ++) {
        /*
           while (x(3) > M_PI) {
           x(3) -= 2 * M_PI;
           }
           while (x(3) < -M_PI) {
           x(3) += 2 * M_PI;
           }
           */
        //x(3) = helper::ToWithIn2PI(x(3));
        //helper::Normalize(x(3));
        diff.col(i) -= x;
        //diff.col(i)(3) = helper::ToWithIn2PI(diff.col(i)(3));
        //helper::Normalize(diff.col(i)(3));
    }
    P += diff * weights_.asDiagonal() * diff.transpose();
    x_ = x;
    P_ = P;
    std::cout << "Predicted Mean And Covariance: " << std::endl;
    std::cout << "x_ = " << std::endl << x_ << std::endl;
    std::cout << "P_ = " << std::endl << P_ << std::endl;
}

//build augmented sigma points
void UKF::BuildAugmentedSigmaPoints(Eigen::MatrixXd* Xaug) {
    Eigen::VectorXd x_aug(n_aug_);
    Eigen::MatrixXd P_aug(n_aug_, n_aug_);
    //Augmented Sigma point matrix
    Eigen::MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
    x_aug.fill(0);
    P_aug.fill(0);
    Xsig_aug.fill(0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
    for (int i = 0; i < 2 * n_aug_ + 1; i ++) {
        Xsig_aug.col(i).head(n_x_) = x_;
    }
    double scale = sqrt(lambda_ + n_aug_);
    Eigen::MatrixXd sqrtMatrix = P_aug.llt().matrixL();
    for (int i = 1; i <= n_aug_; i ++) {
        Eigen::VectorXd vector = scale * sqrtMatrix.col(i - 1);
        Eigen::VectorXd xv = Xsig_aug.col(0);
        Xsig_aug.col(i).head(vector.size()) += vector;
        Xsig_aug.col(i + n_aug_).head(vector.size()) += -vector;
    }
    for (int i = 0; i < 2 * n_aug_ + 1; i ++) {
        //Xsig_aug.col(i)(3) = helper::ToWithIn2PI(Xsig_aug.col(i)(3));
        //helper::Normalize(Xsig_aug.col(i)(3));
    }
    *Xaug = Xsig_aug;
}

void UKF::PredictSigmaPoints(const Eigen::MatrixXd & Xaug,
        const double delta_t) {
    Eigen::MatrixXd Xsig_pred(n_x_, 2 * n_aug_ + 1);
    Xsig_pred = Xaug.topLeftCorner(n_x_, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; i ++) {
        const Eigen::VectorXd & x = Xaug.col(i);
        Eigen::VectorXd noise(n_x_), state(n_x_);
        helper::BuildNoiseVector(x, delta_t, noise);
        helper::BuildStateVector(x, delta_t, state);
        Xsig_pred.col(i) += state + noise;
        //Xsig_pred.col(i)(3) = helper::ToWithIn2PI(Xsig_pred.col(i)(3));
        //helper::Normalize(Xsig_pred.col(i)(3));
    }
    Xsig_pred_ = Xsig_pred;
}

void UKF::UpdateStateToMeasurementTime(
        const MeasurementPackage & meas_package) {
    std::cout << "UpdateStateToMeasurementTime called\n";
    std::cout << "meas_package.time_stamp = " << meas_package.timestamp_ << std::endl;
    std::cout << "previous_measurement_time = " << time_us_ << std::endl;
    //bring the state up to speed.
    double delta_in_ms = 1e-6 * 
        (meas_package.timestamp_ - time_us_);
    time_us_ = meas_package.timestamp_;
    Prediction(delta_in_ms);
}

void UKF::UpdateLidar(const MeasurementPackage & meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief 
     * about the object's position. Modify the state vector, x_, and 
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    const Eigen::VectorXd & z= meas_package.raw_measurements_;
    std::cout << "lidar called with z = \n" << z << std::endl;
    if (!is_initialized_) {
        std::cout << "initializing lidar...\n";
        x_.fill(0.0);
        x_.head(z.size()) = z;
        std::cout << "x_ is now \n" << x_ << std::endl;
        is_initialized_ = true;
        return;
    } 
    if (!use_laser_) {
        return;
    }
    Eigen::VectorXd z_pred(n_lidar_z);
    Eigen::MatrixXd S(n_lidar_z, n_lidar_z);
    Eigen::MatrixXd Zsig(n_lidar_z, 2 * n_aug_ + 1);
    UpdateStateToMeasurementTime(meas_package);
    PredictLidarMeasurement(z_pred, S, Zsig);
    UpdateStateWithData(z, z_pred, S, Zsig, false);
    time_us_ = meas_package.timestamp_;
}

void UKF::UpdateRadar(const MeasurementPackage & meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief 
     * about the object's position. Modify the state vector, x_, and 
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    //std::cout << "update RADAR has been called. \n";
    const Eigen::VectorXd & z = meas_package.raw_measurements_;
    std::cout << "radar received z = \n" << z << std::endl;
    if (!is_initialized_) {
        std::cout << "initializing radar...\n";
        double rho = z[0], phi = z[1];
        x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
        is_initialized_ = true;
        //Prediction(0);
        return;
    }
    if (!use_radar_) {
        return;
    }
    Eigen::VectorXd z_pred(n_radar_z);
    Eigen::MatrixXd S(n_radar_z, n_radar_z);
    Eigen::MatrixXd Zsig(n_radar_z, 2 * n_aug_ + 1);
    UpdateStateToMeasurementTime(meas_package);
    PredictRadarMeasurement(z_pred, S, Zsig);
    UpdateStateWithData(z, z_pred, S, Zsig, true);
    time_us_ = meas_package.timestamp_;
}

void UKF::PredictLidarMeasurement(Eigen::VectorXd & z_pred,
        Eigen::MatrixXd & S,
        Eigen::MatrixXd & Zsig) {
    for (int i = 0; i < 2 *n_aug_  + 1; i ++) {
        Eigen::VectorXd z = helper::convertToLidarMeasurementSpace(Xsig_pred_.col(i));
        Zsig.col(i) = z;
    }
    helper::deduceNormalDistributionParams(Zsig, weights_, 
            lidar_R, -1, z_pred, S);
}

void UKF::PredictRadarMeasurement(Eigen::VectorXd & z_pred,
        Eigen::MatrixXd & S,
        Eigen::MatrixXd & Zsig) {
    //std::cout << "x_ = \n" << x_ << std::endl;
    //std::cout << "PRM: Xsig_pred_ = \n" << Xsig_pred_ << std::endl;
    for (int i = 0; i < 2 *n_aug_  + 1; i ++) {
        Eigen::VectorXd z = helper::convertToRadarMeasurementSpace(Xsig_pred_.col(i));
        Zsig.col(i) = z;
    }
    //std::cout << "PRM: Zsig = \n" << Zsig << std::endl;
    helper::deduceNormalDistributionParams(Zsig, weights_, 
            radar_R, 1, z_pred, S);
    //std::cout << "x_ = \n" << x_ << std::endl;
}

void UKF::UpdateStateWithData(const Eigen::VectorXd & z,
        const Eigen::VectorXd & z_pred,
        const Eigen::MatrixXd & S,
        const Eigen::MatrixXd & Zsig,
        const bool isRadar) {
    helper::mergeMeasurementsWithState(Xsig_pred_,
            Zsig, S, z_pred, z, weights_, isRadar, x_, P_);
}

void UKF::initialize() {
    n_x_ = 5;
    n_aug_ = 7;
    n_radar_z = 3;
    n_lidar_z = 2;
    lambda_ = 3 - n_aug_;
    time_us_ = 0;
    Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);
    Xsig_pred_.fill(0);
    initializeWeights();
    radar_R = Eigen::MatrixXd(n_radar_z, n_radar_z);
    radar_R.fill(0);
    radar_R(0, 0) = std_radr_ * std_radr_;
    radar_R(1, 1) = std_radphi_ * std_radphi_;
    radar_R(2, 2) = std_radrd_ * std_radrd_;
    lidar_R = Eigen::MatrixXd(n_lidar_z, n_lidar_z);
    lidar_R.fill(0);
    lidar_R(0, 0) = std_laspx_ * std_laspx_;
    lidar_R(1, 1) = std_laspy_ * std_laspy_;
    is_initialized_ = false;
    //std::cout << "radar_R = \n " << radar_R << std::endl;
}

void UKF::initializeWeights() {
    weights_ = Eigen::VectorXd(2 * n_aug_ + 1);
    double factor = 1/(lambda_ + n_aug_);
    weights_(0) = lambda_ * factor;
    for (int i = 1; i < weights_.size(); i ++) {
        weights_(i) = 0.5 * factor;
    }
}
