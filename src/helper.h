#ifndef HELPER_
#define HELPER_
#include "Eigen/Dense"
#include <cmath>

namespace helper {
    void Normalize(double & value);

    template<class T>
    void init(T & t) {
        t.fill(0);
    }

    void BuildStateVector(const Eigen::VectorXd & x,
            const double delta,
            Eigen::VectorXd & state) {
        //std::cout << "BuildStateVector -- x = \n" << x << std::endl;
        //std::cout << "delta = \n" << delta << std::endl;
        double psi_rate = x(4), psi = x(3), v = x(2);
        if (fabs(psi_rate) > 0.001) {
            double a = v/psi_rate;
            state << a * (sin(psi + psi_rate * delta) - sin(psi)),
                  a * (-cos(psi + psi_rate * delta) + cos(psi)),
                  0,
                  psi_rate * delta,
                  0;
        } else {
            state << v * cos(psi) * delta,
                  v * sin(psi) * delta,
                  0,
                  psi_rate * delta,
                  0;
        }
        //std::cout << "state = \n" << state << std::endl;
    }

    void BuildNoiseVector(const Eigen::VectorXd & x,
            const double delta,
            Eigen::VectorXd & noise) {
        //std::cout << "BuildNoiseVector -- x = \n" << x << std::endl;
        //std::cout << "noise initial = \n" << noise << std::endl;
        double vak = x(5), vpsik = x(6), psi = x(3);
        noise << 0.5 * delta * delta * cos(psi) * vak,
              0.5 * delta * delta * sin(psi) * vak,
              delta * vak,
              0.5 * delta* delta * vpsik,
              delta * vpsik;
        //std::cout << "noise = \n" << noise << std::endl;
    }
    //use Kalman filter to merge state with the measurements.
    void mergeMeasurementsWithState(const Eigen::MatrixXd & Xsig_pred,
            const Eigen::MatrixXd & Zsig,
            const Eigen::MatrixXd & S,
            const Eigen::VectorXd & z_pred,
            const Eigen::VectorXd & z,
            const Eigen::VectorXd & weights,
            const bool isRadar,
            Eigen::VectorXd & x,
            Eigen::MatrixXd & P) {
        std::cout << "mergeMeasurementsWithState\n" ;
        std::cout << "Xsig_pred = \n" << Xsig_pred << std::endl;
        std::cout << "Zsig = \n" << Zsig << std::endl;
        std::cout << "S = \n" << S << std::endl;
        std::cout << "z_pred = \n" << z_pred << std::endl;
        std::cout << "z = \n" << z << std::endl;
        std::cout << "weights = \n" << weights << std::endl;
        std::cout << "x = \n" << x << std::endl;
        std::cout << "P = \n" << P << std::endl;
        Eigen::MatrixXd xdiff = Xsig_pred;
        Eigen::MatrixXd zdiff = Zsig;
        for (int i = 0; i < Xsig_pred.cols(); i ++) {
            xdiff.col(i) -= x;
            //xdiff.col(i)(3) = Normalize(xdiff.col(i)(3));
            Normalize(xdiff.col(i)(3));
        }
        for (int i = 0; i < Zsig.cols(); i ++) {
            zdiff.col(i) -= z_pred;
            //zdiff.col(i)(1) = Normalize(zdiff.col(i)(1));
            if (isRadar) {
                Normalize(zdiff.col(i)(1));
            }
        }
        //build cross correlation matrix
        Eigen::MatrixXd T = xdiff * weights.asDiagonal() * zdiff.transpose();
        //build K
        Eigen::MatrixXd K = T * S.inverse();
        //update state
        Eigen::VectorXd innovation = z - z_pred;
        Normalize(innovation(1));
        x += K * innovation;
        //update covariance
        P += -K * S * K.transpose();
        //std::cout << weights << std::endl;
        //std::cout << "x = \n" << x << std::endl;
    }

    Eigen::VectorXd convertToRadarMeasurementSpace(const Eigen::VectorXd & x) { 
        Eigen::VectorXd z(3);
        double px = x(0), py = x(1), v = x(2), yaw = x(3); //not -used: yawr = x(4);
        double rho = sqrt(px*px + py*py);
        double phi = std::atan2(py, px);
        //calculate rhodot only if rho != 0
        double rhodot = fabs(rho) > 1e-3 ? 
            (px * cos(yaw) * v + py * sin(yaw) * v)/rho : 0;
        z << rho, phi, rhodot;
        return z;
    }

    Eigen::VectorXd convertToLidarMeasurementSpace(const Eigen::VectorXd & x) {
        Eigen::VectorXd z(2);
        z.fill(0.0);
        z.head(2) = x.head(2);
        return z;
    }

    void deduceNormalDistributionParams(const Eigen::MatrixXd & Zsig,
            const Eigen::VectorXd & weights,
            const Eigen::MatrixXd & R,
            const int entryToNormalize,
            Eigen::VectorXd & z_pred, Eigen::MatrixXd & S) {
        /*std::cout << "deduceRadarNormalDistributionParams: " << std::endl;
        std::cout << "Zsig = \n" << Zsig << std::endl;
        std::cout << "weights = \n" << weights << std::endl;
        std::cout << "R = \n" << R << std::endl;
        */
        z_pred = Zsig * weights;
        Eigen::MatrixXd diff = Zsig;
        for (int i = 0; i < diff.cols(); i ++) {
            diff.col(i) -= z_pred;
            //diff.col(i)(1) = Normalize(diff.col(i)(1));
            if (entryToNormalize != -1) {
                Normalize(diff.col(i)(entryToNormalize));
            }
        }
        //std::cout << "diff = \n" << diff << std::endl;
        S = diff * weights.asDiagonal() * diff.transpose() + R;
    }

    void Normalize(double & value) {
        std::cout << "Normalize called with value = " << value << std::endl;
        if (value >= -M_PI && value <= M_PI) {
            return;
        }
        int sign = value > M_PI ? -1 : +1;
        value += sign * 2 * M_PI;
        Normalize(value);
        /*
        while (x(col) > M_PI) {
            x(col) -= 2 * M_PI;
        }
        while (x(col) < -M_PI) {
        }
        double rv = theta;
        while (rv > M_PI) {
            rv -= 2 * M_PI;
        }
        while (rv < -M_PI) {
            rv += 2 * M_PI;
        }
        return rv;
        */
    }

};
#endif
