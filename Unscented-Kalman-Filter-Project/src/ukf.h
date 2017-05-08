#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  MatrixXd R_radar_;

  MatrixXd R_laser_;

  ///* for Zpred transformation
  MatrixXd H_laser_;

  ///* time when the state is true, in us
  long long previous_timestamp_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const Eigen::MatrixXd& Xsig_pred,
                   const Eigen::MatrixXd& Zsig,
                   const Eigen::VectorXd& z_pred,
                   const Eigen::MatrixXd& S,
                   MeasurementPackage reading);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const Eigen::MatrixXd& Xsig_pred,
                   const Eigen::MatrixXd& Zsig,
                   const Eigen::VectorXd& z_pred,
                   const Eigen::MatrixXd& S,
                   MeasurementPackage reading);

  /**
  * Helper function that updates Lidar or Radar data
  */
  void Update(MeasurementPackage meas_package);

  /**
  *Predict the measurement state of the Lidar sensor
  */
  void PredictLidarMeasurement(const MatrixXd& Xsig_pred,
                               VectorXd* z_out,
                               MatrixXd* S_out,
                               MatrixXd* Zsig_out);

  /**
  *Predict the measurement state of the radar sensor
  */
  void PredictRadarMeasurement(const MatrixXd& Xsig_pred,
                                  VectorXd* z_out,
                                  MatrixXd* S_out,
                                  MatrixXd* Zsig_out);
 /** Helper functions for the prediction function: Makes code
 significantly more readable and easier to debug
**/
  void PredictSigmaPoints(const MatrixXd& Xsig_aug,
                          const double dt,
                          MatrixXd* Xsig_out);

  void PredictMeanAndCovariance(const Eigen::MatrixXd& Xsig_pred);

  void GenerateSigmaPoints(MatrixXd* Xsig_out);

};

#endif /* UKF_H */
