#include "PID.h"
#include <numeric>
#include <vector>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    current_cte = 0;
    prev_cte = 0;
}

void PID::UpdateError(double cte) {
    current_cte += cte;
    p_error = - Kp * cte;
    i_error = - Ki * current_cte;
    d_error = - Kd * (cte - prev_cte);
    prev_cte = cte;
}

double PID::TotalError() {
  return p_error + i_error + d_error;
}

void PID::Twiddle(double cte){
}
