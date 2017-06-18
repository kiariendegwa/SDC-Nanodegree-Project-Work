# Reflection on PID controller project
## Brief discussion on PID Controller components

### Proportional controller
This adjusts the steering angle only based on the current cross track error(cte) according, by multiplying it by a proportional value. Should we have a linear or simple system that we are trying to control, this value would be adequate enough to adjust and control the system based on the incoming cte.

### Integration controller
This is important in reacting to a bias within the steering angle's cte. Should there be a systematic error signal that is muddling the cte, say caused by a slope in the track or strong wind; this coeffecient is used to stop this.

### Differentiable controller
This is used to stop the steering from overshooting given the current cte. This parameters tries to account for the next value of the cte signal.

## Brief discussion on how final parameters where chosen

Since manual tuning resulted in the car successfully navigating the track, the Twiddle algorithm was not implemented. Values were however picked manually in a scheme similar to that carried out by Twiddle. I.e. adjusting the P parameter until the car could manouvre along the track as far as it could. Then adjusting I and D in a similar manner, to see how this affected the car and then D. This resulted in satisfactory results given the project scope. This PID values were tested out using a relatively low throttle. However increased speeds could have been worked up to using a similar scheme. The final values picked given a throttle of 0.1, are P =0.15, I =0.0, D = 4.15