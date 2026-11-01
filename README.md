# 41014_integrated_project

Application code is in `scripts/`. Separate diagnostic experiments are in `testing/`.

`scripts/main.py` runs the paperclip controller diagnostic using ground-truth
pose feedback. This isolates tracking and actuation; it is not the assessed
estimated-state pipeline. Experiment settings are at the top of `main.py`.

From the project folder:

```powershell
.\.venv\Scripts\python.exe scripts/main.py
```

Add `--headless` to skip the viewer, or `--max-time 10` to limit simulated seconds.
Wheel PI runs every physics step; LQR runs at `CONTROL_DT`. The simulation
continues for two seconds after the endpoint stop request, then prints results.

Each run also saves one combined controller figure, numeric `.npz` data and
JSON metrics under `figures/controller_tracking/run_NNN/`. New run indices
preserve previous results. Plots include desired/actual path, cross-track and
heading errors, body speed/yaw rate, wheel speeds and requested/applied torque.
Wheel effort is recorded every physics step to capture PI transients.

To replot saved data without simulating again:

```powershell
.\.venv\Scripts\python.exe scripts/plot_controller_results.py --input figures/controller_tracking/run_001/run_data_001.npz
```

This also creates a new indexed results folder. Add `--no-show` to save only.

The previous robot wrapper is preserved in `scripts/legacy_robot.py`.
The obsolete goal controller and its legacy runner have been removed.
Use the motion experiment below to test the working wheel loop first.

## Pipeline skeleton

- `scripts/robot.py`: simulation, timestamped sensors and body-command/wheel PI boundary.
- `scripts/models.py`: pure motion models and tracking-error linearisation.
- `scripts/estimator.py`: encoder-camera pose EKF and covariance.
- `scripts/controller.py`: LQR controller and tracking-error calculation.
- `scripts/wheel_controller.py`: existing working wheel PI, unchanged.
- `scripts/paperclip_path.py`: five-segment paperclip geometry and progression.

Preview the same path geometry used by the robot reference:

```powershell
.\.venv\Scripts\python.exe scripts/paperclip_path.py
```

Edit the three straight lengths and the two turn radii inside
`PaperclipReference.__init__`. Both turns are left semicircles. The final
straight requests deceleration and a stop near its endpoint. This reference
does not implement the robot controller or guarantee its stopping accuracy.
- `scripts/evaluation.py`: logs and metrics, with truth reserved for evaluation.

Adapt subject tutorial functions first, then reuse suitable Robotics Toolbox and
supporting library routines. Define shared data formats at each boundary during
implementation; units, frames and intended memory are documented in the stubs.

## Wheel motion experiment

From this project folder, run:

```powershell
.\.venv\Scripts\python.exe testing/motor_control_loop/motion_test.py
```

Both wheels follow this 11-second profile:

| Simulated time | Wheel target |
|---|---|
| 0–1 s | Rest at 0 rad/s |
| 1–3 s | Smooth increase from 0 to 4 rad/s |
| 3–5 s | Hold 4 rad/s |
| 5–8 s | Abrupt increase to 6 rad/s, then hold |
| 8–11 s | Abrupt decrease to 0 rad/s; observe stopping |

The smooth increase has zero slope at its start and end. Edit the named speeds
and times near the top of `testing/motor_control_loop/motion_test.py` to adjust the experiment.

`scripts/wheel_controller.py` contains the reusable `WheelPI` class. Each wheel
has a separate instance. It converts wheel-speed error into torque, limits torque
to the XML actuator range, and prevents integral windup during saturation.
There is no derivative term or derivative filter. Reset the controller when
resetting the robot.

The experiment runs the wheel loop once per physics step, using the model's
simulation timestep. The selected gains are **Kp=0.4 and Ki=0.5**. The test keeps
the master/ratio notation:

```python
MASTER_GAIN = 0.4
I_RATIO = 1.25
```

The effective gains are `Kp = MASTER_GAIN` and `Ki = MASTER_GAIN * I_RATIO`.
Set `I_RATIO=0` for P-only control. These gains are selected for continued
development; the complete navigation pipeline has not yet been validated.

Afterwards, graphs compare targets with measured wheel speeds and show tracking
errors, plus requested and limited motor torque. The experiment uses the existing
`robotis_tb3` model. It runs straight to the graphs by default; add `--viewer`
to watch the robot first. `--headless` is still accepted for existing commands.

Logs stay in a dictionary of lists; no CSV or other log files are written.
The log also includes `pi_gains` metadata holding the actual Kp and Ki used
in that run. All graph titles display these values for comparison.
Each test saves a new numbered set in `testing/motor_control_loop/figures`, for example
`wheel_logs_001.png`, `encoder_positions_001.png`, and `wheel_slip_001.png`.
The next test uses `002`, continuing after the highest existing index. Older
graphs are preserved, including the original unnumbered files.
The encoder-position graph shows
encoder-predicted x/y positions against simulator truth and their position
mismatch over time. Odometry starts at the initial true pose and then uses only
wheel-angle increments. Divergence can indicate slip, but geometry errors can
also cause it; this is not a direct slip measurement.

The wheel-slip graph plots both wheels' instantaneous slip-speed
estimates together: measured wheel rotation times radius minus true forward
ground speed at each side of the axle, in m/s. It accounts for left/right speed
differences during turns and assumes planar motion. Positive means excess
forward wheel surface speed; zero means agreement with the rolling model.
The simulation logs true forward speed and turn rate only for this diagnostic.

The scene defines chassis-floor sliding friction as `0.01` in an explicit
contact pair. This approximates rear caster resistance without changing tyre
traction or chassis friction against other obstacles.

To reuse the experiment or plotter from Python running in the project folder:

```python
import matplotlib.pyplot as plt
from testing.motor_control_loop.motion_test import run_motion_test
from testing.motor_control_loop.plot_wheel_logs import plot_wheel_logs

log = run_motion_test(show_viewer=False)
plot_wheel_logs(log)
plt.show()
```

The plotter also accepts logs collected by your controller. Its required keys are
`time` (seconds), `left_target`, `right_target`, `left_measured`, and
`right_measured` (all wheel speeds in rad/s). Extra fields are ignored.
Optional `left_torque`/`right_torque` fields add a torque row; optional
`left_raw_torque`/`right_raw_torque` fields also show the unclipped requests.

For the position comparison, call
`plot_encoder_positions(log, wheel_radius=0.033, wheel_base=0.288)` from the same
plotting module. It additionally requires `left_angle`, `right_angle`, `x`, `y`
and `heading`. Angles are radians and positions are metres in the world frame.
