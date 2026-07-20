# kinova-tele

Quest passthrough teleoperation for a Kinova Gen3 using the Kortex Python API.

## Controls

- Hold the right index trigger and move the controller to translate the robot.
- While holding the trigger, move the right thumbstick left/right to change yaw.
- Use the side squeeze for proportional gripper position; A/B step closed/open.
- On desktop, press `E` to close or `Q` to open the gripper by 10% per
  keypress. The first step starts from measured cyclic gripper feedback.
- Double-tap the index trigger to return to the joint configuration captured
  when the server started. Press the trigger again to cancel auto-home.
- Set `XR_TRANSLATION_GAIN` near the top of `main.py` to map small hand
  displacement to larger robot displacement, then restart the server. For
  example, `3.0` maps 2 cm of Quest movement to a 6 cm robot target.

Movement gain changes target distance, not the robot's configured velocity
limit. XR target offsets are capped at 15 cm per trigger engagement, Cartesian
velocity remains capped at 0.15 m/s per axis, and X/Y commands still pass
through the centered 4 ft by 4 ft application workspace limiter.

## Auto-home behavior

Auto-home stops live Cartesian commands, waits for three near-zero joint-speed
samples, validates a conservative timed joint waypoint, and then executes it.
The headset debug overlay and terminal report the exact Kortex validation or
abort reason if the action fails. An unchanged robot is treated as already
home rather than as a trajectory error.

Auto-home moves every arm joint and is not collision-aware. Configure Kortex
protection zones, keep the area clear, and keep an emergency stop available.

## Run

Set the credentials and start the server from the repository root:

```bash
export KINOVA_USERNAME="your-user"
export KINOVA_PASSWORD="your-password"
python main.py
```

Run offline tests without connecting to the robot:

```bash
python -m unittest discover -s tests -v
```
