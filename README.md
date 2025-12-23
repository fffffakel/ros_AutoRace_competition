# ros_AutoRace_competition_пупуни
# AutoRace 2025
A ROS2 metapackage that has necessary packages for AutoRace 2025 challenge.

## Included packages

* `robot_description` - holds the SDF description of the simulated robot, including sensors.

* `referee_console` - holds a referee node that is required for participants to run.

* `robot_bringup` - holds launch files, worlds and multiple configurations that serve as an example and as a required system for AutoRace to work.

* `autorace_core_pypyni` - our robot for race

## Usage for AutoRace 2025

1. Install WS

    ```bash
    cd ~/template_ws    # your workspace folder
    ```

2. Build the project

    ```bash
    colcon build
    ```

3. Source the workspace

    ```bash
    . ~/template_ws/install/setup.bash
    ```

4. Launch the simulation

    ```bash
    ros2 launch robot_bringup autorace_2025.launch.py
    ```

5. Run your own launch file that controls the robot

    ```bash
    ros2 launch autorace_core_pypyni autorace_core.launch.py
    ```

6. Run the referee

    ```bash
    ros2 run referee_console mission_autorace_2025_referee
    ```

**Good luck !**
