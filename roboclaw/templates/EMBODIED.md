ALWAYS use the embodied tool for any robot, arm, serial, USB, motor, camera, or hardware question.
NEVER use exec to inspect /dev, serial devices, or raw hardware paths.
ALWAYS start hardware questions by calling embodied(action="setup_show").
ALWAYS use embodied(action="identify") when the user wants to connect or name arms.
ALWAYS follow this workflow order: identify -> calibrate -> teleoperate -> record -> train -> run_policy.
ALWAYS refer to arms by aliases from setup. NEVER expose raw /dev paths to the user.
ALWAYS suggest the next workflow step after completing the current step.
ALWAYS treat calibrate as calibrating every uncalibrated arm automatically.
ALWAYS use follower_names and leader_names with arm aliases for teleoperate and record.
ALWAYS use structured setup actions (set_arm, remove_arm, set_camera, remove_camera) to change config.
NEVER ask the user to type raw serial device paths when setup already has scanned ports.
For record: provide dataset_name, task description, and num_episodes.
For train: provide dataset_name and optionally steps and device.
For run_policy: provide checkpoint_path or let it auto-detect the latest.
