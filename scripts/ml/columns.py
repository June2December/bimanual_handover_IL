PHASE_COLS = [f"phase_{i}" for i in range(12)]

STATE_CONT_COLS = [
    "left_pos_x", "left_pos_y", "left_pos_z",
    "left_rot6d_0", "left_rot6d_1", "left_rot6d_2", "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
    "right_pos_x", "right_pos_y", "right_pos_z",
    "right_rot6d_0", "right_rot6d_1", "right_rot6d_2", "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
    "obj_pos_x", "obj_pos_y", "obj_pos_z",
    "obj_rot6d_0", "obj_rot6d_1", "obj_rot6d_2", "obj_rot6d_3", "obj_rot6d_4", "obj_rot6d_5",
]

STATE_BIN_COLS = [
    "left_grip_state",
    "right_grip_state",
]

ARM_ACTION_CONT_COLS = [
    "action_left_pos_x", "action_left_pos_y", "action_left_pos_z",
    "action_left_rot6d_0", "action_left_rot6d_1", "action_left_rot6d_2",
    "action_left_rot6d_3", "action_left_rot6d_4", "action_left_rot6d_5",
    "action_right_pos_x", "action_right_pos_y", "action_right_pos_z",
    "action_right_rot6d_0", "action_right_rot6d_1", "action_right_rot6d_2",
    "action_right_rot6d_3", "action_right_rot6d_4", "action_right_rot6d_5",
]

GRIP_ACTION_BIN_COLS = [
    "action_left_grip_cmd",
    "action_right_grip_cmd",
]

STATE_COLS = STATE_CONT_COLS + STATE_BIN_COLS + PHASE_COLS