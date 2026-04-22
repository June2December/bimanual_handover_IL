import numpy as np

NUM_PHASES = 12

PHASE_NAMES = {
    0: "LEFT_APPROACH_PRE",
    1: "LEFT_APPROACH_GRASP",
    2: "LEFT_SETTLE",
    3: "LEFT_CLOSE",
    4: "LEFT_LIFT",
    5: "LEFT_MOVE_XY_TO_TRANSFER",
    6: "RIGHT_MOVE_TO_RECEIVE",
    7: "RIGHT_CLOSE",
    8: "LEFT_RELEASE",
    9: "LEFT_POST_RELEASE_UP",
    10: "LEFT_GO_HOME",
    11: "DONE",
}


def phase_to_onehot(phase, num_phases=NUM_PHASES, dtype=np.float32):
    onehot = np.zeros(num_phases, dtype=dtype)
    assert 0 <= phase < num_phases, f"invalid phase: {phase}"
    onehot[phase] = 1.0
    return onehot


def phase_feature_names(num_phases=NUM_PHASES):
    return [f"phase_{i}" for i in range(num_phases)]