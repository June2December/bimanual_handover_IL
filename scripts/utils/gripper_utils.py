def grip_to_state(grip_val, close_thresh=0.43):
    """
    gripper 실제 joint 값 -> discrete state
    open=0.0, closed=1.0
    """
    return 1.0 if grip_val >= close_thresh else 0.0


def grip_to_cmd(grip_val, close_thresh=0.43):
    """
    logging용 gripper command를 discrete로 저장
    """
    return 1.0 if grip_val >= close_thresh else 0.0
