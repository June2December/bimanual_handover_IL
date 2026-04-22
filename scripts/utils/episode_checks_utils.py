import numpy as np

def check_max_step(step, max_steps):
    return step >= (max_steps - 1)

def check_object_holder_distance(
    phase,
    left_tcp_pos,
    right_tcp_pos,
    obj_pos,
    left_grasped,
    right_grasped,
    left_max_dist=0.3,
    right_max_dist=0.3,
):
    """
    phase와 grasp 상태를 기준으로
    현재 물체가 있어야 할 팔 TCP 근처에 object가 있는지 확인
    """

    # 아직 grasp 전 구간은 체크 안 함
    if phase in [0, 1, 2]:
        return False

    # 왼팔이 잡고 있어야 하는 구간
    # handover 완료 전까지는 left holder 기준
    if phase in [3, 4, 5, 6, 7]:
        if left_grasped < 0.5:
            return False
        left_dist = np.linalg.norm(obj_pos - left_tcp_pos)
        return left_dist > left_max_dist

    # 오른팔이 넘겨받은 이후 구간
    if phase in [8, 9, 10, 11]:
        if right_grasped < 0.5:
            return False
        right_dist = np.linalg.norm(obj_pos - right_tcp_pos)
        return right_dist > right_max_dist

    return False

def check_stall(progress_hist, stall_window=120, stall_eps=0.002):
    """
    최근 stall_window step 동안 progress 변화량이 거의 없으면 stall 판정
    progress_hist: scalar list
    """
    if len(progress_hist) < stall_window:
        return False

    recent = progress_hist[-stall_window:]
    return (max(recent) - min(recent)) < stall_eps


def check_pick_fail(step, phase, left_grasped, pick_deadline=300):
    """
    일정 step 안에 left grasp가 안 되면 pick 실패
    """
    if step < pick_deadline:
        return False
    if phase <= 3 and left_grasped < 0.5:
        return True
    return False


def check_handover_fail(step, phase, right_grasped, handover_deadline=700):
    """
    일정 step 안에 right grasp가 안 되면 handover 실패
    """
    if step < handover_deadline:
        return False
    if phase <= 7 and right_grasped < 0.5:
        return True
    return False


def check_episode_termination(
    step,
    max_steps,
    progress_hist,
    phase,
    left_grasped,
    right_grasped,
    left_tcp_pos,
    right_tcp_pos,
    obj_pos,
    ):
    """
    종료 조건을 한 번에 확인해서 (should_stop, fail_reason) 반환
    success 종료는 여기서 다루지 않고,
    실패/강제 종료 조건만 담당
    """
    if check_pick_fail(step, phase, left_grasped):
        return True, "pick_fail"

    if check_handover_fail(step, phase, right_grasped):
        return True, "handover_fail"
    if check_object_holder_distance(
        phase=phase,
        left_tcp_pos=left_tcp_pos,
        right_tcp_pos=right_tcp_pos,
        obj_pos=obj_pos,
        left_grasped=left_grasped,
        right_grasped=right_grasped,
    ):
        return True, "object_too_far_from_holder"

    if check_stall(progress_hist):
        return True, "stall"

    if check_max_step(step, max_steps):
        return True, "max_step"

    return False, None