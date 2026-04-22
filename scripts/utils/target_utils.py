import numpy as np


def sample_cylinder_pos():
    return np.array([
        np.random.uniform(0.50, 0.58),
        0.20 + np.random.uniform(-0.03, 0.03),
        0.05,
    ], dtype=np.float64)


def get_right_target_poses(world, right_robot, right_tcp, right_wait_full, right_receive_full, right_home, get_pose_and_rot6d):
    """
    right arm의 WAIT / RECEIVE tcp target pose를 미리 추출
    """
    right_robot.set_joint_positions(right_wait_full)
    for _ in range(2):
        world.step(render=False)
    right_wait_tcp_pos, right_wait_rot6d = get_pose_and_rot6d(right_tcp)

    right_robot.set_joint_positions(right_receive_full)
    for _ in range(2):
        world.step(render=False)
    right_receive_tcp_pos, right_receive_rot6d = get_pose_and_rot6d(right_tcp)

    right_robot.set_joint_positions(right_home)
    for _ in range(2):
        world.step(render=False)

    return (
        right_wait_tcp_pos,
        right_wait_rot6d,
        right_receive_tcp_pos,
        right_receive_rot6d,
    )