import numpy as np


def quat_2_rot6d(quat):
    """
    quat : w, x, y, z
    output : rotmat6d 반환해야 하니가 (6,)반한화면 되나
    """

    # 예외처리
    assert quat.shape == (4,), f"quat shape : {quat.shape}"
    # 정규화 국룰
    norm = np.linalg.norm(quat)
    assert norm > 1e-8, f"quat norm 이 너무 작음: {norm}"
    quat = quat / norm


    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ], dtype=np.float32)
    # 처음 2개 축 정보만
    # return R[:, :2].flatten()
    v1 = R[:, 0]
    v2 = R[:, 1]
    return np.concatenate([v1, v2])


def get_pose_and_rot6d(xform_prim):
    """
    xform_prim.get_world_poses() 결과에서
    - pos : (3,)
    - rot6d : (6,)
    를 뽑아 반환
    """
    positions, orientations = xform_prim.get_world_poses()
    pos = np.array(positions[0], dtype=np.float32)
    quat = np.array(orientations[0], dtype=np.float32)   # Isaac Sim core 기준 (w, x, y, z)
    rot6d = quat_2_rot6d(quat)
    return pos, rot6d


def rot6d_to_rotmat(rot6d):
    """
    rot6d: (6,) = [r00, r10, r20, r01, r11, r21]
    Gram-Schmidt로 회전행렬 복원
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    assert rot6d.shape == (6,), f"rot6d shape : {rot6d.shape}"

    a1 = rot6d[:3]
    a2 = rot6d[3:]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1).astype(np.float32)
    return R


def rotmat_to_quat(R):
    """
    R: (3, 3)
    output quat: (w, x, y, z)
    """
    R = np.asarray(R, dtype=np.float32)
    assert R.shape == (3, 3), f"R shape : {R.shape}"

    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float32)
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    return quat


def rot6d_to_quat(rot6d):
    R = rot6d_to_rotmat(rot6d)
    quat = rotmat_to_quat(R)
    return quat