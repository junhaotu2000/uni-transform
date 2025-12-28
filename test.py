from uni_transform import (
    Transform, Rotation,
    interpolate, interpolate_sequence,
    interpolate_transform, interpolate_transform_sequence,
    compute_spline,
)
import numpy as np

if __name__ == "__main__":
    # =========================================================================
    # 1. 两点 Transform 插值
    # =========================================================================
    tf0 = Transform.identity()
    tf1 = Transform.from_rep(np.array([1.0, 2.0, 0.0, 0.0, 0.0, np.pi/2]), from_rep="euler")
    
    # 单个时间点
    tf_mid = interpolate_transform(tf0, tf1, t=0.5)
    print(f"Transform at t=0.5: trans={tf_mid.translation}")
    
    # 多个时间点
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for ti in t:
        tf = interpolate_transform(tf0, tf1, t=ti)
        print(f"  t={ti:.2f}: trans={tf.translation}")

    # =========================================================================
    # 2. 多点 Transform 序列插值
    # =========================================================================
    print("\n多点 Transform 序列插值:")
    
    keyframes = Transform.stack([
        Transform.from_rep(np.array([0, 0, 0, 0, 0, 0]), from_rep="euler"),
        Transform.from_rep(np.array([1, 0, 0, 0, 0, np.pi/4]), from_rep="euler"),
        Transform.from_rep(np.array([2, 1, 0, 0, 0, np.pi/2]), from_rep="euler"),
        Transform.from_rep(np.array([2, 2, 0, 0, 0, np.pi]), from_rep="euler"),
    ])
    times = np.array([0.0, 1.0, 2.0, 3.0])
    query = np.array([0.5, 1.5, 2.5])
    
    # 默认: slerp + linear
    result = interpolate_transform_sequence(keyframes, times, query)
    print(f"slerp+linear: {result.translation}")
    
    # squad + cubic_spline (更平滑)
    result = interpolate_transform_sequence(
        keyframes, times, query,
        rotation_method="squad",
        translation_method="cubic_spline"
    )
    print(f"squad+spline: {result.translation}")

    # =========================================================================
    # 3. 向量插值
    # =========================================================================
    print("\n向量插值:")
    
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 2.0, 3.0])
    
    # 线性
    pos = interpolate(start, end, t=0.5)
    print(f"Linear t=0.5: {pos}")
    
    # Minimum jerk (平滑，端点速度为0)
    pos = interpolate(start, end, t=0.5, method="minimum_jerk", duration=1.0)
    print(f"MinJerk t=0.5: {pos}")

    # =========================================================================
    # 4. 多点向量序列插值
    # =========================================================================
    print("\n多点向量序列插值:")
    
    waypoints = np.array([[0, 0], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
    times = np.array([0, 1, 2, 3], dtype=np.float64)
    query = np.array([0.5, 1.5, 2.5])
    
    pos = interpolate_sequence(waypoints, times, query, method="linear")
    print(f"Linear: {pos}")
    
    pos = interpolate_sequence(waypoints, times, query, method="cubic_spline")
    print(f"Spline: {pos}")

    # =========================================================================
    # 5. 可复用 Spline (计算一次，多次求值)
    # =========================================================================
    print("\n可复用 Spline:")
    
    spline = compute_spline(waypoints, times)
    
    pos1 = spline.evaluate(np.array([0.5, 1.0, 1.5]))
    pos2 = spline.evaluate(np.array([2.0, 2.5]))
    vel = spline.derivative(np.array([1.0]), order=1)
    
    print(f"Position at [0.5, 1.0, 1.5]: {pos1}")
    print(f"Position at [2.0, 2.5]: {pos2}")
    print(f"Velocity at t=1.0: {vel}")

    print("\n✅ Done!")
