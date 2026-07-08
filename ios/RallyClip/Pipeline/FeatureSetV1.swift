import Foundation

/// Streaming builder of the 362-dim v1 feature vector, one per kept frame.
///
/// Ports `features/feature_engineer.py` (`iter_build_features`) +
/// `training/features/v1.py` (`FeatureSetV1.build_feature_vector`) exactly:
/// per-player = 181 floats, near then far → 362.  Velocity/acceleration use the
/// previous *kept* frame and the previous frame's stored velocities, so only O(1)
/// state is retained.
///
/// PARITY: numpy does the scalar math in float64 and casts on store; keypoint
/// array ops stay float32. Here everything is Float (float32). The scaler
/// normalizes and the LSTM is robust, so this affects segment boundaries only at
/// the sub-frame level — verify against the golden run on-device.
final class FeatureEngineer {
    let numKeypoints = 17
    let perPlayer: Int
    let dt: Float

    private struct Vel { var centroid: (Float, Float)?; var keypoints: [Float]? }
    private var prevPlayers: (near: Player?, far: Player?)?
    private var prevNearVel = Vel()
    private var prevFarVel = Vel()

    init(targetFps: Double) {
        self.dt = targetFps > 0 ? Float(1.0 / targetFps) : 1.0
        self.perPlayer = 1 + 4 + 2 + 2 + 2 + 1 + 1
            + (numKeypoints * 2) + numKeypoints          // keypoints xy + conf
            + (numKeypoints * 2) + (numKeypoints * 2)    // kp_vel + kp_accel
            + numKeypoints + numKeypoints                // kp_speed + kp_accel_mag
            + 14 + 1                                      // limb lengths + box_conf
    }

    /// One (feature_vector) for the current kept frame; advances internal state.
    func build(near: Player?, far: Player?) -> [Float] {
        let prevNear = prevPlayers?.near
        let prevFar = prevPlayers?.far

        var vector = [Float](repeating: -1.0, count: perPlayer * 2)
        let nearFeat = playerFeatures(near, prev: prevNear, prevVel: prevNearVel)
        let farFeat = playerFeatures(far, prev: prevFar, prevVel: prevFarVel)
        vector.replaceSubrange(0..<perPlayer, with: nearFeat)
        vector.replaceSubrange(perPlayer..<(perPlayer * 2), with: farFeat)

        // Advance stored velocities for the next frame (matches feature_engineer).
        let nearVel = Vel(centroid: centroidVelocity(near, prevNear),
                          keypoints: keypointVelocity(near, prevNear))
        let farVel = Vel(centroid: centroidVelocity(far, prevFar),
                         keypoints: keypointVelocity(far, prevFar))
        prevNearVel = nearVel
        prevFarVel = farVel
        prevPlayers = (near, far)
        return vector
    }

    // MARK: - per-player 181-vector

    private func playerFeatures(_ player: Player?, prev: Player?, prevVel: Vel) -> [Float] {
        var f = [Float](repeating: -1.0, count: perPlayer)
        guard let p = player else {
            f[0] = 0.0
            return f
        }

        let centroid = boxCentroid(p.box)
        var velocity: (Float, Float) = (0, 0)
        var acceleration: (Float, Float) = (0, 0)
        var speed: Float = -1
        var accelMag: Float = -1

        var kpVel = [Float](repeating: 0, count: numKeypoints * 2)
        var kpAccel = [Float](repeating: 0, count: numKeypoints * 2)
        var kpSpeed = [Float](repeating: -1, count: numKeypoints)
        var kpAccelMag = [Float](repeating: -1, count: numKeypoints)
        let limbs = limbLengths(p.keypoints)

        if let pp = prev {
            let prevCentroid = boxCentroid(pp.box)
            velocity = ((centroid.0 - prevCentroid.0) / dt, (centroid.1 - prevCentroid.1) / dt)
            speed = (velocity.0 * velocity.0 + velocity.1 * velocity.1).squareRoot()
            if let pcv = prevVel.centroid {
                acceleration = ((velocity.0 - pcv.0) / dt, (velocity.1 - pcv.1) / dt)
                accelMag = (acceleration.0 * acceleration.0 + acceleration.1 * acceleration.1).squareRoot()
            }
            for k in 0..<numKeypoints {
                kpVel[2 * k] = (p.keypoints[2 * k] - pp.keypoints[2 * k]) / dt
                kpVel[2 * k + 1] = (p.keypoints[2 * k + 1] - pp.keypoints[2 * k + 1]) / dt
            }
            for k in 0..<numKeypoints {
                let vx = kpVel[2 * k], vy = kpVel[2 * k + 1]
                kpSpeed[k] = (vx * vx + vy * vy).squareRoot()
            }
            if let pkv = prevVel.keypoints, pkv.count == kpVel.count {
                for i in 0..<kpVel.count { kpAccel[i] = (kpVel[i] - pkv[i]) / dt }
            }
            for k in 0..<numKeypoints {
                let ax = kpAccel[2 * k], ay = kpAccel[2 * k + 1]
                kpAccelMag[k] = (ax * ax + ay * ay).squareRoot()
            }
        }

        var idx = 0
        f[idx] = 1.0; idx += 1
        for v in p.box { f[idx] = v; idx += 1 }
        f[idx] = centroid.0; idx += 1
        f[idx] = centroid.1; idx += 1
        f[idx] = velocity.0; idx += 1
        f[idx] = velocity.1; idx += 1
        f[idx] = acceleration.0; idx += 1
        f[idx] = acceleration.1; idx += 1
        f[idx] = speed; idx += 1
        f[idx] = accelMag; idx += 1
        for v in p.keypoints { f[idx] = v; idx += 1 }        // 34
        for v in p.conf { f[idx] = v; idx += 1 }             // 17
        for v in kpVel { f[idx] = v; idx += 1 }              // 34
        for v in kpAccel { f[idx] = v; idx += 1 }            // 34
        for v in kpSpeed { f[idx] = v; idx += 1 }            // 17
        for v in kpAccelMag { f[idx] = v; idx += 1 }         // 17
        for v in limbs { f[idx] = v; idx += 1 }              // 14
        f[idx] = p.boxConf                                   // 1
        return f
    }

    // MARK: - helpers (mirror v1.py)

    private func boxCentroid(_ box: [Float]) -> (Float, Float) {
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    }

    private func centroidVelocity(_ cur: Player?, _ prev: Player?) -> (Float, Float)? {
        guard let c = cur, let p = prev else { return nil }
        let cc = boxCentroid(c.box), pc = boxCentroid(p.box)
        guard dt > 0 else { return (0, 0) }
        return ((cc.0 - pc.0) / dt, (cc.1 - pc.1) / dt)
    }

    private func keypointVelocity(_ cur: Player?, _ prev: Player?) -> [Float]? {
        guard let c = cur, let p = prev else { return nil }
        guard dt > 0 else { return [Float](repeating: 0, count: numKeypoints * 2) }
        var out = [Float](repeating: 0, count: numKeypoints * 2)
        for i in 0..<out.count { out[i] = (c.keypoints[i] - p.keypoints[i]) / dt }
        return out
    }

    /// 14 limb lengths — same connection order as v1.py `_limb_lengths`.
    private func limbLengths(_ kp: [Float]) -> [Float] {
        let connections: [(Int, Int)] = [
            (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12), (6, 5), (12, 11),
        ]
        return connections.map { (i, j) in
            let (ix, iy) = Player.kp(kp, i)
            let (jx, jy) = Player.kp(kp, j)
            let dx = ix - jx, dy = iy - jy
            return (dx * dx + dy * dy).squareRoot()
        }
    }
}
