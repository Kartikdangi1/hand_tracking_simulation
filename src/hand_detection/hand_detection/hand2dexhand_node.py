#!/usr/bin/env python3
"""
hand2dexpose_node.py
====================
Publish **geometry_msgs/PoseArray** (position **and** orientation) for the 15
DexHand joints directly from the 21‑landmark output of MediaPipe Hands.

*   Removes the landmark IDs that have **no** matching joint in the URDF 
    (0, 1, 5, 9, 13, 17).
*   One–to–one mapping of the remaining landmark IDs to the joints already
    defined on */joint_states*:

    ````text
    ┌────────┬────────────────────┐
    │Lm‑ID   │  DexHand joint     │
    ├────────┼────────────────────┤
    │   2    │  thumb_pitch       │
    │   3    │  thumb_knuckle     │
    │   4    │  thumb_tip         │
    │   6    │  index_pitch       │
    │   7    │  index_knuckle     │
    │   8    │  index_tip         │
    │  10    │  middle_pitch      │
    │  11    │  middle_knuckle    │
    │  12    │  middle_tip        │
    │  14    │  ring_pitch        │
    │  15    │  ring_knuckle      │
    │  16    │  ring_tip          │
    │  18    │  pinky_pitch       │
    │  19    │  pinky_knuckle     │
    │  20    │  pinky_tip         │
    └────────┴────────────────────┘

The node converts every landmark’s *xyz* position into a **Pose** whose
orientation quaternion is derived as follows:

1. **Direction (X‑axis):**  The unit vector **d** from the landmark to its
   child along the finger (or to its parent for fingertip landmarks).
2. **Palm normal (Z‑axis):**  The right‑handed normal **n** of the palm plane
   spanned by wrist→middle_MCP (id 9) and wrist→index_MCP (id 5).
3. **Y‑axis:**  **y = n × d**.
4. Build a 3×3 rotation matrix R = [x y n] and transform to a quaternion with
   *tf.transformations.quaternion_from_matrix*.

The resulting **PoseArray** is published on */dexhand_joint_poses* and can be
consumed by TF broadcasters, RViz Markers, or downstream IK.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, PoseArray
from tf_transformations import quaternion_from_matrix  # tf >= 0.2

# --------------------------------------------------------------------------------------
# Type helpers
# --------------------------------------------------------------------------------------
Vector = Tuple[float, float, float]

# Landmark indices (MediaPipe Hands)
W, T_CMC, T_MCP, T_IP, T_TIP, I_MCP, I_PIP, I_DIP, I_TIP, \
    M_MCP, M_PIP, M_DIP, M_TIP, R_MCP, R_PIP, R_DIP, R_TIP, \
    P_MCP, P_PIP, P_DIP, P_TIP = range(21)

# 15‑element mapping (see table in module doc‑string)
LM_IDS: List[int] = [
    2, 3, 4,    # thumb
    6, 7, 8,    # index
    10, 11, 12, # middle
    14, 15, 16, # ring
    18, 19, 20, # pinky
]

JOINT_NAMES: List[str] = [
    "thumb_pitch", "thumb_knuckle", "thumb_tip",
    "index_pitch", "index_knuckle", "index_tip",
    "middle_pitch","middle_knuckle","middle_tip",
    "ring_pitch",  "ring_knuckle",  "ring_tip",
    "pinky_pitch", "pinky_knuckle", "pinky_tip",
]
assert len(LM_IDS) == len(JOINT_NAMES)

# Child look‑up: for every landmark we need the next landmark along the finger.
_CHILD = {
    2: 3, 3: 4,         # thumb
    6: 7, 7: 8,
    10: 11, 11: 12,
    14: 15, 15: 16,
    18: 19, 19: 20,
}

# --------------------------------------------------------------------------------------
# Maths helpers
# --------------------------------------------------------------------------------------
_vec = np.array  # convenience alias

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v.copy()

# --------------------------------------------------------------------------------------
class Hand2DexPose(Node):
    """ROS 2 node converting 21 landmarks → PoseArray for DexHand."""

    def __init__(self) -> None:
        super().__init__("hand2dexpose")

        # Parameters (tweak via launch YAML or CLI – ros2 run … --ros‑args –p rate:=60)
        self.declare_parameter("publish_topic", "/dexhand_joint_poses")
        self.declare_parameter("rate",           30.0)  # Hz
        self.declare_parameter("frame_id",       "camera")
        self.declare_parameter("mirrored",       False)  # true for left hand / selfie camera

        self._topic    : str   = self.get_parameter("publish_topic").value
        self._rate     : float = float(self.get_parameter("rate").value)
        self._frame_id : str   = self.get_parameter("frame_id").value
        self._mirrored : bool  = bool(self.get_parameter("mirrored").value)

        # I/O
        self._pub = self.create_publisher(PoseArray, self._topic, 10)
        self.create_subscription(Float64MultiArray, "hand_landmarks",
                                 self._on_landmarks, qos_profile_sensor_data)

        self._last_msg: PoseArray | None = None
        self.create_timer(1.0 / self._rate, self._tick)

        self.get_logger().info(f"✓ Up – publishing PoseArray@{self._rate:.1f} Hz on '{self._topic}'")

    # -------------------------------------------------------------------------
    def _on_landmarks(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 63:
            self.get_logger().warn(f"Expected 63 floats, got {len(msg.data)} – ignored")
            return

        # reshape to list[Vector]
        lm: List[np.ndarray] = [
            _vec([msg.data[3*i], msg.data[3*i+1], msg.data[3*i+2]]) for i in range(21)
        ]
        if self._mirrored:
            lm = [_vec([-p[0], p[1], p[2]]) for p in lm]

        # Palm plane normal (wrist –> middle & index MCPs)
        n = _unit(np.cross(lm[M_MCP] - lm[W], lm[I_MCP] - lm[W]))

        poses: List[Pose] = []
        for lid in LM_IDS:
            p = Pose()
            p.position.x, p.position.y, p.position.z = lm[lid]

            # Finger direction along bone – choose child or parent as reference
            if lid in _CHILD:
                d = lm[_CHILD[lid]] - lm[lid]
            else:                               # tip → use previous joint as ref
                parent = lid - 1
                d = lm[lid] - lm[parent]
            x = _unit(d)
            z = n
            y = _unit(np.cross(z, x))
            z = _unit(np.cross(x, y))            # re‑orthonormalise

            R = np.eye(4)
            R[0:3, 0] = x
            R[0:3, 1] = y
            R[0:3, 2] = z
            q = quaternion_from_matrix(R)
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            poses.append(p)

        pa            = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self._frame_id
        pa.poses       = poses
        self._last_msg = pa

    # -------------------------------------------------------------------------
    def _tick(self) -> None:
        if self._last_msg is not None:
            self._pub.publish(self._last_msg)

# --------------------------------------------------------------------------------------

def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Hand2DexPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down …")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
