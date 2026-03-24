#!/usr/bin/env python3
"""
Redesigned hand tracking node:
- Uses MediaPipe Hands to get 21 landmarks.
- Smooths landmarks with One-Euro filter (tuned for less jitter).
- Computes DexHand joint angles from landmark positions (position-based).
- Publishes JointState to /joint_states and optionally a PoseArray for visualization.
"""

import time
import math
from typing import List
from rclpy.executors import MultiThreadedExecutor
import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

# Helper: One-Euro filter class for smoothing (unchanged structure, tuned params)
class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=0.5, beta=0.2, d_cutoff=1.0):
        # Lower min_cutoff -> more smoothing of slow changes; beta higher -> more responsive to fast moves
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x: np.ndarray, t: float = None) -> np.ndarray:
        if t is None:
            t = time.time()
        if self.x_prev is None:
            # Initialize on first call
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0.0:
            return x
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        # Low-pass filter the derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = self._low_pass(dx, self.dx_prev, alpha_d)
        # Adaptive cutoff for position, higher if moving fast
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        # Low-pass filter the position
        alpha = self._alpha(cutoff, dt)
        x_filt = self._low_pass(x, self.x_prev, alpha)
        # Store and return
        self.x_prev = x_filt
        self.dx_prev = edx
        self.t_prev = t
        return x_filt

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(x: np.ndarray, x_prev: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * x + (1.0 - alpha) * x_prev

# Landmark indices for clarity (MediaPipe Hands model indexes)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

class HandDetectionNode(Node):
    """Node that detects one hand and publishes filtered landmarks."""
    def __init__(self):
        super().__init__("hand_detection_node")
        self.bridge = CvBridge()
        # Subscribers and publishers
        self.create_subscription(Image, "/image_raw", self._on_image, qos_profile_sensor_data)
        self.pub_landmarks = self.create_publisher(Float64MultiArray, "hand_landmarks", 1)
        self.pub_annotated = self.create_publisher(Image, "hand_detection/image_annotated", 1)
        # Set up MediaPipe Hands (use GPU if available, otherwise CPU)
        self.mp_hands = mp.solutions.hands.Hands(  
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
            # Note: The Python API uses CPU by default. For GPU, ensure proper installation or use MediaPipe Tasks:contentReference[oaicite:17]{index=17}.
        )
        self.mp_draw = mp.solutions.drawing_utils
        # One-Euro filters for each of the 21 landmarks (for smoothing X,Y,Z)
        self.filters = [OneEuro(freq=30.0, min_cutoff=0.5, beta=0.2) for _ in range(21)]
        self.get_logger().info("HandDetectionNode initialized: subscribed to /image_raw")

    def _on_image(self, msg: Image):
        # Receive image frame and run hand landmark detection
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CVBridge conversion failed: {e}")
            return
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)
        # Draw landmarks on the image for visualization
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(cv_img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        # Prepare landmark array for publishing (63 floats = 21 points * 3 coords)
        landmark_data = [0.0] * 63
        if results.multi_hand_landmarks:
            # We have a hand detected
            lm = results.multi_hand_landmarks[0].landmark
            h, w, _ = cv_img.shape
            # Extract and filter each landmark coordinate
            for i in range(21):
                raw_point = np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float32)
                # Apply one-euro filter to smooth jitter
                filtered_point = self.filters[i](raw_point)
                # Optionally, convert to pixel units (e.g., for 2D display) – here we keep normalized for consistency
                landmark_data[3*i:3*i+3] = filtered_point.tolist()
        else:
            # No hand detected in this frame: do not update landmarks (retain last). 
            # We simply do not publish new data to avoid spikes.
            # Optionally, could count missed frames and if many in a row, send a "hand lost" signal.
            return
        # Publish the filtered landmarks
        self.pub_landmarks.publish(Float64MultiArray(data=landmark_data))
        # Publish the annotated image (for debug viewing)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.pub_annotated.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

class HandPosePublisher(Node):
    """Node that converts landmarks to DexHand PoseArray and JointState."""
    def __init__(self):
        super().__init__("hand_pose_publisher")
        # Parameters
        self.declare_parameter("publish_pose_array", False)  # whether to publish PoseArray for visualization
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("mirrored", False)  # set True if using left hand to control a right-hand model
        self.publish_pose_array = bool(self.get_parameter("publish_pose_array").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.mirrored = bool(self.get_parameter("mirrored").value)
        # Publishers
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 1)
        self.pose_pub = self.create_publisher(PoseArray, "/dexhand_joint_poses", 1) if self.publish_pose_array else None
        # Subscribe to the landmark array from HandDetectionNode
        self.create_subscription(Float64MultiArray, "hand_landmarks", self._on_landmarks, qos_profile_sensor_data)
        # Joint names for DexHand (from URDF) in expected order:
        self.joint_names = [
            "thumb_yaw", "thumb_pitch", "thumb_knuckle",
            "index_yaw", "index_pitch", "index_knuckle", "index_tip",
            "middle_yaw", "middle_pitch", "middle_knuckle", "middle_tip",
            "ring_yaw", "ring_pitch", "ring_knuckle", "ring_tip",
            "pinky_yaw", "pinky_pitch", "pinky_knuckle", "pinky_tip",
            # Fixed (non-moving) joints:
            "wrist_pitch_lower", "wrist_yaw", "wrist_pitch_upper", "thumb_roll"
        ]
        # Initialize previous joint angles for smoothing (if needed)
        self.prev_joint_angles = [0.0] * len(self.joint_names)
        self.get_logger().info("HandPosePublisher initialized: publishing joint states (and PoseArray)")


    def _on_landmarks(self, msg: Float64MultiArray):
        data = msg.data
        if len(data) != 63:
            return  # sanity check
        # Convert flat array into list of 3D points (as numpy arrays for math)
        points = [np.array(data[i:i+3], dtype=np.float32) for i in range(0, 63, 3)]
        # If mirrored, flip the X coordinate (to mirror the hand across YZ plane)
        if self.mirrored:
            for p in points:
                p[0] = -p[0]

        # Compute palm plane normal using the base of fingers (index to pinky MCPs)
        mcp_points = np.stack([points[i] for i in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)])
        # Use PCA/SVD to find plane normal
        _, _, vh = np.linalg.svd(mcp_points - mcp_points.mean(axis=0), full_matrices=False)
        palm_normal = vh[2].astype(np.float32)
        # Ensure palm_normal points in a consistent direction (flip if needed for stability)
        # (For example, make it point outward from palm by checking thumb side or camera direction - omitted for brevity)

        # Define a forward reference direction in the palm plane (from wrist to middle MCP)
        wrist_to_middle = points[MIDDLE_MCP] - points[WRIST]
        # Project it onto palm plane
        forward_dir = wrist_to_middle - np.dot(wrist_to_middle, palm_normal) * palm_normal
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)

        # Prepare PoseArray for visualization (if enabled)
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.frame_id

        # Prepare JointState message
        joint_state = JointState()
        joint_state.header.stamp = pose_array.header.stamp

        # Helper lambda: compute angle between two vectors
        def angle_between(v1, v2):
            cosang = np.dot(v1, v2) / ((np.linalg.norm(v1)+1e-8)*(np.linalg.norm(v2)+1e-8))
            cosang = np.clip(cosang, -1.0, 1.0)  # numerical safety
            return math.acos(cosang)

        # Compute joint angles for each finger
        # Thumb (uses landmarks 1,2,3,4)
        # Thumb yaw: angle between (wrist->thumb CMC) and (wrist->index MCP) in palm plane
        thumb_yaw = angle_between(
            points[THUMB_CMC] - points[WRIST], 
            points[INDEX_MCP] - points[WRIST]
        )
        # Thumb pitch (MCP flexion): angle between (CMC->MCP) and (MCP->IP)
        thumb_pitch = angle_between(
            points[THUMB_MCP] - points[THUMB_CMC],
            points[THUMB_IP] - points[THUMB_MCP]
        )
        # Thumb IP flexion (knuckle): angle between (MCP->IP) and (IP->tip)
        thumb_knuckle = angle_between(
            points[THUMB_IP] - points[THUMB_MCP],
            points[THUMB_TIP] - points[THUMB_IP]
        )
        # Index finger
        # Index yaw: angle in plane between index direction and forward_dir
        index_dir = points[INDEX_PIP] - points[INDEX_MCP]
        index_proj = index_dir - np.dot(index_dir, palm_normal) * palm_normal
        index_proj = index_proj / (np.linalg.norm(index_proj)+1e-8)
        index_yaw = angle_between(index_proj, forward_dir)
        # Determine sign based on horizontal position relative to middle finger
        if points[INDEX_MCP][0] < points[MIDDLE_MCP][0]:
            index_yaw *= 1  # thumb side (e.g. right-hand index finger moves outward → positive yaw)
        else:
            index_yaw *= -1  # if on opposite side, make angle negative
        # Index pitch (MCP flexion): angle between finger direction and its projection (out-of-plane angle)
        index_pitch = angle_between(index_dir, index_proj)
        # Index PIP flexion
        index_knuckle = angle_between(points[INDEX_MCP] - points[INDEX_PIP], points[INDEX_DIP] - points[INDEX_PIP])
        # Index DIP flexion
        index_tip = angle_between(points[INDEX_PIP] - points[INDEX_DIP], points[INDEX_TIP] - points[INDEX_DIP])
        # Middle finger
        middle_dir = points[MIDDLE_PIP] - points[MIDDLE_MCP]
        middle_proj = middle_dir - np.dot(middle_dir, palm_normal) * palm_normal
        middle_proj = middle_proj / (np.linalg.norm(middle_proj)+1e-8)
        middle_yaw = angle_between(middle_proj, forward_dir)
        # Middle finger likely centered; assign sign based on slight deviation (if any)
        if points[MIDDLE_MCP][0] < points[INDEX_MCP][0]:
            middle_yaw *= 1 
        else:
            middle_yaw *= -1
        middle_pitch = angle_between(middle_dir, middle_proj)
        middle_knuckle = angle_between(points[MIDDLE_MCP] - points[MIDDLE_PIP], points[MIDDLE_DIP] - points[MIDDLE_PIP])
        middle_tip = angle_between(points[MIDDLE_PIP] - points[MIDDLE_DIP], points[MIDDLE_TIP] - points[MIDDLE_DIP])
        # Ring finger
        ring_dir = points[RING_PIP] - points[RING_MCP]
        ring_proj = ring_dir - np.dot(ring_dir, palm_normal) * palm_normal
        ring_proj = ring_proj / (np.linalg.norm(ring_proj)+1e-8)
        ring_yaw = angle_between(ring_proj, forward_dir)
        # Ring: if its MCP is to the right of middle MCP (for right hand), mark negative
        if points[RING_MCP][0] > points[MIDDLE_MCP][0]:
            ring_yaw *= -1
        else:
            ring_yaw *= 1
        ring_pitch = angle_between(ring_dir, ring_proj)
        ring_knuckle = angle_between(points[RING_MCP] - points[RING_PIP], points[RING_DIP] - points[RING_PIP])
        ring_tip = angle_between(points[RING_PIP] - points[RING_DIP], points[RING_TIP] - points[RING_DIP])
        # Pinky finger
        pinky_dir = points[PINKY_PIP] - points[PINKY_MCP]
        pinky_proj = pinky_dir - np.dot(pinky_dir, palm_normal) * palm_normal
        pinky_proj = pinky_proj / (np.linalg.norm(pinky_proj)+1e-8)
        pinky_yaw = angle_between(pinky_proj, forward_dir)
        if points[PINKY_MCP][0] > points[MIDDLE_MCP][0]:
            pinky_yaw *= -1
        else:
            pinky_yaw *= 1
        pinky_pitch = angle_between(pinky_dir, pinky_proj)
        pinky_knuckle = angle_between(points[PINKY_MCP] - points[PINKY_PIP], points[PINKY_DIP] - points[PINKY_PIP])
        pinky_tip = angle_between(points[PINKY_PIP] - points[PINKY_DIP], points[PINKY_TIP] - points[PINKY_DIP])

        # Collect all joint angles in the DexHand joint order
        joint_angles = [
            thumb_yaw, thumb_pitch, thumb_knuckle,
            index_yaw, index_pitch, index_knuckle, index_tip,
            middle_yaw, middle_pitch, middle_knuckle, middle_tip,
            ring_yaw, ring_pitch, ring_knuckle, ring_tip,
            pinky_yaw, pinky_pitch, pinky_knuckle, pinky_tip,
        ]
        # Append fixed joints (wrist and thumb roll) as 0.0
        joint_angles += [0.0, 0.0, 0.0, 0.0]

        # Optional smoothing of joint angles: ignore tiny changes (deadband)
        for j in range(len(joint_angles) - 4):  # exclude fixed joints
            if abs(joint_angles[j] - self.prev_joint_angles[j]) < math.radians(2):  # less than 2° change
                joint_angles[j] = self.prev_joint_angles[j]  # hold previous value
            else:
                # Low-pass filter the angle change (simple exponential smoothing)
                joint_angles[j] = self.prev_joint_angles[j] + 0.5 * (joint_angles[j] - self.prev_joint_angles[j])
        self.prev_joint_angles = joint_angles

        # Clip angles to safe range (-90 to 90 degrees) for sanity
        for j in range(len(joint_angles) - 4):
            joint_angles[j] = float(np.clip(joint_angles[j], -math.pi/2, math.pi/2))

        # Fill JointState message
        joint_state.name = self.joint_names
        joint_state.position = joint_angles

        # Fill PoseArray if needed (using only positions, orientation as identity)
        if self.publish_pose_array:
            for pid in range(1, 21):  # landmark 1..20 (skipping wrist) to visualize finger joints
                pose = Pose()
                pose.position.x = float(points[pid][0])
                pose.position.y = float(points[pid][1])
                pose.position.z = float(points[pid][2])
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
            self.pose_pub.publish(pose_array)

        # Publish the joint state for the hand
        self.joint_pub.publish(joint_state)

        # (At this point, robot_state_publisher will use /joint_states to update the DexHand model in RViz)
        # Log a warning if our orientation fallback happened frequently (omitted here for brevity).



# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main(args: List[str] | None = None) -> None:  # noqa: D401
    """Launch both nodes under a single multi-threaded executor."""
    rclpy.init(args=args)
    det = HandDetectionNode()
    dex = HandPosePublisher()
    exe = MultiThreadedExecutor()
    exe.add_node(det)
    exe.add_node(dex)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        det.destroy_node()
        dex.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
