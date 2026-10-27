"""ROS 2 (Jazzy) adapters: an independent integration layer over the same Python modules (master prompt §18).

Nodes (std / sensor / geometry messages only; no custom interfaces yet):
  SimBridge      publishes /aura/fixed/{rgb, depth, camera_info} (sensor-data QoS), /joint_states and TF world->camera
                 (the BELIEVED extrinsics). The parameter `drift_rot_deg` bumps the true camera at runtime (fault injection).
  Perception     subscribes rgb+depth+camera_info (+ TF) -> /aura/part_poses (geometry_msgs/PoseArray) and
                 /aura/part_classes (std_msgs/String, JSON list aligned with the PoseArray).
  CalibMonitor   depth -> /aura/calib_residual_mm (Float32), /aura/calib_alarm (Bool); thresholds from commissioning.
Units: metres, radians; frame ids "world" and "aura/fixed_optical" (OpenCV optical convention).
Not claimed: compatibility with any vendor controller or with Palladyne IQ.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float32, String
from tf2_ros import TransformBroadcaster

from aura3d.calibration.monitor import DriftMonitor
from aura3d.contracts import CameraFrame
from aura3d.geometry import se3
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.faults import CameraDrift, apply_camera_drift

OPTICAL = "aura/fixed_optical"


def _stamp(node, t_sim: float):
    return node.get_clock().now().to_msg()


def _img(arr, enc, header):
    m = Image()
    m.header = header
    m.height, m.width = arr.shape[:2]
    m.encoding = enc
    m.is_bigendian = 0
    m.step = arr.strides[0]
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


class SimBridge(Node):
    def __init__(self, env, seed: int = 0, rate_hz: float = 5.0):
        super().__init__("aura_sim_bridge")
        self.env = env
        env.reset(seed)
        self.declare_parameter("drift_rot_deg", 0.0)
        self.pub_rgb = self.create_publisher(Image, "/aura/fixed/rgb", qos_profile_sensor_data)
        self.pub_d = self.create_publisher(Image, "/aura/fixed/depth", qos_profile_sensor_data)
        self.pub_ci = self.create_publisher(CameraInfo, "/aura/fixed/camera_info", qos_profile_sensor_data)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)
        self._drift = 0.0
        self.create_timer(1.0 / rate_hz, self.tick)

    def tick(self):
        d = float(self.get_parameter("drift_rot_deg").value)
        if d != self._drift:
            apply_camera_drift(self.env, CameraDrift(d, 0.0, seed=1))
            self._drift = d
        self.env.step(10)
        f = self.env.render("fixed")
        hdr = Image().header
        hdr.stamp = _stamp(self, f.stamp)
        hdr.frame_id = OPTICAL
        self.pub_rgb.publish(_img(f.rgb, "rgb8", hdr))
        self.pub_d.publish(_img(f.depth.astype(np.float32), "32FC1", hdr))
        ci = CameraInfo()
        ci.header = hdr
        ci.height, ci.width = f.rgb.shape[:2]
        ci.k = f.K.astype(float).ravel().tolist()
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0] * 5
        self.pub_ci.publish(ci)
        js = JointState()
        js.header.stamp = hdr.stamp
        js.name = list(self.env.meta["robot_cfg"]["joints"])
        js.position = self.env.robot.q().tolist()
        self.pub_js.publish(js)
        T = f.T_world_cam
        q = se3.rot_to_quat(T[:3, :3])
        ts = TransformStamped()
        ts.header.stamp, ts.header.frame_id, ts.child_frame_id = hdr.stamp, "world", OPTICAL
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = map(float, T[:3, 3])
        ts.transform.rotation.w, ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z = map(float, q)
        self.tf.sendTransform(ts)


def _frame_from(rgb_msg, depth_msg, ci_msg, T_world_cam) -> CameraFrame:
    h, w = depth_msg.height, depth_msg.width
    rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(h, w, 3)
    depth = np.frombuffer(depth_msg.data, np.float32).reshape(h, w)
    t = depth_msg.header.stamp.sec + 1e-9 * depth_msg.header.stamp.nanosec
    return CameraFrame(t, OPTICAL, rgb, depth, np.array(ci_msg.k).reshape(3, 3), T_world_cam)


class _Synced(Node):
    """Keeps the latest rgb/depth/camera_info with matching stamps (exact-time pairing)."""

    def __init__(self, name, T_world_cam):
        super().__init__(name)
        self.T = T_world_cam
        self.last = {}
        for topic, typ in (("rgb", Image), ("depth", Image), ("camera_info", CameraInfo)):
            self.create_subscription(typ, f"/aura/fixed/{topic}", lambda m, k=topic: self._got(k, m),
                                     qos_profile_sensor_data)
        self.latency_s = []

    def _got(self, key, msg):
        self.last[key] = msg
        st = {k: (m.header.stamp.sec, m.header.stamp.nanosec) for k, m in self.last.items()}
        if len(st) == 3 and len(set(st.values())) == 1:
            now = self.get_clock().now().nanoseconds * 1e-9
            s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            self.latency_s.append(now - s)
            self.on_frame(_frame_from(self.last["rgb"], self.last["depth"], self.last["camera_info"], self.T))
            self.last = {}


class Perception(_Synced):
    def __init__(self, layout, T_world_cam):
        super().__init__("aura_perception", T_world_cam)
        self.layout = layout
        self.pub = self.create_publisher(PoseArray, "/aura/part_poses", 10)
        self.pub_c = self.create_publisher(String, "/aura/part_classes", 10)

    def on_frame(self, f):
        ests = detect_and_localize(f, self.layout)
        pa = PoseArray()
        pa.header.frame_id = "world"
        pa.header.stamp = self.get_clock().now().to_msg()
        for e in ests:
            p = Pose()
            p.position.x, p.position.y, p.position.z = map(float, e.translation)
            q = se3.rot_to_quat(e.rotation)
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = map(float, q)
            pa.poses.append(p)
        self.pub.publish(pa)
        self.pub_c.publish(String(data=json.dumps([e.features["cls"] for e in ests])))


class CalibMonitor(_Synced):
    def __init__(self, layout, T_world_cam, monitor: DriftMonitor):
        super().__init__("aura_calib_monitor", T_world_cam)
        self.layout, self.mon = layout, monitor
        self.pub_r = self.create_publisher(Float32, "/aura/calib_residual_mm", 10)
        self.pub_a = self.create_publisher(Bool, "/aura/calib_alarm", 10)

    def on_frame(self, f):
        alarm, r = self.mon.check(f, self.layout)
        self.pub_r.publish(Float32(data=float(r)))
        self.pub_a.publish(Bool(data=bool(alarm)))


__all__ = ["SimBridge", "Perception", "CalibMonitor", "rclpy"]
