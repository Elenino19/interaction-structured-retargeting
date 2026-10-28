"""Per-frame diagnostics and synchronized comparison rendering for EXT-JC refinement.

Everything reads saved trajectories only.  Rendering reuses the frozen v41
mesh renderer (`mujoco_scene`): same G1 MJCF, same object mesh, same
resolution (960x720 per tile), same fps (15), same azimuth/elevation; the
camera look-at/distance is *fixed from the reference method* so all tiles
share one camera.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .audit import load_npz_mapping
from .contact import PHASE_NAMES, hysteresis_contact, resolve_object_model
from .coordinates import points_robot_to_core4d, poses_core4d_to_robot
from .coordination_eval import (
    ROBOT_CONTACT_ENTER_M,
    ROBOT_CONTACT_EXIT_M,
    SOURCE_CONTACT_ENTER_M,
    SOURCE_CONTACT_EXIT_M,
)
from .metrics import rubber_foot_world_volume_points, rubber_hand_world_volume_points
from .retarget.balance import (
    LEFT_FOOT_VOLUME_OFFSETS_M,
    LEFT_HAND_VOLUME_OFFSETS_M,
    points_in_object_frame,
)

DEFAULT_URDF = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf")
DEFAULT_G1_XML = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist")
WAIST_JOINT_TOKENS = ("waist",)
LEG_JOINT_TOKENS = ("hip", "knee", "ankle")


@dataclass
class FrameDiagnostics:
    """Per-frame signals for one method on one sequence (all arrays length T)."""

    label: str
    sequence_id: str
    fps: float
    phases: np.ndarray
    expected_joint: np.ndarray
    predicted: dict[int, np.ndarray]
    hand_pen_cm: dict[int, np.ndarray]  # (T,) deepest hull penetration of hand samples
    hand_pen_per_hand_cm: dict[int, np.ndarray]  # (T, 2)
    hand_dist_m: dict[int, np.ndarray]  # (T, 2) nearest-vertex distance per hand
    soft_min_signed_m: dict[int, np.ndarray]  # (T,) soft-min signed hull distance (beta 0.01)
    soft_indicator: dict[int, np.ndarray]  # (T,) sigma((tau - softmin)/beta)
    true_min_signed_m: dict[int, np.ndarray]  # (T,) exact min signed hull distance
    root_speed_m_s: dict[int, np.ndarray]
    root_acc_m_s2: dict[int, np.ndarray]
    root_jerk_m_s3: dict[int, np.ndarray]
    joint_vel_max: dict[int, np.ndarray]  # max over joints |dq/dt|
    joint_acc_max: dict[int, np.ndarray]
    joint_jerk_max: dict[int, np.ndarray]
    arm_jerk_max: dict[int, np.ndarray]
    waist_jerk_max: dict[int, np.ndarray]
    leg_jerk_max: dict[int, np.ndarray]
    foot_speed_cm_s: dict[int, np.ndarray]  # (T-1, 2) centroid horizontal speed
    foot_zmin_m: dict[int, np.ndarray]  # (T, 2)
    joint_positions: dict[int, np.ndarray]
    root_translation: dict[int, np.ndarray]
    root_yaw: dict[int, np.ndarray]
    joint_names: np.ndarray

    @property
    def predicted_joint(self) -> np.ndarray:
        return self.predicted[1] & self.predicted[2]


def _derivatives(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.gradient(x, 1.0 / fps, axis=0)
    a = np.gradient(v, 1.0 / fps, axis=0)
    j = np.gradient(a, 1.0 / fps, axis=0)
    return v, a, j


def _resolve(result_root: Path, stem: str, person: int) -> Path:
    for candidate in (
        result_root / stem / f"robot{person}.npz",
        result_root / f"robot{person}.npz",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(result_root / stem / f"robot{person}.npz")


def load_frame_diagnostics(
    sequence_id: str,
    result_root: Path,
    *,
    label: str,
    motion_root: Path = Path("data/core4d/motions"),
    object_models_root: Path = Path("data/core4d/object_models"),
    contact_root: Path = Path("outputs/contact_analysis"),
    urdf_path: Path = DEFAULT_URDF,
    fps: float = 15.0,
    tau_m: float = 0.04,
    beta_m: float = 0.01,
) -> FrameDiagnostics:
    stem = sequence_id.replace("/", "_")
    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not pose_path.exists():
        pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(pose_path)
    mesh = trimesh.load_mesh(resolve_object_model(sequence_dir, object_models_root), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hull = ConvexHull(vertices)
    tree = cKDTree(vertices)
    with np.load(contact_root / f"{stem}.npz") as contact:
        phases = np.asarray(contact["phase"])
        fingertip = {p: np.asarray(contact[f"person{p}_fingertip_distances"]) for p in (1, 2)}
    expected = {}
    for p in (1, 2):
        per_hand = [
            hysteresis_contact(
                fingertip[p][:, h * 5 : (h + 1) * 5].min(axis=1),
                enter_threshold=SOURCE_CONTACT_ENTER_M,
                exit_threshold=SOURCE_CONTACT_EXIT_M,
            )
            for h in range(2)
        ]
        expected[p] = per_hand[0] | per_hand[1]
    expected_joint = expected[1] & expected[2]

    n_left = len(LEFT_HAND_VOLUME_OFFSETS_M)
    n_left_foot = len(LEFT_FOOT_VOLUME_OFFSETS_M)
    out: dict[str, dict[int, np.ndarray]] = {
        k: {}
        for k in (
            "predicted",
            "hand_pen_cm",
            "hand_pen_per_hand_cm",
            "hand_dist_m",
            "soft_min_signed_m",
            "soft_indicator",
            "true_min_signed_m",
            "root_speed_m_s",
            "root_acc_m_s2",
            "root_jerk_m_s3",
            "joint_vel_max",
            "joint_acc_max",
            "joint_jerk_max",
            "arm_jerk_max",
            "waist_jerk_max",
            "leg_jerk_max",
            "foot_speed_cm_s",
            "foot_zmin_m",
            "joint_positions",
            "root_translation",
            "root_yaw",
        )
    }
    joint_names = None
    for p in (1, 2):
        result = load_npz_mapping(_resolve(result_root, stem, p))
        q = np.asarray(result["joint_positions"], dtype=np.float64)
        root = np.asarray(result["root_translation"], dtype=np.float64)
        yaw = np.asarray(result["root_yaw"], dtype=np.float64)
        joint_names = np.asarray(result["joint_names"])
        frames = len(q)
        hand_world = rubber_hand_world_volume_points(urdf_path, q, root, yaw)
        local = points_in_object_frame(points_robot_to_core4d(hand_world), object_poses[:frames])
        flat = local.reshape(-1, 3)
        dist, _ = tree.query(flat, k=1)
        dist = dist.reshape(frames, -1)
        out["hand_dist_m"][p] = np.stack(
            [dist[:, :n_left].min(axis=1), dist[:, n_left:].min(axis=1)], axis=1
        )
        out["predicted"][p] = hysteresis_contact(
            out["hand_dist_m"][p].min(axis=1),
            enter_threshold=ROBOT_CONTACT_ENTER_M,
            exit_threshold=ROBOT_CONTACT_EXIT_M,
        )
        signed = (
            (flat @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1).reshape(frames, -1)
        )
        depth = np.maximum(-signed, 0.0) * 100.0
        out["hand_pen_cm"][p] = depth.max(axis=1)
        out["hand_pen_per_hand_cm"][p] = np.stack(
            [depth[:, :n_left].max(axis=1), depth[:, n_left:].max(axis=1)], axis=1
        )
        soft_min = -beta_m * (
            np.log(
                np.exp(-signed / beta_m - (-signed / beta_m).max(axis=1, keepdims=True)).sum(axis=1)
            )
            + (-signed / beta_m).max(axis=1)
        )
        out["soft_min_signed_m"][p] = soft_min
        out["soft_indicator"][p] = 1.0 / (1.0 + np.exp(-(tau_m - soft_min) / beta_m))
        out["true_min_signed_m"][p] = signed.min(axis=1)
        rv, ra, rj = _derivatives(root, fps)
        out["root_speed_m_s"][p] = np.linalg.norm(rv, axis=1)
        out["root_acc_m_s2"][p] = np.linalg.norm(ra, axis=1)
        out["root_jerk_m_s3"][p] = np.linalg.norm(rj, axis=1)
        qv, qa, qj = _derivatives(q, fps)
        out["joint_vel_max"][p] = np.abs(qv).max(axis=1)
        out["joint_acc_max"][p] = np.abs(qa).max(axis=1)
        out["joint_jerk_max"][p] = np.abs(qj).max(axis=1)
        names = [str(n) for n in joint_names]
        arm = np.array([any(t in n for t in ARM_JOINT_TOKENS) for n in names])
        waist = np.array([any(t in n for t in WAIST_JOINT_TOKENS) for n in names])
        leg = np.array([any(t in n for t in LEG_JOINT_TOKENS) for n in names])
        out["arm_jerk_max"][p] = np.abs(qj[:, arm]).max(axis=1)
        out["waist_jerk_max"][p] = (
            np.abs(qj[:, waist]).max(axis=1) if waist.any() else np.zeros(frames)
        )
        out["leg_jerk_max"][p] = np.abs(qj[:, leg]).max(axis=1)
        foot_world = rubber_foot_world_volume_points(urdf_path, q, root, yaw)
        left, right = foot_world[:, :n_left_foot], foot_world[:, n_left_foot:]
        out["foot_zmin_m"][p] = np.stack(
            [left[..., 2].min(axis=1), right[..., 2].min(axis=1)], axis=1
        )
        centroids = np.stack([left.mean(axis=1), right.mean(axis=1)], axis=1)
        out["foot_speed_cm_s"][p] = (
            np.linalg.norm(np.diff(centroids[:, :, :2], axis=0), axis=-1) * fps * 100
        )
        out["joint_positions"][p] = q
        out["root_translation"][p] = root
        out["root_yaw"][p] = yaw
    return FrameDiagnostics(
        label=label,
        sequence_id=sequence_id,
        fps=fps,
        phases=phases,
        expected_joint=expected_joint,
        joint_names=joint_names,
        **out,
    )


def phase_windows(phases: np.ndarray) -> dict[str, tuple[int, int]]:
    """First contiguous window of each phase name present (inclusive-exclusive)."""
    windows: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(PHASE_NAMES):
        idx = np.flatnonzero(phases == index)
        if idx.size:
            windows[name] = (int(idx[0]), int(idx[-1]) + 1)
    return windows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _camera_from_reference(result_root: Path, stem: str, object_poses_robot: np.ndarray):
    import mujoco

    centers = []
    for p in (1, 2):
        centers.append(
            np.asarray(load_npz_mapping(_resolve(result_root, stem, p))["root_translation"])
        )
    all_centers = np.concatenate(centers + [object_poses_robot[:, :3, 3]])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(all_centers, axis=0)
    camera.distance = max(np.ptp(all_centers[:, :2], axis=0).max() * 1.8, 3.0)
    camera.azimuth = 135
    camera.elevation = -22
    return camera


class SceneRenderer:
    """Frozen-v41-style G1 mesh renderer with a camera fixed from a reference result."""

    def __init__(
        self,
        sequence_id: str,
        reference_root: Path,
        *,
        motion_root: Path = Path("data/core4d/motions"),
        object_models_root: Path = Path("data/core4d/object_models"),
        g1_xml: Path = DEFAULT_G1_XML,
        width: int = 960,
        height: int = 720,
        camera_override: dict[str, object] | None = None,
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        from .mujoco_scene import build_dual_g1_scene_xml

        self.stem = sequence_id.replace("/", "_")
        sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
        pose_path = sequence_dir / "smooth_objposes_so3.npy"
        if not pose_path.exists():
            pose_path = sequence_dir / "smooth_objposes.npy"
        self.object_poses = poses_core4d_to_robot(np.load(pose_path))
        scene_xml = Path("outputs/mujoco") / self.stem / "dual_g1_render.xml"
        build_dual_g1_scene_xml(
            g1_xml, resolve_object_model(sequence_dir, object_models_root), scene_xml
        )
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shared_object")
        ]
        self.camera = _camera_from_reference(reference_root, self.stem, self.object_poses)
        if camera_override:
            for key, value in camera_override.items():
                if key == "lookat":
                    self.camera.lookat[:] = value
                else:
                    setattr(self.camera, key, value)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width, self.height = width, height

    def camera_params(self) -> dict[str, object]:
        return {
            "lookat": [float(v) for v in self.camera.lookat],
            "distance": float(self.camera.distance),
            "azimuth": float(self.camera.azimuth),
            "elevation": float(self.camera.elevation),
            "width": self.width,
            "height": self.height,
        }

    def load(self, result_root: Path) -> dict[int, dict[str, np.ndarray]]:
        return {p: load_npz_mapping(_resolve(result_root, self.stem, p)) for p in (1, 2)}

    def render_frame(
        self, trajectories: dict[int, dict[str, np.ndarray]], frame: int
    ) -> np.ndarray:
        import mujoco

        from .mujoco_scene import _set_robot_qpos

        for person, prefix in ((1, "r1_"), (2, "r2_")):
            t = trajectories[person]
            _set_robot_qpos(
                self.model,
                self.data,
                prefix,
                t["joint_names"],
                t["joint_positions"][frame],
                t["root_translation"][frame],
                float(t["root_yaw"][frame]),
            )
        self.data.mocap_pos[self.mocap_id] = self.object_poses[frame, :3, 3]
        quat = Rotation.from_matrix(self.object_poses[frame, :3, :3]).as_quat()
        self.data.mocap_quat[self.mocap_id] = quat[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate_tile(
    image: np.ndarray,
    *,
    title: str,
    frame: int,
    fps: float,
    phase: str,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.rectangle([0, 0, pil.width, 46], fill=(0, 0, 0, 150))
    draw.text((12, 8), title, font=_font(26), fill=(255, 255, 255))
    draw.text(
        (12, pil.height - 34),
        f"frame {frame:3d}  t={frame / fps:5.2f}s  phase: {phase}",
        font=_font(20),
        fill=(255, 255, 255),
    )
    y = 56
    for text, color in lines:
        draw.rectangle([8, y - 2, 8 + 14 * len(text) + 8, y + 24], fill=(0, 0, 0, 110))
        draw.text((12, y), text, font=_font(20), fill=color)
        y += 30
    return np.asarray(pil)


def contact_lines(diag: FrameDiagnostics, frame: int) -> list[tuple[str, tuple[int, int, int]]]:
    exp = bool(diag.expected_joint[frame])
    pred = bool(diag.predicted_joint[frame])
    pen = max(diag.hand_pen_cm[1][frame], diag.hand_pen_cm[2][frame])
    lines = [
        (
            f"joint contact: source {'ON ' if exp else 'off'} | robot {'ON ' if pred else 'off'}",
            (120, 255, 120) if exp == pred else (255, 170, 60),
        ),
        (
            f"hand penetration: R1 {diag.hand_pen_cm[1][frame]:.1f} cm  R2 {diag.hand_pen_cm[2][frame]:.1f} cm",
            (255, 90, 90) if pen > 0.5 else (200, 200, 200),
        ),
    ]
    return lines


def render_comparison_video(
    sequence_id: str,
    methods: list[tuple[str, Path, FrameDiagnostics]],
    output_path: Path,
    *,
    reference_root: Path,
    frames: tuple[int, int] | None = None,
    fps: float = 15.0,
    slow_factor: int = 1,
    tile_width: int = 960,
    tile_height: int = 720,
    strip_height: int = 70,
) -> dict[str, object]:
    """Side-by-side video with one shared camera plus a joint-contact timeline strip."""
    from PIL import Image, ImageDraw

    renderer = SceneRenderer(sequence_id, reference_root, width=tile_width, height=tile_height)
    try:
        trajectories = [renderer.load(root) for _, root, _ in methods]
        total = min(len(t[1]["joint_positions"]) for t in trajectories)
        start, end = frames if frames else (0, total)
        end = min(end, total)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            output_path, fps=fps / slow_factor, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            for frame in range(start, end):
                tiles = []
                for (label, _root, diag), traj in zip(methods, trajectories, strict=True):
                    image = renderer.render_frame(traj, frame)
                    phase = (
                        PHASE_NAMES[int(diag.phases[frame])] if frame < len(diag.phases) else "?"
                    )
                    tiles.append(
                        annotate_tile(
                            image,
                            title=label,
                            frame=frame,
                            fps=fps,
                            phase=phase,
                            lines=contact_lines(diag, frame),
                        )
                    )
                canvas = np.concatenate(tiles, axis=1)
                strip = Image.new(
                    "RGB", (canvas.shape[1], strip_height * (len(methods) + 1)), (20, 20, 20)
                )
                draw = ImageDraw.Draw(strip)
                lane_w = canvas.shape[1] - 260
                x0 = 240
                scale = lane_w / max(total, 1)
                rows = [("source joint contact", methods[0][2].expected_joint, (180, 180, 180))] + [
                    (f"{label}: robot joint contact", diag.predicted_joint, (90, 170, 255))
                    for label, _r, diag in methods
                ]
                for li, (name, mask, color) in enumerate(rows):
                    y = li * strip_height + 10
                    draw.text((10, y + 12), name, font=_font(18), fill=(230, 230, 230))
                    draw.rectangle(
                        [x0, y + 8, x0 + lane_w, y + strip_height - 18], outline=(90, 90, 90)
                    )
                    for t in np.flatnonzero(mask):
                        draw.rectangle(
                            [x0 + t * scale, y + 10, x0 + (t + 1) * scale, y + strip_height - 20],
                            fill=color,
                        )
                    if li > 0:
                        diag = methods[li - 1][2]
                        bad = (diag.hand_pen_cm[1] > 0.5) | (diag.hand_pen_cm[2] > 0.5)
                        for t in np.flatnonzero(bad):
                            draw.rectangle(
                                [
                                    x0 + t * scale,
                                    y + strip_height - 20,
                                    x0 + (t + 1) * scale,
                                    y + strip_height - 14,
                                ],
                                fill=(255, 80, 80),
                            )
                xf = x0 + frame * scale
                draw.line([xf, 0, xf, strip.height], fill=(255, 255, 0), width=3)
                draw.text(
                    (x0 + lane_w + 8, 20), "red = pen>0.5cm", font=_font(14), fill=(255, 120, 120)
                )
                writer.append_data(np.concatenate([canvas, np.asarray(strip)], axis=0))
        params = renderer.camera_params()
    finally:
        renderer.close()
    return {
        "output": str(output_path),
        "frames": [start, end],
        "camera": params,
        "fps": fps / slow_factor,
    }


def render_keyframes(
    sequence_id: str,
    methods: list[tuple[str, Path, FrameDiagnostics]],
    frames: list[tuple[str, int]],
    output_path: Path,
    *,
    reference_root: Path,
    tile_width: int = 640,
    tile_height: int = 480,
    zoom: float | None = None,
) -> None:
    """Grid: rows = methods, columns = named frames; identical camera for all cells."""
    from PIL import Image, ImageDraw

    override = None
    renderer = SceneRenderer(sequence_id, reference_root, width=tile_width, height=tile_height)
    if zoom is not None:
        renderer.camera.distance = renderer.camera.distance * zoom
    try:
        trajectories = [renderer.load(root) for _, root, _ in methods]
        header = 40
        label_w = 220
        canvas = Image.new(
            "RGB",
            (label_w + tile_width * len(frames), header + tile_height * len(methods)),
            (255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        for ci, (name, frame) in enumerate(frames):
            draw.text(
                (label_w + ci * tile_width + 10, 8),
                f"{name} (frame {frame})",
                font=_font(22),
                fill=(0, 0, 0),
            )
        for ri, ((label, _root, diag), traj) in enumerate(zip(methods, trajectories, strict=True)):
            words, lines_, cur = label.split(), [], ""
            for w in words:
                if len(cur) + len(w) + 1 > 16 and cur:
                    lines_.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            lines_.append(cur)
            for k, line in enumerate(lines_):
                draw.text((10, header + ri * tile_height + 10 + 26 * k), line, font=_font(20), fill=(0, 0, 0))
            for ci, (_name, frame) in enumerate(frames):
                if zoom is not None:
                    # Object-centred close-up; identical for every method in this column.
                    renderer.camera.lookat[:] = renderer.object_poses[frame, :3, 3]
                image = renderer.render_frame(traj, frame)
                pen = max(diag.hand_pen_cm[1][frame], diag.hand_pen_cm[2][frame])
                exp = bool(diag.expected_joint[frame])
                pred = bool(diag.predicted_joint[frame])
                tile = annotate_tile(
                    image,
                    title="",
                    frame=frame,
                    fps=diag.fps,
                    phase=PHASE_NAMES[int(diag.phases[frame])],
                    lines=[
                        (
                            f"joint contact src {'ON' if exp else 'off'} / robot {'ON' if pred else 'off'}",
                            (120, 255, 120) if exp == pred else (255, 170, 60),
                        ),
                        (
                            f"max hand pen {pen:.1f} cm",
                            (255, 90, 90) if pen > 0.5 else (200, 200, 200),
                        ),
                    ],
                )
                canvas.paste(
                    Image.fromarray(tile), (label_w + ci * tile_width, header + ri * tile_height)
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        renderer.close()
    del override
