
import time

import mujoco
import numpy as np

from mujoco.studio import app as studio_app
from mujoco.studio import launch_web
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import viewer_protocol


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "universal_robots_ur5e/scene.xml"

DT = 0.002

DAMPING = 1e-4
INTEGRATION_DT = 1.0

MAX_ANGVEL = 0.0

GRAVITY_COMPENSATION = True

HTTP_PORT = 0

# Circle trajectory
CIRCLE_RADIUS = 0.1
CIRCLE_CENTER_X = 0.5
CIRCLE_CENTER_Y = 0.0
CIRCLE_FREQUENCY = 0.5

# Orientation control
# True: maintain initial target orientation
# False: use current mocap target orientation
FIXED_ORIENTATION = True


# ============================================================
# Circle trajectory
# ============================================================

def circle(t, r, h, k, f):

    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k

    return np.array([x, y])


# ============================================================
# Viewer controls
# ============================================================

class ViewerControls(plugin.SimPlugin):

    def __init__(self):
        self.paused = False
        self.speed = 100.0

    @messages.handler
    def on_step_control(
        self,
        snapshot: messages.StepControlSnapshot,
    ):

        self.paused = (
            snapshot.pause_state != sim.PauseState.UNPAUSED
        )

        self.speed = snapshot.speed


# ============================================================
# Load model
# ============================================================

def load_model():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    model.opt.timestep = DT

    # End-effector site
    site_id = model.site("attachment_site").id

    # Gravity compensation
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    body_ids = [
        model.body(name).id
        for name in body_names
    ]

    if GRAVITY_COMPENSATION:
        model.body_gravcomp[body_ids] = 1.0

    # Joint configuration
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    joint_ids = np.array([
        model.joint(name).id
        for name in joint_names
    ])

    dof_ids = model.jnt_dofadr[joint_ids]
    qpos_ids = model.jnt_qposadr[joint_ids]

    actuator_ids = np.array([
        model.actuator(name).id
        for name in joint_names
    ])

    # Mocap target
    mocap_id = int(
        model.body("target").mocapid[0]
    )

    if mocap_id < 0:
        raise ValueError(
            "target must be a mocap body."
        )

    # Initial keyframe
    key_id = model.key("home").id

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        key_id,
    )

    mujoco.mj_forward(model, data)

    return (
        model,
        data,
        site_id,
        mocap_id,
        joint_ids,
        dof_ids,
        qpos_ids,
        actuator_ids,
    )


# ============================================================
# Differential IK
# ============================================================

class DifferentialIK:

    def __init__(self, model, site_id):

        self.model = model
        self.site_id = site_id

        self.jac = np.zeros(
            (6, model.nv)
        )

        self.diag = DAMPING * np.eye(6)

        self.error = np.zeros(6)

        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]

        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def compute(
        self,
        data,
        target_pos,
        target_quat,
    ):

        # Position error
        self.error_pos[:] = (
            target_pos -
            data.site(self.site_id).xpos
        )

        # Orientation error
        mujoco.mju_mat2Quat(
            self.site_quat,
            data.site(self.site_id).xmat,
        )

        mujoco.mju_negQuat(
            self.site_quat_conj,
            self.site_quat,
        )

        mujoco.mju_mulQuat(
            self.error_quat,
            target_quat,
            self.site_quat_conj,
        )

        mujoco.mju_quat2Vel(
            self.error_ori,
            self.error_quat,
            1.0,
        )

        # Jacobian
        mujoco.mj_jacSite(
            self.model,
            data,
            self.jac[:3],
            self.jac[3:],
            self.site_id,
        )

        # Damped Least Squares
        dq = (
            self.jac.T
            @ np.linalg.solve(
                self.jac @ self.jac.T + self.diag,
                self.error,
            )
        )

        return dq


# ============================================================
# Main
# ============================================================

def main():

    (
        model,
        data,
        site_id,
        mocap_id,
        joint_ids,
        dof_ids,
        qpos_ids,
        actuator_ids,
    ) = load_model()

    ik = DifferentialIK(
        model,
        site_id,
    )

    controls = ViewerControls()

    config = viewer_protocol.ViewerConfig(
        title="UR5e Circle Tracking",
        http_port=HTTP_PORT,
        open_browser=True,
    )

    # Preserve initial target height and orientation
    target_height = data.mocap_pos[mocap_id, 2]

    initial_target_quat = (
        data.mocap_quat[mocap_id].copy()
    )

    print("Starting UR5e Circle Tracking...")

    with launch_web.launch(
        config,
        plugins=[
            studio_app.App(),
            controls,
        ],
    ) as handle:

        while handle.is_running():

            loop_start = time.perf_counter()

            # Synchronize browser input and simulation state
            model, data = handle.sync(
                model,
                data,
            )

            if model is None or data is None:
                continue

            # Pause
            if controls.paused:

                time.sleep(0.01)
                continue

            # ------------------------------------------------
            # 1. Circle trajectory
            # ------------------------------------------------

            t = data.time

            xy = circle(
                t,
                CIRCLE_RADIUS,
                CIRCLE_CENTER_X,
                CIRCLE_CENTER_Y,
                CIRCLE_FREQUENCY,
            )

            data.mocap_pos[mocap_id, 0] = xy[0]
            data.mocap_pos[mocap_id, 1] = xy[1]
            data.mocap_pos[mocap_id, 2] = target_height

            # ------------------------------------------------
            # 2. Target orientation
            # ------------------------------------------------

            if FIXED_ORIENTATION:

                data.mocap_quat[mocap_id] = (
                    initial_target_quat
                )

            target_pos = (
                data.mocap_pos[mocap_id].copy()
            )

            target_quat = (
                data.mocap_quat[mocap_id].copy()
            )

            # ------------------------------------------------
            # 3. Update kinematics
            # ------------------------------------------------

            mujoco.mj_forward(model, data)

            # ------------------------------------------------
            # 4. Differential IK
            # ------------------------------------------------

            dq = ik.compute(
                data,
                target_pos,
                target_quat,
            )

            # ------------------------------------------------
            # 5. Velocity limitation
            # ------------------------------------------------

            controlled_dq = dq[dof_ids]

            if MAX_ANGVEL > 0:

                dq_abs_max = np.max(
                    np.abs(controlled_dq)
                )

                if dq_abs_max > MAX_ANGVEL:

                    dq *= (
                        MAX_ANGVEL / dq_abs_max
                    )

            # ------------------------------------------------
            # 6. Integrate joint velocities
            # ------------------------------------------------

            q_desired = data.qpos.copy()

            mujoco.mj_integratePos(
                model,
                q_desired,
                dq,
                INTEGRATION_DT,
            )

            # Joint limits
            for joint_id, qpos_id in zip(
                joint_ids,
                qpos_ids,
            ):

                if model.jnt_limited[joint_id]:

                    lower, upper = (
                        model.jnt_range[joint_id]
                    )

                    q_desired[qpos_id] = np.clip(
                        q_desired[qpos_id],
                        lower,
                        upper,
                    )

            # ------------------------------------------------
            # 7. Actuator control
            # ------------------------------------------------

            data.ctrl[actuator_ids] = (
                q_desired[qpos_ids]
            )

            # ------------------------------------------------
            # 8. Physics
            # ------------------------------------------------

            mujoco.mj_step(
                model,
                data,
            )

            # ------------------------------------------------
            # 9. Real-time pacing
            # ------------------------------------------------

            speed = max(
                controls.speed,
                1e-6,
            )

            period = (
                model.opt.timestep
                * 100.0
                / speed
            )

            elapsed = (
                time.perf_counter()
                - loop_start
            )

            time.sleep(
                max(0.0, period - elapsed)
            )

    print("Simulation finished.")


if __name__ == "__main__":
    main()
