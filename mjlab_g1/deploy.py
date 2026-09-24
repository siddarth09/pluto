"""Run a trained policy on the real G1 over unitree_hg, or on a MuJoCo stand-in.

SAFE ONLY WITH THE ROBOT HOISTED, FEET CLEAR, unless you know the policy stands.

    python pluto/mjlab_g1/deploy.py --dry-run     # read state, run policy, publish nothing
    python pluto/mjlab_g1/deploy.py --pose-only   # ramp to the default pose and hold
    python pluto/mjlab_g1/deploy.py --hold        # policy, zero velocity command
    python pluto/mjlab_g1/deploy.py --teleop      # keyboard driving
    python pluto/mjlab_g1/deploy.py --sim         # MuJoCo instead of the robot

Run as a FILE, not with -m: no mjlab dependency, and -m would import it.

Conventions, all read out of mjlab rather than assumed:
  obs layout    per-term blocks, each term's history flattened oldest->newest
  history init  backfilled with the first frame, not zeros
  gravity       R(q)^-1 @ [0,0,-1], unit vector; quaternion is (w,x,y,z)
  joint_pos     relative to the default pose
  ordering      obs AND actions are both identity with the SDK motor order
  gains         uncorrected nominal Kp; the robot supplies the shortfall itself
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/sid/unitree_sdk2_python")
# import sim_backend without the package __init__, which pulls in mjlab
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnxruntime as ort  # noqa: E402
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,  # noqa: E402
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient)
except ImportError:  # older SDK layout
    from unitree_sdk2py.g1.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient)

N_MOTOR = 29
N_SLOT = 35
HZ = 50.0
DT = 1.0 / HZ

# position error is how the policy modulates torque, not a fault; ceiling only
MAX_TRACK_ERR = 2.5     # rad, per joint
# per-joint guards, as a fraction of each motor's rated speed/effort
VEL_GUARD_FRAC = 0.85
TAU_GUARD_FRAC = 0.95
MAX_RUN_S = 60.0
# teleop exits on idle rather than a fixed duration
TELEOP_IDLE_EXIT_S = 60.0

# Ctrl-C: settle under the policy, then hold pose. Second Ctrl-C releases gains.
SETTLE_S = 2.0
HOLD_MAX_S = 120.0

# wz limit = the curriculum's trained range; vy kept low, lateral was never curriculum'd
TELEOP_VX = 0.8
TELEOP_VY = 0.4
TELEOP_WZ = 0.7
TELEOP_STEP = 0.1        # per keypress
TELEOP_HOLD_S = 0.4      # decay to zero this long after the last keypress


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """R(q)^-1 @ [0,0,-1]. Matches mjlab's quat_apply_inverse with a unit gravity."""
    w, x, y, z = quat_wxyz
    # -R[:,2] in body frame
    return np.array([
        -(2.0 * (x * z - w * y)),
        -(2.0 * (y * z + w * x)),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


class Keyboard:
    """Non-blocking single-key reader. Hold-to-move: terminal key repeat keeps the
    command alive, and it decays to zero TELEOP_HOLD_S after the last keypress so
    a walk-away cannot leave the robot driving."""

    HELP = ("  w/s forward/back   a/d turn left/right   <- -> strafe   "
            "space stop   ^C settle+hold   x exit")


    def __init__(self):
        import termios
        self._termios = termios
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        self.cmd = np.zeros(3)
        self.last_key = 0.0
        self.quit = False
        self._esc = 0          # arrow keys arrive as ESC [ A/B/C/D

    def __enter__(self):
        import tty
        # raw, not cbreak: cbreak leaves ISIG on and Ctrl-Z would suspend publishing
        tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.saved)

    def poll(self) -> np.ndarray:
        import select
        got = False
        while select.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1)
            got = True
            # arrows are ESC [ A/B/C/D, possibly split across reads
            if self._esc == 0 and c == "\x1b":
                self._esc = 1
                continue
            if self._esc == 1:
                self._esc = 2 if c == "[" else 0
                continue
            if self._esc == 2:
                self._esc = 0
                if c == "D":                      # left arrow -> strafe left
                    self.cmd[1] += TELEOP_STEP
                elif c == "C":                    # right arrow -> strafe right
                    self.cmd[1] -= TELEOP_STEP
                elif c == "A":                    # up arrow, same as w
                    self.cmd[0] += TELEOP_STEP
                elif c == "B":                    # down arrow, same as s
                    self.cmd[0] -= TELEOP_STEP
                continue
            c = c.lower()
            if c == "w":
                self.cmd[0] += TELEOP_STEP        # forward
            elif c == "s":
                self.cmd[0] -= TELEOP_STEP        # back
            elif c == "a":
                self.cmd[2] += TELEOP_STEP        # turn left (CCW)
            elif c == "d":
                self.cmd[2] -= TELEOP_STEP        # turn right
            elif c == "q":
                self.cmd[1] += TELEOP_STEP        # strafe left (alias for <-)
            elif c == "e":
                self.cmd[1] -= TELEOP_STEP        # strafe right
            elif c == " ":
                self.cmd[:] = 0.0
            elif c in ("x", "\x03"):        # x or Ctrl-C: settle and hold
                self.quit = True
            # all other bytes ignored
        now = time.time()
        if got:
            self.last_key = now
        elif now - self.last_key > TELEOP_HOLD_S:
            self.cmd *= 0.85          # decay toward zero when keys are released
            self.cmd[np.abs(self.cmd) < 0.02] = 0.0
        lim = np.array([TELEOP_VX, TELEOP_VY, TELEOP_WZ])
        self.cmd = np.clip(self.cmd, -lim, lim)
        return self.cmd


def smoothstep(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class History:
    """Per-term ring buffer, backfilled with the first frame like mjlab's."""

    def __init__(self, dim: int, length: int):
        self.length = length
        self.buf: np.ndarray | None = None
        self.dim = dim

    def append(self, x: np.ndarray) -> None:
        if self.buf is None:
            self.buf = np.tile(x, (self.length, 1))
        else:
            self.buf = np.roll(self.buf, -1, axis=0)
            self.buf[-1] = x

    def flat(self) -> np.ndarray:
        assert self.buf is not None
        return self.buf.reshape(-1)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--onnx", default=None, help="policy .onnx; default = newest run's")
    ap.add_argument("--map", default=str(here / "deploy_map.json"))
    ap.add_argument("--iface", default="enp130s0")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0, help="policy phase seconds")
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="scale the deployment gains; start below 1 for the first test")
    ap.add_argument("--hard-abort", action="store_true",
                    help="watchdog trips go straight to limp instead of settling "
                         "and holding. Limp drops a standing robot.")
    ap.add_argument("--vel-frac", type=float, default=VEL_GUARD_FRAC,
                    help="velocity guard as a fraction of each motor's rated speed")
    ap.add_argument("--pose-only", action="store_true",
                    help="ramp to the default pose and HOLD it, no policy. Isolates "
                         "'does the robot accept lowcmd' from 'is the policy sane'.")
    ap.add_argument("--max-err", type=float, default=MAX_TRACK_ERR,
                    help="tracking-error ceiling, rad")
    ap.add_argument("--limit-margin", type=float, default=0.05,
                    help="rad kept clear of each joint's physical limit")
    ap.add_argument("--hold", action="store_true", help="force zero velocity command")
    ap.add_argument("--idle-exit", type=float, default=TELEOP_IDLE_EXIT_S,
                    help="teleop: exit after this many seconds with no keypress. "
                         "0 disables (not recommended).")
    ap.add_argument("--teleop", action="store_true",
                    help=f"keyboard driving. vx +-{TELEOP_VX}, vy +-{TELEOP_VY}, "
                         f"wz +-{TELEOP_WZ} rad/s")
    ap.add_argument("--sim", action="store_true",
                    help="drive a MuJoCo stand-in instead of the robot. Same "
                         "observation assembly, action mapping, teleop and "
                         "safety logic -- rehearsal for the real thing.")
    ap.add_argument("--no-viewer", action="store_true",
                    help="--sim without the MuJoCo window")
    ap.add_argument("--dry-run", action="store_true",
                    help="read state and run the policy but PUBLISH NOTHING")
    args = ap.parse_args()

    if args.hold:
        args.vx = args.vy = args.wz = 0.0

    dm = json.loads(Path(args.map).read_text())
    assert dm["obs_is_identity_with_sdk"], "observation ordering is not SDK identity"
    assert dm["policy"]["control_hz"] == HZ, f"map says {dm['policy']['control_hz']} Hz"
    # actions are in joint order == SDK order; actuator order would be wrong
    assert dm["sdk_from_action"] == list(range(N_MOTOR)), "action order not identity"
    default = np.array(dm["default_pose_sdk_order"])
    scale = np.array(dm["action_scale_sdk_order"])
    kp = np.array(dm["deploy_kp_sdk_order"]) * args.kp_scale
    jlo = np.array(dm["joint_lower_sdk_order"]) + args.limit_margin
    jhi = np.array(dm["joint_upper_sdk_order"]) - args.limit_margin
    vmax = np.array(dm["vel_limit_sdk_order"]) * args.vel_frac
    taumax = np.array(dm["effort_limit_sdk_order"]) * TAU_GUARD_FRAC
    kd = np.array(dm["deploy_kd_sdk_order"]) * args.kp_scale
    hist_len = dm["policy"]["history_length"]

    onnx_path = args.onnx
    if onnx_path is None:
        runs = sorted(Path("/home/sid/projects25/src/logs/rsl_rl/"
                           "pluto_g1_velocity_identified").glob("*/*.onnx"))
        onnx_path = str(runs[-1])
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_dim = sess.get_inputs()[0].shape[-1]
    print(f"policy : {Path(onnx_path).name}  input {in_dim}  history {hist_len}")
    expect = (3 + 3 + N_MOTOR + N_MOTOR + N_MOTOR + 3) * hist_len
    assert in_dim == expect, f"onnx expects {in_dim}, layout gives {expect}"

    # per-term history, concatenated in this order
    h_ang = History(3, hist_len)
    h_grav = History(3, hist_len)
    h_qpos = History(N_MOTOR, hist_len)
    h_qvel = History(N_MOTOR, hist_len)
    h_act = History(N_MOTOR, hist_len)
    h_cmd = History(3, hist_len)

    state: dict = {}
    sim = viewer = None
    if args.sim:
        import mujoco.viewer

        import sim_backend
        sim = sim_backend.load(
            Path(args.map),
            Path(args.map).parent.parent / "sysid/results/g1_legs_identified.json",
        )
        state.update(sim.read())
        print("SIM backend: MuJoCo stand-in, no robot involved")
    else:
        ChannelFactoryInitialize(0, args.iface)
    if not args.sim:
        ChannelSubscriber("rt/lowstate", LowState_).Init(
            lambda m: state.update(
            q=np.array([m.motor_state[i].q for i in range(N_MOTOR)]),
            dq=np.array([m.motor_state[i].dq for i in range(N_MOTOR)]),
            tau=np.array([m.motor_state[i].tau_est for i in range(N_MOTOR)]),
            gyro=np.array(m.imu_state.gyroscope),
            quat=np.array(m.imu_state.quaternion),
            mode_machine=m.mode_machine), 20)
    t0 = time.time()
    while "q" not in state and time.time() - t0 < 5.0:
        time.sleep(0.02)
    if "q" not in state:
        raise SystemExit("no lowstate -- check --iface and the cable")
    if args.sim and not args.no_viewer:
        import os
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            print("viewer open; press x or Ctrl-C to stop")
        else:
            print("no display found, running headless")

    # a live motion service also publishes rt/lowcmd; check every run, it re-arms
    if args.sim:
        mode = {"name": ""}
    else:
        ms = MotionSwitcherClient()
        ms.SetTimeout(5.0)
        ms.Init()
        _, mode = ms.CheckMode()
    active = (mode or {}).get("name", "")
    print(f"motion service: {mode}"
          + (f"   <-- '{active}' IS ACTIVE" if active else "   (released, lowcmd is ours)"))
    if active and not args.dry_run:
        raise SystemExit(
            f"\nRefusing to run: the '{active}' controller owns the motors and will "
            f"fight every command.\nRelease it first:\n"
            f"  cd ~/projects25/src/pluto/sysid && env -u PYTHONPATH "
            f"~/g1_real_env/bin/python scripts/release_mode.py"
        )

    q_start = state["q"].copy()
    print(f"mode_machine {state['mode_machine']}   |quat| {np.linalg.norm(state['quat']):.4f}")
    print(f"gravity_b    {projected_gravity(state['quat']).round(3)}  (upright ~ [0,0,-1])")
    print(f"command      vx={args.vx} vy={args.vy} wz={args.wz}")
    print(f"kp (legs)    {kp[:6].round(1)}")

    cmd_vec = np.array([args.vx, args.vy, args.wz])
    if args.teleop:
        print(f"\nTELEOP  {Keyboard.HELP}")
        idle = (f"exits after {args.idle_exit:.0f}s idle" if args.idle_exit > 0
                else "NO idle timeout")
        print(f"  runs until you press x  ({idle})")
        print(f"  limits: vx +-{TELEOP_VX}  vy +-{TELEOP_VY}  wz +-{TELEOP_WZ} rad/s"
              f"   (release keys -> decays to stop)")
        print("  holonomic: vx and vy combine, so w+a walks diagonally forward-left")
    last_action = np.zeros(N_MOTOR)

    def build_obs() -> np.ndarray:
        h_ang.append(state["gyro"])
        h_grav.append(projected_gravity(state["quat"]))
        h_qpos.append(state["q"] - default)
        h_qvel.append(state["dq"])
        h_act.append(last_action)
        h_cmd.append(cmd_vec)
        return np.concatenate([h_ang.flat(), h_grav.flat(), h_qpos.flat(),
                               h_qvel.flat(), h_act.flat(), h_cmd.flat()])

    if args.dry_run:
        print("\n--- DRY RUN: 3 s of inference, nothing published ---")
        acts = []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            obs = build_obs().astype(np.float32)[None, :]
            a = sess.run(None, {in_name: obs})[0][0]
            last_action = a
            acts.append(a)
            time.sleep(DT)
        A = np.stack(acts)
        q_des = default + A * scale
        print(f"inferences         {len(acts)}")
        print(f"|action| max       {np.abs(A).max():.3f}   mean {np.abs(A).mean():.3f}")
        print(f"q_des legs range   {(q_des[:, :6].max(0) - q_des[:, :6].min(0)).round(3)}")
        print(f"q_des - q now      {np.abs(q_des[-1] - state['q']).max():.3f} rad max")
        outside = int(np.sum((q_des < jlo) | (q_des > jhi)))
        print(f"commands outside limits: {outside} of {q_des.size} "
              f"({100 * outside / q_des.size:.1f}%) -- clipped at run time")
        print("\nsanity: |action| should be O(1), and q_des near the default pose.")
        return

    if not args.sim:
        print("\nThis RELEASES nothing and COMMANDS the legs at the gains above.")
        print("Both feet must be clear of the floor. Keep the stop reachable.")
        if input("Type RUN to proceed: ").strip() != "RUN":
            print("aborted")
            return

    if args.sim:
        pub = crc = cmd = None
    else:
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()
        crc = CRC()
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = 0
        cmd.mode_machine = state["mode_machine"]

    def publish(q_des: np.ndarray, gain: float) -> None:
        if args.sim:
            sim.step(q_des, gain, int(round(DT / sim_backend.PHYSICS_DT)))
            state.update(sim.read())
            if viewer is not None and viewer.is_running():
                viewer.sync()
            return
        for j in range(N_MOTOR):
            mc = cmd.motor_cmd[j]
            mc.mode, mc.dq, mc.tau = 1, 0.0, 0.0
            mc.q = float(q_des[j])
            mc.kp = float(kp[j] * gain)
            mc.kd = float(kd[j] * gain)
        for j in range(N_MOTOR, N_SLOT):
            mc = cmd.motor_cmd[j]
            mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd = 1, 0.0, 0.0, 0.0, 0.0, 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    class _PoseOnlyDone(Exception):
        pass

    log, aborted, kb = [], None, None
    import signal
    stop = {"n": 0}

    def _on_sigint(signum, frame):
        stop["n"] += 1
        if stop["n"] >= 2:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)
    n_ramp = int(args.ramp / DT)
    try:
        next_t = time.time()
        # gains in, holding where it is
        for g in np.linspace(0.0, 1.0, n_ramp):
            publish(q_start, float(g))
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
        # move to the policy's default pose
        for a in smoothstep(n_ramp):
            publish(q_start + a * (default - q_start), 1.0)
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
        if args.pose_only:
            print(f"holding the default pose for {args.duration}s (no policy)")
            t_start = time.time()
            while time.time() - t_start < min(args.duration, MAX_RUN_S):
                publish(default, 1.0)
                log.append((time.time() - t_start, state["q"].copy(),
                            state["dq"].copy(), default.copy(),
                            np.zeros(N_MOTOR), state["tau"].copy(), 0))
                next_t += DT
                time.sleep(max(0.0, next_t - time.time()))
            raise _PoseOnlyDone

        # policy
        kb = Keyboard() if args.teleop else None
        t_start = time.time()
        if kb:
            kb.__enter__()
            kb.last_key = t_start
        while True:
            if kb is None and time.time() - t_start >= min(args.duration, MAX_RUN_S):
                break
            if kb is not None:
                if stop["n"] == 0:
                    cmd_vec[:] = kb.poll()
                    if kb.quit:
                        stop["n"], stop["t"] = 1, time.time()
                else:
                    cmd_vec[:] = 0.0        # settling: zero command, policy still on
                idle_s = time.time() - kb.last_key
                if (stop["n"] == 0 and args.idle_exit > 0
                        and idle_s > args.idle_exit):
                    aborted = f"idle for {idle_s:.0f}s, no keypress"
                    stop["n"], stop["t"] = 1, time.time()
                print(f"\r  vx {cmd_vec[0]:+.2f}  vy {cmd_vec[1]:+.2f}  "
                      f"wz {cmd_vec[2]:+.2f}   t {time.time() - t_start:5.0f}s  "
                      f"idle {idle_s:4.1f}s   ", end="", flush=True)
            elif stop["n"]:
                cmd_vec[:] = 0.0
            if stop["n"]:
                stop.setdefault("t", time.time())
                if time.time() - stop["t"] > SETTLE_S:
                    break
            obs = build_obs().astype(np.float32)[None, :]
            action = sess.run(None, {in_name: obs})[0][0]
            last_action = action
            q_des = default + action * scale
            n_clipped = int(np.sum((q_des < jlo) | (q_des > jhi)))
            q_des = np.clip(q_des, jlo, jhi)
            err = np.abs(q_des - state["q"])
            vfrac = np.abs(state["dq"]) / vmax
            tfrac = np.abs(state["tau"]) / taumax
            if err.max() > args.max_err or vfrac.max() > 1.0 or tfrac.max() > 1.0:
                j = int(np.argmax(np.maximum(vfrac, tfrac)))
                aborted = (f"watchdog on {dm['sdk_joint_order'][j]}: "
                           f"|dq|={abs(state['dq'][j]):.1f}/{vmax[j]:.0f} "
                           f"|tau|={abs(state['tau'][j]):.1f}/{taumax[j]:.0f} "
                           f"max|err|={err.max():.3f}")
                # settle and hold; --hard-abort goes limp instead
                if args.hard_abort:
                    break
                stop["n"] = max(stop["n"], 1)
                stop.setdefault("t", time.time())
            publish(q_des, 1.0)
            log.append((time.time() - t_start, state["q"].copy(), state["dq"].copy(),
                        q_des.copy(), action.copy(), state["tau"].copy(), n_clipped))
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
        if stop["n"]:
            # hold on gains; second Ctrl-C drops through to the ramp-out
            hold_q = state["q"].copy()
            print(f"\n  settled -- holding pose. Ctrl-C again to release "
                  f"(auto after {HOLD_MAX_S:.0f}s).")
            t_hold = time.time()
            while time.time() - t_hold < HOLD_MAX_S:
                publish(hold_q, 1.0)
                next_t += DT
                time.sleep(max(0.0, next_t - time.time()))

        # back to where we started, then gains out
        q_last = state["q"].copy()
        for a in smoothstep(n_ramp):
            publish(q_last + a * (q_start - q_last), 1.0)
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
        for g in np.linspace(1.0, 0.0, n_ramp):
            publish(q_start, float(g))
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        aborted = "Ctrl-C (second) -- ramping gains out"
        try:
            q_now = state["q"].copy()
            for g in np.linspace(1.0, 0.0, int(1.0 / DT)):
                publish(q_now, float(g))
                time.sleep(DT)
        except Exception:
            pass
    except _PoseOnlyDone:
        q_last = state["q"].copy()
        for a in smoothstep(n_ramp):
            publish(q_last + a * (q_start - q_last), 1.0)
            next_t += DT
            time.sleep(max(0.0, next_t - time.time()))
    finally:
        try:
            kb.__exit__() if kb else None
        except Exception:
            pass
        for _ in range(100):
            publish(np.zeros(N_MOTOR), 0.0)
            time.sleep(DT)
        print("\ngains zeroed, joints limp")

    if aborted:
        print("ABORTED:", aborted)
    if log:
        t = np.array([r[0] for r in log])
        q = np.stack([r[1] for r in log]); dq = np.stack([r[2] for r in log])
        qd = np.stack([r[3] for r in log]); tau = np.stack([r[5] for r in log])
        out = here / "deploy_log.npz"
        np.savez(out, t=t, q=q, dq=dq, q_des=qd,
                 action=np.stack([r[4] for r in log]), tau_est=tau,
                 sdk_order=dm["sdk_joint_order"], kp=kp, kd=kd)
        print(f"{len(log)} steps at {len(log) / max(t[-1], 1e-9):.0f} Hz -> {out.name}")
        print(f"track err mean {np.abs(qd - q).mean():.4f} max {np.abs(qd - q).max():.3f}")
        print(f"peak |dq| {np.abs(dq).max(0)[:6].round(2)} (legs), "
              f"as % of motor limit {(100 * np.abs(dq).max(0)[:6] / vmax[:6]).round(0)}")
        print(f"peak |tau| {np.abs(tau).max(0)[:6].round(1)} (legs), "
              f"as % of effort limit {(100 * np.abs(tau).max(0)[:6] / taumax[:6]).round(0)}")
        # total command->response lag: transport + servo, not transport alone
        legs = slice(0, 12)
        cmd_var = float(np.abs(qd[:, legs] - qd[:, legs].mean(0)).mean())
        if cmd_var < 0.01:
            print(f"command->response lag: NO SIGNAL (commanded motion {cmd_var:.4f} "
                  f"rad mean deviation). Time-shifting a constant command changes "
                  f"nothing; needs a varying command to estimate.")
            best = None
        else:
            best = min(
                range(0, 11),
                key=lambda k: np.abs(qd[: len(qd) - k, legs] - q[k:, legs]).mean()
                if len(qd) > k else np.inf,
            )
            errs = {k: float(np.abs(qd[: len(qd) - k, legs] - q[k:, legs]).mean())
                    for k in range(0, 11)}
            print(f"command->response lag {best} steps = {best * DT * 1000:.0f} ms")
            print("  mean |err| by shift: "
                  + " ".join(f"{k}:{errs[k]:.4f}" for k in range(0, 11, 2)))
        clipped = sum(r[6] for r in log)
        print(f"limit-clipped commands: {clipped} joint-steps "
              f"({100 * clipped / (len(log) * N_MOTOR):.2f}%)")


if __name__ == "__main__":
    main()
