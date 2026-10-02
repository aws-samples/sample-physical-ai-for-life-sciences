from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass

from labflow_isaac.robots import JETBOT_CONFIG

ENVIRONMENT_VERSION = 1


class Action:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    PICK_UP = 4
    DROP_OFF = 5
    CHARGE = 6
    WAIT = 7


class Station:
    LOAD = 0
    LIQUID_HANDLER = 1
    INCUBATOR = 2
    READER = 3
    ARCHIVE = 4
    AGV = 5


@configclass
class LabFlowSceneCfg(InteractiveSceneCfg):
    """Scene entities cloned into every environment."""

    robot: ArticulationCfg = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


@configclass
class LabFlowIsaacEnvCfg(DirectRLEnvCfg):
    decimation = 12
    episode_length_s = 30.0
    action_space = gym.spaces.Discrete(8)
    observation_space = 20
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )
    scene: LabFlowSceneCfg = LabFlowSceneCfg(
        num_envs=64,
        env_spacing=8.0,
    )

    dof_names = ["left_wheel_joint", "right_wheel_joint"]
    wheel_radius = 0.0325
    wheel_base = 0.118
    max_linear_speed = 0.75
    max_angular_speed = 3.0
    heading_gain = 2.5
    station_radius = 0.38
    lab_half_extent = 2.8
    initial_deadline = 240.0
    process_times = (12, 24, 14)


class LabFlowIsaacEnv(DirectRLEnv):
    """Physics-backed JetBot transport with logical wet-lab instruments."""

    cfg: LabFlowIsaacEnvCfg

    def __init__(
        self,
        cfg: LabFlowIsaacEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.dof_idx = torch.tensor(dof_idx, device=self.device)

        self.actions = torch.full(
            (self.num_envs,), Action.WAIT, dtype=torch.long, device=self.device
        )
        self.wheel_targets = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.station_positions = torch.tensor(
            [
                [-2.0, -2.0],
                [2.0, -2.0],
                [2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.charger_position = torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=self.device
        )
        self.route = torch.tensor(
            [
                Station.LIQUID_HANDLER,
                Station.INCUBATOR,
                Station.READER,
                Station.ARCHIVE,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.process_times = torch.tensor(
            self.cfg.process_times, dtype=torch.long, device=self.device
        )

        self.plate_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.plate_location = torch.full(
            (self.num_envs,),
            Station.LOAD,
            dtype=torch.long,
            device=self.device,
        )
        self.plate_ready = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.process_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.carrying = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.battery = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.deadline = torch.full(
            (self.num_envs,),
            self.cfg.initial_deadline,
            dtype=torch.float32,
            device=self.device,
        )
        self.exposure = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.previous_target_distance = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.terminal_observation = torch.zeros(
            (self.num_envs, self.cfg.observation_space),
            dtype=torch.float32,
            device=self.device,
        )
        self.episode_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.reshape(-1).long().clamp(0, 7)
        positions = self._local_positions()
        self.previous_target_distance = torch.linalg.vector_norm(
            self._current_target(positions) - positions,
            dim=-1,
        )

        yaw = self._yaw()
        desired_headings = torch.tensor(
            [math.pi / 2, 0.0, -math.pi / 2, math.pi],
            dtype=torch.float32,
            device=self.device,
        )
        movement = self.actions < 4
        desired = desired_headings[self.actions.clamp(max=3)]
        heading_error = torch.atan2(
            torch.sin(desired - yaw),
            torch.cos(desired - yaw),
        )
        linear_speed = self.cfg.max_linear_speed * torch.clamp(
            1.0 - torch.abs(heading_error) / (math.pi / 2),
            min=0.0,
            max=1.0,
        )
        angular_speed = torch.clamp(
            self.cfg.heading_gain * heading_error,
            min=-self.cfg.max_angular_speed,
            max=self.cfg.max_angular_speed,
        )
        linear_speed = torch.where(
            movement, linear_speed, torch.zeros_like(linear_speed)
        )
        angular_speed = torch.where(
            movement, angular_speed, torch.zeros_like(angular_speed)
        )
        left = (
            linear_speed - angular_speed * self.cfg.wheel_base / 2
        ) / self.cfg.wheel_radius
        right = (
            linear_speed + angular_speed * self.cfg.wheel_base / 2
        ) / self.cfg.wheel_radius
        self.wheel_targets = torch.stack([left, right], dim=-1)

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target_index(
            target=self.wheel_targets,
            joint_ids=self.dof_idx,
        )

    def _get_observations(self) -> dict:
        observations = self._compute_observation()
        self.extras["terminal_observation"] = self.terminal_observation
        self.extras["action_mask"] = self.action_mask()
        self.extras["completed"] = self.episode_completed
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        positions = self._local_positions()
        current_distance = torch.linalg.vector_norm(
            self._current_target(positions) - positions,
            dim=-1,
        )
        rewards = torch.full(
            (self.num_envs,), -0.01, dtype=torch.float32, device=self.device
        )
        movement = self.actions < 4
        rewards += torch.where(
            movement,
            0.15 * (self.previous_target_distance - current_distance),
            torch.zeros_like(rewards),
        )

        active_process = self.process_remaining > 0
        self.process_remaining = torch.where(
            active_process,
            self.process_remaining - 1,
            self.process_remaining,
        )
        became_ready = active_process & (self.process_remaining == 0)
        self.plate_ready |= became_ready
        self.plate_stage += became_ready.long()
        rewards += became_ready.float()

        mask = self.action_mask()
        pick_requested = self.actions == Action.PICK_UP
        valid_pick = pick_requested & mask[:, Action.PICK_UP]
        invalid_pick = pick_requested & ~valid_pick
        self.carrying |= valid_pick
        self.plate_ready &= ~valid_pick
        self.plate_location = torch.where(
            valid_pick,
            torch.full_like(self.plate_location, Station.AGV),
            self.plate_location,
        )
        rewards += 0.2 * valid_pick.float() - 0.5 * invalid_pick.float()

        drop_requested = self.actions == Action.DROP_OFF
        valid_drop = drop_requested & mask[:, Action.DROP_OFF]
        invalid_drop = drop_requested & ~valid_drop
        target_station = self.route[self.plate_stage.clamp(max=3)]
        archive_drop = valid_drop & (target_station == Station.ARCHIVE)
        process_drop = valid_drop & ~archive_drop
        self.carrying &= ~valid_drop
        self.plate_location = torch.where(
            valid_drop, target_station, self.plate_location
        )
        self.completed |= archive_drop
        self.plate_ready &= ~valid_drop
        process_indices = (target_station - 1).clamp(0, 2)
        process_duration = self.process_times[process_indices]
        self.process_remaining = torch.where(
            process_drop, process_duration, self.process_remaining
        )
        rewards += (
            2.0 * process_drop.float()
            + 25.0 * archive_drop.float()
            - 0.5 * invalid_drop.float()
        )

        charge_requested = self.actions == Action.CHARGE
        valid_charge = charge_requested & mask[:, Action.CHARGE]
        invalid_charge = charge_requested & ~valid_charge
        self.battery = torch.where(
            valid_charge,
            torch.clamp(self.battery + 0.08, max=1.0),
            self.battery,
        )
        rewards += 0.02 * valid_charge.float() - 0.5 * invalid_charge.float()

        wheel_cost = 0.00015 * torch.mean(
            torch.abs(self.wheel_targets), dim=-1
        )
        self.battery = torch.clamp(
            self.battery - 0.0005 - wheel_cost,
            min=0.0,
            max=1.0,
        )
        self.deadline -= 1.0
        rewards -= 0.02 * (self.deadline < 0).float()
        self.exposure = torch.where(
            self.carrying, self.exposure + 1.0, torch.zeros_like(self.exposure)
        )
        rewards -= 0.02 * (self.exposure > 40).float()
        rewards += 10.0 * archive_drop.float()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.episode_completed.copy_(self.completed)
        position = self._local_positions()
        out_of_bounds = torch.any(
            torch.abs(position) > self.cfg.lab_half_extent,
            dim=-1,
        )
        terminated = self.completed | (self.battery <= 0.0) | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        if (
            hasattr(self, "terminal_observation")
            and env_ids.numel() > 0
            and torch.any(self.episode_length_buf[env_ids] > 0)
        ):
            terminal = self._compute_observation()
            self.terminal_observation[env_ids] = terminal[env_ids]

        super()._reset_idx(env_ids)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0:2] += self.station_positions[Station.LOAD]
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(
            root_pose=root_pose,
            env_ids=env_ids,
        )
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            env_ids=env_ids,
        )

        self.actions[env_ids] = Action.WAIT
        self.wheel_targets[env_ids] = 0.0
        self.plate_stage[env_ids] = 0
        self.plate_location[env_ids] = Station.LOAD
        self.plate_ready[env_ids] = True
        self.process_remaining[env_ids] = 0
        self.carrying[env_ids] = False
        self.completed[env_ids] = False
        self.battery[env_ids] = 1.0
        self.deadline[env_ids] = self.cfg.initial_deadline
        self.exposure[env_ids] = 0.0

    def action_mask(self) -> torch.Tensor:
        positions = self._local_positions()
        mask = torch.ones(
            (self.num_envs, 8), dtype=torch.bool, device=self.device
        )
        plate_position = self._plate_position(positions)
        near_plate = (
            torch.linalg.vector_norm(plate_position - positions, dim=-1)
            <= self.cfg.station_radius
        )
        mask[:, Action.PICK_UP] = (
            ~self.carrying
            & self.plate_ready
            & ~self.completed
            & near_plate
        )

        target_station = self.route[self.plate_stage.clamp(max=3)]
        target_position = self.station_positions[target_station]
        near_target = (
            torch.linalg.vector_norm(target_position - positions, dim=-1)
            <= self.cfg.station_radius
        )
        mask[:, Action.DROP_OFF] = self.carrying & near_target
        near_charger = (
            torch.linalg.vector_norm(
                self.charger_position.unsqueeze(0) - positions,
                dim=-1,
            )
            <= self.cfg.station_radius
        )
        mask[:, Action.CHARGE] = near_charger & (self.battery < 0.98)
        mask[:, Action.WAIT] = True

        margin = self.cfg.lab_half_extent - 0.1
        mask[:, Action.NORTH] &= positions[:, 1] < margin
        mask[:, Action.EAST] &= positions[:, 0] < margin
        mask[:, Action.SOUTH] &= positions[:, 1] > -margin
        mask[:, Action.WEST] &= positions[:, 0] > -margin
        return mask

    def heuristic_actions(self) -> torch.Tensor:
        mask = self.action_mask()
        actions = torch.full(
            (self.num_envs,), Action.WAIT, dtype=torch.long, device=self.device
        )
        actions = torch.where(
            mask[:, Action.PICK_UP],
            torch.full_like(actions, Action.PICK_UP),
            actions,
        )
        actions = torch.where(
            mask[:, Action.DROP_OFF],
            torch.full_like(actions, Action.DROP_OFF),
            actions,
        )

        needs_navigation = (
            (actions == Action.WAIT)
            & (self.carrying | self.plate_ready)
            & ~self.completed
        )
        positions = self._local_positions()
        delta = self._current_target(positions) - positions
        horizontal = torch.abs(delta[:, 0]) >= torch.abs(delta[:, 1])
        direction = torch.where(
            horizontal,
            torch.where(
                delta[:, 0] >= 0,
                torch.full_like(actions, Action.EAST),
                torch.full_like(actions, Action.WEST),
            ),
            torch.where(
                delta[:, 1] >= 0,
                torch.full_like(actions, Action.NORTH),
                torch.full_like(actions, Action.SOUTH),
            ),
        )
        actions = torch.where(needs_navigation, direction, actions)
        low_battery = (self.battery < 0.2) & ~self.carrying
        actions = torch.where(
            low_battery & mask[:, Action.CHARGE],
            torch.full_like(actions, Action.CHARGE),
            actions,
        )
        return actions

    def _compute_observation(self) -> torch.Tensor:
        positions = self._local_positions()
        target_delta = (
            self._current_target(positions) - positions
        ) / self.cfg.lab_half_extent
        yaw = self._yaw()
        velocity = self.robot.data.root_com_vel_w.torch
        busy = torch.stack(
            [
                (self.plate_location == station)
                & (self.process_remaining > 0)
                for station in (
                    Station.LIQUID_HANDLER,
                    Station.INCUBATOR,
                    Station.READER,
                )
            ],
            dim=-1,
        ).float()
        observation = torch.cat(
            [
                positions / self.cfg.lab_half_extent,
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
                velocity[:, 0:2] / self.cfg.max_linear_speed,
                velocity[:, 5:6] / self.cfg.max_angular_speed,
                self.battery.unsqueeze(-1),
                self.carrying.float().unsqueeze(-1),
                target_delta,
                (self.plate_stage.float() / 4.0).unsqueeze(-1),
                (self.plate_location.float() / Station.AGV).unsqueeze(-1),
                self.plate_ready.float().unsqueeze(-1),
                (
                    self.process_remaining.float()
                    / float(max(self.cfg.process_times))
                ).unsqueeze(-1),
                torch.clamp(
                    self.deadline / self.cfg.initial_deadline,
                    min=-1.0,
                    max=1.0,
                ).unsqueeze(-1),
                self.completed.float().unsqueeze(-1),
                busy,
            ],
            dim=-1,
        )
        return torch.clamp(observation, min=-1.0, max=1.0)

    def _current_target(self, positions: torch.Tensor) -> torch.Tensor:
        stage_target = self.station_positions[
            self.route[self.plate_stage.clamp(max=3)]
        ]
        plate_target = self._plate_position(positions)
        target = torch.where(
            self.carrying.unsqueeze(-1),
            stage_target,
            plate_target,
        )
        inactive = ~self.carrying & ~self.plate_ready
        target = torch.where(inactive.unsqueeze(-1), positions, target)
        low_battery = (self.battery < 0.2) & ~self.carrying
        return torch.where(
            low_battery.unsqueeze(-1),
            self.charger_position.unsqueeze(0).expand_as(target),
            target,
        )

    def _plate_position(self, positions: torch.Tensor) -> torch.Tensor:
        station_index = self.plate_location.clamp(max=Station.ARCHIVE)
        station_position = self.station_positions[station_index]
        return torch.where(
            self.carrying.unsqueeze(-1),
            positions,
            station_position,
        )

    def _local_positions(self) -> torch.Tensor:
        return (
            self.robot.data.root_pos_w.torch[:, 0:2]
            - self.scene.env_origins[:, 0:2]
        )

    def _yaw(self) -> torch.Tensor:
        # Isaac Lab 3.0 stores quaternions as XYZW.
        quaternion = self.robot.data.root_link_quat_w.torch
        x, y, z, w = quaternion.unbind(dim=-1)
        return torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
