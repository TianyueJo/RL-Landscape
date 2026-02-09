# gridworld_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class GridSpec:
    width: int
    height: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    walls: Tuple[Tuple[int, int], ...] = ()
    # Backward-compatible single candy (preferred: use `candies`)
    candy: Optional[Tuple[int, int]] = None           # Candy cell: grants reward and terminates
    # Support multiple candy cells (case3 etc.)
    candies: Tuple[Tuple[int, int], ...] = ()
    risk: Optional[Tuple[int, int]] = None            # Risk cell: fails with some probability
    risk_p: float = 0.0                               # Failure probability
    step_penalty: float = -0.01
    goal_reward: float = 10.0
    candy_reward: float = 9.0
    fail_reward: float = -5.0
    max_steps: int = 200


def _default_specs_dir() -> Path:
    # Make spec loading robust to current working directory.
    return Path(__file__).resolve().parent / "env_specs"


def load_grid_spec(case_id: int, specs_dir: Optional[Path] = None) -> GridSpec:
    """
    Load a GridWorld environment definition from a JSON file under env_specs/
    and construct a GridSpec.

    JSON conventions:
      - start/goal/candy/risk: [x, y] or null
      - walls: [[x, y], ...]
    """
    specs_dir = specs_dir or _default_specs_dir()
    spec_path = specs_dir / f"case_{int(case_id)}.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing env spec file: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    def _to_xy(v, name: str):
        if v is None:
            return None
        if (
            not isinstance(v, (list, tuple))
            or len(v) != 2
            or not all(isinstance(x, (int, float)) for x in v)
        ):
            raise ValueError(f"Invalid {name} in {spec_path}: {v}")
        return (int(v[0]), int(v[1]))

    def _to_xy_list(v, name: str) -> Tuple[Tuple[int, int], ...]:
        """
        Accept either:
          - null
          - [x, y]
          - [[x, y], [x, y], ...]
        """
        if v is None:
            return ()
        # single [x,y]
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
        ):
            return (int(v[0]), int(v[1])),
        # list of [x,y]
        if isinstance(v, list) and v and all(isinstance(p, (list, tuple)) for p in v):
            out = []
            for p in v:
                if (
                    not isinstance(p, (list, tuple))
                    or len(p) != 2
                    or not all(isinstance(x, (int, float)) for x in p)
                ):
                    raise ValueError(f"Invalid {name} entry in {spec_path}: {p}")
                out.append((int(p[0]), int(p[1])))
            return tuple(out)
        raise ValueError(f"Invalid {name} in {spec_path}: {v}")

    walls_raw = payload.get("walls", [])
    if not isinstance(walls_raw, list):
        raise ValueError(f"Invalid walls in {spec_path}: expected list, got {type(walls_raw)}")
    walls: Tuple[Tuple[int, int], ...] = tuple(
        (int(p[0]), int(p[1])) for p in walls_raw
    )

    candies = _to_xy_list(payload.get("candy"), "candy")
    candy_primary = candies[0] if candies else None

    return GridSpec(
        width=int(payload["width"]),
        height=int(payload["height"]),
        start=_to_xy(payload["start"], "start"),
        goal=_to_xy(payload["goal"], "goal"),
        walls=walls,
        candy=candy_primary,
        candies=candies,
        risk=_to_xy(payload.get("risk"), "risk"),
        risk_p=float(payload.get("risk_p", 0.0)),
        step_penalty=float(payload.get("step_penalty", -0.01)),
        goal_reward=float(payload.get("goal_reward", 10.0)),
        candy_reward=float(payload.get("candy_reward", 9.0)),
        fail_reward=float(payload.get("fail_reward", -5.0)),
        max_steps=int(payload.get("max_steps", 200)),
    )


class ObservationWrapper:
    """
    Convert MultiDiscrete observations to a Box (float) and normalize (x, y) to [0, 1].
    This ensures training/evaluation/embeddings/jump&retrain share the same preprocessing.
    """

    def __init__(self, env):
        self.env = env
        if hasattr(env, "spec") and hasattr(env.spec, "width") and hasattr(env.spec, "height"):
            self.width = env.spec.width
            self.height = env.spec.height
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            )
        else:
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
        if isinstance(obs, np.ndarray) and obs.dtype in (np.int64, np.int32):
            return obs.astype(np.float32) / np.array(
                [self.width - 1, self.height - 1], dtype=np.float32
            )
        return obs.astype(np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(
    seed: int = 0,
    case_id: int = 1,
    obs_mode: str = "pos",
    render_mode: Optional[str] = None,
    specs_dir: Optional[Path] = None,
    wrap_obs: bool = True,
):
    spec = load_grid_spec(case_id=case_id, specs_dir=specs_dir)
    env = GridWorldEnv(spec, obs_mode=obs_mode, render_mode=render_mode, seed=seed)
    if wrap_obs:
        env = ObservationWrapper(env)
    return env


class GridWorldEnv(gym.Env):
    """
    A simple discrete GridWorld:
    - Actions: 0=up, 1=right, 2=down, 3=left
    - Observations: by default the agent (x, y) position (discrete); can also be one-hot over the grid
    - Supports: walls, goal, candy (grants reward and resets), risk (terminates with some probability)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        spec: GridSpec,
        obs_mode: str = "pos",          # "pos" or "onehot"
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert obs_mode in ("pos", "onehot")
        self.spec = spec
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        # action space: 4 directions
        self.action_space = spaces.Discrete(4)

        # observation space
        if obs_mode == "pos":
            # agent position as two ints: x in [0,w-1], y in [0,h-1]
            self.observation_space = spaces.MultiDiscrete([spec.width, spec.height])
        else:
            # one-hot over all grid cells
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(spec.width * spec.height,), dtype=np.float32
            )

        self.walls = set(spec.walls)
        # normalize candies
        if getattr(spec, "candies", ()):
            self.candies = set(spec.candies)
        elif spec.candy is not None:
            self.candies = {spec.candy}
        else:
            self.candies = set()
        self._validate_layout()

        self.agent_pos = spec.start
        self.steps = 0

    def _validate_layout(self) -> None:
        w, h = self.spec.width, self.spec.height
        def in_bounds(p): return 0 <= p[0] < w and 0 <= p[1] < h

        assert in_bounds(self.spec.start), "start out of bounds"
        assert in_bounds(self.spec.goal), "goal out of bounds"
        for p in self.walls:
            assert in_bounds(p), f"wall {p} out of bounds"
        for c in self.candies:
            assert in_bounds(c), "candy out of bounds"
        if self.spec.risk is not None:
            assert in_bounds(self.spec.risk), "risk out of bounds"
        assert self.spec.start not in self.walls, "start cannot be a wall"
        assert self.spec.goal not in self.walls, "goal cannot be a wall"

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "pos":
            return np.array(self.agent_pos, dtype=np.int64)
        idx = self.agent_pos[1] * self.spec.width + self.agent_pos[0]
        onehot = np.zeros((self.spec.width * self.spec.height,), dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.spec.start
        self.steps = 0
        info: Dict[str, Any] = {}
        return self._obs(), info

    
    def step(self, action: int):
        self.steps += 1

        x, y = self.agent_pos
        if action == 0:   # up
            nx, ny = x, y - 1
        elif action == 1: # right
            nx, ny = x + 1, y
        elif action == 2: # down
            nx, ny = x, y + 1
        elif action == 3: # left
            nx, ny = x - 1, y
        else:
            raise ValueError("Invalid action")

        # stay if out of bounds or hits wall
        if not (0 <= nx < self.spec.width and 0 <= ny < self.spec.height):
            nx, ny = x, y
        if (nx, ny) in self.walls:
            nx, ny = x, y

        self.agent_pos = (nx, ny)

        reward = self.spec.step_penalty
        terminated = False
        truncated = False
        info = {}

        # ---------- risk cell ----------
        if self.spec.risk is not None and self.agent_pos == self.spec.risk:
            if self.rng.random() < self.spec.risk_p:
                reward += self.spec.fail_reward
                terminated = True
                info["event"] = "risk_fail"

        # ---------- candy cell (CHANGED) ----------
        if (
            not terminated
            and self.candies
            and self.agent_pos in self.candies
        ):
            reward += self.spec.candy_reward
            terminated = True
            info["event"] = "candy_terminal"

        # ---------- goal ----------
        if (
            not terminated
            and self.agent_pos == self.spec.goal
        ):
            reward += self.spec.goal_reward
            terminated = True
            info["event"] = "goal"

        # ---------- time limit ----------
        if self.steps >= self.spec.max_steps and not terminated:
            truncated = True
            info["event"] = "timeout"

        if self.render_mode == "human":
            print(self.render())

        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        # simple ASCII render
        w, h = self.spec.width, self.spec.height
        grid = [["." for _ in range(w)] for _ in range(h)]
        for (wx, wy) in self.walls:
            grid[wy][wx] = "#"
        sx, sy = self.spec.start
        gx, gy = self.spec.goal
        grid[sy][sx] = "S"
        grid[gy][gx] = "G"
        for (cx, cy) in self.candies:
            grid[cy][cx] = "C"
        if self.spec.risk is not None:
            rx, ry = self.spec.risk
            grid[ry][rx] = "R"
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"
        return "\n".join("".join(row) for row in grid)

    def close(self):
        pass
