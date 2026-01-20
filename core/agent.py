#!/usr/bin/env python3
"""
Living AI - Main Agent

진정한 자율 에이전트:
- 가상 OS 완전 통제
- 비전으로 화면 이해
- Plastic LNN으로 실시간 학습
- 경험에서 진화

이건 wrapper가 아님. 실제로 배우는 AI.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F

from .plastic_lnn import (
    PlasticLNN,
    PlasticVisionLNN,
    LNNConfig,
    SimpleVisionEncoder,
    create_plastic_lnn,
    create_vision_lnn,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Living AI 에이전트 설정."""
    # Vision
    screen_width: int = 1280
    screen_height: int = 720
    vision_dim: int = 512

    # LNN
    hidden_dim: int = 256
    output_dim: int = 64

    # Learning
    online_learning: bool = True
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor

    # Exploration
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01

    # Goals
    goal_embedding_dim: int = 128

    # Performance
    action_interval: float = 0.1  # seconds between actions
    max_steps_per_episode: int = 1000

    # Checkpointing
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"


@dataclass
class AgentState:
    """에이전트 내부 상태."""
    step: int = 0
    episode: int = 0
    total_reward: float = 0.0
    episode_reward: float = 0.0
    exploration_rate: float = 0.1

    # History
    recent_rewards: List[float] = field(default_factory=list)
    recent_actions: List[Dict] = field(default_factory=list)

    # Learning metrics
    avg_value: float = 0.0
    avg_loss: float = 0.0


class LivingAgent:
    """
    Living AI Agent.

    진짜 배우는 AI:
    1. 화면을 보고
    2. 행동을 결정하고
    3. 결과를 관찰하고
    4. 가중치를 업데이트

    Frozen 모델 위에 껍데기 씌운 게 아님.
    """

    def __init__(
        self,
        config: AgentConfig = None,
        device: str = 'cuda',
    ):
        self.config = config or AgentConfig()
        self.device = device

        # State
        self.state = AgentState(exploration_rate=self.config.exploration_rate)

        # Models
        self._vision_encoder: Optional[torch.nn.Module] = None
        self._plastic_lnn: Optional[PlasticLNN] = None
        self._vision_lnn: Optional[PlasticVisionLNN] = None

        # Environment (lazy loaded)
        self._env = None

        # Goal system
        self._current_goal: Optional[str] = None
        self._goal_embedding: Optional[torch.Tensor] = None

        # Callbacks
        self.on_action: Optional[Callable] = None
        self.on_reward: Optional[Callable] = None
        self.on_learn: Optional[Callable] = None

        # Initialize
        self._init_models()

    def _init_models(self):
        """모델 초기화."""
        print("Initializing Living AI models...")

        # Vision encoder (경량 버전, 나중에 LFM-VL로 교체)
        self._vision_encoder = SimpleVisionEncoder(
            output_dim=self.config.vision_dim
        ).to(self.device)

        # Plastic LNN
        lnn_config = LNNConfig(
            input_dim=self.config.vision_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            online_learning=self.config.online_learning,
            learning_rate=self.config.learning_rate,
        )

        self._plastic_lnn = PlasticLNN(lnn_config).to(self.device)

        # Combined model
        self._vision_lnn = PlasticVisionLNN(
            self._vision_encoder,
            lnn_config,
            freeze_vision=True,  # Vision은 일단 freeze
        ).to(self.device)

        total_params = sum(p.numel() for p in self._plastic_lnn.parameters())
        print(f"✓ Plastic LNN initialized ({total_params:,} params)")

    async def connect_environment(self, env):
        """환경 연결."""
        self._env = env
        print(f"✓ Connected to environment: {env.resolution}")

    def set_goal(self, goal: str):
        """
        목표 설정.

        나중에 language encoder로 embedding 변환.
        지금은 단순 해시.
        """
        self._current_goal = goal

        # Simple goal embedding (placeholder)
        # TODO: Use language model for proper embedding
        hash_val = hash(goal) % (2**31)
        torch.manual_seed(hash_val)
        self._goal_embedding = torch.randn(
            1, self.config.goal_embedding_dim
        ).to(self.device)

        print(f"Goal set: {goal}")

    def _preprocess_screen(self, screen: np.ndarray) -> torch.Tensor:
        """화면 전처리."""
        # Resize if needed
        from PIL import Image
        img = Image.fromarray(screen)
        img = img.resize((224, 224))  # Standard vision size

        # To tensor
        arr = np.array(img)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float().to(self.device) / 255.0

        return tensor

    def _compute_reward(
        self,
        prev_screen: np.ndarray,
        curr_screen: np.ndarray,
        action: Dict[str, Any],
    ) -> float:
        """
        보상 계산.

        여러 신호 결합:
        1. 화면 변화 (행동이 효과가 있었나?)
        2. 목표 진행도 (나중에 구현)
        3. 탐색 보너스
        """
        # Screen change reward
        diff = np.abs(curr_screen.astype(float) - prev_screen.astype(float))
        change_ratio = np.mean(diff) / 255.0

        # 너무 많은 변화도 안 좋고, 아무 변화 없는 것도 안 좋음
        if change_ratio < 0.01:
            change_reward = -0.1  # 아무 일도 안 일어남
        elif change_ratio > 0.5:
            change_reward = -0.05  # 너무 급격한 변화
        else:
            change_reward = change_ratio * 0.5  # 적당한 변화

        # Action cost (클릭은 비용이 있음)
        action_cost = 0.0
        if action.get('click', 'none') != 'none':
            action_cost = -0.01

        # Exploration bonus
        exploration_bonus = 0.0
        if np.random.random() < self.state.exploration_rate:
            exploration_bonus = 0.02

        total_reward = change_reward + action_cost + exploration_bonus

        return total_reward

    async def step(self) -> Dict[str, Any]:
        """
        한 스텝 실행.

        1. 화면 캡처
        2. 행동 결정
        3. 행동 실행
        4. 결과 관찰
        5. 학습

        Returns:
            Dict with step info
        """
        if self._env is None:
            raise RuntimeError("Environment not connected")

        self.state.step += 1

        # 1. Capture screen
        prev_screen = self._env.capture_screen()
        screen_tensor = self._preprocess_screen(prev_screen)

        # 2. Get action from model
        self._vision_lnn.train()
        output = self._vision_lnn(screen_tensor)

        # 3. Exploration vs Exploitation
        if np.random.random() < self.state.exploration_rate:
            # Random action
            action = {
                'x': np.random.random(),
                'y': np.random.random(),
                'click': np.random.choice(['none', 'left', 'right']),
                'keys': [],
            }
        else:
            # Model action - use output directly (already computed)
            action_tensor = output['action']
            action = {
                'x': torch.sigmoid(action_tensor[0, 0]).item(),
                'y': torch.sigmoid(action_tensor[0, 1]).item(),
                'click': 'none',
                'keys': [],
            }
            # Decode click from action tensor if present
            if action_tensor.shape[1] >= 4:
                click_probs = torch.softmax(action_tensor[0, 2:5], dim=0)
                click_idx = click_probs.argmax().item()
                action['click'] = ['none', 'left', 'right'][click_idx]

        # 4. Execute action
        self._env.inject_action(action)

        if self.on_action:
            self.on_action(action)

        # Wait for effect
        await asyncio.sleep(self.config.action_interval)

        # 5. Observe result
        curr_screen = self._env.capture_screen()
        curr_tensor = self._preprocess_screen(curr_screen)
        curr_output = self._vision_lnn(curr_tensor)

        # 6. Compute reward
        reward = self._compute_reward(prev_screen, curr_screen, action)

        self.state.total_reward += reward
        self.state.episode_reward += reward
        self.state.recent_rewards.append(reward)
        self.state.recent_actions.append(action)

        # Keep only recent
        if len(self.state.recent_rewards) > 100:
            self.state.recent_rewards.pop(0)
            self.state.recent_actions.pop(0)

        if self.on_reward:
            self.on_reward(reward)

        # 7. Online learning (핵심!)
        self._vision_lnn.plastic_lnn.online_update(
            reward,
            output,
            curr_output,
        )

        if self.on_learn:
            self.on_learn(self.state.step)

        # 8. Update exploration rate
        self.state.exploration_rate = max(
            self.config.min_exploration,
            self.state.exploration_rate * self.config.exploration_decay
        )

        # 9. Checkpoint
        if self.state.step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        return {
            'step': self.state.step,
            'action': action,
            'reward': reward,
            'total_reward': self.state.total_reward,
            'exploration_rate': self.state.exploration_rate,
            'value': output['value'].item(),
        }

    async def run_episode(
        self,
        max_steps: int = None,
        goal: str = None,
    ) -> Dict[str, Any]:
        """
        에피소드 실행.

        Returns:
            Episode summary
        """
        max_steps = max_steps or self.config.max_steps_per_episode

        if goal:
            self.set_goal(goal)

        self.state.episode += 1
        self.state.episode_reward = 0.0

        print(f"\n=== Episode {self.state.episode} ===")
        if self._current_goal:
            print(f"Goal: {self._current_goal}")

        for step in range(max_steps):
            step_info = await self.step()

            # Progress
            if step % 50 == 0:
                avg_reward = np.mean(self.state.recent_rewards[-50:]) if self.state.recent_rewards else 0
                print(f"  Step {step}: reward={step_info['reward']:.4f}, "
                      f"avg={avg_reward:.4f}, explore={self.state.exploration_rate:.3f}")

        return {
            'episode': self.state.episode,
            'total_steps': self.state.step,
            'episode_reward': self.state.episode_reward,
            'exploration_rate': self.state.exploration_rate,
        }

    async def run_forever(self):
        """
        무한 실행.

        진정한 continuous learning.
        """
        print("\n" + "=" * 60)
        print("Living AI - Starting continuous learning")
        print("=" * 60 + "\n")

        self._plastic_lnn.reset_state(1, self.device)

        step = 0
        try:
            while True:
                step_info = await self.step()
                step += 1

                if step % 100 == 0:
                    avg_reward = np.mean(self.state.recent_rewards[-100:])
                    print(f"[Step {step}] avg_reward={avg_reward:.4f}, "
                          f"total={self.state.total_reward:.2f}, "
                          f"explore={self.state.exploration_rate:.3f}")

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self._save_checkpoint()

    def _save_checkpoint(self):
        """체크포인트 저장."""
        path = Path(self.config.checkpoint_dir)
        path.mkdir(exist_ok=True)

        checkpoint_path = path / f"living_ai_step_{self.state.step}.pt"

        self._plastic_lnn.save(str(checkpoint_path))
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드."""
        self._plastic_lnn = PlasticLNN.load(path, self.device)
        print(f"✓ Checkpoint loaded: {path}")


# ============================================================================
# Factory
# ============================================================================

def create_living_agent(
    screen_size: tuple = (1280, 720),
    hidden_dim: int = 256,
    online_learning: bool = True,
    device: str = 'cuda',
) -> LivingAgent:
    """Living Agent 생성."""
    config = AgentConfig(
        screen_width=screen_size[0],
        screen_height=screen_size[1],
        hidden_dim=hidden_dim,
        online_learning=online_learning,
    )

    return LivingAgent(config, device)


# ============================================================================
# Main
# ============================================================================

async def main():
    """메인 실행."""
    from .plastic_lnn import create_vision_lnn

    print("=" * 60)
    print("Living AI - Self-Learning Agent")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create agent
    agent = create_living_agent(device=device)

    # Without environment, just test the model
    print("\n=== Model Test (no environment) ===")

    # Fake screen
    fake_screen = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    tensor = agent._preprocess_screen(fake_screen)

    print(f"Input shape: {tensor.shape}")

    output = agent._vision_lnn(tensor)
    print(f"Action shape: {output['action'].shape}")
    print(f"Value: {output['value'].item():.4f}")

    action = agent._vision_lnn.plastic_lnn.get_action(output['features'])
    print(f"Decoded: x={action['x']:.3f}, y={action['y']:.3f}, click={action['click']}")

    print("\n✓ Model test passed!")
    print("\nTo run with environment:")
    print("  python run.py")


if __name__ == "__main__":
    asyncio.run(main())
