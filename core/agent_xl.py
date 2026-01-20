#!/usr/bin/env python3
"""
LNN-AGI: XL Agent

2B 파라미터 대규모 모델 + 트레이딩/자동화 기능.
목표: 돈 버는 AI
"""

import asyncio
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn.functional as F
from PIL import Image

from .plastic_lnn_xl import PlasticVisionLNNXL, LNNXLConfig, create_lnn_xl


@dataclass
class AgentXLConfig:
    """XL Agent 설정."""
    # 화면
    screen_width: int = 1280
    screen_height: int = 720

    # 모델
    model_size: str = 'large'  # 'base', 'large', 'xl'

    # 학습
    online_learning: bool = True
    exploration_rate: float = 0.2
    exploration_decay: float = 0.9995
    min_exploration: float = 0.05

    # 액션
    action_interval: float = 0.3  # 초

    # 체크포인트
    checkpoint_dir: str = "checkpoints_xl"
    checkpoint_interval: int = 100

    # 트레이딩
    trading_enabled: bool = True
    trading_log_file: str = "trading_signals.jsonl"

    # 태스크
    active_tasks: List[str] = field(default_factory=lambda: [
        'action', 'trading', 'analysis', 'planning'
    ])


@dataclass
class AgentXLState:
    """Agent 상태."""
    step: int = 0
    total_reward: float = 0.0
    exploration_rate: float = 0.2
    session_start: float = field(default_factory=time.time)

    # 통계
    actions_taken: int = 0
    clicks_made: int = 0
    trading_signals: int = 0
    profitable_signals: int = 0


class LivingAgentXL:
    """
    2B 파라미터 Living Agent.

    특징:
    - 대규모 Plastic LNN (2B params, 16GB VRAM)
    - 실시간 온라인 학습
    - 트레이딩 신호 생성
    - 화면 분석 및 자동화
    """

    def __init__(
        self,
        config: Optional[AgentXLConfig] = None,
        device: str = 'cuda',
    ):
        self.config = config or AgentXLConfig()
        self.device = device
        self.state = AgentXLState(exploration_rate=self.config.exploration_rate)

        # 환경
        self._env = None

        # 모델
        print("=" * 60)
        print("  LNN-AGI: Initializing 2B Parameter Agent")
        print("=" * 60)
        self._model = create_lnn_xl(device, self.config.model_size)
        self._model.train()

        # 체크포인트 디렉토리
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)

        # 트레이딩 로그
        if self.config.trading_enabled:
            self._trading_log = open(self.config.trading_log_file, 'a')

        # 콜백
        self.on_action: Optional[Callable] = None
        self.on_reward: Optional[Callable] = None
        self.on_trading_signal: Optional[Callable] = None

        # 이전 출력 저장
        self._prev_output: Optional[Dict] = None
        self._prev_screen: Optional[np.ndarray] = None

        print(f"✓ Agent initialized on {device}")

    async def connect_environment(self, env):
        """환경 연결."""
        self._env = env
        print(f"✓ Connected to environment: ({env.width}, {env.height})")

    def _preprocess_screen(self, screen: np.ndarray) -> torch.Tensor:
        """화면 전처리."""
        # Resize if needed
        img = Image.fromarray(screen)
        if img.size != (self.config.screen_width, self.config.screen_height):
            img = img.resize((self.config.screen_width, self.config.screen_height))

        # To tensor
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def _compute_reward(
        self,
        prev_screen: np.ndarray,
        curr_screen: np.ndarray,
        action: Dict[str, Any],
    ) -> float:
        """보상 계산."""
        reward = 0.0

        # 1. 화면 변화 보상
        prev_gray = np.mean(prev_screen, axis=2)
        curr_gray = np.mean(curr_screen, axis=2)
        diff = np.abs(curr_gray - prev_gray).mean() / 255.0

        if diff > 0.01:  # 의미있는 변화
            reward += diff * 2.0  # 변화에 비례하는 보상

        # 2. 클릭 엔트로피 (불필요한 클릭 패널티)
        if action.get('click') != 'none':
            if diff < 0.01:  # 클릭했는데 변화 없음
                reward -= 0.1
            else:  # 클릭으로 변화 발생
                reward += 0.2

        # 3. 탐험 보너스
        x, y = action.get('x', 0.5), action.get('y', 0.5)
        edge_bonus = 0.0
        if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:
            edge_bonus = 0.02  # 가장자리 탐험 보너스
        reward += edge_bonus

        return reward

    def _decode_action(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """모델 출력에서 액션 추출."""
        action_tensor = output['action'][0]  # [output_dim]

        # x, y 좌표 (0~1)
        x = torch.sigmoid(action_tensor[0]).item()
        y = torch.sigmoid(action_tensor[1]).item()

        # 클릭 타입
        click_logits = action_tensor[2:5]
        click_idx = torch.softmax(click_logits, dim=0).argmax().item()
        click = ['none', 'left', 'right'][click_idx]

        # 키 입력 (아직 미구현)
        keys = []

        return {
            'x': x,
            'y': y,
            'click': click,
            'keys': keys,
        }

    def _log_trading_signal(self, signal: Dict[str, Any]):
        """트레이딩 신호 로깅."""
        if not self.config.trading_enabled:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': self.state.step,
            **signal,
        }
        self._trading_log.write(json.dumps(log_entry) + '\n')
        self._trading_log.flush()

        self.state.trading_signals += 1

        if self.on_trading_signal:
            self.on_trading_signal(signal)

    async def step(self) -> Dict[str, Any]:
        """한 스텝 실행."""
        assert self._env is not None, "Environment not connected"

        self.state.step += 1
        self.state.actions_taken += 1

        # 1. 화면 캡처
        screen = self._env.capture_screen()
        screen_tensor = self._preprocess_screen(screen)

        # 2. 모델 추론
        self._model.train()
        with torch.amp.autocast('cuda'):
            output = self._model(screen_tensor, self.config.active_tasks)

        # 3. 탐험 vs 활용
        if np.random.random() < self.state.exploration_rate:
            # 랜덤 탐험
            action = {
                'x': np.random.random(),
                'y': np.random.random(),
                'click': np.random.choice(['none', 'left', 'right'], p=[0.7, 0.2, 0.1]),
                'keys': [],
            }
        else:
            # 모델 액션
            action = self._decode_action(output)

        # 4. 액션 실행
        self._env.inject_action(action)
        if action['click'] != 'none':
            self.state.clicks_made += 1

        if self.on_action:
            self.on_action(action)

        # 대기
        await asyncio.sleep(self.config.action_interval)

        # 5. 결과 관찰
        next_screen = self._env.capture_screen()
        next_tensor = self._preprocess_screen(next_screen)

        with torch.amp.autocast('cuda'):
            next_output = self._model(next_tensor, self.config.active_tasks)

        # 6. 보상 계산
        reward = self._compute_reward(screen, next_screen, action)
        self.state.total_reward += reward

        if self.on_reward and abs(reward) > 0.1:
            self.on_reward(reward)

        # 7. 온라인 학습
        # TD-learning: V(s) + reward -> V(s')
        # output has gradients (current), next_output is target (detached)
        if self.config.online_learning:
            # Detach next_output for use as target
            next_out_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v
                                 for k, v in next_output.items()}
            self._model.plastic_lnn.online_update(
                reward,
                output,  # Has gradients
                next_out_detached,  # Target (no grad)
            )

        # 8. 트레이딩 신호 처리
        if 'trading' in output and self.config.trading_enabled:
            signal = self._model.plastic_lnn.get_trading_signal(output)
            if signal['confidence'] > 0.6:  # 신뢰도 60% 이상만
                self._log_trading_signal(signal)

        # 9. 탐험율 감소
        self.state.exploration_rate = max(
            self.config.min_exploration,
            self.state.exploration_rate * self.config.exploration_decay
        )

        return {
            'step': self.state.step,
            'action': action,
            'reward': reward,
            'exploration_rate': self.state.exploration_rate,
        }

    async def run_forever(self):
        """무한 실행 루프."""
        print("\n" + "=" * 60)
        print("LNN-AGI: Starting 2B Parameter Agent")
        print("=" * 60 + "\n")

        while True:
            try:
                step_info = await self.step()

                # 주기적 출력
                if self.state.step % 50 == 0:
                    elapsed = time.time() - self.state.session_start
                    steps_per_sec = self.state.step / elapsed if elapsed > 0 else 0

                    print(f"\n[Step {self.state.step}]")
                    print(f"  Total reward: {self.state.total_reward:.2f}")
                    print(f"  Exploration: {self.state.exploration_rate:.3f}")
                    print(f"  Speed: {steps_per_sec:.2f} steps/sec")
                    print(f"  Trading signals: {self.state.trading_signals}")

                # 체크포인트
                if self.state.step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error at step {self.state.step}: {e}")
                await asyncio.sleep(1)

        print("\n✓ Agent stopped")

    def _save_checkpoint(self):
        """체크포인트 저장."""
        path = Path(self.config.checkpoint_dir) / f"lnn_agi_step_{self.state.step}.pt"
        self._model.plastic_lnn.save(str(path))
        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드."""
        from .plastic_lnn_xl import PlasticVisionLNNXL
        self._model = PlasticVisionLNNXL.load(path, self.device)
        print(f"✓ Checkpoint loaded: {path}")

    def set_goal(self, goal: str):
        """목표 설정 (추후 goal-conditioned learning)."""
        print(f"🎯 Goal set: {goal}")
        # TODO: goal encoding and conditioning

    def close(self):
        """리소스 정리."""
        if hasattr(self, '_trading_log'):
            self._trading_log.close()


def create_agent_xl(device: str = 'cuda') -> LivingAgentXL:
    """XL Agent 생성."""
    config = AgentXLConfig(
        model_size='large',
        online_learning=True,
        trading_enabled=True,
    )
    return LivingAgentXL(config, device)


if __name__ == "__main__":
    # 테스트
    agent = create_agent_xl('cuda')
    print("\n✓ Agent created successfully")
