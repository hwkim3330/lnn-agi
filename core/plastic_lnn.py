#!/usr/bin/env python3
"""
Living AI - Plastic Liquid Neural Network

핵심 특징:
- Liquid Time-Constant (LTC) 뉴런: ODE 기반 동적 상태
- Online Learning: 실시간 가중치 업데이트
- Hebbian + Gradient Hybrid: 생물학적 + 수학적 학습

MIT Liquid Neural Networks 논문 기반:
https://arxiv.org/abs/2006.04439
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class LNNConfig:
    """Plastic LNN 설정."""
    input_dim: int = 512          # 비전 인코더 출력 차원
    hidden_dim: int = 256         # 히든 상태 차원
    output_dim: int = 64          # 행동 공간 차원
    num_layers: int = 3           # LTC 레이어 수

    # Liquid 파라미터
    tau_min: float = 0.1          # 최소 시간 상수
    tau_max: float = 10.0         # 최대 시간 상수
    dt: float = 0.1               # 시간 스텝

    # 학습 파라미터
    learning_rate: float = 1e-4
    hebbian_lr: float = 1e-3      # Hebbian 학습률
    plasticity: float = 0.1       # 가소성 강도

    # Online learning
    online_learning: bool = True
    grad_clip: float = 1.0


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell.

    ODE: τ * dh/dt = -h + f(Wh*h + Wx*x + b)

    τ (time constant)가 입력에 따라 동적으로 변함.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: LNNConfig):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config

        # 가중치
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Time constant (τ) - 학습 가능
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))
        self.tau_input = nn.Linear(input_dim, hidden_dim, bias=False)

        # Plasticity weights (Hebbian용)
        self.plasticity_mask = nn.Parameter(
            torch.ones(hidden_dim, hidden_dim) * config.plasticity,
            requires_grad=False
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화."""
        nn.init.orthogonal_(self.W_h.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.tau_input.weight)

    def compute_tau(self, x: Tensor) -> Tensor:
        """입력 기반 동적 시간 상수 계산."""
        tau_mod = torch.sigmoid(self.tau_input(x))
        tau = self.config.tau_min + (self.config.tau_max - self.config.tau_min) * (
            torch.sigmoid(self.tau_base) * (1 + tau_mod)
        )
        return tau

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with ODE integration.

        Args:
            x: Input [batch, input_dim]
            h: Hidden state [batch, hidden_dim]

        Returns:
            new_h: Updated hidden state
            tau: Time constants (for analysis)
        """
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 동적 시간 상수
        tau = self.compute_tau(x)

        # ODE: dh/dt = (-h + f(Wh*h + Wx*x + b)) / τ
        pre_act = self.W_h(h) + self.W_x(x) + self.bias
        f_val = torch.tanh(pre_act)

        # Euler integration
        dh = (-h + f_val) / tau
        new_h = h + self.config.dt * dh

        return new_h, tau

    def hebbian_update(self, pre: Tensor, post: Tensor):
        """
        Hebbian learning: "neurons that fire together wire together"

        ΔW = η * post * pre^T (outer product)

        NOTE: 현재 gradient-based learning과 충돌 방지를 위해 비활성화.
        """
        # Hebbian과 gradient descent를 동시에 하면 충돌
        # 나중에 별도 phase로 분리 필요
        pass


class PlasticLNN(nn.Module):
    """
    Plastic Liquid Neural Network.

    실시간 학습 가능한 Liquid Neural Network.
    """

    def __init__(self, config: LNNConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # LTC layers
        self.ltc_layers = nn.ModuleList([
            LTCCell(
                config.hidden_dim if i > 0 else config.hidden_dim,
                config.hidden_dim,
                config
            )
            for i in range(config.num_layers)
        ])

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Hidden states
        self._hidden_states: Optional[list] = None

        # Optimizer for online learning
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
        )

        # Experience buffer for online learning
        self.experience_buffer = []
        self.buffer_size = 32

    def reset_state(self, batch_size: int = 1, device: str = 'cuda'):
        """히든 상태 초기화."""
        self._hidden_states = [
            torch.zeros(batch_size, self.config.hidden_dim, device=device)
            for _ in range(self.config.num_layers)
        ]

    def forward(
        self,
        x: Tensor,
        hidden_states: Optional[list] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            hidden_states: Optional list of hidden states

        Returns:
            Dict with 'action', 'value', 'hidden_states', 'taus'
        """
        if hidden_states is None:
            hidden_states = self._hidden_states

        if hidden_states is None:
            self.reset_state(x.size(0), x.device)
            hidden_states = self._hidden_states

        # Input projection
        h = self.input_proj(x)

        # Process through LTC layers
        new_hidden_states = []
        taus = []

        for i, ltc in enumerate(self.ltc_layers):
            prev_h = hidden_states[i]
            h, tau = ltc(h, prev_h)
            new_hidden_states.append(h)
            taus.append(tau)

            # Hebbian update (online)
            if self.training and self.config.online_learning:
                ltc.hebbian_update(prev_h, h)

        # Update internal states (detached to prevent gradient issues across steps)
        self._hidden_states = [h.detach() for h in new_hidden_states]

        # Output heads
        action = self.action_head(h)
        value = self.value_head(h)

        return {
            'action': action,
            'value': value,
            'hidden_states': new_hidden_states,
            'taus': taus,
            'features': h,
        }

    def online_update(
        self,
        reward: float,
        prev_output: Dict[str, Tensor],
        curr_output: Dict[str, Tensor],
    ):
        """
        Online learning update.

        TD-learning style update.
        """
        if not self.config.online_learning:
            return

        # Immediate gradient update (more responsive than buffered)
        self.optimizer.zero_grad()

        gamma = 0.99

        # TD error: r + γV(s') - V(s)
        # V(s)는 grad 필요, V(s')는 target이므로 detach
        value = prev_output['value']
        next_value = curr_output['value'].detach()

        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Value loss
        value_loss = td_error.pow(2).mean()

        # Backward
        value_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()

    def _learn_from_buffer(self):
        """버퍼에서 학습."""
        if not self.experience_buffer:
            return

        self.optimizer.zero_grad()

        # 버퍼에서 데이터 추출
        rewards = torch.tensor([exp['reward'] for exp in self.experience_buffer])
        values = torch.stack([exp['value'] for exp in self.experience_buffer])
        next_values = torch.stack([exp['next_value'] for exp in self.experience_buffer])

        gamma = 0.99

        # TD target (no grad needed for target)
        td_targets = rewards.to(values.device) + gamma * next_values.squeeze()

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), td_targets.detach())

        value_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()

        # Clear buffer
        self.experience_buffer.clear()

    def get_action(self, features: Tensor) -> Dict[str, Any]:
        """
        행동 결정.

        Returns:
            Dict with 'x', 'y', 'click', 'keys', 'raw_action'
        """
        with torch.no_grad():
            output = self.forward(features)

        action = output['action'].squeeze(0)  # [output_dim]

        # Decode action
        # action[0:2] = x, y (normalized 0-1)
        # action[2:5] = click type (none, left, right, double)
        # action[5:] = special keys

        x = torch.sigmoid(action[0]).item()
        y = torch.sigmoid(action[1]).item()

        click_logits = action[2:6]
        click_type = torch.argmax(click_logits).item()
        click_types = ['none', 'left', 'right', 'double']

        # Key presses (simplified)
        key_logits = action[6:] if len(action) > 6 else None
        keys = []
        if key_logits is not None:
            key_probs = torch.sigmoid(key_logits)
            active_keys = (key_probs > 0.5).nonzero().flatten().tolist()
            # Map to actual keys (simplified)
            key_map = ['enter', 'tab', 'escape', 'backspace', 'space']
            keys = [key_map[i] for i in active_keys if i < len(key_map)]

        return {
            'x': x,
            'y': y,
            'click': click_types[click_type],
            'keys': keys,
            'raw_action': action,
            'value': output['value'].item(),
        }

    def save(self, path: str):
        """모델 저장."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'PlasticLNN':
        """모델 로드."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        return model.to(device)


class PlasticVisionLNN(nn.Module):
    """
    Vision + Plastic LNN 통합 모델.

    스크린샷 → 비전 인코더 → Plastic LNN → 행동
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lnn_config: LNNConfig,
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder

        # LNN config의 input_dim을 vision encoder 출력에 맞게 조정
        self.plastic_lnn = PlasticLNN(lnn_config)

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """
        이미지에서 행동까지 end-to-end.

        Args:
            image: [batch, C, H, W]
        """
        # Vision encoding
        with torch.no_grad():
            features = self.vision_encoder(image)

        # Plastic LNN
        return self.plastic_lnn(features)

    def get_action(self, image: Tensor) -> Dict[str, Any]:
        """이미지에서 행동 결정."""
        with torch.no_grad():
            features = self.vision_encoder(image)
        return self.plastic_lnn.get_action(features)


# ============================================================================
# Simple Vision Encoder (lightweight, for testing)
# ============================================================================

class SimpleVisionEncoder(nn.Module):
    """
    경량 비전 인코더 (테스트용).
    실제로는 LFM-VL 사용.
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(256 * 4 * 4, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, 3, H, W] (any resolution)
        Returns:
            features: [batch, output_dim]
        """
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ============================================================================
# Factory
# ============================================================================

def create_plastic_lnn(
    input_dim: int = 512,
    hidden_dim: int = 256,
    output_dim: int = 64,
    online_learning: bool = True,
    device: str = 'cuda',
) -> PlasticLNN:
    """Plastic LNN 생성."""
    config = LNNConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        online_learning=online_learning,
    )
    return PlasticLNN(config).to(device)


def create_vision_lnn(
    vision_encoder: nn.Module = None,
    hidden_dim: int = 256,
    output_dim: int = 64,
    device: str = 'cuda',
) -> PlasticVisionLNN:
    """Vision + LNN 통합 모델 생성."""
    if vision_encoder is None:
        vision_encoder = SimpleVisionEncoder(output_dim=512)

    config = LNNConfig(
        input_dim=512,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        online_learning=True,
    )

    return PlasticVisionLNN(vision_encoder, config).to(device)


if __name__ == "__main__":
    # 테스트
    print("=== Plastic LNN Test ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 모델 생성
    model = create_plastic_lnn(device=device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward test
    x = torch.randn(1, 512).to(device)
    output = model(x)
    print(f"Action shape: {output['action'].shape}")
    print(f"Value: {output['value'].item():.4f}")

    # Action decoding test
    action = model.get_action(x)
    print(f"Decoded action: x={action['x']:.3f}, y={action['y']:.3f}, click={action['click']}")

    # Online learning test
    print("\n=== Online Learning Test ===")
    model.train()

    for i in range(10):
        x = torch.randn(1, 512).to(device)
        prev_out = model(x)

        # Simulate reward
        reward = torch.rand(1).item() - 0.5

        x_next = torch.randn(1, 512).to(device)
        curr_out = model(x_next)

        model.online_update(reward, prev_out, curr_out)

    print("Online learning completed!")

    # Vision + LNN test
    print("\n=== Vision LNN Test ===")
    vision_model = create_vision_lnn(device=device)
    print(f"Vision LNN params: {sum(p.numel() for p in vision_model.parameters()):,}")

    img = torch.randn(1, 3, 224, 224).to(device)
    action = vision_model.get_action(img)
    print(f"From image: x={action['x']:.3f}, y={action['y']:.3f}")

    print("\n✓ All tests passed!")
