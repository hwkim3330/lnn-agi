#!/usr/bin/env python3
"""
LNN-AGI: Plastic Liquid Neural Network XL

VRAM 20GB+ 사용하는 대규모 모델.
목표: 돈 버는 AI - 자동화, 트레이딩, 작업 수행.

특징:
- 대규모 LTC Network (2048 hidden, 12 layers)
- Multi-head Attention + LTC 결합
- 장기 메모리 뱅크
- 멀티태스크 헤드 (트레이딩, 자동화, 분석)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class LNNXLConfig:
    """대규모 Plastic LNN 설정."""
    # 입출력
    input_dim: int = 1024           # 비전 인코더 출력
    hidden_dim: int = 2048          # 히든 차원 (대규모)
    output_dim: int = 256           # 행동 공간
    num_layers: int = 12            # LTC 레이어 수

    # Attention
    num_heads: int = 16             # Multi-head attention
    head_dim: int = 128             # head당 차원
    dropout: float = 0.1

    # Liquid 파라미터
    tau_min: float = 0.1
    tau_max: float = 10.0
    dt: float = 0.1

    # 메모리
    memory_size: int = 10000        # 장기 메모리 슬롯
    memory_dim: int = 512           # 메모리 차원

    # 학습
    learning_rate: float = 3e-5
    hebbian_lr: float = 1e-4
    plasticity: float = 0.05
    online_learning: bool = True
    grad_clip: float = 1.0

    # 태스크 헤드
    task_heads: List[str] = field(default_factory=lambda: [
        'action',      # 마우스/키보드 액션
        'trading',     # 트레이딩 신호
        'analysis',    # 화면 분석
        'planning',    # 계획 수립
    ])


class RotaryEmbedding(nn.Module):
    """RoPE - 위치 인코딩."""

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)


class MultiHeadLTCAttention(nn.Module):
    """
    Multi-Head Attention + LTC 결합.

    Attention으로 중요 정보 선택 + LTC로 시간적 통합.
    """

    def __init__(self, config: LNNXLConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim, bias=False)

        # LTC time constants per head
        self.tau_base = nn.Parameter(torch.ones(config.num_heads, config.head_dim))
        self.tau_gate = nn.Linear(config.hidden_dim, config.num_heads, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(config.head_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """RoPE 적용."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward with attention + LTC dynamics.

        Args:
            x: [batch, seq, hidden]
            h: Previous hidden state [batch, num_heads, head_dim]
            mask: Attention mask

        Returns:
            output: [batch, seq, hidden]
            new_h: [batch, num_heads, head_dim]
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(seq_len, x.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]

        # LTC dynamics on attention output
        if h is None:
            h = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device)

        # Dynamic tau based on input
        tau_gate = torch.sigmoid(self.tau_gate(x.mean(dim=1)))  # [batch, heads]
        tau = self.config.tau_min + (self.config.tau_max - self.config.tau_min) * (
            torch.sigmoid(self.tau_base) * tau_gate.unsqueeze(-1)
        )

        # ODE: dh/dt = (-h + out) / tau
        out_pooled = out.mean(dim=2)  # [batch, heads, head_dim]
        dh = (-h + out_pooled) / tau
        new_h = h + self.config.dt * dh

        # Output projection
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.o_proj(out)

        return out, new_h


class LTCFFNBlock(nn.Module):
    """LTC + FFN Block."""

    def __init__(self, config: LNNXLConfig):
        super().__init__()
        self.config = config

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # LTC gate
        self.ltc_gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.tau = nn.Parameter(torch.ones(config.hidden_dim))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward with residual + LTC."""
        # FFN with residual
        x = x + self.ffn(self.norm1(x))

        # LTC dynamics
        if h is None:
            h = torch.zeros_like(x[:, 0, :])

        x_pooled = x.mean(dim=1)  # [batch, hidden]
        gate = torch.sigmoid(self.ltc_gate(x_pooled))
        tau = self.config.tau_min + (self.config.tau_max - self.config.tau_min) * torch.sigmoid(self.tau)

        dh = (-h + gate * x_pooled) / tau
        new_h = h + self.config.dt * dh

        # Modulate output with hidden state
        x = self.norm2(x + new_h.unsqueeze(1))

        return x, new_h


class PlasticLNNXL(nn.Module):
    """
    대규모 Plastic Liquid Neural Network.

    20GB+ VRAM 사용, 멀티태스크 헤드.
    """

    def __init__(self, config: LNNXLConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)

        # LTC-Attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadLTCAttention(config),
                'ffn': LTCFFNBlock(config),
            })
            for _ in range(config.num_layers)
        ])

        # Long-term memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(config.memory_size, config.memory_dim) * 0.02
        )
        self.memory_proj = nn.Linear(config.hidden_dim, config.memory_dim)
        self.memory_read = nn.Linear(config.memory_dim, config.hidden_dim)

        # Task heads
        self.task_heads = nn.ModuleDict()
        for task in config.task_heads:
            if task == 'action':
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim // 2, config.output_dim),
                )
            elif task == 'trading':
                # 트레이딩: buy/sell/hold + confidence
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim // 2, 4),  # buy, sell, hold, confidence
                )
            elif task == 'analysis':
                # 분석: 화면 이해
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, 512),  # 분석 임베딩
                )
            elif task == 'planning':
                # 계획: 다음 단계들
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, 256),  # 계획 임베딩
                )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Hidden states
        self._attn_states: Optional[List[Tensor]] = None
        self._ffn_states: Optional[List[Tensor]] = None

        # 파라미터 수 먼저 출력 (옵티마이저 선택 전)
        self._count_params()

        # Optimizer - 큰 모델은 vanilla SGD 사용 (최소 메모리)
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > 1e9:  # > 1B params
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=config.learning_rate * 100,  # Vanilla SGD needs even higher LR
                momentum=0,  # No momentum = no extra memory
            )
            print(f"  Using vanilla SGD (minimal memory for {total_params/1e9:.1f}B params)")
        else:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01,
            )
            print(f"  Using AdamW optimizer")

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PlasticLNN-XL: {total:,} params ({trainable:,} trainable)")
        print(f"  Estimated VRAM: {total * 4 / 1e9:.2f} GB (fp32)")

    def reset_state(self, batch_size: int, device: torch.device):
        """Hidden state 초기화."""
        self._attn_states = [
            torch.zeros(batch_size, self.config.num_heads, self.config.head_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        self._ffn_states = [
            torch.zeros(batch_size, self.config.hidden_dim, device=device)
            for _ in range(self.config.num_layers)
        ]

    def read_memory(self, query: Tensor) -> Tensor:
        """장기 메모리에서 읽기."""
        # Query projection
        q = self.memory_proj(query)  # [batch, memory_dim]

        # Attention over memory
        scores = torch.matmul(q, self.memory_bank.T)  # [batch, memory_size]
        attn = F.softmax(scores / math.sqrt(self.config.memory_dim), dim=-1)

        # Read
        read = torch.matmul(attn, self.memory_bank)  # [batch, memory_dim]
        return self.memory_read(read)  # [batch, hidden_dim]

    def write_memory(self, key: Tensor, value: Tensor, idx: int):
        """장기 메모리에 쓰기 (Hebbian style)."""
        with torch.no_grad():
            # 가장 유사한 슬롯 찾기
            k = self.memory_proj(key)
            sim = torch.matmul(k, self.memory_bank.T)
            slot_idx = sim.argmax(dim=-1)

            # Hebbian update
            for b in range(key.size(0)):
                self.memory_bank.data[slot_idx[b]] = (
                    0.99 * self.memory_bank.data[slot_idx[b]] +
                    0.01 * self.memory_proj(value[b:b+1]).squeeze(0)
                )

    def forward(
        self,
        x: Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, seq, input_dim] or [batch, input_dim]
            tasks: Which task heads to compute

        Returns:
            Dict with outputs per task + value
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        batch = x.size(0)
        device = x.device

        # Initialize states if needed
        if self._attn_states is None:
            self.reset_state(batch, device)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Process through layers
        new_attn_states = []
        new_ffn_states = []

        for i, layer in enumerate(self.layers):
            # Attention + LTC
            h, attn_state = layer['attention'](h, self._attn_states[i])
            new_attn_states.append(attn_state.detach())

            # FFN + LTC
            h, ffn_state = layer['ffn'](h, self._ffn_states[i])
            new_ffn_states.append(ffn_state.detach())

        # Update states
        self._attn_states = new_attn_states
        self._ffn_states = new_ffn_states

        # Pool to single vector
        h_pooled = h.mean(dim=1)  # [batch, hidden]

        # Memory augmentation
        memory_context = self.read_memory(h_pooled)
        h_pooled = h_pooled + 0.1 * memory_context

        # Task outputs
        outputs = {}
        tasks = tasks or self.config.task_heads

        for task in tasks:
            if task in self.task_heads:
                outputs[task] = self.task_heads[task](h_pooled)

        # Value
        outputs['value'] = self.value_head(h_pooled)
        outputs['features'] = h_pooled
        outputs['hidden'] = h

        return outputs

    def online_update(
        self,
        reward: float,
        prev_output: Dict[str, Tensor],
        curr_output: Dict[str, Tensor],
        task_loss: Optional[Tensor] = None,
    ):
        """Online learning update."""
        if not self.config.online_learning:
            return

        self.optimizer.zero_grad()

        # TD error
        gamma = 0.99
        value = prev_output['value']
        next_value = curr_output['value'].detach()

        td_target = reward + gamma * next_value
        td_error = td_target - value
        value_loss = td_error.pow(2).mean()

        # Total loss
        total_loss = value_loss
        if task_loss is not None:
            total_loss = total_loss + task_loss

        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

        self.optimizer.step()

        return {
            'value_loss': value_loss.item(),
            'td_error': td_error.mean().item(),
        }

    def get_action(self, output: Dict[str, Tensor]) -> Dict[str, Any]:
        """액션 출력에서 실제 액션 추출."""
        action_tensor = output['action']

        return {
            'x': torch.sigmoid(action_tensor[0, 0]).item(),
            'y': torch.sigmoid(action_tensor[0, 1]).item(),
            'click': ['none', 'left', 'right'][
                torch.softmax(action_tensor[0, 2:5], dim=0).argmax().item()
            ] if action_tensor.shape[1] >= 5 else 'none',
            'keys': [],
        }

    def get_trading_signal(self, output: Dict[str, Tensor]) -> Dict[str, Any]:
        """트레이딩 신호 추출."""
        if 'trading' not in output:
            return {'signal': 'hold', 'confidence': 0.0}

        t = output['trading'][0]
        probs = torch.softmax(t[:3], dim=0)
        signal_idx = probs.argmax().item()

        return {
            'signal': ['buy', 'sell', 'hold'][signal_idx],
            'confidence': torch.sigmoid(t[3]).item(),
            'probs': {
                'buy': probs[0].item(),
                'sell': probs[1].item(),
                'hold': probs[2].item(),
            }
        }

    def save(self, path: str):
        """체크포인트 저장."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'PlasticLNNXL':
        """체크포인트 로드."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        return model.to(device)


class VisionEncoderXL(nn.Module):
    """
    대규모 비전 인코더.

    ResNet-style + Attention, 1024 dim output.
    """

    def __init__(self, output_dim: int = 1024):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ResNet blocks
        self.layer1 = self._make_layer(64, 128, 3, stride=2)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 1024, 3, stride=2)

        # Global attention pooling
        self.attn_pool = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
        )

        # 파라미터 수
        total = sum(p.numel() for p in self.parameters())
        print(f"VisionEncoderXL: {total:,} params")

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        for _ in range(blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: [batch, 3, H, W]

        Returns:
            features: [batch, output_dim]
        """
        # CNN features
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 1024, H', W']

        # Flatten spatial
        batch = x.size(0)
        x = x.flatten(2).transpose(1, 2)  # [batch, H'*W', 1024]

        # Attention pooling
        query = self.query.expand(batch, -1, -1)
        x, _ = self.attn_pool(query, x, x)  # [batch, 1, 1024]
        x = x.squeeze(1)

        # Output
        return self.out_proj(x)


class PlasticVisionLNNXL(nn.Module):
    """
    Vision + PlasticLNN-XL 통합.

    End-to-end 화면 → 액션 + 트레이딩 + 분석.
    """

    def __init__(self, config: Optional[LNNXLConfig] = None, device: str = 'cuda'):
        super().__init__()
        self.config = config or LNNXLConfig()
        self.device = device

        # Vision encoder
        self.vision = VisionEncoderXL(output_dim=self.config.input_dim)

        # Plastic LNN
        self.plastic_lnn = PlasticLNNXL(self.config)

        self.to(device)

        # Total params
        total = sum(p.numel() for p in self.parameters())
        print(f"\n=== PlasticVisionLNN-XL Total: {total:,} params ===")
        print(f"=== Estimated VRAM: {total * 4 / 1e9:.2f} GB (fp32) ===\n")

    def forward(
        self,
        screen: Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward: 화면 → 액션/트레이딩/분석.

        Args:
            screen: [batch, 3, H, W] normalized
            tasks: which heads to compute

        Returns:
            Dict with task outputs
        """
        # Vision encoding
        features = self.vision(screen)  # [batch, input_dim]

        # LNN
        outputs = self.plastic_lnn(features, tasks)
        outputs['vision_features'] = features

        return outputs


def create_lnn_xl(device: str = 'cuda', size: str = 'medium') -> PlasticVisionLNNXL:
    """
    대규모 모델 생성.

    Args:
        device: 'cuda' or 'cpu'
        size: 'base' (700M), 'medium' (1.2B - 학습 가능), 'large' (2B), 'xl' (3.4B - inference only)

    RTX 3090 24GB 권장:
    - 학습: 'medium' (1.2B params, ~18GB with training)
    - 추론: 'large' or 'xl'
    """
    configs = {
        'base': LNNXLConfig(
            input_dim=1024,
            hidden_dim=2048,
            output_dim=256,
            num_layers=12,
            num_heads=16,
            head_dim=128,
            memory_size=10000,
            memory_dim=512,
        ),
        'medium': LNNXLConfig(  # 1.2B params, ~18GB with training (RTX 3090 optimal)
            input_dim=1280,
            hidden_dim=2560,
            output_dim=320,
            num_layers=14,
            num_heads=20,
            head_dim=128,
            memory_size=15000,
            memory_dim=640,
        ),
        'large': LNNXLConfig(  # 2B params, inference or gradient checkpointing
            input_dim=1536,
            hidden_dim=3072,
            output_dim=384,
            num_layers=16,
            num_heads=24,
            head_dim=128,
            memory_size=20000,
            memory_dim=768,
        ),
        'xl': LNNXLConfig(  # 3.4B params, inference only
            input_dim=1536,
            hidden_dim=3584,
            output_dim=384,
            num_layers=20,
            num_heads=28,
            head_dim=128,
            memory_size=30000,
            memory_dim=896,
            online_learning=False,
        ),
    }
    return PlasticVisionLNNXL(configs[size], device)


if __name__ == "__main__":
    # 테스트
    print("Creating PlasticLNN-XL...")
    model = create_lnn_xl('cuda')

    # Fake input
    x = torch.randn(1, 3, 720, 1280).cuda()

    print("\nForward pass...")
    with torch.cuda.amp.autocast():
        out = model(x)

    print("\nOutputs:")
    for k, v in out.items():
        if isinstance(v, Tensor):
            print(f"  {k}: {v.shape}")

    print("\nVRAM usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
