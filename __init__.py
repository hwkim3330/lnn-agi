"""
Living AI - Self-Learning Agent

진정한 자율 학습 AI:
- Plastic Liquid Neural Network (가중치 실시간 업데이트)
- 가상 OS 완전 통제
- 경험에서 진화

이건 wrapper가 아님. 실제로 배우는 AI.
"""

__version__ = "0.1.0"

from .core import (
    PlasticLNN,
    PlasticVisionLNN,
    LNNConfig,
    LivingAgent,
    AgentConfig,
    create_plastic_lnn,
    create_vision_lnn,
    create_living_agent,
)

from .vm import (
    VirtualEnvironment,
    VMConfig,
)

__all__ = [
    "__version__",

    # LNN
    "PlasticLNN",
    "PlasticVisionLNN",
    "LNNConfig",
    "create_plastic_lnn",
    "create_vision_lnn",

    # Agent
    "LivingAgent",
    "AgentConfig",
    "create_living_agent",

    # VM
    "VirtualEnvironment",
    "VMConfig",
]
