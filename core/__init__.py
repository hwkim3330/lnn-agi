"""Living AI - Core Module."""

from .plastic_lnn import (
    PlasticLNN,
    PlasticVisionLNN,
    LNNConfig,
    LTCCell,
    SimpleVisionEncoder,
    create_plastic_lnn,
    create_vision_lnn,
)

from .plastic_lnn_xl import (
    PlasticLNNXL,
    PlasticVisionLNNXL,
    LNNXLConfig,
    VisionEncoderXL,
    create_lnn_xl,
)

from .agent import (
    LivingAgent,
    AgentConfig,
    AgentState,
    create_living_agent,
)

__all__ = [
    # LNN Base
    "PlasticLNN",
    "PlasticVisionLNN",
    "LNNConfig",
    "LTCCell",
    "SimpleVisionEncoder",
    "create_plastic_lnn",
    "create_vision_lnn",

    # LNN XL (2B params)
    "PlasticLNNXL",
    "PlasticVisionLNNXL",
    "LNNXLConfig",
    "VisionEncoderXL",
    "create_lnn_xl",

    # Agent
    "LivingAgent",
    "AgentConfig",
    "AgentState",
    "create_living_agent",
]
