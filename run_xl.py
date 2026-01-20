#!/usr/bin/env python3
"""
LNN-AGI: XL Agent Runner

2B 파라미터 모델로 실행.
목표: 돈 버는 AI - 트레이딩, 자동화, 작업 수행.

사용법:
    python run_xl.py                    # 기본 실행 (2B model)
    python run_xl.py --vnc              # VNC로 관찰
    python run_xl.py --app firefox      # Firefox로 시작
    python run_xl.py --trading          # 트레이딩 모드 강조
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch


async def run_xl_agent(args):
    """XL Agent 실행."""
    from vm.environment import VirtualEnvironment, VMConfig
    from core.agent_xl import LivingAgentXL, AgentXLConfig

    print("=" * 60)
    print("       LNN-AGI: 2B Parameter Self-Learning Agent")
    print("       돈 버는 AI - Trading + Automation")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # VM 설정
    vm_config = VMConfig(
        width=args.width,
        height=args.height,
        display=args.display,
        enable_vnc=args.vnc,
        vnc_port=args.vnc_port,
    )

    # Agent 설정
    agent_config = AgentXLConfig(
        screen_width=args.width,
        screen_height=args.height,
        model_size=args.model_size,
        online_learning=True,
        exploration_rate=args.exploration,
        checkpoint_dir=args.checkpoint_dir,
        trading_enabled=args.trading,
    )

    # 시작
    print(f"\n📺 Starting virtual environment ({args.width}x{args.height})...")

    async with VirtualEnvironment(config=vm_config) as env:
        print(f"✓ Virtual display ready on {env.display}")

        if args.vnc:
            print(f"✓ VNC available on port {args.vnc_port}")
            print(f"  Connect: vncviewer localhost:{args.vnc_port}")

        # Agent 생성
        print("\n🧠 Initializing 2B Parameter Agent...")
        agent = LivingAgentXL(agent_config, device)
        await agent.connect_environment(env)

        # 체크포인트 로드
        if args.checkpoint:
            agent.load_checkpoint(args.checkpoint)

        # 앱 실행
        if args.app:
            print(f"\n🚀 Launching: {args.app}")
            env.launch_app(args.app)
            await asyncio.sleep(2)

        # 목표 설정
        if args.goal:
            agent.set_goal(args.goal)

        # 콜백
        def on_action(action):
            if args.verbose:
                print(f"  Action: ({action['x']:.2f}, {action['y']:.2f}) {action['click']}")

        def on_reward(reward):
            if args.verbose:
                print(f"  Reward: {reward:+.4f}")

        def on_trading(signal):
            if signal['confidence'] > 0.7:
                print(f"  💰 Trading: {signal['signal'].upper()} (conf: {signal['confidence']:.2f})")

        agent.on_action = on_action
        agent.on_reward = on_reward
        agent.on_trading_signal = on_trading

        # 실행
        print("\n" + "=" * 60)
        print("🎯 Starting 2B parameter continuous learning...")
        if args.trading:
            print("💰 Trading signals enabled")
        print("   Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        try:
            await agent.run_forever()
        except KeyboardInterrupt:
            print("\n\n⏹ Stopping...")

        # 최종 저장
        print("\n💾 Saving final checkpoint...")
        agent._save_checkpoint()
        agent.close()

        print("\n✓ LNN-AGI session ended")
        print(f"  Total steps: {agent.state.step}")
        print(f"  Total reward: {agent.state.total_reward:.2f}")
        print(f"  Trading signals: {agent.state.trading_signals}")


async def test_xl(model_size: str = 'base'):
    """테스트 모드."""
    from core.plastic_lnn_xl import create_lnn_xl
    import numpy as np

    print(f"=== LNN-AGI XL Test Mode ({model_size}) ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clear GPU cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = create_lnn_xl(device, model_size)

    # Fake input
    print("Running 50 fake steps...")

    for i in range(50):
        # Reset hidden states to avoid gradient issues across iterations
        model.plastic_lnn.reset_state(1, device)

        # Fake screen
        screen = torch.randn(1, 3, 720, 1280).to(device)

        # Forward with gradients
        model.train()
        with torch.amp.autocast('cuda'):
            output = model(screen)

        # Fake reward
        reward = np.random.randn() * 0.1

        # Get next state value as target (no grad, fresh states)
        model.plastic_lnn.reset_state(1, device)
        next_screen = torch.randn(1, 3, 720, 1280).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                next_output = model(next_screen)

        # Online update
        model.plastic_lnn.online_update(reward, output, next_output)

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}: value={output['value'].item():.4f}")

            # Trading signal
            signal = model.plastic_lnn.get_trading_signal(output)
            print(f"    Trading: {signal['signal']} (conf: {signal['confidence']:.2f})")

    print("\n✓ XL Test completed!")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="LNN-AGI: 2B Parameter Self-Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_xl.py                         # 기본 실행 (2B model)
  python run_xl.py --vnc                   # VNC로 관찰
  python run_xl.py --app firefox           # Firefox로 시작
  python run_xl.py --trading               # 트레이딩 모드
  python run_xl.py --model-size base       # 작은 모델 (700M)
  python run_xl.py --test                  # 테스트 모드
        """
    )

    # Environment
    parser.add_argument("--width", type=int, default=1280, help="Screen width")
    parser.add_argument("--height", type=int, default=720, help="Screen height")
    parser.add_argument("--display", default=":99", help="X display")
    parser.add_argument("--vnc", action="store_true", help="Enable VNC")
    parser.add_argument("--vnc-port", type=int, default=5999, help="VNC port")

    # Model
    parser.add_argument("--model-size", choices=['base', 'medium', 'large', 'xl'], default='medium',
                        help="Model size: base (700M), medium (1.2B trainable), large (2B), xl (3.4B inference)")

    # Agent
    parser.add_argument("--exploration", type=float, default=0.2, help="Exploration rate")
    parser.add_argument("--checkpoint", type=str, help="Load checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints_xl", help="Checkpoint directory")

    # Trading
    parser.add_argument("--trading", action="store_true", default=True, help="Enable trading signals")
    parser.add_argument("--no-trading", action="store_false", dest="trading", help="Disable trading")

    # Task
    parser.add_argument("--app", type=str, help="App to launch")
    parser.add_argument("--goal", type=str, help="Goal description")

    # Debug
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", action="store_true", help="Test mode (no environment)")

    args = parser.parse_args()

    # Signal handler
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    if args.test:
        asyncio.run(test_xl(args.model_size))
    else:
        asyncio.run(run_xl_agent(args))


if __name__ == "__main__":
    main()
