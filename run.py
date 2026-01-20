#!/usr/bin/env python3
"""
Living AI - 실행 스크립트

가상 OS에서 스스로 배우는 AI 실행.

사용법:
    python run.py                    # 기본 실행
    python run.py --vnc              # VNC 활성화 (원격 관찰)
    python run.py --app firefox      # 특정 앱으로 시작
    python run.py --goal "open terminal and type hello"
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import torch


async def run_living_ai(args):
    """Living AI 실행."""
    from vm.environment import VirtualEnvironment, VMConfig
    from core.agent import LivingAgent, AgentConfig

    print("=" * 60)
    print("         Living AI - Self-Learning Agent")
    print("         실시간 학습하는 진짜 AI")
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
    agent_config = AgentConfig(
        screen_width=args.width,
        screen_height=args.height,
        online_learning=True,
        exploration_rate=args.exploration,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 시작
    print(f"\n📺 Starting virtual environment ({args.width}x{args.height})...")

    async with VirtualEnvironment(config=vm_config) as env:
        print(f"✓ Virtual display ready on {env.display}")

        if args.vnc:
            print(f"✓ VNC available on port {args.vnc_port}")
            print(f"  Connect with: vncviewer localhost:{args.vnc_port}")

        # Agent 생성
        print("\n🧠 Initializing Living AI agent...")
        agent = LivingAgent(agent_config, device)
        await agent.connect_environment(env)

        # 체크포인트 로드
        if args.checkpoint:
            agent.load_checkpoint(args.checkpoint)

        # 앱 실행
        if args.app:
            print(f"\n🚀 Launching: {args.app}")
            env.launch_app(args.app)
            await asyncio.sleep(2)  # Wait for app to start

        # 목표 설정
        if args.goal:
            agent.set_goal(args.goal)

        # Callbacks
        def on_action(action):
            if args.verbose:
                print(f"  Action: ({action['x']:.2f}, {action['y']:.2f}) {action['click']}")

        def on_reward(reward):
            if args.verbose and abs(reward) > 0.1:
                print(f"  Reward: {reward:+.4f}")

        agent.on_action = on_action
        agent.on_reward = on_reward

        # 실행
        print("\n" + "=" * 60)
        print("🎯 Starting continuous learning...")
        print("   Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        try:
            await agent.run_forever()
        except KeyboardInterrupt:
            print("\n\n⏹ Stopping...")

        # 최종 저장
        print("\n💾 Saving final checkpoint...")
        agent._save_checkpoint()

        print("\n✓ Living AI session ended")
        print(f"  Total steps: {agent.state.step}")
        print(f"  Total reward: {agent.state.total_reward:.2f}")


async def test_mode():
    """테스트 모드 (환경 없이 모델만)."""
    from core.agent import create_living_agent
    import numpy as np

    print("=== Test Mode (no virtual environment) ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = create_living_agent(device=device)

    # Fake interaction
    print("Running 100 fake steps...")

    agent._vision_lnn.train()

    for i in range(100):
        # Fake screen
        screen = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        tensor = agent._preprocess_screen(screen)

        # Forward (need gradients for value)
        output = agent._vision_lnn(tensor)

        # Decode action (detached for use)
        action = {
            'x': torch.sigmoid(output['action'][0, 0]).item(),
            'y': torch.sigmoid(output['action'][0, 1]).item(),
            'click': 'none',
            'keys': [],
        }

        # Fake reward
        reward = np.random.randn() * 0.1

        # Next state (for TD target, will be detached in online_update)
        next_screen = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        next_tensor = agent._preprocess_screen(next_screen)

        with torch.no_grad():
            next_output = agent._vision_lnn(next_tensor)

        # Online update (gradient flows through output['value'])
        agent._vision_lnn.plastic_lnn.online_update(reward, output, next_output)

        if (i + 1) % 20 == 0:
            with torch.no_grad():
                test_out = agent._vision_lnn(tensor)
            print(f"  Step {i+1}: value={test_out['value'].item():.4f}")

    print("\n✓ Test completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Living AI - Self-Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                         # 기본 실행
  python run.py --vnc                   # VNC로 관찰 가능
  python run.py --app "firefox"         # Firefox로 시작
  python run.py --goal "open terminal"  # 목표 설정
  python run.py --test                  # 환경 없이 테스트
        """
    )

    # Environment
    parser.add_argument("--width", type=int, default=1280, help="Screen width")
    parser.add_argument("--height", type=int, default=720, help="Screen height")
    parser.add_argument("--display", default=":99", help="X display")
    parser.add_argument("--vnc", action="store_true", help="Enable VNC")
    parser.add_argument("--vnc-port", type=int, default=5999, help="VNC port")

    # Agent
    parser.add_argument("--exploration", type=float, default=0.3, help="Exploration rate")
    parser.add_argument("--checkpoint", type=str, help="Load checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")

    # Task
    parser.add_argument("--app", type=str, help="App to launch (e.g., 'firefox', 'xterm')")
    parser.add_argument("--goal", type=str, help="Goal description")

    # Debug
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", action="store_true", help="Test mode (no environment)")

    args = parser.parse_args()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    if args.test:
        asyncio.run(test_mode())
    else:
        asyncio.run(run_living_ai(args))


if __name__ == "__main__":
    main()
