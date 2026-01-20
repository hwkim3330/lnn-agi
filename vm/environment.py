#!/usr/bin/env python3
"""
Living AI - Virtual OS Environment

가상 데스크톱 환경:
- Docker 컨테이너 + Xvfb (가상 디스플레이)
- VNC로 접근 가능
- 스크린 캡처 + 마우스/키보드 주입

AI가 완전히 통제하는 샌드박스 환경.
"""

import asyncio
import subprocess
import time
import os
import signal
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    from PIL import Image
    import mss
except ImportError:
    Image = None
    mss = None


@dataclass
class VMConfig:
    """가상 환경 설정."""
    # Display
    width: int = 1280
    height: int = 720
    depth: int = 24
    display: str = ":99"

    # Docker (optional)
    use_docker: bool = False
    docker_image: str = "ubuntu:22.04"
    container_name: str = "living_ai_vm"

    # VNC (optional)
    vnc_port: int = 5999
    enable_vnc: bool = False

    # Performance
    capture_fps: int = 10


class XvfbEnvironment:
    """
    Xvfb 기반 가상 디스플레이 환경.

    Docker 없이 로컬에서 가상 디스플레이 생성.
    """

    def __init__(self, config: VMConfig = None):
        self.config = config or VMConfig()
        self._xvfb_proc: Optional[subprocess.Popen] = None
        self._display = self.config.display
        self._started = False

    async def start(self):
        """가상 디스플레이 시작."""
        if self._started:
            return

        print(f"Starting Xvfb on {self._display}...")

        # Xvfb 시작
        cmd = [
            "Xvfb", self._display,
            "-screen", "0",
            f"{self.config.width}x{self.config.height}x{self.config.depth}",
            "-ac",  # Disable access control
            "-nolisten", "tcp",
        ]

        self._xvfb_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for display
        await asyncio.sleep(0.5)

        # Set environment
        os.environ["DISPLAY"] = self._display

        # Start window manager (optional, for better rendering)
        try:
            subprocess.Popen(
                ["openbox", "--replace"],
                env={"DISPLAY": self._display},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass  # openbox not installed

        self._started = True
        print(f"✓ Xvfb started on {self._display}")

        # VNC server (optional)
        if self.config.enable_vnc:
            await self._start_vnc()

    async def _start_vnc(self):
        """VNC 서버 시작."""
        try:
            subprocess.Popen(
                [
                    "x11vnc",
                    "-display", self._display,
                    "-forever",
                    "-shared",
                    "-rfbport", str(self.config.vnc_port),
                    "-nopw",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"✓ VNC server on port {self.config.vnc_port}")
        except FileNotFoundError:
            print("x11vnc not found, VNC disabled")

    async def stop(self):
        """환경 종료."""
        if self._xvfb_proc:
            self._xvfb_proc.terminate()
            try:
                self._xvfb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._xvfb_proc.kill()
            self._xvfb_proc = None

        self._started = False
        print("✓ Xvfb stopped")

    def capture_screen(self) -> np.ndarray:
        """
        화면 캡처.

        Returns:
            np.ndarray: RGB 이미지 [H, W, 3]
        """
        if not self._started:
            raise RuntimeError("Environment not started")

        # mss를 사용한 캡처
        if mss is None:
            raise ImportError("mss not installed: pip install mss")

        with mss.mss() as sct:
            # 특정 디스플레이 캡처
            monitor = {
                "top": 0,
                "left": 0,
                "width": self.config.width,
                "height": self.config.height,
            }

            screenshot = sct.grab(monitor)
            img = np.array(screenshot)[:, :, :3]  # BGRA -> RGB
            img = img[:, :, ::-1]  # BGR -> RGB

        return img

    def capture_pil(self) -> 'Image.Image':
        """PIL Image로 캡처."""
        arr = self.capture_screen()
        return Image.fromarray(arr)

    def inject_mouse(
        self,
        x: int,
        y: int,
        button: str = 'left',
        action: str = 'click'
    ):
        """
        마우스 이벤트 주입.

        Args:
            x, y: 좌표
            button: 'left', 'right', 'middle'
            action: 'click', 'down', 'up', 'move', 'double'
        """
        try:
            import pyautogui
            pyautogui.FAILSAFE = False

            if action == 'move':
                pyautogui.moveTo(x, y)
            elif action == 'click':
                pyautogui.click(x, y, button=button)
            elif action == 'double':
                pyautogui.doubleClick(x, y, button=button)
            elif action == 'down':
                pyautogui.mouseDown(x, y, button=button)
            elif action == 'up':
                pyautogui.mouseUp(x, y, button=button)

        except ImportError:
            # xdotool fallback
            self._xdotool_mouse(x, y, button, action)

    def _xdotool_mouse(self, x: int, y: int, button: str, action: str):
        """xdotool로 마우스 제어."""
        button_map = {'left': 1, 'middle': 2, 'right': 3}
        btn = button_map.get(button, 1)

        env = {"DISPLAY": self._display}

        if action == 'move':
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], env=env)
        elif action == 'click':
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], env=env)
            subprocess.run(["xdotool", "click", str(btn)], env=env)
        elif action == 'double':
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], env=env)
            subprocess.run(["xdotool", "click", "--repeat", "2", str(btn)], env=env)

    def inject_keyboard(self, keys: str, action: str = 'type'):
        """
        키보드 이벤트 주입.

        Args:
            keys: 키 문자열 또는 특수키 (예: 'enter', 'ctrl+c')
            action: 'type', 'press', 'down', 'up'
        """
        try:
            import pyautogui
            pyautogui.FAILSAFE = False

            if action == 'type':
                pyautogui.typewrite(keys, interval=0.05)
            elif action == 'press':
                pyautogui.press(keys)
            elif action == 'hotkey':
                pyautogui.hotkey(*keys.split('+'))

        except ImportError:
            self._xdotool_keyboard(keys, action)

    def _xdotool_keyboard(self, keys: str, action: str):
        """xdotool로 키보드 제어."""
        env = {"DISPLAY": self._display}

        if action == 'type':
            subprocess.run(["xdotool", "type", "--", keys], env=env)
        elif action == 'press':
            subprocess.run(["xdotool", "key", keys], env=env)
        elif action == 'hotkey':
            subprocess.run(["xdotool", "key", keys.replace('+', '+')], env=env)

    def launch_app(self, command: str) -> subprocess.Popen:
        """
        앱 실행.

        Args:
            command: 실행할 명령

        Returns:
            Popen 객체
        """
        env = os.environ.copy()
        env["DISPLAY"] = self._display

        proc = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return proc

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()


class DockerEnvironment:
    """
    Docker 기반 완전 격리 환경.

    더 안전한 샌드박스 (파일시스템, 네트워크 격리).
    """

    def __init__(self, config: VMConfig = None):
        self.config = config or VMConfig()
        self.config.use_docker = True
        self._container_id: Optional[str] = None
        self._started = False

    async def start(self):
        """Docker 컨테이너 시작."""
        if self._started:
            return

        print("Starting Docker environment...")

        # Dockerfile 생성
        dockerfile = f"""
FROM {self.config.docker_image}

RUN apt-get update && apt-get install -y \\
    xvfb \\
    x11vnc \\
    openbox \\
    xterm \\
    firefox \\
    xdotool \\
    scrot \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install mss pillow pyautogui

ENV DISPLAY=:99

CMD ["bash", "-c", "Xvfb :99 -screen 0 {self.config.width}x{self.config.height}x{self.config.depth} & sleep 1 && openbox & x11vnc -display :99 -forever -shared -rfbport 5900 -nopw & tail -f /dev/null"]
"""

        # 컨테이너 실행
        # (실제 구현에서는 docker-py 사용)
        cmd = [
            "docker", "run", "-d",
            "--name", self.config.container_name,
            "-p", f"{self.config.vnc_port}:5900",
            "--shm-size=2g",
            self.config.docker_image,
        ]

        # 간단한 버전: 직접 실행
        subprocess.run(cmd, capture_output=True)

        self._started = True
        print(f"✓ Docker container started: {self.config.container_name}")

    async def stop(self):
        """컨테이너 종료."""
        subprocess.run(
            ["docker", "stop", self.config.container_name],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "rm", self.config.container_name],
            capture_output=True,
        )
        self._started = False

    def capture_screen(self) -> np.ndarray:
        """컨테이너 화면 캡처."""
        # VNC를 통해 캡처하거나 docker exec으로 scrot 사용
        result = subprocess.run(
            ["docker", "exec", self.config.container_name,
             "scrot", "-o", "/tmp/screen.png"],
            capture_output=True,
        )

        # 파일 복사
        subprocess.run(
            ["docker", "cp",
             f"{self.config.container_name}:/tmp/screen.png",
             "/tmp/docker_screen.png"],
            capture_output=True,
        )

        img = Image.open("/tmp/docker_screen.png")
        return np.array(img)


# ============================================================================
# Environment Wrapper (통합 인터페이스)
# ============================================================================

class VirtualEnvironment:
    """
    가상 환경 통합 인터페이스.

    사용법:
        async with VirtualEnvironment() as env:
            env.launch_app("firefox")
            screen = env.capture_screen()
            env.inject_mouse(100, 100, action='click')
    """

    def __init__(
        self,
        use_docker: bool = False,
        config: VMConfig = None,
    ):
        self.config = config or VMConfig(use_docker=use_docker)

        if use_docker:
            self._env = DockerEnvironment(self.config)
        else:
            self._env = XvfbEnvironment(self.config)

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    async def start(self):
        await self._env.start()

    async def stop(self):
        await self._env.stop()

    def capture_screen(self) -> np.ndarray:
        """화면 캡처 [H, W, 3]."""
        return self._env.capture_screen()

    def capture_tensor(self, device: str = 'cuda') -> 'torch.Tensor':
        """PyTorch 텐서로 캡처 [1, 3, H, W]."""
        import torch
        arr = self.capture_screen()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(device) / 255.0

    def inject_action(self, action: Dict[str, Any]):
        """
        행동 주입.

        Args:
            action: {
                'x': 0.0-1.0,
                'y': 0.0-1.0,
                'click': 'none'|'left'|'right'|'double',
                'keys': ['enter', 'tab', ...]
            }
        """
        # Denormalize coordinates
        x = int(action.get('x', 0.5) * self.config.width)
        y = int(action.get('y', 0.5) * self.config.height)

        # Mouse
        click = action.get('click', 'none')
        if click != 'none':
            self._env.inject_mouse(x, y, button='left', action=click)
        else:
            self._env.inject_mouse(x, y, action='move')

        # Keyboard
        keys = action.get('keys', [])
        for key in keys:
            self._env.inject_keyboard(key, action='press')

    def launch_app(self, command: str):
        """앱 실행."""
        return self._env.launch_app(command)

    @property
    def display(self) -> str:
        return self.config.display

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.config.width, self.config.height)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()


# ============================================================================
# Test
# ============================================================================

async def test_environment():
    """환경 테스트."""
    print("=== Virtual Environment Test ===\n")

    config = VMConfig(
        width=1280,
        height=720,
        display=":99",
        enable_vnc=False,
    )

    async with VirtualEnvironment(use_docker=False, config=config) as env:
        print(f"Resolution: {env.resolution}")

        # Launch terminal
        print("Launching xterm...")
        env.launch_app("xterm")
        await asyncio.sleep(2)

        # Capture
        print("Capturing screen...")
        screen = env.capture_screen()
        print(f"Screen shape: {screen.shape}")

        # Save
        if Image:
            img = Image.fromarray(screen)
            img.save("/tmp/living_ai_test.png")
            print("Saved to /tmp/living_ai_test.png")

        # Click test
        print("Clicking at (640, 360)...")
        env.inject_action({
            'x': 0.5,
            'y': 0.5,
            'click': 'left',
        })
        await asyncio.sleep(0.5)

        # Type test
        print("Typing 'hello'...")
        env._env.inject_keyboard("hello", action='type')
        await asyncio.sleep(0.5)

        # Final capture
        final = env.capture_screen()
        if Image:
            Image.fromarray(final).save("/tmp/living_ai_final.png")
            print("Saved to /tmp/living_ai_final.png")

    print("\n✓ Test completed!")


if __name__ == "__main__":
    asyncio.run(test_environment())
