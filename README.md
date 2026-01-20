# LNN-AGI

**1.2B Plastic Neural Network** - 실시간 학습하는 자율 AI

## 핵심

```
frozen LLM (추론만) → Plastic LNN (실시간 학습)
API 비용 → 로컬 GPU
스크립트 실행 → 화면 보고 판단
```

## 구조

```
Screen → VisionEncoder(45M) → PlasticLNN(1.2B) → Action/Trading/Analysis
            ↓                      ↓
         화면 캡처            실시간 가중치 업데이트
```

**Liquid Time-Constant Network**: `τ * dh/dt = -h + f(Wx*x + Wh*h)`
- 동적 시간상수: 입력에 따라 반응속도 변화
- Multi-head Attention + LTC 결합
- TD-Learning 온라인 업데이트

## 실행

```bash
# 의존성
pip install torch numpy pillow mss pyautogui
sudo apt install xvfb xdotool x11vnc

# 테스트
python run_xl.py --test

# 실행 (1.2B model, 18GB VRAM)
python run_xl.py --vnc --app xterm

# VNC 연결
vncviewer localhost:5999
```

## 모델 크기

| Size | Params | VRAM (학습) | 용도 |
|------|--------|-------------|------|
| base | 725M | 11GB | 실험 |
| **medium** | **1.2B** | **18GB** | **RTX 3090** |
| large | 2B | 24GB+ | 추론 |

## 출력

```python
{
    'action': [x, y, click],      # 마우스/키보드
    'trading': {                   # 트레이딩 신호
        'signal': 'buy/sell/hold',
        'confidence': 0.85
    },
    'analysis': [...],            # 화면 분석
    'value': 0.5                  # 상태 가치
}
```

## 파일

```
├── core/
│   ├── plastic_lnn.py      # Base LNN
│   ├── plastic_lnn_xl.py   # XL LNN (1.2B)
│   └── agent_xl.py         # Agent + Trading
├── vm/environment.py       # Virtual OS
├── run_xl.py               # 실행
└── checkpoints_xl/         # 학습된 가중치
```

## TODO

- [x] Plastic LNN + Attention
- [x] 1.2B 학습 가능 모델
- [x] Trading Head
- [ ] 실제 트레이딩 연동
- [ ] Goal-conditioned Learning

---

MIT License
