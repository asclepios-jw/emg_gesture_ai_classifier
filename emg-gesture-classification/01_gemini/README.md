# 01. Gemini (사람 + LLM 협업)

Docker 컨테이너 위에 Jupyter Notebook을 띄우고, **Gemini LLM과 대화하며 직접 작성**한 코드입니다. 다른 두 폴더(Antigravity, Claude Cowork)와 달리 에이전트가 자동으로 결과물을 뽑은 것이 아니라, 사람이 흐름을 통제하고 LLM은 코드 작성/리뷰를 보조하는 방식으로 진행했습니다.

## 환경

- 워크스테이션 + Docker + Jupyter Notebook
- PyTorch + CUDA (GPU 학습)
- Gemini LLM과 대화하며 셀 단위로 코드 작성·수정

## 공통 설계 (v1, v2 모두)

| 항목 | 값 |
|---|---|
| 평가 프로토콜 | **Subject-independent** — 학습 1~30번 피험자, 검증 31~36번 피험자 |
| 정규화 | 피험자별 Z-score (`mean/std` per subject) |
| 클래스 | 7 (원본 1~7 → 0~6으로 remap, class 0 unmarked는 제외) |
| Optimizer | AdamW (weight_decay=1e-3) |
| Scheduler | CosineAnnealingLR |
| Early stopping | val loss 기준 |

이 protocol은 *"학습에 본 적 없는 새 사용자에게도 동작하는가"* 를 평가하는 가장 엄격한 설정입니다. 같은 사용자의 데이터를 train/test에 섞는 mixed-subject 방식보다 일반적으로 정확도가 5~10%p 낮게 나옵니다.

---

## v1: Baseline 1D-CNN

**파일:** `emg_gesture_gm_v1.ipynb`
**결과: Validation Accuracy 91.51%**

가장 단순한 1D-CNN으로 baseline을 잡았습니다.

### 모델 구조

```
Input (8 ch, 64 timesteps)
    ↓
Conv1d(8→32, k=5) → BN → ReLU → MaxPool(2)
    ↓
Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
    ↓
Conv1d(64→128, k=3) → BN → ReLU → Dropout(0.5)
    ↓
Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→7)
```

### 하이퍼파라미터

- 윈도우: 64 samples, stride 32
- Batch size: 512
- LR: 0.001
- Epochs: 40 (early stopping 15 epoch에서 발동)

### 클래스별 결과

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.9915 | 0.9977 | 0.9946 | 1,282 |
| 1 | 0.9094 | 0.9254 | 0.9173 | 1,247 |
| 2 | 0.8530 | 0.8666 | 0.8597 | 1,252 |
| 3 | 0.9400 | 0.9268 | 0.9334 | 1,285 |
| 4 | 0.8987 | 0.9163 | 0.9074 | 1,278 |
| 5 | 0.8971 | 0.8555 | 0.8758 | 1,253 |
| 6 | — | — | — | 0 *(검증셋에 없음)* |

→ class 2, 5에서 confusion이 가장 많음 (손목 굽힘 ↔ ulnar deviation 등 해부학적으로 유사한 동작들)

---

## v2: SOTA Hybrid CNN + BiLSTM

**파일:** `emg_gesture_gm_v2.ipynb`
**결과: Validation Accuracy 93.18%** *(v1 대비 +1.67%p)*

v1의 한계를 분석하고 **시간적 dependency를 더 잘 잡기 위해 BiLSTM을 추가**한 하이브리드 모델로 업그레이드했습니다.

### v1 → v2 변경 사항

| 항목 | v1 | v2 |
|---|---|---|
| 윈도우 크기 | 64 | **128** (시야 2배) |
| Stride | 32 | 64 |
| 모델 구조 | CNN-only | **CNN + Bi-LSTM hybrid** |
| 손실 함수 | CrossEntropy | **CrossEntropy with Label Smoothing (0.1)** |
| LR | 0.001 | 0.0005 (모델이 무거워서 안정 진입 위해 ↓) |
| Epochs | 40 (patience 7) | 50 (patience 10) |

### 새 모델 구조 (HybridEMGNet)

```
Input (8 ch, 128 timesteps)
    ↓
[Spatial Feature Extractor — 1D-CNN]
Conv1d(8→32) → BN → ReLU → MaxPool(2)
Conv1d(32→64) → BN → ReLU → MaxPool(2)
Conv1d(64→128) → BN → ReLU → MaxPool(2)
    ↓ (Batch, 128 features, 16 timesteps)
[Temporal Sequence Learner — Bidirectional LSTM]
BiLSTM(input=128, hidden=128, num_layers=2)
    ↓
Global Average Pooling (전체 timestep 평균, 마지막 step만 쓰지 않음)
    ↓
Classifier (Linear → ReLU → Dropout → Linear)
```

### 개선의 핵심 아이디어

1. **윈도우 2배 확장 (64 → 128)**: 짧은 윈도우는 동작의 시작/중간/끝을 구분 못 함. 시야를 넓혀 동작 전체 패턴을 보게 함.
2. **CNN + BiLSTM 결합**: CNN은 채널 간 공간적 패턴(spatial)을 잡고, BiLSTM은 시간 축의 순차적 dependency를 잡음. 두 정보를 함께 활용.
3. **Global Average Pooling**: BiLSTM의 마지막 hidden state만 쓰면 앞부분 정보 손실이 큼. 전체 시퀀스 평균을 써서 정보 손실 방지.
4. **Label Smoothing (0.1)**: 비슷한 동작 클래스 (특히 손가락/손목 계열) 사이의 hard one-hot 학습을 부드럽게 만들어 과적합 방지.

### 클래스별 결과

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.9938 | 1.0000 | 0.9969 | 636 |
| 1 | 0.9331 | 0.9467 | 0.9399 | 619 |
| 2 | 0.8636 | 0.9179 | 0.8899 | 621 |
| 3 | 0.9219 | 0.9639 | 0.9424 | 637 |
| 4 | 0.9410 | 0.9039 | 0.9221 | 635 |
| 5 | 0.9416 | 0.8567 | 0.8971 | 621 |

→ 모든 클래스에서 v1 대비 F1이 상승. 특히 가장 약했던 class 2, 5의 precision/recall 균형이 개선됨.

---

## v1 → v2 개선 요약

| 지표 | v1 (CNN) | v2 (CNN + BiLSTM) | Δ |
|---|---|---|---|
| Val Accuracy | 91.51% | 93.18% | **+1.67%p** |
| Macro F1 | 0.7840 | 0.9314 | **+0.147** |
| 학습 에폭 | 15 (early stop) | 14 (early stop) | — |
| 에폭당 시간 | 1.3초 | 3.3초 | +2배 (모델 무거워짐) |

> Macro F1이 크게 오른 이유: v1은 class 6 support가 0이라 macro 계산이 왜곡됐고, v2는 윈도우 크기를 2배로 늘려 데이터가 줄면서 class 6 자체가 검증셋에 안 들어옴 — 두 버전 모두 6-class 실질 평가.

## 실행 방법

```bash
# Docker 컨테이너 안에서 (CUDA + PyTorch 환경)
jupyter notebook

# emg_gesture_gm_v1.ipynb 또는 v2.ipynb 열고 위에서부터 셀 실행
```

⚠️ 데이터 경로 (`/workspace/medical_datasets/EMG-data.csv`)는 본인 환경에 맞게 수정 필요.

## 다른 접근법과의 차이

- 최상위 [README.md](../README.md)의 3-way 비교 표 참고
- 가장 큰 차이: **subject-independent 평가**를 채택해 실용적으로 의미있는 일반화 성능을 측정
- 정확도 숫자만 보면 다른 두 접근(96~97%)이 더 높아 보이지만, 그쪽은 mixed-subject random split이라 평가 난이도가 훨씬 낮음
