# AI 모델 가이드

재생에너지 발전량 예측을 위한 딥러닝 모델 사용 가이드

## 📋 목차

1. [개요](#개요)
2. [사용 가능한 모델](#사용-가능한-모델)
3. [모델 사용법](#모델-사용법)
4. [모델 비교](#모델-비교)
5. [고급 기능](#고급-기능)

---

## 개요

이 시스템은 시계열 데이터 예측에 특화된 여러 딥러닝 모델을 제공합니다:

- **LSTM 기반 모델**: 순환 신경망으로 시계열 패턴 학습
- **Transformer 기반 모델**: Attention 메커니즘으로 장기 의존성 포착
- **Ensemble 모델**: 여러 모델의 예측을 결합

---

## 사용 가능한 모델

### LSTM 모델군

#### 1. LSTMModel (기본 LSTM)
- 표준 LSTM 아키텍처
- 양방향 LSTM 지원
- 빠른 학습 속도

**장점:**
- 구현이 간단하고 안정적
- 적은 데이터로도 학습 가능
- 빠른 추론 속도

**단점:**
- 장기 의존성 포착 제한적
- 병렬 처리 어려움

**사용 예:**
```python
from src.models import LSTMModel

model = LSTMModel(
    input_dim=50,
    hidden_dim=128,
    num_layers=2,
    output_dim=24,
    sequence_length=168,
    dropout=0.2,
    bidirectional=False
)
```

#### 2. LSTMAttentionModel (LSTM + Attention)
- LSTM에 Attention 메커니즘 추가
- 중요한 시점에 집중
- 해석 가능한 예측

**장점:**
- 기본 LSTM보다 정확
- Attention 가중치로 해석 가능
- 장기 의존성 개선

**사용 예:**
```python
from src.models import LSTMAttentionModel

model = LSTMAttentionModel(
    input_dim=50,
    hidden_dim=128,
    num_layers=2,
    output_dim=24,
    sequence_length=168,
    dropout=0.2,
    attention_dim=128
)
```

#### 3. MultiOutputLSTM (다중 출력 헤드)
- 각 시간대별 독립적인 출력 헤드
- 다양한 예측 불확실성 처리

**사용 예:**
```python
from src.models import MultiOutputLSTM

model = MultiOutputLSTM(
    input_dim=50,
    hidden_dim=128,
    num_layers=2,
    output_dim=24,
    sequence_length=168,
    dropout=0.2
)
```

#### 4. StackedLSTM (깊은 LSTM)
- 여러 LSTM 레이어 적층
- Residual connection 지원
- 복잡한 패턴 학습

**사용 예:**
```python
from src.models import StackedLSTM

model = StackedLSTM(
    input_dim=50,
    hidden_dims=[128, 128, 64],  # 3개 레이어
    output_dim=24,
    sequence_length=168,
    dropout=0.2,
    use_residual=True
)
```

### Transformer 모델군

#### 1. TransformerModel (표준 Transformer)
- Multi-head self-attention
- Positional encoding
- 병렬 처리 가능

**장점:**
- 장기 의존성 우수
- 병렬 학습 가능
- 높은 정확도

**단점:**
- 많은 데이터 필요
- 메모리 사용량 높음
- 학습 시간 김

**사용 예:**
```python
from src.models import TransformerModel

model = TransformerModel(
    input_dim=50,
    d_model=128,
    n_heads=8,
    n_layers=4,
    d_ff=512,
    output_dim=24,
    sequence_length=168,
    dropout=0.1
)
```

#### 2. TimeSeriesTransformer (시계열 최적화)
- 시계열 예측에 특화
- 간소화된 구조
- PyTorch 기본 구현 활용

**사용 예:**
```python
from src.models import TimeSeriesTransformer

model = TimeSeriesTransformer(
    input_dim=50,
    d_model=128,
    n_heads=8,
    n_layers=4,
    output_dim=24,
    sequence_length=168,
    dropout=0.1
)
```

#### 3. InformerModel (장기 예측)
- ProbSparse attention
- Distillation 레이어
- 긴 시퀀스 처리 최적화

**장점:**
- 매우 긴 시퀀스 처리 가능
- 메모리 효율적
- 장기 예측에 강함

**사용 예:**
```python
from src.models import InformerModel

model = InformerModel(
    input_dim=50,
    d_model=128,
    n_heads=8,
    n_layers=4,
    output_dim=24,
    sequence_length=168,
    dropout=0.1,
    factor=5
)
```

### Ensemble 모델

여러 모델의 예측을 결합하여 더 안정적인 예측:

```python
from src.models import EnsembleModel, LSTMModel, TransformerModel

# 개별 모델 생성
lstm_model = LSTMModel(...)
transformer_model = TransformerModel(...)

# 앙상블 생성
ensemble = EnsembleModel(
    models=[lstm_model, transformer_model],
    weights=[0.5, 0.5]  # 동일한 가중치
)
```

---

## 모델 사용법

### 1. 간단한 생성 (Factory 함수)

```python
from src.models import create_model

model = create_model(
    model_type='lstm',  # 'lstm', 'transformer', 'timeseries' 등
    input_dim=50,
    output_dim=24,
    sequence_length=168,
    config={'hidden_dim': 128, 'num_layers': 2}
)
```

### 2. 설정 파일로 생성

```yaml
# configs/config.yaml
model:
  lstm:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
    bidirectional: False
```

```python
model = create_model(
    model_type='lstm',
    input_dim=50,
    output_dim=24,
    sequence_length=168,
    config_path='configs/config.yaml'
)
```

### 3. 모델 정보 확인

```python
from src.models import print_model_summary, count_parameters

# 상세 요약
print_model_summary(model)

# 파라미터 수
params = count_parameters(model)
print(f"Total parameters: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")

# 모델 크기
from src.models import get_model_size
size = get_model_size(model)
print(f"Model size: {size['mb']:.2f} MB")
```

### 4. 모델 저장/로드

```python
from src.models import save_model, load_model, get_device

# 저장
save_model(
    model=model,
    path='models/checkpoints/best_model.pth',
    epoch=100,
    optimizer=optimizer,
    metrics={'rmse': 0.05, 'mae': 0.03}
)

# 로드
device = get_device()
model = load_model(
    path='models/checkpoints/best_model.pth',
    model=model,
    device=device
)
```

### 5. 추론

```python
import torch

# 모델을 평가 모드로
model.eval()

# 추론
with torch.no_grad():
    # x: (batch_size, sequence_length, input_dim)
    predictions = model(x)
    # predictions: (batch_size, output_dim)

# 원래 스케일로 변환
from src.preprocessing import TargetScaler
target_scaler = TargetScaler().load('models/scalers/target_scaler.pkl')
predictions_original = target_scaler.inverse_transform(predictions.cpu().numpy())
```

---

## 모델 비교

### 성능 비교 (일반적인 경향)

| 모델 | 정확도 | 학습 속도 | 추론 속도 | 메모리 | 데이터 요구량 |
|------|--------|-----------|-----------|---------|---------------|
| LSTM | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| LSTM + Attention | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Transformer | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Informer | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 사용 시나리오별 권장 모델

#### 시나리오 1: 빠른 프로토타이핑
**추천:** LSTMModel
- 빠른 학습과 추론
- 안정적인 수렴
- 적은 리소스

#### 시나리오 2: 최고 정확도 필요
**추천:** TransformerModel 또는 Ensemble
- 충분한 데이터 확보 시
- GPU 리소스 사용 가능
- 학습 시간 여유

#### 시나리오 3: 장기 예측 (48시간 이상)
**추천:** InformerModel
- 긴 시퀀스 처리
- 메모리 효율적
- 장기 의존성

#### 시나리오 4: 제한된 데이터
**추천:** LSTMAttentionModel
- 적은 데이터로 학습 가능
- Attention으로 해석 가능
- 과적합 방지

#### 시나리오 5: 실시간 추론
**추천:** LSTMModel (단일 레이어)
- 빠른 추론 속도
- 낮은 지연시간
- 경량 모델

---

## 고급 기능

### 1. 가중치 초기화

```python
from src.models import initialize_weights

# Xavier 초기화
initialize_weights(model, method='xavier')

# Kaiming 초기화
initialize_weights(model, method='kaiming')
```

### 2. 레이어 동결 (Transfer Learning)

```python
from src.models.model_utils import freeze_layers, unfreeze_layers

# 특정 레이어 동결
freeze_layers(model, ['lstm', 'encoder'])

# 동결 해제
unfreeze_layers(model, ['fc', 'output'])
```

### 3. 그래디언트 확인

```python
from src.models.model_utils import get_gradient_info

# 역전파 후
loss.backward()

# 그래디언트 통계
grad_info = get_gradient_info(model)
print(f"Gradient mean: {grad_info['mean']:.6f}")
print(f"Gradient norm: {grad_info['norm']:.6f}")
```

### 4. 여러 모델 비교

```python
from src.models import compare_models

models = {
    'LSTM': lstm_model,
    'Transformer': transformer_model,
    'LSTM+Attention': lstm_attention_model
}

compare_models(models)
```

출력 예:
```
================================================================================
MODEL COMPARISON
================================================================================
Model                Parameters      Size (MB)    Type
--------------------------------------------------------------------------------
LSTM                    156,824         0.60     LSTMModel
Transformer             987,432         3.77     TransformerModel
LSTM+Attention          234,512         0.89     LSTMAttentionModel
================================================================================
```

### 5. 커스텀 모델 생성

BaseModel을 상속하여 커스텀 모델 생성:

```python
from src.models import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, input_dim, output_dim, sequence_length, **kwargs):
        super().__init__(input_dim, output_dim, sequence_length)

        # 커스텀 레이어 정의
        self.encoder = nn.LSTM(input_dim, 128, 2, batch_first=True)
        self.decoder = nn.Linear(128, output_dim)

    def forward(self, x):
        # Forward 로직
        lstm_out, _ = self.encoder(x)
        output = self.decoder(lstm_out[:, -1, :])
        return output
```

---

## 모델 선택 가이드

### 질문으로 선택하기

1. **데이터가 충분한가? (>10,000 샘플)**
   - YES → Transformer 고려
   - NO → LSTM 추천

2. **예측 기간은? (prediction_horizon)**
   - 단기 (<24시간) → LSTM
   - 중기 (24-48시간) → LSTM + Attention
   - 장기 (>48시간) → Informer

3. **학습 시간 제약은?**
   - 빠르게 → LSTM
   - 시간 여유 → Transformer

4. **해석 가능성 필요?**
   - YES → LSTM + Attention (Attention 가중치 시각화)
   - NO → 모든 모델 가능

5. **GPU 사용 가능?**
   - YES → Transformer (병렬 처리)
   - NO → LSTM (순차 처리도 빠름)

---

## 다음 단계

모델을 선택했다면:

1. **학습**: `src/training/` 모듈로 모델 학습
2. **평가**: 성능 지표 계산 (RMSE, MAE, MAPE)
3. **최적화**: 하이퍼파라미터 튜닝
4. **배포**: API로 서빙

관련 가이드:
- 학습: `TRAINING_GUIDE.md`
- 평가: `EVALUATION_GUIDE.md`
- 배포: `DEPLOYMENT_GUIDE.md`
