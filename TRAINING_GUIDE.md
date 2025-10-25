# 학습 가이드

재생에너지 발전량 예측 모델 학습 가이드

## 📋 목차

1. [개요](#개요)
2. [빠른 시작](#빠른-시작)
3. [학습 프로세스](#학습-프로세스)
4. [Loss Functions](#loss-functions)
5. [Metrics](#metrics)
6. [고급 설정](#고급-설정)

---

## 개요

이 시스템은 재생에너지 발전량 예측 모델 학습을 위한 완전한 파이프라인을 제공합니다:

- ✅ 자동 데이터 전처리
- ✅ 다양한 Loss functions
- ✅ Early stopping & Learning rate scheduling
- ✅ 체크포인트 관리
- ✅ 실시간 모니터링
- ✅ 평가 지표 계산

---

## 빠른 시작

### 1. 기본 학습

```bash
python scripts/train_model.py
```

기본 설정:
- Model: LSTM
- Location: Seoul Solar Farm
- Energy Type: Solar
- Epochs: 100
- Batch Size: 32

### 2. 옵션 지정

```bash
python scripts/train_model.py \
  --model-type transformer \
  --location "Seoul Solar Farm" \
  --energy-type solar \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.0001 \
  --gpu 0
```

**사용 가능한 옵션:**
- `--model-type`: lstm, lstm_attention, transformer, timeseries
- `--location`: 위치 이름 (config.yaml에 정의)
- `--energy-type`: solar, wind
- `--epochs`: 학습 에폭 수
- `--batch-size`: 배치 크기
- `--lr`: 학습률
- `--gpu`: GPU ID (None이면 자동 선택)
- `--resume`: 체크포인트에서 재개

### 3. 학습 재개

```bash
python scripts/train_model.py --resume models/checkpoints/last_model.pth
```

---

## 학습 프로세스

### 전체 파이프라인

```
1. 데이터 준비 (DataPipeline)
   ├── 데이터베이스에서 로드
   ├── 전처리 및 특성 생성
   ├── 스케일링
   └── 시퀀스 생성

2. 모델 생성 (create_model)
   └── 지정된 타입의 모델 인스턴스 생성

3. Trainer 초기화
   ├── Loss function 설정
   ├── Optimizer 설정
   ├── Early stopping 설정
   └── Learning rate scheduler 설정

4. 학습 (Trainer.train)
   ├── Epoch 루프
   │   ├── 학습 (train_epoch)
   │   ├── 검증 (validate)
   │   ├── Metrics 계산
   │   ├── Learning rate 조정
   │   ├── 체크포인트 저장
   │   └── Early stopping 체크
   └── 학습 완료

5. 평가 (Trainer.evaluate)
   └── 테스트 세트에서 최종 평가
```

### Python 코드로 학습

```python
from src.preprocessing import DataPipeline
from src.models import create_model
from src.training import Trainer, get_loss_function
from src.models import get_device

# 1. 데이터 준비
pipeline = DataPipeline(
    location_name='Seoul Solar Farm',
    energy_type='solar',
    latitude=37.5665,
    longitude=126.9780,
    sequence_length=168,
    prediction_horizon=24
)

result = pipeline.run_pipeline(batch_size=32)
train_loader = result['train_loader']
val_loader = result['val_loader']
test_loader = result['test_loader']

# 2. 모델 생성
model = create_model(
    model_type='lstm',
    input_dim=result['n_features'],
    output_dim=24,
    sequence_length=168
)

# 3. Trainer 설정
device = get_device()
criterion = get_loss_function('rmse')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

config = {
    'early_stopping': {'enabled': True, 'patience': 10},
    'scheduler': {'type': 'ReduceLROnPlateau', 'patience': 5},
    'checkpoint': {'save_best': True, 'save_dir': 'models/checkpoints'}
}

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    config=config
)

# 4. 학습
history = trainer.train(epochs=100)

# 5. 평가
metrics = trainer.evaluate(test_loader)
```

---

## Loss Functions

### 사용 가능한 Loss Functions

#### 1. MSE (Mean Squared Error)
```python
criterion = get_loss_function('mse')
```
- 가장 기본적인 회귀 손실
- 큰 오차에 더 큰 페널티

#### 2. MAE (Mean Absolute Error)
```python
criterion = get_loss_function('mae')
```
- 이상치에 덜 민감
- 모든 오차에 동일한 가중치

#### 3. RMSE (Root Mean Squared Error)
```python
criterion = get_loss_function('rmse')
```
- MSE의 제곱근
- 원래 스케일로 해석 가능

#### 4. Huber Loss
```python
criterion = get_loss_function('huber', delta=1.0)
```
- MSE와 MAE의 결합
- 이상치에 강건

#### 5. MAPE (Mean Absolute Percentage Error)
```python
criterion = get_loss_function('mape')
```
- 백분율 오차
- 스케일 독립적

#### 6. Weighted MSE
```python
criterion = get_loss_function('weighted_mse')
```
- 시간대별 가중치 적용
- 단기 예측에 더 큰 가중치

#### 7. Asymmetric Loss
```python
criterion = get_loss_function('asymmetric', beta=2.0)
```
- 과소 예측에 더 큰 페널티
- 재생에너지 예측에 유용

### Loss Function 비교

| Loss | 장점 | 단점 | 사용 시나리오 |
|------|------|------|---------------|
| MSE | 수학적으로 간단 | 이상치에 민감 | 일반적인 회귀 |
| MAE | 이상치에 강건 | 최적화 느림 | 이상치 많은 데이터 |
| RMSE | 해석 용이 | MSE와 유사 | 표준 평가 |
| Huber | 강건성 | 하이퍼파라미터 필요 | 이상치 존재 |
| MAPE | 스케일 독립 | 0에 가까운 값 문제 | 비율 중요 |
| Weighted MSE | 시간대별 차별화 | 가중치 설정 필요 | 단기 예측 중요 |
| Asymmetric | 비대칭 페널티 | 불균형 문제 | 과소예측 위험 |

---

## Metrics

### 평가 지표

#### 1. RMSE (Root Mean Squared Error)
```
RMSE = sqrt(mean((y_pred - y_true)^2))
```
- **의미**: 예측과 실제 값의 평균 편차 (제곱근)
- **단위**: 원래 데이터와 동일 (kW)
- **좋은 값**: 낮을수록 좋음

#### 2. MAE (Mean Absolute Error)
```
MAE = mean(abs(y_pred - y_true))
```
- **의미**: 절대 오차의 평균
- **단위**: 원래 데이터와 동일 (kW)
- **좋은 값**: 낮을수록 좋음

#### 3. MAPE (Mean Absolute Percentage Error)
```
MAPE = mean(abs((y_true - y_pred) / y_true)) * 100
```
- **의미**: 백분율 오차의 평균
- **단위**: %
- **좋은 값**: <10% (우수), 10-20% (좋음), >20% (개선 필요)

#### 4. R² (R-squared)
```
R² = 1 - (SS_res / SS_tot)
```
- **의미**: 모델이 설명하는 분산의 비율
- **범위**: -∞ ~ 1
- **좋은 값**: 1에 가까울수록 좋음 (1 = 완벽한 예측)

#### 5. NRMSE (Normalized RMSE)
```
NRMSE = RMSE / (max(y_true) - min(y_true)) * 100
```
- **의미**: 데이터 범위로 정규화된 RMSE
- **단위**: %
- **좋은 값**: 낮을수록 좋음

#### 6. MBE (Mean Bias Error)
```
MBE = mean(y_pred - y_true)
```
- **의미**: 체계적 과대/과소 예측
- **양수**: 과대 예측
- **음수**: 과소 예측
- **0**: 편향 없음

### Metrics 계산

```python
from src.training import MetricsCalculator

calculator = MetricsCalculator()

# 모든 지표 계산
metrics = calculator.calculate_all(y_true, y_pred)

# 시간대별 지표
per_horizon_metrics = calculator.calculate_per_horizon(
    y_true, y_pred,
    horizon_names=['1h', '2h', ..., '24h']
)

# 출력
calculator.print_metrics(metrics)
```

---

## 고급 설정

### Early Stopping

```yaml
# configs/config.yaml
training:
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0001
```

- **patience**: 개선 없이 기다릴 에폭 수
- **min_delta**: 개선으로 간주할 최소 변화량

### Learning Rate Scheduler

```yaml
training:
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 5
    factor: 0.5
    min_lr: 0.00001
```

- **patience**: LR 감소 전 기다릴 에폭
- **factor**: LR 감소 비율
- **min_lr**: 최소 학습률

### Gradient Clipping

```yaml
training:
  grad_clip: 1.0  # 0이면 비활성화
```

### 체크포인트

```yaml
training:
  checkpoint:
    save_best: true
    save_last: true
    save_dir: "./models/checkpoints"
```

저장되는 파일:
- `best_model.pth`: 검증 손실이 가장 낮은 모델
- `last_model.pth`: 마지막 에폭 모델

### 커스텀 학습 루프

```python
from src.training import Trainer

class CustomTrainer(Trainer):
    def train_epoch(self):
        # 커스텀 학습 로직
        pass

    def validate(self):
        # 커스텀 검증 로직
        pass
```

---

## 학습 모니터링

### 실시간 로그

학습 중 실시간으로 출력:

```
Epoch 10/100 | Train RMSE: 0.0542 | Val RMSE: 0.0618 | LR: 0.001000 | Time: 23.45s
EarlyStopping counter: 3/10
Validation score improved by 0.001234
Saved best model (Val RMSE: 0.0618)
```

### 학습 히스토리

```python
history = trainer.train(epochs=100)

# 접근 가능한 데이터
train_losses = history['train_loss']
val_losses = history['val_loss']
learning_rates = history['learning_rate']
epoch_times = history['epoch_time']

# 시각화
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.legend()
plt.show()
```

---

## 문제 해결

### GPU 메모리 부족
```
RuntimeError: CUDA out of memory
```
**해결책:**
- Batch size 줄이기: `--batch-size 16`
- Sequence length 줄이기: config.yaml에서 수정
- 작은 모델 사용: `--model-type lstm` (대신 transformer)

### 학습이 느림
**해결책:**
- GPU 사용: `--gpu 0`
- Batch size 증가: `--batch-size 64`
- 작은 모델 사용

### Overfitting
**증상:** Train loss는 감소, Val loss는 증가

**해결책:**
- Early stopping patience 줄이기
- Dropout 증가: config.yaml에서 `dropout: 0.3`
- 더 많은 데이터 수집
- 정규화: `weight_decay` 증가

### Underfitting
**증상:** Train/Val loss 모두 높음

**해결책:**
- 더 큰 모델 사용: `--model-type transformer`
- 더 긴 학습: `--epochs 200`
- Learning rate 조정: `--lr 0.01`

---

## 다음 단계

학습이 완료되면:

1. **모델 평가**: 테스트 세트에서 상세 평가
2. **하이퍼파라미터 튜닝**: 최적 설정 찾기
3. **앙상블**: 여러 모델 결합
4. **배포**: API로 서빙

관련 가이드:
- 모델 선택: `MODEL_GUIDE.md`
- 전처리: `DATA_PREPROCESSING_GUIDE.md`
- 배포: `DEPLOYMENT_GUIDE.md`
