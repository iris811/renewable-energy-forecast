# 데이터 전처리 가이드

재생에너지 발전량 예측을 위한 데이터 전처리 파이프라인 사용법

## 📋 목차

1. [개요](#개요)
2. [전처리 파이프라인 구조](#전처리-파이프라인-구조)
3. [사용 방법](#사용-방법)
4. [모듈 상세 설명](#모듈-상세-설명)

---

## 개요

데이터 전처리는 원시 데이터를 AI 모델이 학습할 수 있는 형태로 변환하는 과정입니다. 이 시스템은 다음 단계를 자동화합니다:

1. **데이터 로딩**: 데이터베이스에서 기상 및 발전량 데이터 로드
2. **데이터 정제**: 결측치 처리, 이상치 제거, 데이터 검증
3. **특성 엔지니어링**: 시간, 태양 위치, 상호작용 특성 생성
4. **스케일링**: 데이터 정규화
5. **시퀀스 생성**: LSTM/Transformer를 위한 시계열 윈도우 생성
6. **데이터 분할**: Train/Validation/Test 세트 분할

---

## 전처리 파이프라인 구조

```
src/preprocessing/
├── data_loader.py          # 데이터베이스에서 데이터 로드
├── data_cleaner.py         # 결측치, 이상치 처리
├── feature_engineering.py  # 특성 생성
├── scaler.py              # 데이터 스케일링
├── sequence_generator.py   # 시계열 시퀀스 생성
└── data_pipeline.py       # 통합 파이프라인
```

---

## 사용 방법

### 방법 1: 스크립트 사용 (권장)

```bash
# 기본 사용
python scripts/preprocess_data.py

# 옵션 지정
python scripts/preprocess_data.py \
  --location "Seoul Solar Farm" \
  --energy-type solar \
  --latitude 37.5665 \
  --longitude 126.9780 \
  --sequence-length 168 \
  --prediction-horizon 24 \
  --batch-size 32 \
  --save-processed
```

**옵션 설명:**
- `--location`: 위치 이름
- `--energy-type`: 에너지 타입 (solar/wind)
- `--latitude`: 위도
- `--longitude`: 경도
- `--sequence-length`: 입력 시퀀스 길이 (시간 단위)
- `--prediction-horizon`: 예측 기간 (시간 단위)
- `--batch-size`: 배치 크기
- `--save-processed`: 전처리된 데이터를 CSV로 저장

### 방법 2: Python 코드에서 직접 사용

```python
from src.preprocessing import DataPipeline

# 파이프라인 생성
pipeline = DataPipeline(
    location_name='Seoul Solar Farm',
    energy_type='solar',
    latitude=37.5665,
    longitude=126.9780,
    sequence_length=168,  # 1주일
    prediction_horizon=24  # 24시간
)

# 전체 파이프라인 실행
result = pipeline.run_pipeline(batch_size=32)

# 결과 사용
train_loader = result['train_loader']
val_loader = result['val_loader']
test_loader = result['test_loader']
n_features = result['n_features']
```

---

## 모듈 상세 설명

### 1. DataLoader (데이터 로딩)

데이터베이스에서 기상 및 발전량 데이터를 로드합니다.

```python
from src.preprocessing import DataLoader

loader = DataLoader()

# 기상 데이터 로드
weather_df = loader.load_weather_data(
    location_name='Seoul Solar Farm',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# 발전량 데이터 로드
power_df = loader.load_power_generation(
    location_name='Seoul Solar Farm',
    energy_type='solar'
)

# 기상 + 발전량 병합
merged_df = loader.merge_weather_and_power(
    location_name='Seoul Solar Farm',
    energy_type='solar'
)
```

### 2. DataCleaner (데이터 정제)

결측치, 이상치, 데이터 품질 문제를 처리합니다.

```python
from src.preprocessing import DataCleaner

cleaner = DataCleaner()

# 결측치 처리
df = cleaner.handle_missing_values(df)

# 이상치 제거 (IQR 방법)
df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)

# 값 범위 검증
df = cleaner.validate_ranges(df)

# 시계열 리샘플링
df = cleaner.resample_timeseries(df, freq='1H')

# 전체 정제 파이프라인
df = cleaner.clean(
    df,
    fill_missing=True,
    remove_outliers=True,
    validate_ranges=True,
    resample='1H',
    remove_duplicates=True
)

# 데이터 품질 리포트
report = cleaner.get_data_quality_report(df)
```

### 3. FeatureEngineer (특성 엔지니어링)

재생에너지 예측에 필요한 특성을 생성합니다.

```python
from src.preprocessing import FeatureEngineer

engineer = FeatureEngineer()

# 시간 특성 (시간, 일, 월, 계절, 주말 등)
df = engineer.add_time_features(df)

# 태양 위치 (고도각, 방위각)
df = engineer.add_solar_position(df, latitude=37.5665, longitude=126.9780)

# Lag 특성 (과거 값)
df = engineer.add_lag_features(
    df,
    columns=['temperature', 'wind_speed', 'solar_irradiance'],
    lags=[1, 2, 3, 24]
)

# Rolling window 특성 (이동 평균, 표준편차 등)
df = engineer.add_rolling_features(
    df,
    columns=['temperature', 'wind_speed'],
    windows=[3, 6, 12, 24],
    functions=['mean', 'std']
)

# 상호작용 특성 (태양 효율, 풍력 포텐셜 등)
df = engineer.add_interaction_features(df)

# 모든 특성 한번에 생성
df = engineer.create_all_features(
    df,
    latitude=37.5665,
    longitude=126.9780,
    include_lags=True,
    include_rolling=True
)
```

**생성되는 주요 특성:**
- **시간 특성**: hour, day, month, day_of_week, season, is_weekend
- **순환 특성**: hour_sin/cos, day_sin/cos, month_sin/cos
- **태양 특성**: solar_elevation, solar_azimuth, is_daytime
- **상호작용 특성**:
  - `solar_efficiency`: 구름량을 고려한 태양 효율
  - `wind_power_potential`: 풍속의 3승 (풍력 에너지)
  - `solar_temp_efficiency`: 온도를 고려한 태양광 패널 효율
  - `heat_index`, `wind_chill`

### 4. DataScaler (스케일링)

특성과 타겟을 정규화합니다.

```python
from src.preprocessing import DataScaler, TargetScaler

# 특성 스케일링
feature_scaler = DataScaler(method='standard')  # 'standard', 'minmax', 'robust'
df_scaled = feature_scaler.fit_transform(df, columns=['temperature', 'humidity'])

# 새 데이터에 적용
df_new_scaled = feature_scaler.transform(df_new)

# 원래 스케일로 역변환
df_original = feature_scaler.inverse_transform(df_scaled)

# 스케일러 저장/로드
feature_scaler.save('models/scalers/feature_scaler.pkl')
feature_scaler.load('models/scalers/feature_scaler.pkl')

# 타겟 스케일링
target_scaler = TargetScaler(method='standard')
y_scaled = target_scaler.fit_transform(y)
y_original = target_scaler.inverse_transform(y_scaled)
```

### 5. SequenceGenerator (시퀀스 생성)

LSTM/Transformer 모델을 위한 시계열 윈도우를 생성합니다.

```python
from src.preprocessing import SequenceGenerator, create_dataloaders

# 시퀀스 생성기 초기화
generator = SequenceGenerator(
    sequence_length=168,  # 1주일의 과거 데이터 사용
    prediction_horizon=24,  # 24시간 예측
    stride=1  # 슬라이딩 윈도우 스텝
)

# 시퀀스 생성
X_seq, y_seq = generator.create_sequences(X, y)
# X_seq shape: (n_sequences, 168, n_features)
# y_seq shape: (n_sequences, 24)

# Train/Val/Test 분할
train_data, val_data, test_data = generator.split_train_val_test(
    X_seq, y_seq,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# PyTorch DataLoader 생성
train_loader, val_loader, test_loader = create_dataloaders(
    train_data, val_data, test_data,
    batch_size=32
)
```

### 6. DataPipeline (통합 파이프라인)

모든 전처리 단계를 하나로 통합합니다.

```python
from src.preprocessing import DataPipeline

# 파이프라인 생성
pipeline = DataPipeline(
    location_name='Seoul Solar Farm',
    energy_type='solar',
    latitude=37.5665,
    longitude=126.9780,
    sequence_length=168,
    prediction_horizon=24
)

# 전체 파이프라인 실행
result = pipeline.run_pipeline(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    batch_size=32,
    save_scalers=True
)

# 결과 활용
train_loader = result['train_loader']
val_loader = result['val_loader']
test_loader = result['test_loader']
n_features = result['n_features']
feature_scaler = result['feature_scaler']
target_scaler = result['target_scaler']

# 데이터 요약
summary = pipeline.get_data_summary()
print(summary)
```

---

## 데이터 흐름

```
원시 데이터 (데이터베이스)
    ↓
[DataLoader] 데이터 로드 및 병합
    ↓
[DataCleaner] 결측치, 이상치 처리
    ↓
[FeatureEngineer] 특성 생성 (시간, 태양, 상호작용)
    ↓
[DataScaler] 정규화
    ↓
[SequenceGenerator] 시계열 윈도우 생성
    ↓
[DataLoader] PyTorch DataLoader
    ↓
모델 학습
```

---

## 예제: 완전한 전처리 워크플로우

```python
from datetime import datetime
from src.preprocessing import DataPipeline

# 1. 파이프라인 초기화
pipeline = DataPipeline(
    location_name='Seoul Solar Farm',
    energy_type='solar',
    latitude=37.5665,
    longitude=126.9780,
    sequence_length=168,  # 7일 입력
    prediction_horizon=24   # 24시간 예측
)

# 2. 데이터 로드
df = pipeline.load_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# 3. 전처리
df = pipeline.preprocess(
    df,
    resample_freq='1H',
    include_lags=True,
    include_rolling=True
)

# 4. 특성 준비
feature_columns, target_column = pipeline.prepare_features(df)

# 5. 스케일링
df_scaled = pipeline.scale_data(df, feature_columns, target_column, fit=True)

# 6. 시퀀스 생성 및 분할
sequences = pipeline.create_sequences(
    df_scaled,
    feature_columns,
    target_column,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 7. 모델 학습 시작!
from torch.utils.data import DataLoader
train_loader, val_loader, test_loader = create_dataloaders(
    sequences['train'],
    sequences['val'],
    sequences['test'],
    batch_size=32
)
```

---

## 출력 데이터 형태

### 시퀀스 데이터

**입력 (X):**
- Shape: `(batch_size, sequence_length, n_features)`
- 예: `(32, 168, 50)` - 32개 샘플, 168시간 과거 데이터, 50개 특성

**타겟 (y):**
- Single-step: `(batch_size,)` - 1시간 예측
- Multi-step: `(batch_size, prediction_horizon)` - 24시간 예측

### 데이터 분할

기본 비율:
- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

시간 순서 유지 (no shuffle) - 미래 데이터 누출 방지

---

## 저장되는 파일

```
models/scalers/
├── Seoul_Solar_Farm_solar_feature_scaler.pkl
└── Seoul_Solar_Farm_solar_target_scaler.pkl

data/processed/
└── Seoul_Solar_Farm_solar_processed.csv
```

---

## 다음 단계

전처리가 완료되면:

1. **모델 학습**: `src/models/`, `src/training/` 모듈 사용
2. **예측 실행**: `src/inference/` 모듈 사용
3. **평가**: 예측 정확도 측정 (RMSE, MAE, MAPE)

---

## 문제 해결

### 데이터가 없음
```
Error: No data loaded
```
→ 먼저 `python scripts/collect_data.py`로 데이터 수집

### 메모리 부족
→ `batch_size` 줄이기, `sequence_length` 줄이기

### 특성이 너무 많음
→ `include_lags=False`, `include_rolling=False`로 특성 수 줄이기

---

## 참고

- 시계열 데이터는 shuffle하지 않음 (데이터 누출 방지)
- 스케일러는 Train set에만 fit하고 Val/Test에는 transform만 적용
- NaN 값은 자동으로 처리되지만, 너무 많으면 데이터 품질 확인 필요
