# 데이터 수집 가이드

재생에너지 발전량 예측 시스템의 데이터 수집 모듈 사용법

## 📋 목차

1. [개요](#개요)
2. [초기 설정](#초기-설정)
3. [데이터 수집 방법](#데이터-수집-방법)
4. [데이터 소스](#데이터-소스)

---

## 개요

이 시스템은 두 가지 주요 데이터 소스에서 재생에너지 예측에 필요한 데이터를 수집합니다:

- **OpenWeatherMap API**: 실시간 기상 데이터 및 5일 예보
- **NASA POWER API**: 태양 복사량, 기상 데이터 (무료, API 키 불필요)

## 초기 설정

### 1. API 키 설정

```bash
# API 키 설정 파일 복사
copy configs\api_keys.yaml.example configs\api_keys.yaml
```

`configs/api_keys.yaml` 파일을 열고 OpenWeatherMap API 키를 입력하세요:

```yaml
openweathermap:
  api_key: "여기에_실제_API_키를_입력하세요"
```

**OpenWeatherMap API 키 받기:**
1. https://openweathermap.org/api 방문
2. 무료 계정 생성
3. API 키 발급 (Free tier: 60 calls/minute, 1,000,000 calls/month)

### 2. 수집 위치 설정

`configs/config.yaml` 파일에서 데이터를 수집할 위치를 설정:

```yaml
data:
  collection:
    locations:
      - name: "Seoul Solar Farm"
        latitude: 37.5665
        longitude: 126.9780
        type: "solar"

      - name: "Gangwon Wind Farm"
        latitude: 37.8228
        longitude: 128.5569
        type: "wind"
```

### 3. 데이터베이스 초기화

```bash
python scripts/setup_database.py
```

이 명령은 다음을 수행합니다:
- SQLite 데이터베이스 생성 (`data/renewable_energy.db`)
- 필요한 테이블 생성 (weather_data, power_generation, predictions)

---

## 데이터 수집 방법

### 방법 1: 수동 데이터 수집

한 번만 데이터를 수집하고 싶을 때:

```bash
# 모든 소스에서 데이터 수집
python scripts/collect_data.py --source all

# OpenWeatherMap에서만 수집
python scripts/collect_data.py --source weather

# NASA POWER에서만 수집 (최근 30일)
python scripts/collect_data.py --source nasa --days-back 30
```

### 방법 2: 자동 스케줄러

주기적으로 자동으로 데이터를 수집:

```bash
python scripts/run_scheduler.py
```

스케줄러는 `config.yaml`의 설정에 따라 주기적으로 데이터를 수집합니다:

```yaml
data:
  collection:
    sources:
      - name: "weather_api"
        enabled: true
        update_interval: 3600  # 1시간마다

      - name: "nasa_power"
        enabled: true
        update_interval: 86400  # 1일마다
```

**스케줄러 기능:**
- 시작 시 즉시 초기 데이터 수집
- 설정된 간격으로 자동 수집
- 에러 발생 시 로그 기록 및 계속 실행
- Ctrl+C로 안전하게 종료

---

## 데이터 소스

### 1. OpenWeatherMap API

**수집 데이터:**
- 현재 기상 데이터
- 5일 예보 (3시간 간격)
- 기온, 습도, 기압, 풍속, 풍향, 구름량, 강수량

**수집 파일:** `src/data_collection/weather_collector.py`

**주요 기능:**
```python
from src.data_collection.weather_collector import WeatherCollector

collector = WeatherCollector()

# 현재 날씨 수집
weather = collector.get_current_weather(37.5665, 126.9780, "Seoul")

# 예보 수집
forecast = collector.get_forecast(37.5665, 126.9780, "Seoul")

# 모든 설정된 위치에서 수집
collector.collect_all_locations()
```

### 2. NASA POWER API

**수집 데이터:**
- 전역 수평 일사량 (GHI)
- 직달 일사량 (DNI)
- 산란 일사량 (DHI)
- 기온, 풍속, 습도, 강수량
- 구름량, 기압

**수집 파일:** `src/data_collection/nasa_power_collector.py`

**특징:**
- API 키 불필요 (무료)
- 일별 데이터
- 1981년부터 현재까지 역사적 데이터 제공
- 태양광 발전 예측에 최적화된 데이터

**주요 기능:**
```python
from src.data_collection.nasa_power_collector import NASAPowerCollector

collector = NASAPowerCollector()

# 최근 30일 데이터 수집
collector.collect_and_save(37.5665, 126.9780, "Seoul", days_back=30)

# 모든 설정된 위치에서 수집
collector.collect_all_locations(days_back=30)
```

---

## 데이터베이스 구조

### weather_data 테이블

수집된 모든 기상 데이터를 저장:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | Integer | 기본 키 |
| timestamp | DateTime | 데이터 시간 |
| location_name | String | 위치 이름 |
| latitude | Float | 위도 |
| longitude | Float | 경도 |
| temperature | Float | 기온 (°C) |
| humidity | Float | 습도 (%) |
| pressure | Float | 기압 (hPa) |
| wind_speed | Float | 풍속 (m/s) |
| wind_direction | Float | 풍향 (°) |
| cloud_cover | Float | 구름량 (%) |
| precipitation | Float | 강수량 (mm) |
| solar_irradiance | Float | 태양 복사량 (W/m²) |
| ghi | Float | 전역 수평 일사량 |
| dni | Float | 직달 일사량 |
| dhi | Float | 산란 일사량 |
| source | String | 데이터 소스 |
| created_at | DateTime | 생성 시간 |

---

## 로그 확인

모든 데이터 수집 활동은 로그에 기록됩니다:

```bash
# 로그 파일 위치
logs/app.log
```

로그 레벨은 `configs/config.yaml`에서 설정:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file:
    enabled: true
    path: "./logs/app.log"
```

---

## 문제 해결

### API 키 오류
```
Error: OpenWeatherMap API key not found!
```
→ `configs/api_keys.yaml` 파일에 API 키가 설정되었는지 확인

### 데이터베이스 오류
```
Error: no such table: weather_data
```
→ `python scripts/setup_database.py` 실행

### 네트워크 오류
```
Error fetching current weather: Connection timeout
```
→ 인터넷 연결 확인, 방화벽 설정 확인

---

## 다음 단계

데이터 수집이 완료되면:

1. **데이터 전처리**: `src/preprocessing/` 모듈 구현
2. **AI 모델 학습**: `src/models/`, `src/training/` 모듈 구현
3. **예측 실행**: `src/inference/` 모듈 구현
4. **API 서버**: `api/` 모듈로 REST API 제공
5. **대시보드**: `dashboard/` 모듈로 시각화

---

## 참고 자료

- [OpenWeatherMap API 문서](https://openweathermap.org/api)
- [NASA POWER API 문서](https://power.larc.nasa.gov/docs/services/api/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [APScheduler 문서](https://apscheduler.readthedocs.io/)
