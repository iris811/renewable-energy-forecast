# 🚀 Google Colab 배포 및 실행 가이드

Google Colab에서 프로젝트를 실행하는 **3가지 방법**을 소개합니다.

---

## 방법 1: GitHub 사용 (가장 추천!) ⭐⭐⭐⭐⭐

가장 깔끔하고 전문적인 방법입니다.

### 📋 준비물
- GitHub 계정
- 프로젝트를 GitHub에 업로드

### 🔧 단계별 가이드

#### Step 1: GitHub에 프로젝트 업로드

```bash
# 로컬 컴퓨터에서 (Git Bash 또는 터미널)

# 1. GitHub에서 새 Repository 생성
# - https://github.com/new
# - Repository 이름: renewable-energy-forecast
# - Public 또는 Private 선택

# 2. 프로젝트 폴더로 이동
cd /path/to/renewable-energy-forecast

# 3. Git 초기화 (처음 한 번만)
git init
git add .
git commit -m "Initial commit: Renewable Energy Forecasting project"

# 4. GitHub에 연결 (본인의 username으로 변경!)
git remote add origin https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
git branch -M main
git push -u origin main
```

#### Step 2: Colab에서 노트북 열기

**옵션 A: 직접 링크로 열기**
```
https://colab.research.google.com/github/YOUR_USERNAME/renewable-energy-forecast/blob/main/notebooks/train_on_colab.ipynb
```

**옵션 B: Colab에서 GitHub 연동**
1. Google Colab 접속: https://colab.research.google.com
2. `파일` → `노트북 열기`
3. `GitHub` 탭 선택
4. Repository URL 입력: `YOUR_USERNAME/renewable-energy-forecast`
5. `train_on_colab.ipynb` 선택

#### Step 3: Colab에서 실행

노트북의 첫 번째 셀을 다음과 같이 수정:

```python
# GitHub에서 프로젝트 클론
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
%cd renewable-energy-forecast

# 나머지는 그대로 실행!
```

✅ **장점:**
- 코드 버전 관리 가능
- 언제든 최신 버전으로 업데이트
- 여러 사람과 협업 가능
- 포트폴리오로 활용

---

## 방법 2: Google Drive 사용 ⭐⭐⭐⭐

GitHub 없이 간편하게 사용할 수 있습니다.

### 📋 준비물
- Google Drive
- 프로젝트 zip 파일

### 🔧 단계별 가이드

#### Step 1: 프로젝트를 zip으로 압축

**Windows:**
```cmd
# 프로젝트 폴더를 마우스 우클릭
# "압축" → "renewable-energy-forecast.zip"
```

**Linux/Mac:**
```bash
cd /path/to
zip -r renewable-energy-forecast.zip renewable-energy-forecast/
```

#### Step 2: Google Drive에 업로드

1. Google Drive 접속: https://drive.google.com
2. 폴더 생성: `Colab Projects` (또는 원하는 이름)
3. `renewable-energy-forecast.zip` 업로드

#### Step 3: Colab 노트북 수정

`train_on_colab.ipynb`를 Google Drive에도 업로드하고, 다음과 같이 수정:

```python
# 첫 번째 셀에 추가

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 압축 해제
!unzip /content/drive/MyDrive/Colab\ Projects/renewable-energy-forecast.zip
%cd renewable-energy-forecast

# 나머지는 그대로!
```

✅ **장점:**
- GitHub 계정 불필요
- 간단하고 빠름
- Drive 용량만 있으면 OK

---

## 방법 3: 직접 업로드 (가장 간단, 소규모) ⭐⭐⭐

작은 프로젝트나 테스트용으로 적합합니다.

### 🔧 단계별 가이드

#### Step 1: 프로젝트 zip 압축
(위와 동일)

#### Step 2: Colab에서 직접 업로드

```python
# Colab 노트북 첫 번째 셀

# 파일 업로드
from google.colab import files
uploaded = files.upload()  # 여기서 zip 파일 선택

# 압축 해제
!unzip renewable-energy-forecast.zip
%cd renewable-energy-forecast

# 나머지 진행!
```

⚠️ **단점:**
- 세션마다 다시 업로드 필요
- 큰 파일은 업로드 시간 오래 걸림
- 12시간 후 세션 종료되면 다시 업로드

---

## 방법별 비교표

| 방법 | 난이도 | 재사용성 | 속도 | 추천 상황 |
|------|--------|---------|------|-----------|
| **GitHub** | 중 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 장기 프로젝트, 협업 |
| **Google Drive** | 하 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 개인 프로젝트 |
| **직접 업로드** | 하 | ⭐⭐ | ⭐⭐ | 1회성 테스트 |

---

## 📝 완전한 Colab 노트북 예제

어떤 방법을 선택하든, 다음 구조로 노트북을 만드세요:

```python
# ========================================
# 1. GPU 확인
# ========================================
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# 2. 프로젝트 가져오기 (3가지 중 1개 선택)
# ========================================

# 옵션 A: GitHub
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
%cd renewable-energy-forecast

# 옵션 B: Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# !unzip /content/drive/MyDrive/renewable-energy-forecast.zip
# %cd renewable-energy-forecast

# 옵션 C: 직접 업로드
# from google.colab import files
# uploaded = files.upload()
# !unzip renewable-energy-forecast.zip
# %cd renewable-energy-forecast

# ========================================
# 3. 패키지 설치
# ========================================
!pip install -q torch pandas numpy scikit-learn sqlalchemy pyyaml tqdm loguru

# ========================================
# 4. 데이터베이스 초기화
# ========================================
!python scripts/setup_database.py

# ========================================
# 5. 샘플 데이터 생성
# ========================================
!python scripts/generate_sample_data.py --days 365

# ========================================
# 6. 모델 학습
# ========================================
!python scripts/train_model.py --model-type lstm --epochs 100 --gpu 0

# ========================================
# 7. 모델 평가
# ========================================
!python scripts/evaluate_model.py \
    --model-path models/checkpoints/best_model.pth \
    --visualize

# ========================================
# 8. 결과 확인
# ========================================
!cat models/checkpoints/training_results.txt

# ========================================
# 9. 결과 다운로드
# ========================================
!zip -r training_results.zip models/ evaluation_results/ logs/

# 방법 1: 파일 직접 다운로드
from google.colab import files
files.download('training_results.zip')

# 방법 2: Google Drive에 저장 (선택사항)
# !cp training_results.zip /content/drive/MyDrive/
```

---

## 🎯 추천 워크플로우

### 처음 사용하는 경우:
1. **Google Drive 방법** 사용 (가장 쉬움)
2. zip 파일 업로드
3. 노트북 실행
4. 결과 다운로드

### 계속 사용할 경우:
1. **GitHub에 프로젝트 업로드** (한 번만)
2. Colab에서 GitHub 링크로 열기
3. 코드 수정하면 GitHub에 push
4. Colab에서 자동으로 최신 버전 사용

---

## 💡 팁과 트릭

### 1. Google Drive 자동 마운트
노트북 시작 부분에 추가:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 2. GitHub Personal Access Token (Private Repo)
```python
# Private repository인 경우
!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/renewable-energy-forecast.git
```

### 3. 중간 저장 자동화
```python
# 학습 중간에 Drive에 자동 저장
import shutil
shutil.copy('models/checkpoints/best_model.pth',
            '/content/drive/MyDrive/backups/best_model_backup.pth')
```

### 4. 세션 유지
```python
# 주기적으로 실행되는 더미 코드 (세션 유지용)
import time
for i in range(100):
    print(f"세션 유지 중... {i}")
    time.sleep(300)  # 5분마다
```

---

## 🔍 문제 해결

### "Module not found" 에러
```python
# 패키지 다시 설치
!pip install --upgrade torch pandas numpy scikit-learn
```

### "No such file or directory" 에러
```python
# 현재 디렉토리 확인
!pwd
!ls -la

# 올바른 디렉토리로 이동
%cd renewable-energy-forecast
```

### GitHub 클론 실패
```python
# 기존 폴더 삭제 후 다시 클론
!rm -rf renewable-energy-forecast
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
```

### Drive 마운트 실패
```python
# 강제 재마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## 📚 다음 단계

1. ✅ 방법 선택 (GitHub 추천)
2. ✅ 프로젝트 업로드
3. ✅ Colab에서 노트북 실행
4. ✅ 결과 다운로드
5. ✅ 로컬에서 API/Dashboard 실행

---

**Happy Colab Training! 🚀☁️**
