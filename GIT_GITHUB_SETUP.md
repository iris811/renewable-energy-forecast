# 🔧 Git & GitHub 설정 가이드

프로젝트를 GitHub에 올리고 Google Colab에서 사용하는 완벽한 가이드입니다.

---

## 📋 사전 준비

### 1. Git 설치 확인

```bash
git --version
```

**Git이 없다면:**
- Windows: https://git-scm.com/download/win
- Mac: `brew install git` 또는 Xcode 설치
- Linux: `sudo apt-get install git`

### 2. Git 설정 (처음 한 번만)

```bash
# 사용자 정보 설정
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 설정 확인
git config --list
```

### 3. GitHub 계정
- https://github.com 에서 계정 생성

---

## 🚀 Step-by-Step: 프로젝트를 GitHub에 올리기

### Step 1: 프로젝트 폴더로 이동

```bash
# Windows (Git Bash 또는 PowerShell)
cd C:\Users\RAN\Downloads\renewable-energy-forecast

# Mac/Linux
cd /path/to/renewable-energy-forecast
```

### Step 2: Git 저장소 초기화

```bash
# Git 초기화
git init

# 현재 상태 확인
git status
```

**출력 예시:**
```
Initialized empty Git repository in C:/Users/RAN/Downloads/renewable-energy-forecast/.git/
```

### Step 3: 파일 추가 및 커밋

```bash
# 모든 파일 스테이징 (추가)
git add .

# 커밋 (첫 번째 저장)
git commit -m "Initial commit: Renewable Energy Forecasting Project"
```

**출력 예시:**
```
[master (root-commit) a1b2c3d] Initial commit: Renewable Energy Forecasting Project
 XX files changed, XXXX insertions(+)
 create mode 100644 README.md
 create mode 100644 requirements.txt
 ...
```

### Step 4: GitHub에 새 Repository 생성

**웹 브라우저에서:**

1. https://github.com 로그인
2. 우측 상단 `+` → `New repository` 클릭
3. 설정:
   - **Repository name**: `renewable-energy-forecast`
   - **Description**: `Deep learning-based renewable energy forecasting system`
   - **Public** 또는 **Private** 선택
   - ⚠️ **중요**: `Add a README file`, `.gitignore`, `license` 모두 체크 **안 함** (이미 로컬에 있음)
4. `Create repository` 클릭

### Step 5: 로컬 저장소와 GitHub 연결

GitHub에서 생성 후 나오는 화면에서 코드를 복사하거나, 아래처럼 입력:

```bash
# GitHub 저장소와 연결 (본인의 username으로 변경!)
git remote add origin https://github.com/YOUR_USERNAME/renewable-energy-forecast.git

# 브랜치 이름을 main으로 변경 (GitHub 기본값)
git branch -M main

# GitHub에 업로드 (push)
git push -u origin main
```

**⚠️ 주의**: `YOUR_USERNAME`을 본인의 GitHub 계정명으로 바꾸세요!

**예시:**
```bash
git remote add origin https://github.com/johndoe/renewable-energy-forecast.git
```

### Step 6: 인증

**HTTPS 방식 (추천):**

Push 시 username과 password 요구:
- **Username**: GitHub 계정명
- **Password**: ⚠️ **Personal Access Token** 사용 (비밀번호 아님!)

**Personal Access Token 생성:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. `Generate new token` → `Generate new token (classic)`
3. Note: `Renewable Energy Forecast`
4. Expiration: `90 days` 또는 원하는 기간
5. Scopes: `repo` 전체 체크 ✅
6. `Generate token` 클릭
7. 생성된 토큰 복사 (⚠️ 다시 볼 수 없으니 저장!)
8. Git push 시 password에 토큰 입력

**또는 SSH 방식:**
```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "your.email@example.com"

# 공개 키 복사
cat ~/.ssh/id_ed25519.pub

# GitHub Settings → SSH and GPG keys → New SSH key에 추가

# 원격 저장소 URL 변경
git remote set-url origin git@github.com:YOUR_USERNAME/renewable-energy-forecast.git
```

### Step 7: 업로드 확인

```bash
# Push 성공 후
git remote -v
```

**출력:**
```
origin  https://github.com/YOUR_USERNAME/renewable-energy-forecast.git (fetch)
origin  https://github.com/YOUR_USERNAME/renewable-energy-forecast.git (push)
```

브라우저에서 `https://github.com/YOUR_USERNAME/renewable-energy-forecast` 접속하여 확인!

---

## ✅ Git 완료! 이제 Colab에서 사용하기

### 방법 1: Colab에서 직접 노트북 열기 (가장 쉬움!)

**URL로 바로 접속:**
```
https://colab.research.google.com/github/YOUR_USERNAME/renewable-energy-forecast/blob/main/notebooks/train_on_colab_UPDATED.ipynb
```

브라우저 북마크에 저장하면 한 번에 열림!

### 방법 2: Colab에서 GitHub 연동

1. https://colab.research.google.com 접속
2. `파일` → `노트북 열기`
3. `GitHub` 탭 선택
4. Repository URL 입력: `YOUR_USERNAME/renewable-energy-forecast`
5. `train_on_colab_UPDATED.ipynb` 선택

### 방법 3: 새 Colab 노트북에서 클론

```python
# 첫 번째 셀에서
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
%cd renewable-energy-forecast

# GPU 확인
import torch
print(f"GPU: {torch.cuda.is_available()}")
```

---

## 🔄 코드 수정 및 업데이트

### 로컬에서 코드 수정 후 GitHub에 반영

```bash
# 1. 변경사항 확인
git status

# 2. 변경된 파일 추가
git add .

# 또는 특정 파일만:
# git add src/models/lstm_model.py

# 3. 커밋 (변경 내용 저장)
git commit -m "Update: LSTM model architecture improved"

# 4. GitHub에 업로드
git push origin main
```

### Colab에서 최신 버전 가져오기

```python
# Colab 노트북에서
%cd renewable-energy-forecast
!git pull origin main

# 또는 처음부터 다시:
!rm -rf renewable-energy-forecast
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
%cd renewable-energy-forecast
```

---

## 📝 유용한 Git 명령어

### 상태 확인
```bash
git status              # 현재 상태
git log                 # 커밋 히스토리
git log --oneline      # 간단한 히스토리
git diff               # 변경 내용 확인
```

### 브랜치 관리
```bash
git branch             # 브랜치 목록
git branch feature-x   # 새 브랜치 생성
git checkout feature-x # 브랜치 전환
git merge feature-x    # 브랜치 병합
```

### 실수 복구
```bash
# 마지막 커밋 취소 (변경사항 유지)
git reset --soft HEAD~1

# 파일 스테이징 취소
git restore --staged filename

# 파일 변경사항 취소 (⚠️ 주의: 복구 불가)
git restore filename
```

---

## 🎯 권장 워크플로우

### 일상적인 작업

```bash
# 1. 작업 시작 전 최신 버전 받기
git pull origin main

# 2. 코드 작성/수정

# 3. 변경사항 확인
git status
git diff

# 4. 커밋
git add .
git commit -m "Descriptive commit message"

# 5. GitHub에 업로드
git push origin main
```

### Colab에서 학습

```bash
# 1. Colab에서 최신 코드 가져오기
!git clone https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
%cd renewable-energy-forecast

# 2. 학습 실행
!python scripts/train_model.py --model-type lstm --epochs 100

# 3. 결과 다운로드
!zip -r results.zip models/ evaluation_results/
from google.colab import files
files.download('results.zip')

# 4. 로컬에 결과 저장 및 커밋 (선택사항)
# 로컬 컴퓨터에서:
# git add models/
# git commit -m "Add trained LSTM model"
# git push origin main
```

---

## 🔐 .gitignore 확인

이미 `.gitignore` 파일이 생성되어 있지만, 확인해보세요:

```bash
cat .gitignore
```

**포함되어야 할 항목:**
- `data/` - 데이터 파일
- `models/checkpoints/` - 학습된 모델 (용량 큼)
- `*.pkl` - 스케일러 파일
- `logs/` - 로그 파일
- `configs/api_keys.yaml` - API 키 (보안!)
- `__pycache__/` - Python 캐시

**⚠️ 중요**:
- 학습된 모델은 용량이 커서 GitHub에 올리지 않는 것이 좋습니다
- GitHub 무료 계정은 파일당 100MB, 저장소당 1GB 제한
- 큰 파일은 Git LFS 사용하거나 Google Drive에 별도 저장

---

## 💡 팁과 트릭

### 1. Colab에서 Private Repository 사용

```python
# Personal Access Token 사용
TOKEN = "ghp_your_token_here"  # ⚠️ 절대 공개하지 말 것!
USERNAME = "your_username"
REPO = "renewable-energy-forecast"

!git clone https://{TOKEN}@github.com/{USERNAME}/{REPO}.git
```

### 2. 커밋 메시지 컨벤션

```bash
git commit -m "Add: 새 기능 추가"
git commit -m "Update: 기존 기능 개선"
git commit -m "Fix: 버그 수정"
git commit -m "Docs: 문서 업데이트"
git commit -m "Refactor: 코드 리팩토링"
```

### 3. README.md에 Badge 추가

GitHub 프로젝트 페이지를 멋지게!

```markdown
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/renewable-energy-forecast)
```

---

## 🆘 문제 해결

### "Permission denied" 에러
```bash
# HTTPS에서 SSH로 변경
git remote set-url origin git@github.com:YOUR_USERNAME/renewable-energy-forecast.git
```

### "Remote origin already exists"
```bash
# 기존 원격 저장소 제거 후 재추가
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
```

### 큰 파일 업로드 에러
```bash
# .gitignore에 추가
echo "models/checkpoints/*.pth" >> .gitignore
git rm --cached models/checkpoints/*.pth
git commit -m "Remove large model files"
```

### 커밋 메시지 수정
```bash
# 마지막 커밋 메시지 수정
git commit --amend -m "New commit message"

# ⚠️ 이미 push한 경우 (신중하게!)
# git push --force origin main
```

---

## ✅ 완료 체크리스트

- [ ] Git 설치 및 설정
- [ ] 프로젝트 폴더에서 `git init`
- [ ] 파일 추가 및 커밋 (`git add .`, `git commit`)
- [ ] GitHub에서 Repository 생성
- [ ] 로컬과 GitHub 연결 (`git remote add origin`)
- [ ] GitHub에 업로드 (`git push`)
- [ ] Colab에서 테스트 (`git clone`)
- [ ] 북마크 저장 (Colab 노트북 URL)

---

## 🎉 축하합니다!

이제 프로젝트가 GitHub에 있습니다!

**다음 단계:**
1. ✅ Colab에서 노트북 열기
2. ✅ GPU 활성화
3. ✅ 학습 실행
4. ✅ 결과 다운로드

**URL:**
```
https://colab.research.google.com/github/YOUR_USERNAME/renewable-energy-forecast/blob/main/notebooks/train_on_colab_UPDATED.ipynb
```

---

**Happy Coding! 🚀💻**
