# 🚀 Google Colab 3분 시작 가이드

## 가장 쉬운 방법: Google Drive 사용 (추천!)

### Step 1: 프로젝트 압축 (로컬 컴퓨터)

**Windows:**
```
renewable-energy-forecast 폴더 우클릭 → 압축
```

**Mac/Linux:**
```bash
zip -r renewable-energy-forecast.zip renewable-energy-forecast/
```

### Step 2: Google Drive에 업로드

1. https://drive.google.com 접속
2. `renewable-energy-forecast.zip` 파일 업로드 (드래그 앤 드롭)

### Step 3: Colab 노트북 업로드

1. `notebooks/train_on_colab.ipynb` 파일도 Drive에 업로드

### Step 4: Colab에서 열기

1. Drive에서 `train_on_colab.ipynb` 우클릭
2. "연결 앱" → "Google Colaboratory"
3. GPU 활성화: 런타임 → 런타임 유형 변경 → GPU ✅

### Step 5: 실행

1. **방법 2: Google Drive** 섹션의 셀만 실행
2. Drive 경로 확인 (필요시 수정):
   ```python
   ZIP_PATH = '/content/drive/MyDrive/renewable-energy-forecast.zip'
   ```
3. 나머지 모든 셀을 순서대로 실행 (Shift + Enter)

### Step 6: 결과 다운로드

- 학습 완료 후 `training_results.zip` 다운로드
- 로컬 프로젝트에 압축 해제

---

## 더 나은 방법: GitHub 사용 (장기 프로젝트)

### Step 1: GitHub에 업로드 (한 번만)

```bash
# 로컬 컴퓨터에서
cd renewable-energy-forecast
git init
git add .
git commit -m "Initial commit"

# GitHub에서 새 Repository 생성 후
git remote add origin https://github.com/YOUR_USERNAME/renewable-energy-forecast.git
git push -u origin main
```

### Step 2: Colab에서 바로 열기

URL 접속:
```
https://colab.research.google.com/github/YOUR_USERNAME/renewable-energy-forecast/blob/main/notebooks/train_on_colab.ipynb
```

### Step 3: 실행

1. GPU 활성화
2. **방법 1: GitHub** 섹션 실행 (username 수정)
3. 나머지 셀 실행

---

## ⏱️ 예상 시간

- 설정: 3-5분
- LSTM 학습: 10-15분
- 평가 및 다운로드: 2-3분
- **총 20분 이내!**

---

## 💡 핵심 포인트

1. ✅ **GPU 활성화 필수!** (런타임 → GPU)
2. ✅ **방법 1개만 선택** (GitHub 또는 Drive 또는 업로드)
3. ✅ **순서대로 실행** (건너뛰지 말 것)
4. ✅ **결과 다운로드** (세션 종료 전에!)

---

## 🆘 문제 해결

### "GPU를 찾을 수 없습니다"
→ 런타임 → 런타임 유형 변경 → GPU 선택

### "파일을 찾을 수 없습니다"
→ Drive 경로 확인: `/content/drive/MyDrive/파일명.zip`

### "메모리 부족"
→ 배치 크기 줄이기: `--batch-size 16`

### "세션 타임아웃"
→ 중간 저장 자주 하기, Drive에 백업

---

## 📱 요약 체크리스트

- [ ] 프로젝트 zip으로 압축
- [ ] Google Drive에 업로드
- [ ] Colab에서 노트북 열기
- [ ] GPU 활성화
- [ ] 셀 순서대로 실행
- [ ] 결과 다운로드
- [ ] 로컬에서 압축 해제
- [ ] API/Dashboard 실행

---

**이제 시작하세요! 🚀**
