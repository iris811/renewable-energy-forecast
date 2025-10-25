# ☁️ 클라우드에서 학습하기

GPU가 없어도 걱정 마세요! 무료 클라우드 서비스를 이용해서 학습할 수 있습니다.

## 🎯 추천 옵션

### 1. Google Colab (가장 추천!) ⭐

**장점:**
- ✅ **완전 무료** GPU 제공 (Tesla T4)
- ✅ 설치 필요 없음
- ✅ 바로 사용 가능
- ✅ Jupyter Notebook 환경

**제한:**
- ⏱️ 12시간 연속 실행 제한
- 💾 일정 시간 후 데이터 삭제
- 🔄 세션 타임아웃 주의

**사용 방법:**

1. **Google Colab 접속**
   - https://colab.research.google.com

2. **노트북 업로드**
   - `notebooks/train_on_colab.ipynb` 파일 업로드
   - 또는 GitHub에서 직접 열기

3. **GPU 활성화**
   - 메뉴: `런타임` → `런타임 유형 변경` → `하드웨어 가속기: GPU` 선택

4. **노트북 실행**
   - 셀을 순서대로 실행 (Shift + Enter)
   - 모든 과정이 자동으로 진행됩니다

5. **결과 다운로드**
   - 학습 완료 후 `training_results.zip` 다운로드
   - 로컬 컴퓨터에 압축 해제

**예상 학습 시간:**
- LSTM: 10-15분
- LSTM + Attention: 15-20분
- Transformer: 25-35분

---

### 2. Kaggle Notebooks

**장점:**
- ✅ 무료 GPU (Tesla P100)
- ✅ **주당 30시간** GPU 사용 가능
- ✅ 데이터셋 공유 가능

**제한:**
- ⏱️ 12시간 세션 제한
- 🌐 인터넷 접근 제한 (데이터셋은 업로드 필요)

**사용 방법:**

1. **Kaggle 계정 생성**
   - https://www.kaggle.com

2. **New Notebook 생성**
   - Settings → Accelerator → GPU 선택

3. **코드 업로드**
   - 프로젝트를 zip으로 압축
   - Kaggle에 데이터셋으로 업로드
   - 노트북에서 압축 해제

4. **학습 실행**
   ```python
   !unzip ../input/your-dataset/renewable-energy-forecast.zip
   %cd renewable-energy-forecast
   !pip install -r requirements.txt
   !python scripts/train_model.py --model-type lstm --epochs 100
   ```

---

### 3. AWS SageMaker (유료)

**장점:**
- ✅ 강력한 GPU 선택 가능
- ✅ 안정적인 학습 환경
- ✅ 확장성 좋음

**비용:**
- 💰 시간당 요금 발생
- 💰 ml.p3.2xlarge (V100 GPU): ~$3/시간

**사용 방법:**
- AWS 계정 필요
- SageMaker Studio 또는 Notebook Instance 생성
- Jupyter 환경에서 학습

---

### 4. Paperspace Gradient

**장점:**
- ✅ 무료 티어 제공 (제한적)
- ✅ 사용하기 쉬운 인터페이스

**제한:**
- ⏱️ 월 6시간 무료 GPU

**사용 방법:**
- https://www.paperspace.com/gradient
- Free GPU 노트북 생성
- GitHub에서 프로젝트 클론

---

## 🔧 로컬 컴퓨터에서 CPU로 학습하기

GPU가 없어도 CPU로 학습할 수 있습니다!

**장점:**
- ✅ 시간 제한 없음
- ✅ 추가 비용 없음

**단점:**
- ⏱️ 매우 느림 (GPU 대비 5-10배)

**최적화 팁:**

1. **작은 모델 사용**
   ```bash
   python scripts/train_model.py \
       --model-type lstm \
       --epochs 50 \
       --batch-size 16 \
       --gpu -1
   ```

2. **데이터 줄이기**
   ```bash
   # 6개월치만 생성
   python scripts/generate_sample_data.py --days 180
   ```

3. **에폭 수 줄이기**
   ```bash
   python scripts/train_model.py --epochs 30 --gpu -1
   ```

---

## 📊 학습 환경 비교

| 환경 | GPU | 무료 | 시간 제한 | 추천도 |
|------|-----|------|-----------|--------|
| **Google Colab** | Tesla T4 | ✅ | 12시간 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | Tesla P100 | ✅ | 12시간 (주 30시간) | ⭐⭐⭐⭐ |
| **Paperspace** | M4000 | ✅ (제한) | 월 6시간 | ⭐⭐⭐ |
| **로컬 CPU** | 없음 | ✅ | 무제한 | ⭐⭐ |
| **AWS SageMaker** | V100 등 | ❌ | 무제한 | ⭐⭐⭐⭐ |

---

## 🎓 추천 워크플로우

### 초보자용 (Google Colab)

1. `notebooks/train_on_colab.ipynb` 업로드
2. GPU 활성화
3. 모든 셀 실행
4. 결과 다운로드
5. 로컬에서 API/Dashboard 실행

### 고급 사용자용 (하이브리드)

1. **Colab에서 학습**
   - 무료 GPU로 빠른 학습
   - 결과 다운로드

2. **로컬에서 운영**
   - API 서버 실행
   - Dashboard 실행
   - 추론은 CPU로도 충분히 빠름

3. **필요시 재학습**
   - 새로운 데이터로 Colab에서 재학습
   - 모델만 업데이트

---

## 💡 팁과 주의사항

### Google Colab 사용 팁

1. **주기적으로 저장**
   - 2-3시간마다 중간 결과 다운로드
   - Early stopping이 작동하면 바로 다운로드

2. **세션 유지**
   - 브라우저 탭 닫지 않기
   - 코드 실행 중에는 연결 유지

3. **GPU 할당량 관리**
   - 하루에 너무 많이 사용하면 제한될 수 있음
   - 효율적으로 사용하기

### 결과 다운로드 체크리스트

```bash
# 다운로드해야 할 파일들
models/checkpoints/best_model.pth        # 최고 성능 모델
models/checkpoints/training_results.txt  # 학습 결과
models/scalers/*.pkl                     # 스케일러
evaluation_results/                      # 평가 결과
logs/app.log                            # 로그
```

### 로컬에 적용하기

```bash
# 1. 다운로드한 zip 압축 해제
unzip training_results.zip

# 2. 파일 복사
cp -r models/ /path/to/your/project/
cp -r evaluation_results/ /path/to/your/project/

# 3. API/Dashboard 실행
python -m uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

---

## 🆘 문제 해결

### "GPU 할당량 초과" 에러
- 다음날까지 기다리기
- Kaggle이나 다른 플랫폼 사용

### "세션 타임아웃"
- 중간 저장 자주 하기
- 학습 중에는 브라우저 활성화 유지

### "메모리 부족" 에러
- 배치 크기 줄이기: `--batch-size 16`
- 작은 모델 사용: `--model-type lstm`

### "다운로드 실패"
- Google Drive에 먼저 저장
- 작은 파일로 나눠서 다운로드

---

## 📚 추가 자료

- [Google Colab 공식 문서](https://colab.research.google.com/notebooks/intro.ipynb)
- [Kaggle GPU 사용 가이드](https://www.kaggle.com/docs/notebooks)
- [PyTorch GPU 최적화](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Happy Cloud Training! ☁️⚡**
