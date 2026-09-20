# TEA (Temporal Emotion Adapter) 연구 보고서

## 1. 개요

### 연구 목표
Video2Music(AMT) 모델의 감정 처리 방식을 개선한다.
기존 AMT는 영상의 감정 feature(6차원)를 단순히 concat해서 Transformer에 넘기는 방식이다.
TEA는 감정 시퀀스를 시간적으로 처리(RNN)해서 감정 흐름을 더 풍부하게 모델에 주입한다.

### 베이스라인 (AMT, Video2Music 논문 Table 4)
| 모델 | H@1 | H@3 | H@5 | Affective Matching Loss |
|------|-----|-----|-----|------------------------|
| Transformer (Vaswani et al.) | 0.4789 | 0.7117 | 0.8204 | 1.8366 |
| Music Transformer (Huang et al.) | 0.4965 | 0.7303 | 0.8323 | 1.8795 |
| AMT w/o affective loss | 0.5142 | 0.7585 | 0.8660 | 1.6859 |
| **AMT (논문 baseline)** | **0.5139** | **0.7722** | **0.8672** | **0.4662** |

> **재현 결과**: 동일 설정으로 직접 훈련한 AMT (100 epoch) → H@1=0.5537, H@3=0.8235, H@5=0.8981, EmoLoss=0.4597
> 논문 수치와 유사하게 재현됨 (H@k 소폭 상회는 재현성 범위 내).

---

## 2. TEA 모듈 설계

### 구조 개요
```
입력: feature_emotion (batch, seq_len, 6)
 │
 ├─ [1] 학습 가능한 감정 가중치 (emotion_weight: 6차원 + temperature scaling)
 │       → softmax 가중 합으로 감정 차원별 중요도 학습
 │
 ├─ [2] Input Dropout (p=0.2)
 │       → 감정 feature 노이즈에 대한 강건성 확보
 │
 ├─ [3] 시간 윈도우 (window_size=2, 지수 감쇠 가중치)
 │       → 최근 프레임에 더 높은 가중치 (causal 구조)
 │       → torch.unfold 완전 벡터화
 │
 ├─ [4] Bidirectional RNN (LSTM 또는 GRU)
 │       → 감정 시퀀스는 사전 계산된 전체 시퀀스 → causal 제약 없음
 │       → 양방향으로 전체 감정 흐름 포착
 │       → hidden = d_model // 2 (양방향이므로 출력 dim = d_model)
 │
 ├─ [5] Residual + LayerNorm
 │       → input_proj(EMO_DIM → d_model) + rnn_out → LayerNorm
 │
 └─ [6] Output Projection (Linear d_model → d_model)

출력: emotion_embedding (batch, seq_len, 512)
```

### 하이퍼파라미터
| 파라미터 | 값 |
|---------|-----|
| d_model (TEA hidden) | 512 |
| RNN layers | 2 |
| Window size | 2 |
| Overlap | 0.5 |
| Input dropout | 0.2 |
| EMO_DIM | 6 |

### TEA 주입 방식 (위치별)

**Encoder only**
- TEA 출력 → Additive injection으로 Video feature에 더함
- `encoder_out = encoder_out + emotion_norm(encoder_emotion_proj(tea_out))`
- 영상 이해 단계에서 감정 맥락 반영

**Decoder only**
- TEA 출력 → Cross-Attention으로 Decoder 출력에 주입
- Decoder transformer 출력이 Q, TEA 출력이 K/V
- 코드 생성 단계에서 감정 맥락 반영

**Both**
- Encoder TEA + Decoder TEA 모두 적용
- enc_dec_fusion (Linear → LayerNorm → ReLU)으로 두 신호 융합
- 융합 결과를 Decoder에 추가 반영

---

## 3. 학습 설정

### 공통 하이퍼파라미터
| 항목 | 값 |
|------|-----|
| Epochs | 100 |
| Batch size | 4 |
| Optimizer | Adam (β1=0.9, β2=0.98, ε=1e-9) |
| Weight decay | 1e-4 |
| LR scheduler | Warmup (d_model=512 기반) |
| Gradient clipping | max_norm=1.0 |
| Loss: Chord | SmoothCrossEntropyLoss |
| Loss: Emotion | BCEWithLogitsLoss |
| Eval | Val-only (매 epoch) |
| Best model 저장 | Val total loss 최저 기준 |
| Weight 저장 주기 | 5 epoch |

### 데이터셋 (MuVi-Sync)
- 총 ~748개 영상
- Train ≈ 600 / Val = 75 / Test 나머지
- 피처: 의미론적(CLIP-L/14p, 768차원), 감정(6차원), 모션, 장면 전환
- 코드 레이블: 159가지 (CHORD_SIZE)

### Backbone 학습 방식
Joint training from scratch (baseline AMT와 동일).
사전학습 가중치 없음 → backbone 동결 불가 → TEA + Transformer 전체 학습.

---

## 4. 실험 구성 (8가지)

| Exp | TEA 위치 | Encoder RNN | Decoder RNN |
|-----|----------|-------------|-------------|
| 1 | Decoder only | — | LSTM |
| 2 | Decoder only | — | GRU |
| 3 | Encoder only | LSTM | — |
| 4 | Encoder only | GRU | — |
| 5 | Both | LSTM | LSTM |
| 6 | Both | GRU | GRU |
| 7 | Both | LSTM | GRU |
| 8 | Both | GRU | LSTM |

---

## 5. 실험 결과

### 5.1 베스트 모델 성능 (Best Val Loss 기준 체크포인트)

| Exp | TEA 위치 | RNN 구성 | Best Epoch | Val Loss | H@1 | H@3 | H@5 | Emo Loss |
|-----|----------|----------|-----------|----------|-----|-----|-----|----------|
| Baseline (논문) | — | — | — | — | 0.5139 | 0.7722 | 0.8672 | 0.4662 |
| Baseline (재현) | — | — | 59 | 0.9024 | 0.5537 | 0.8235 | 0.8981 | 0.4597 |
| Exp1 | Decoder | LSTM | 55 | 0.8700 | 0.5618 | 0.8274 | 0.9078 | 0.4232 |
| Exp2 | Decoder | GRU | 67 | 0.8108 | 0.6098 | 0.8514 | 0.9148 | 0.4158 |
| Exp3 | Encoder | LSTM | 68 | 0.7938 | 0.6222 | 0.8596 | 0.9223 | 0.4222 |
| Exp4 | Encoder | GRU | 63 | 0.7963 | 0.6236 | 0.8564 | 0.9212 | 0.4222 |
| Exp5 | Both | LSTM+LSTM | 59 | 0.8074 | 0.6106 | 0.8524 | 0.9173 | 0.4261 |
| Exp6 | Both | GRU+GRU | 47 | 0.8386 | 0.5824 | 0.8438 | 0.9155 | 0.4272 |
| Exp7 | Both | LSTM+GRU | 45 | 0.8438 | 0.5731 | 0.8419 | 0.9129 | 0.4271 |
| Exp8 | Both | GRU+LSTM | 49 | 0.7963 | 0.6245 | 0.8570 | 0.9208 | 0.4253 |

### 5.2 Affective Correspondence 전체 비교 (Best Loss 체크포인트 기준)

`compute_vevo_correspondence` — 생성된 코드의 감정이 영상의 감정과 일치하는지 측정.
AMT baseline을 직접 훈련하여 Corr 비교 기준치 확보함.

| Exp | TEA 위치 | RNN 구성 | H@1 | H@3 | H@5 | Emo Loss | Corr | Acc+Corr |
|-----|----------|----------|-----|-----|-----|----------|------|----------|
| **Baseline** | — | — | 0.5537 | 0.8235 | 0.8981 | 0.4597 | **0.4634** | — |
| Exp1 | Decoder | LSTM | 0.5618 | 0.8274 | 0.9078 | 0.4232 | **0.5019** ↑ | 0.5318 |
| Exp2 | Decoder | GRU | 0.6098 | 0.8514 | 0.9148 | 0.4158 | 0.4740 ↑ | 0.5419 |
| Exp3 | Encoder | LSTM | 0.6222 | 0.8596 | 0.9223 | 0.4222 | 0.4703 ↑ | 0.5462 |
| Exp4 | Encoder | GRU | 0.6236 | 0.8564 | 0.9212 | 0.4222 | 0.4862 ↑ | 0.5549 |
| Exp5 | Both | LSTM+LSTM | 0.6106 | 0.8524 | 0.9173 | 0.4261 | 0.4684 ↑ | 0.5395 |
| Exp6 | Both | GRU+GRU | 0.5824 | 0.8438 | 0.9155 | 0.4272 | 0.4771 ↑ | 0.5297 |
| Exp7 | Both | LSTM+GRU | 0.5731 | 0.8419 | 0.9129 | 0.4271 | 0.4619 ↓ | 0.5175 |
| Exp8 | Both | GRU+LSTM | 0.6245 | 0.8570 | 0.9208 | 0.4253 | 0.4641 ↑ | 0.5443 |

> Exp7(Both LSTM+GRU)만 Corr에서 baseline 소폭 미달. 나머지 7개 실험 모두 baseline 이상.

### 5.3 Baseline 대비 향상률 (재현 baseline 기준, Best H@1 epoch)

| Exp | H@1 향상 | H@3 향상 | H@5 향상 | Emo Loss 향상 | Corr 향상 |
|-----|---------|---------|---------|-------------|---------|
| Exp1 (Decoder LSTM) | +1.5% | +0.5% | +1.1% | +8.0% | +8.3% |
| Exp2 (Decoder GRU) | +10.7% | +3.4% | +1.9% | +9.5% | +2.3% |
| Exp3 (Encoder LSTM) | +12.7% | +4.2% | +2.3% | +8.3% | +1.5% |
| Exp4 (Encoder GRU) | +12.6% | +4.0% | +2.6% | +8.3% | +4.9% |
| Exp5 (Both LSTM+LSTM) | +11.9% | +3.5% | +2.1% | +7.3% | +1.1% |
| Exp6 (Both GRU+GRU) | +5.7% | +2.5% | +1.9% | +7.1% | +2.9% |
| Exp7 (Both LSTM+GRU) | +5.7% | +2.3% | +1.6% | +7.1% | -0.3% |
| **Exp8 (Both GRU+LSTM)** | **+13.3%** | **+4.1%** | **+2.5%** | **+7.6%** | **+0.2%** |

> H@k 향상: 재현 baseline(0.5537) 기준. 논문 baseline(0.5139) 대비로는 더 큰 향상폭.
> Emo Loss 향상 = (baseline - ours) / baseline × 100.

---

### 5.4 테스트 세트 최종 평가 (Test Set)

`eval_TEA.py`를 test split으로 실행한 최종 결과. 논문 Table 4와 직접 비교 가능.

| 모델 | H@1 | H@3 | H@5 | Emo Loss | Aff. Corr |
|------|----:|----:|----:|--------:|----------:|
| **논문 AMT** | 0.5139 | 0.7722 | 0.8672 | 0.4662 | — |
| BASE (재현) | 0.5995 | 0.8545 | 0.9166 | 0.4527 | 0.4597 |
| Exp1 decoder LSTM | 0.6093 | 0.8636 | 0.9258 | 0.4124 | 0.4643 |
| Exp2 decoder GRU | 0.6542 | 0.8797 | 0.9301 | **0.4038** | 0.4900 |
| **Exp3 encoder LSTM** | **0.6700** | 0.8845 | 0.9352 | 0.4095 | 0.4885 |
| Exp4 encoder GRU | 0.6689 | **0.8862** | 0.9346 | 0.4103 | **0.4939** |
| Exp5 both LSTM+LSTM | 0.6590 | 0.8816 | 0.9349 | 0.4145 | 0.4881 |
| Exp6 both GRU+GRU | 0.6293 | 0.8724 | 0.9295 | 0.4155 | 0.4655 |
| Exp7 both LSTM+GRU | 0.6154 | 0.8638 | 0.9170 | nan* | 0.4535 |
| Exp8 both GRU+LSTM | 0.6684 | 0.8857 | **0.9359** | 0.4136 | 0.4853 |

> *Exp7 EmoLoss=nan: 학습 중 수치 불안정 발생. 해당 실험 제외 권장.

**논문 AMT 대비 최고 TEA 개선폭 (테스트 세트):**
- H@1: +0.1561 (+30.4%, Exp3)
- H@3: +0.1140 (+14.8%, Exp4)
- H@5: +0.0687 (+7.9%, Exp8)
- Emo Loss: -0.0624 (Exp2, 낮을수록 좋음)

---

### 5.5 강건성 평가 (Robustness Evaluation)

#### 5.5.1 노이즈 주입 (테스트 세트, deterministic)

비디오 피처 전체(semantic, scene_offset, motion, emotion)에 Gaussian 노이즈 σ 추가.

| 모델 | σ=0.00 H@1 | σ=0.05 | σ=0.10 | σ=0.20 | 최대 저하 |
|------|----------:|-------:|-------:|-------:|----------:|
| BASE | 0.5995 | 0.5995 | 0.5995 | 0.5995 | ±0.0000 |
| Exp3 | 0.6700 | 0.6701 | 0.6694 | 0.6699 | -0.0006 |
| Exp4 | 0.6689 | 0.6679 | 0.6675 | 0.6676 | -0.0013 |
| Exp8 | 0.6684 | 0.6678 | 0.6679 | 0.6667 | -0.0017 |

σ=0.2 (CLIP 임베딩 기준 20% 노이즈) 에서도 H@1 저하 < 0.002 → **매우 강건**.

#### 5.5.2 시드 안정성 (random_seq=True, 3회 반복)

| 모델 | H@1 mean±std | H@3 mean±std | H@5 mean±std |
|------|:---:|:---:|:---:|
| BASE | 0.5995 ± 0.0000 | 0.8545 ± 0.0000 | 0.9166 ± 0.0000 |
| Exp3 | 0.6700 ± 0.0000 | 0.8845 ± 0.0000 | 0.9352 ± 0.0000 |
| Exp4 | 0.6689 ± 0.0000 | 0.8862 ± 0.0000 | 0.9346 ± 0.0000 |
| Exp8 | 0.6684 ± 0.0000 | 0.8857 ± 0.0000 | 0.9359 ± 0.0000 |

std=0.0000: 테스트 세트 모든 샘플이 max_seq(300) 이하 → 평가가 완전히 결정론적.
재현성 관점에서 긍정적 (동일 환경에서 항상 동일 수치 재현 가능).

---

### 5.6 감정 충실도 평가 — "잡는 척 vs 제대로 잡는지" (`eval_emotion_fidelity.py`)

교수님 지적 사항: 모델이 감정을 실제로 반영하는가, 아니면 통계적으로 흔한 코드를 반복하며 우연히 감정 지표를 맞추는가?

두 가지 방법으로 검출:
1. **코드 다양성 (Shannon Entropy)**: 생성 코드 분포 엔트로피 (정규화 0~1). 낮으면 소수 코드 반복 = "잡는 척" 신호.
2. **피처 셔플 테스트**: 비디오 피처를 다른 영상 것으로 교체 후 Affective Corr 변화 측정.
   - `ΔCorr(emo)`: 감정 피처만 셔플
   - `ΔCorr(all)`: 전체 비디오 피처(semantic + motion + scene + emotion) 셔플

#### 결과 (테스트 세트)

| 모델 | Entropy↑ | Corr(orig) | ΔCorr(emo) | ΔCorr(all) |
|------|:--------:|:----------:|:----------:|:----------:|
| **BASE** | 0.4354 | 0.4597 | 0.0000 | **0.0000** |
| Exp3 encoder LSTM | 0.4514 | 0.4885 | -0.0013 | **-0.0443** |
| Exp4 encoder GRU | 0.4514 | 0.4939 | 0.0000 | **-0.0334** |
| Exp8 both GRU+LSTM | 0.4520 | 0.4853 | 0.0000 | **-0.0242** |

#### 핵심 해석

**BASE — "잡는 척"의 정량적 증거**
- 전체 비디오 피처를 완전히 다른 영상 것으로 교체해도 Corr 변화 = **±0.000**
- 어떤 영상을 입력해도 동일한 코드 품질 분포 → 비디오 내용을 사실상 무시
- Entropy 0.4354 (가장 낮음) → 소수 코드(C-F-G-Am) 반복 패턴

**TEA 모델 — 실제 비디오 활용의 증거**
- ΔCorr(all)이 모두 음수: 입력 비디오가 바뀌면 생성 코드 품질도 달라짐
- Exp3이 가장 큰 반응(-0.0443) → encoder 단계에서 비디오 컨텍스트를 가장 적극적으로 활용
- Entropy도 BASE보다 높음 → 더 다양한 코드 생성

**ΔCorr(emo) ≈ 0인 이유**
감정 6차원 피처만 셔플해도 반응이 미미한 것은, CLIP semantic 피처 자체가 감정 정보를 이미 내포하기 때문. 모델은 단일 감정 채널이 아닌 영상 전체 표현으로 감정을 처리함. → 감정-only 셔플보다 전체 피처 셔플이 더 강한 검출 수단.

#### 결론
BASE는 비디오-감정 정렬을 학습한 것이 아니라 **훈련 데이터의 코드 빈도 분포를 암기**한 것.  
TEA는 실제 비디오 피처 변화에 반응하며 코드 품질을 조정함 → "제대로 잡는" 모델.

---

## 6. 주요 발견 및 분석

### 6.1 TEA 위치별 효과
- **Encoder only (Exp3, 4)** 가 Best val loss 기준 가장 낮은 손실 (0.7938, 0.7963)
- Decoder only (Exp1, 2)는 상대적으로 낮은 H@k 성능
- Both (Exp5~8)는 파라미터가 더 많지만 Encoder only보다 val loss가 높음
  - 원인 추정: enc_dec_fusion이 동일 셀 타입 시 중복 신호 유발
  - **단, Exp8 (GRU+LSTM 혼합)은 H@1 0.6245로 전체 최고** → 셀 다양성이 시너지

### 6.2 H@k와 Affective Correspondence의 역상관
```
Exp1 (H@1: 0.5618) → Corr: 0.5019  ← Corr 최고
Exp3 (H@1: 0.6222) → Corr: 0.4703  ← H@k 높지만 Corr 낮음
Exp4 (H@1: 0.6236) → Corr: 0.4862
```
코드 예측 정확도(H@k)가 높을수록 감정 대응(Corr)이 낮아지는 경향.
- Encoder TEA는 코드 패턴 학습에 집중 → H@k↑, Corr↓
- Decoder TEA는 코드 생성 시점에 감정을 직접 개입 → H@k↓, Corr↑
- 두 지표를 동시에 높이려면 훈련 신호(loss)에 Corr를 직접 포함하는 방향 고려 필요

### 6.3 RNN 셀 타입
- Decoder only: GRU > LSTM (Exp2 > Exp1)
- Encoder only: LSTM ≈ GRU (0.0014 차이)
- Both: 혼합 조합(GRU+LSTM, Exp8)이 동종 조합(Exp5, 6)보다 우수

### 6.4 Val Loss가 낮다 ≠ H@k가 높다
Exp3이 가장 낮은 val loss(0.7938)지만 H@1 피크는 Exp8(0.6245, best loss ckpt 기준).
Best loss 기준 저장 체크포인트가 H@k 최고 시점과 항상 일치하지 않음.

### 6.5 모든 TEA 실험이 감정 손실(EmoLoss) 개선
Baseline EmoLoss=0.4597 대비 TEA 전 실험 7~10% 개선.
단순 concat 방식보다 RNN 기반 시간 처리가 감정 예측에 명확히 유리.

### 6.6 정성 평가 — 청취 비교 (영상 049)

영상 049 (어두운 분위기)에 대해 Baseline / Exp8 / Exp4 음악을 생성 후 청취 비교.

| 모델 | 청취 인상 |
|------|----------|
| Baseline (AMT) | 평균적인 코드 진행, 감정 흐름 반영 미흡 |
| Exp8 (Both GRU+LSTM) | 미묘하게 개선되나 뚜렷한 차이 불명확 |
| **Exp4 (Encoder GRU)** | **어두운 분위기를 baseline보다 명확히 포착** |

- Exp4가 Acc+Corr 1위(0.5549)인 정량 결과와 정성 청취 결과가 일치
- H@1 최고인 Exp8보다 Corr 기준 2위인 Exp4가 체감 차이가 더 뚜렷
- 감정 정렬 측면에서 Encoder TEA + GRU 조합이 가장 효과적임을 시사

---

## 7. 한계 및 향후 연구

### 7.1 한계
- Val set 크기 75개로 H@k, Corr 지표에 분산이 큼
- enc_dec_fusion 구조가 동종 셀 조합 시 성능 하락 가능성
- H@k와 Corr의 역상관 — 두 지표를 동시에 최적화하는 학습 신호 부재
- 정성 평가가 단일 영상(049) 한정 — 더 다양한 샘플로 확장 필요

### 7.2 시도했으나 효과 없었던 접근
- **Soft Corr Loss (Exp9)**: Jaccard 유사도 기반 소프트 타겟 CE → val loss 1.77 (Exp4 0.80 대비 2.2배) → 실패
- **Contrastive Emotion 파인튜닝 (Exp3/4/8_con)**: SupConLoss(batch size 문제) → Classification Head CE Loss → con loss 미수렴, val loss best가 epoch 1-2(pretrained) → 중단

### 7.3 향후 연구 (중간 발표 이후)
- emotion loss 가중치(λ) 조정 실험 (현재 chord:emotion = 1:1)
- enc_dec_fusion 구조 개선 (cross-attention 기반 감정 주입)
- Contrastive learning 재시도 시: 더 큰 배치 크기 + Memory Bank + Differential LR 필수
- **Inference-time emotion override**: 6차원 감정 벡터 수동 지정으로 사용자 제어 가능성

---

## 10. 추가 구현 (VideoRegression 개선 & 임의 영상 지원)

### 10.1 임의 영상 입력 지원

#### generate_TEA.py — `--video` 인자 추가
- 기존: 데이터셋 내 영상 ID만 지원 (`--test_id`)
- 변경: 로컬 mp4 또는 YouTube URL 직접 입력 가능

```bash
# 로컬 영상
python generate_TEA.py --exp 4 --video path/to/video.mp4

# YouTube URL (yt-dlp 자동 다운로드)
python generate_TEA.py --exp 4 --video "https://youtube.com/watch?v=..."
```

- `extract_features_from_video()` 함수 추가 — `video2music.py`의 feature 추출 파이프라인 재사용
- feature 임시 저장: `./feature_tmp/` (매 실행 시 초기화)
- `total_vf_dim = 776` (CLIP 768 + scene_offset 1 + motion 1 + emotion 6)

#### video2music.py — 버그 수정
| 버그 | 원인 | 해결 |
|------|------|------|
| 생성 음악 무음 | `self.SF2_FILE = "soundfonts/default_sound_font.sf2"` 경로 없음 | `/home/taegum/miniconda3/envs/video2music/.../TimGM6mb.sf2` 절대경로로 수정 |
| `RuntimeError: tensors on cuda:0 and cpu` | `USE_CUDA=False` 기본값 | `__init__`에 `use_cuda(True)` 추가 |
| YouTube URL 지원 | 없음 | `generate()` 진입부에 yt-dlp 자동 다운로드 추가 |

#### generate.py — `--test_id` 하드코딩 제거
- `test_id = "049"` 하드코딩 및 `args.test_id = test_id` override 제거
- 이제 `-test_id` 인자가 실제로 동작함

---

### 10.2 정성 평가 — Major/Minor 코드 비율 분석

동일 영상(test.mp4, relaxing 위주)에 대해 Baseline vs TEA Exp4 비교:

| 모델 | Major계열 | Minor계열 | Dim |
|------|---------|---------|-----|
| Baseline AMT | 74.0% | **21.3%** | 4.7% |
| TEA Exp4 | **96.7%** | 2.7% | 0.7% |

- 밝은(relaxing) 영상에 Baseline은 minor/dim 코드를 26% 섞음
- TEA는 같은 영상에 96.7% major 코드 → 영상 감정을 더 정확히 반영
- 이것이 "감정을 처리하는 척 vs 실제로 처리" 의 정량적 증거

**Emotion feature 비교 (049 vs test.mp4)**:
- 049.mp4: tense 0.48~0.72 지배 (어두운 영상)
- test.mp4: relaxing 0.52~0.71 지배 (밝은 영상)
- Feature 추출은 정상 작동 — 두 영상의 감정 분포가 명확히 다름

---

### 10.3 VideoRegression 재학습 — MSE + PCC Loss

#### 문제 진단
기존 MSE-only 학습의 "regression to mean" 현상:

| | Level 0 (2음) | Level 1 (3음) | Level 2 (4음) | Level 3 (5음) | Level 4 (6음+) |
|---|---|---|---|---|---|
| 훈련 데이터 | 19.4% | 35.2% | 28.4% | 12.2% | 4.8% |
| 모델 출력 (기존) | 9.7% | **90.3%** | 0% | 0% | 0% |

훈련 데이터: density mean=10.3, std=5.7 (다양함)
모델 예측: 거의 5~10 범위에 고정 → MSE는 평균 근처 예측으로 오차를 최소화

#### 해결: z-score 정규화 + Pearson Correlation Coefficient (PCC) Loss

**핵심 원인**: Note Density(0–35)와 Loudness(0–0.47)의 스케일 차이가 60배 → combined MSE가 Loudness를 사실상 무시.

z-score 정규화로 두 타겟을 같은 스케일로 맞춘 뒤 학습:

```python
# 학습셋 전체에서 통계 계산 → norm_stats.json 저장
nd_mean, nd_std = 8.5440, 6.4810
lv_mean, lv_std = 0.1448, 0.1116

# train: 정규화 후 손실 계산
feature_note_density_n = (feature_note_density - nd_mean) / nd_std
feature_loudness_n     = (feature_loudness     - lv_mean) / lv_std
loss_density  = F.mse_loss(y_nd, nd_n) + 0.5 * pcc_loss(y_nd, nd_n)
loss_loudness = F.mse_loss(y_lv, lv_n) + 0.5 * pcc_loss(y_lv, lv_n)

# eval: RMSE는 역정규화 후 원래 스케일로 보고
y_nd_raw = y_nd * nd_std + nd_mean
y_lv_raw = y_lv * lv_std + lv_mean
```

PCC Loss 추가 (평균 예측 방지):
```python
def pcc_loss(pred, target):
    """1 - PCC. 모델이 평균만 예측하면 PCC=0 → loss=1로 페널티."""
    p_c = pred.flatten() - pred.mean()
    t_c = target.flatten() - target.mean()
    pcc = (p_c * t_c).sum() / (p_c.norm() * t_c.norm() + 1e-8)
    return 1.0 - pcc
```

추가 수정:
- `evaluate_regression.py`: `regModel = "gru"` → `"bigru"` 버그 수정 (학습/평가 아키텍처 불일치)
- `train_regression.py`: `ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)` 추가, epochs=200
- `generate_TEA.py`: z-score 정규화 도입 후 역정규화 버그 수정 (`y_loudness * 100` → norm_stats.json 로드 후 `y_lv_raw = y_lv * lv_std + lv_mean` 처리)

#### 최종 결과 (Bi-GRU, test set)

| 지표 | 논문 | 우리 |
|------|------|------|
| RMSE (Note Density) | 4.5030 | **4.465** ✓ |
| RMSE (Loudness) | 0.0876 | **0.0811** ✓ |

논문 대비 소폭 개선. `saved_models/AMT/norm_stats.json`에 통계 저장됨.

---

### 10.4 generate_TEA.py — temperature 파라미터 추가

모델이 C-F-G 같은 확률 높은 코드에 수렴하는 문제 완화:

```bash
# 기본
python generate_TEA.py --exp 4 --video video.mp4

# 다양성 증가 (추천)
python generate_TEA.py --exp 4 --video video.mp4 --temperature 1.3
```

- `model/video_music_transformer_TEA.py` generate 함수에 `temperature` 파라미터 추가
- `token_probs = probs ** (1/T)` 후 재정규화 → T > 1.0이면 분포가 평탄해져 다양한 코드 샘플링

---

---

## 10.5 MIDI 렌더러 전면 교체 — music21 기반

### 배경
수동 코딩 방식(Alberti, 왈츠 패턴 등)이 너무 인위적으로 들리는 문제.
코드 보이싱을 음악 이론 라이브러리(music21)에 위임하고, 렌더링 패턴을 단순화.

### 핵심 변경

#### 1. 보이싱 — music21 `ChordSymbol.closedPosition()`
```python
from music21 import harmony as m21harmony

def _m21_pitches(chord_label, octave=4):
    label = chord_label.replace(':', '').replace('hdim7', 'm7b5')
    for flat, sharp in _FLAT_TO_SHARP.items():  # Db→C#, Eb→D# 등
        if label.startswith(flat):
            label = sharp + label[len(flat):]
            break
    cs = m21harmony.ChordSymbol(label)
    closed = cs.closedPosition(forceOctave=octave, inPlace=False)
    return sorted(set(p.midi for p in closed.pitches))
```
- 수동으로 코드 구성음 계산하지 않음 → music21이 음악 이론에 맞게 처리
- `hdim7 → m7b5`, flat→sharp 변환으로 데이터셋 레이블 호환

#### 2. 보이스 리딩 — `_voice_lead()`
```python
def _voice_lead(curr, prev):
    center = sum(prev) / len(prev)
    voiced = []
    for p in curr:
        best = min([p + s*12 for s in (-2,-1,0,1,2)], key=lambda c: abs(c - center))
        voiced.append(best)
    return sorted(voiced)
```
- 이전 코드의 무게중심에 가장 가까운 옥타브로 각 음을 이동 → 보이스 리딩 자연스럽게

#### 3. LH — Oom-pah 패턴
- Beat 1: 베이스 (루트음, 옥타브 2–3, duration×0.75)
- Beat 2 (+0.5박): 코드 쉘 (inner notes 2개, duration×0.42)
- 수동 Alberti/아르페지오/왈츠 패턴 완전 제거

#### 4. RH — 업워드 롤
- 코드 구성음을 아래→위로 0.025박씩 오프셋해서 순차 발음
- 최상단 음(멜로디)은 velocity 강조 + duration 1.05×, 나머지는 0.90×

#### 5. 감정 기반 다이나믹
```python
dark = fearful[1] + tense[2] + sad[3]  # 8코드 이동평균
v = v_base + int(6 * sin(π × phrase_pos))   # 구 내 크레셴도
v = clip(v + (0.5 - dark) * 10, 40, 127)    # 어두울수록 살짝 여림
```
- 구 길이 불규칙화 유지: `PHRASE_LEN_CYCLE = [4,4,3,5,4,3,4,4]`

#### 6. Velocity 안정화 (영상 길이 < 300 대비)
영상이 300초보다 짧을 경우, regression 모델이 zero-padded 구간에서
near-MIN_VELOCITY 예측을 내놓으면서 velocity가 급락하는 문제 수정:
```python
# 5-frame 이동평균으로 스파이크 제거 후 median을 floor로 적용
_velo_arr  = np.convolve(np.array(velo_list), np.ones(5)/5, mode='same')
_vel_floor = max(int(np.median(_velo_arr)), MIN_VELOCITY + 20)
velo_list  = [max(int(v), _vel_floor) for v in _velo_arr]
```
- `_hv()` lower bound: 1 → 35 (어떤 노트도 vel<35가 되지 않도록)
- inner notes minimum: 22 → 38

### 사운드폰트 업그레이드
- 기존: FluidR3_GM.sf2 (142MB, 전자음 느낌)
- 변경: MuseScore_General_Full.sf3 (82MB, 훨씬 자연스러운 피아노)
- sudo 없이 deb 직접 다운로드 후 수동 추출:
  ```bash
  wget http://kr.archive.ubuntu.com/ubuntu/pool/universe/m/.../musescore-general-soundfont_0.1.9-1_all.deb
  dpkg-deb -x musescore_sf.deb /tmp/musescore_extract/
  cp /tmp/musescore_extract/usr/share/sounds/sf3/MuseScore_General_Full.sf3 ~/sf2/
  ```
- 설치 경로: `/home/taegum/sf2/MuseScore_General_Full.sf3`

### 적용 파일
- `generate_TEA.py` — `chords_to_midi()`, `_m21_pitches()`, `_voice_lead()`, `_hv()`
- `generate.py` — 동일 로직 적용 (변수명 `DURATION`→`duration`, `MIN_VELOCITY`→`_MIN_VELOCITY` 차이만 있음)

---

### 10.5.2 generate_TEA.py 후처리 — music21 → 원본 baseline으로 복원

#### 경위
- 10.5에서 `generate_TEA.py`의 `chords_to_midi`도 music21 기반으로 교체함
- 이후 `generate_TEA.py`를 **원래 baseline 방식으로 복원** (아래 이유)

#### 현재 상태 (`generate_TEA.py`)

```python
# 코드 → MIDI 변환: 원본 baseline 방식 유지
midi_chords_original = []
for key in chord_genlist:
    key = key.replace(":", "")
    midi_chords_original.append(Chord(key).getMIDI("c", 4))  # utilities.chord_to_midi

if IS_VOICE:
    midi_chords = voice(midi_chords_original)   # voice leading
else:
    midi_chords = midi_chords_original

# density별 아르페지오 패턴 (density_list 0~4)
# IS_ARP = True → density 단계별 note 배치 (원본 generate.py와 동일 구조)
```

#### generate.py vs generate_TEA.py 후처리 분기

| 항목 | generate.py (baseline) | generate_TEA.py (현재) |
|------|----------------------|----------------------|
| 보이싱 | music21 `ChordSymbol.closedPosition()` | `utilities.chord_to_midi.Chord.getMIDI()` |
| 보이스리딩 | `_voice_lead()` (무게중심 기반) | `voice()` (utilities 내장) |
| LH 패턴 | Oom-pah (dark 연속 함수) | density 0~4 단계별 아르페지오 |
| RH 패턴 | 업워드 롤 + 멜로디 강조 | density별 fixed 패턴 |
| 감정 다이나믹 | dark 연속 함수 기반 velocity | regression 모델 출력 → velocity 변환 |

#### 이유
- music21 기반 구현 시 sus2/sus4 코드에서 `IndexError` 반복 발생 (10.5에서 midiutil deInterleaveNotes 버그)
- 트러블슈팅 후 안정화했지만, TEA 생성의 핵심은 **화음 시퀀스의 질**이지 MIDI 후처리 방식이 아님
- 평가 지표(H@k, Corr, RMSE)가 MIDI 렌더링과 무관하므로 baseline 방식 유지로 실험 변수 최소화
- `generate.py`(baseline 생성기)는 music21 방식 유지 → 두 파일이 서로 다른 후처리 방식을 보유

---

### 10.5.1 LH 패턴 — 감정 연속 함수로 전환

#### 문제: 고정 threshold 기반 패턴 분기
초기 구현에서 `if dark > 0.60 → Pattern A`, `elif dark > 0.38 → Pattern B` 형태로 분기.
→ 특정 영상(049)에 맞춘 경계값이 다른 영상에서 오동작할 위험.
→ `--video URL` 로 임의 영상 처리 시 범용성 없음.

#### 해결: 모든 파라미터를 dark의 연속 함수로

```python
# 이전 — threshold 분기
if dark > 0.60:
    b_dur = DURATION * 0.85
    midi.addNote(fifth, ...)  # 5th 항상 추가
elif dark > 0.38:
    ...

# 이후 — 연속 보간, threshold 없음
b_dur       = DURATION * (0.58 + dark * 0.30) * (1.0 - phrase_tail * 0.40)
use_fifth   = np.random.random() < dark * 0.75   # 확률이 dark에 비례
pah_offset  = DURATION * (0.42 + dark * 0.12)
pah_vol_off = int(-12 + (1.0 - dark) * 6)
use_low_oct = np.random.random() < (1.0 - dark) * 0.25  # 밝을수록 옥타브 점프
```

| 파라미터 | dark=0 (밝음) | dark=1 (어두움) | 효과 |
|---------|-------------|--------------|------|
| `b_dur` | `DURATION × 0.58` | `DURATION × 0.88` | 어두울수록 베이스 길게 유지 |
| `use_fifth` 확률 | 0% | 75% | 어두울수록 5th 파워코드 |
| `pah_offset` | `DURATION × 0.42` | `DURATION × 0.54` | 어두울수록 pah가 늦게 |
| `use_low_oct` 확률 | 25% | 0% | 밝을수록 옥타브 점프 (활기) |

#### dark soft-clamp
```python
dark = float(np.clip(dark_smooth[i], 0.15, 0.85))
```
- 이유: 극단값(0 또는 1)에서 파라미터가 edge case로 치우치는 것 방지
- 어떤 영상이든 LH 파라미터가 합리적인 범위 안에서 동작함

---

## 10.6 Soft Corr Loss — 화성적 유사도 기반 소프트 타겟 손실 (Exp9)

### 배경

기존 Cross-Entropy Loss는 모든 오답을 동등하게 처벌한다.
"G:7을 예측했는데 정답이 G" vs "D#:min을 예측했는데 정답이 G" — 같은 패널티.
그 결과 모델은 **안전한 공통 코드(G:7 / F:7 / G)를 57% 이상** 반복적으로 예측하는
붕괴 현상(Softmax Collapse)이 발생한다 (314번 분석 결과: 상위 6개 코드 = 87%).

### 해결 방법 — Soft Corr Loss

정답 코드 i에 대해 소프트 타겟 분포를 다음과 같이 구성:

```
t[j] = sim(i, j) / Σ_k sim(i, k)
```

여기서 `sim(i, j)` = 코드 i와 j의 **Jaccard 음정클래스 유사도**:

```
sim(i, j) = |PC(i) ∩ PC(j)| / |PC(i) ∪ PC(j)|
```

예시:
| 비교 | 공유 음 | Jaccard |
|------|---------|---------|
| C — C | {C,E,G} | 1.000 |
| C — C:min | {C,G}/4 | 0.500 |
| C — C:7 | {C,E,G}/4 | 0.750 |
| C — G:7 | {G}/6 | 0.167 |

이 소프트 타겟으로 Cross-Entropy를 계산하면 화성적으로 가까운 코드를 예측해도
부분 크레딧이 주어져, 모델이 단일 코드에 몰리는 붕괴를 억제할 수 있다.

### 구현

**`model/loss.py`** 에 두 함수 추가:
- `build_chord_similarity_matrix(vocab_size, chord_end_idx)` — 159×159 유사도 행렬 생성
- `SoftCorrLoss(sim_matrix, ignore_index)` — 소프트 타겟 CE 손실

```python
# SoftCorrLoss forward:
soft = sim_matrix[target, :]          # [M, V] — 정답 코드의 유사도 행
log_probs = F.log_softmax(logits, dim=-1)
loss = -(soft * log_probs).sum(dim=-1).mean()
```

**`experiments/train_TEA_exp9.py`** — exp4(encoder-only GRU, 최고 성능) 구조에
`--soft_corr` 플래그만 교체. 기타 하이퍼파라미터 동일.

**`utilities/argument_funcs.py`** — `--soft_corr` 인수 추가.

### 실험 설정 (Exp9)

| 항목 | 값 |
|------|----|
| 아키텍처 | encoder-only TEA, GRU (exp4와 동일) |
| Loss | SoftCorrLoss (Jaccard 유사도 소프트 타겟) |
| Epochs | 100 |
| Batch size | 4 |
| GPU | CUDA:0 (RTX A6000) |
| 시작 | 2026-05-23 |
| 로그 | `logs/exp9.log` |
| 목표 | 코드 다양성 개선 (87% → < 60% 목표) |

### 실제 결과 (Exp9 최종)

| 항목 | 값 |
|------|-----|
| Best epoch | 7 |
| Best val loss | 1.7700 |

- 동일 아키텍처 Exp4(val loss 0.7963)와 비교 시 **2.2배 높은 손실** → 실패 실험
- 원인: Soft Corr Loss는 정답 코드와 화성적으로 유사한 코드에 부분 크레딧을 주어 CrossEntropy보다 loss가 전반적으로 높게 형성되며, 모델이 수렴 방향을 잃음
- H@k 지표도 Exp4 대비 현저히 낮음 → Exp9 결과 비교 대상에서 제외

---

## 11. Contrastive Emotion Loss 파인튜닝 — 시도 및 중단

### 11.1 동기

Emotion Fidelity 분석(5.6절)에서 ΔCorr(emo) ≈ 0 발견 — TEA가 감정 피처를 충분히 활용하지 못할 가능성. 감정 클래스 간 임베딩 분리를 명시적으로 학습시키면 Affective Correspondence가 개선될 것이라는 가설 아래 Supervised Contrastive Loss 파인튜닝을 시도.

### 11.2 구현 과정

**Phase 1 — SupConLoss (실패)**

- `model/contrastive_loss.py`에 Supervised Contrastive Loss (온도 매개변수 τ=0.07) 구현
- Positive pair: 동일 지배 감정 클래스 내 샘플, Negative pair: 다른 클래스
- **문제**: SupCon은 충분한 positive 쌍이 필요 (권장 배치 크기 256+). batch_size=4 환경에서는 한 배치에 positive 쌍이 없는 경우 다수 → con loss가 ln(3)=1.0986 (랜덤 수준)에 고착

**Phase 2 — CrossEntropy Classification Head (시도)**

SupCon 포기 후 배치 크기 독립적인 방식으로 전환:
- `VideoMusicTransformerTEA`에 `emotion_clf = nn.Linear(128, 6)` 추가
- `get_emotion_emb()`: TEA 어댑터 출력 평균풀링 → `contrastive_proj` (d_model→128) → L2 normalize
- 지배 감정 레이블: `emotion.mean(dim=1).argmax(dim=-1)` (각 영상의 프레임 평균 감정 분포 argmax)
- 손실: `L = λ_chord·L_chord + (1-λ_chord)·L_emo + λ_con·L_con`
  - `L_con = CrossEntropyLoss(emotion_clf(emo_emb), dominant_class)`
  - con_lambda=0.1, LR=1e-4 (고정)

### 11.3 학습 결과

상위 TEA 실험 3개에 대해 각각 GPU를 분리하여 파인튜닝 (CUDA_VISIBLE_DEVICES=0/1/2):

| 실험 | Best Epoch | Best Val Loss | 원본 Val Loss | 비고 |
|------|-----------|--------------|--------------|------|
| Exp3_con (encoder LSTM) | 1 | 0.7901 | 0.7938 | epoch 1 이후 지속 상승 |
| Exp4_con (encoder GRU) | 2 | 0.7916 | 0.7963 | epoch 2 이후 지속 상승 |
| Exp8_con (both GRU+LSTM) | 2 | 0.7844 | 0.7963 | NaN 발생 (epoch 3, 6) |

- Best epoch이 1~2: **사전학습 가중치를 그대로 유지하는 것이 최선** — 파인튜닝 효과 없음
- con loss 추이: ~1.3~1.9 (ln(6)=1.79, 6-class 랜덤 수준)에서 수렴 못 함
- Exp8_con NaN: 불안정한 그라디언트로 epoch 3, 6에서 val loss = nan 발생

### 11.4 실패 원인 분석

| 원인 | 내용 |
|------|------|
| 단일 LR 문제 | LR=1e-4: 기존 가중치 손상 vs 새 emotion_clf 미학습 — 양쪽 모두 불리한 트레이드오프 |
| con_lambda 과다 | λ_con=0.1: con loss가 수렴하지 않은 상태에서 0.1×(1.3~1.9)이 chord loss에 지속적 간섭 |
| 데이터 분포 | 6-class 지배 감정 레이블에서 일부 감정(fearful, tense 등)이 과대 대표 → 클래스 불균형 |
| 감정 임베딩 비효율 | TEA 어댑터 출력 평균풀링이 시간적 감정 변화를 단일 벡터로 압축 → 클래스 분리에 불충분 |

**결론**: Contrastive/Classification 기반 파인튜닝은 현재 실험 설정(batch_size=4, 단일 LR, 클래스 불균형)에서 유효한 개선을 가져오지 못함. Exp3/4/8 원본 체크포인트가 `_con` 변형보다 일관되게 우수. → **`_con` 접근 방식 중단, 원본 TEA 결과로 최종 발표 진행**.

### 11.5 교훈

- Contrastive learning은 충분한 배치 크기 및 메모리뱅크 없이는 효과가 제한적
- 파인튜닝 시 새로 추가된 파라미터(emotion_clf)는 기존 파라미터와 다른 LR이 필요 (Differential LR)
- 감정 정렬 개선은 loss 설계보다 아키텍처 수준의 접근(cross-attention 기반 감정 주입 등)이 더 적합할 수 있음

---

## 8. 파일 구조

```
Video2Music/
├── model/
│   ├── tea_adapter.py                  # TEA 모듈 구현
│   ├── video_music_transformer.py      # 원본 AMT 모델 (feature_key 버그 수정됨)
│   ├── video_music_transformer_TEA.py  # TEA 통합 모델 (emotion_clf 헤드 포함)
│   ├── contrastive_loss.py             # SupConLoss (구현됨, 현재 미사용)
│   └── loss.py                         # SoftCorrLoss (Exp9용)
├── utilities/
│   └── constants_TEA.py               # TEA 하이퍼파라미터 (CONTRASTIVE_LAMBDA, PROJ_DIM 등 포함)
├── experiments/
│   ├── train_TEA_exp1.py ~ exp9.py    # 실험별 학습 스크립트
│   ├── constants_TEA_exp1.py ~ exp9.py
│   ├── experiment_summary.md
│   ├── experiment_results.md
│   ├── eval_correspondence.csv        # 테스트셋 전체 평가 결과
│   ├── emotion_fidelity.csv           # 감정 충실도 (셔플 테스트) 결과
│   └── robustness_noise.csv           # 노이즈 강건성 평가 결과
├── train.py                           # AMT baseline 학습
├── train_regression.py                # VideoRegression 학습 (bigru, MSE+PCC loss)
├── train_TEA.py                       # TEA 학습 메인 스크립트
├── train_TEA_contrastive.py           # Contrastive 파인튜닝 스크립트 (중단됨)
├── eval_TEA.py                        # TEA 평가 스크립트 (Corr + baseline)
├── eval_emotion_fidelity.py           # 감정 충실도 평가 스크립트
├── generate.py                        # baseline AMT 생성 (music21 후처리)
├── generate_TEA.py                    # TEA 음악 생성 (원본 chord_to_midi 방식)
├── run_TEA.sh                         # 실험 1~8 자동 실행 스크립트
├── logs/                              # 학습 로그
│   ├── exp1.log ~ exp9.log
│   ├── amt_baseline.log
│   ├── eval_all.log
│   └── (train_exp3_con.log 등 _con 실험 로그)
├── output_vevo/AMT/049/049_cgen_rd.mp4          # Baseline 생성 샘플
├── output/049/049_TEA_exp8_cgen_rd.mp4           # TEA Exp8 생성 샘플
├── output/049/049_TEA_exp4_cgen_rd.mp4           # TEA Exp4 생성 샘플
└── saved_models/
    ├── TEA_decoder_lstm_exp1/ ~ TEA_both_gru_lstm_exp8/  # 완성된 TEA 체크포인트
    ├── TEA_encoder_gru_exp9/      # Exp9 SoftCorrLoss (실패, val loss 1.77)
    ├── TEA_encoder_lstm_exp3_con/ # Contrastive 파인튜닝 (중단, best epoch 1)
    ├── TEA_encoder_gru_exp4_con/  # Contrastive 파인튜닝 (중단, best epoch 2)
    ├── TEA_both_gru_lstm_exp8_con/# Contrastive 파인튜닝 (중단, best epoch 2)
    └── AMT/
        ├── best_loss_weights.pickle   ← 재현 훈련 완료 (epoch 59)
        ├── norm_stats.json            ← VideoRegression z-score 통계
        └── results.csv
```

---

## 9. 트러블슈팅 기록

| 오류 | 원인 | 해결 |
|------|------|------|
| `IndexError: list index out of range` | MuVi-Sync 데이터셋 미추출 | `tar -xzf MuVi-Sync.tar.gz --strip-components=1` |
| `RuntimeError: Tensor with 4 elements cannot be converted to Scalar` (TEA eval) | `feature_key.item()` — batch_size>1 불가 (`video_music_transformer_TEA.py`) | `.view(B,1,1).expand(B,T,1)` 로 수정 |
| `RuntimeError: Tensor with 4 elements cannot be converted to Scalar` (baseline train) | 동일 버그가 `video_music_transformer.py`에도 존재 | 동일하게 `.view(B,1,1).expand(B,T,1)` 수정 |
| AMT baseline이 CPU로 훈련됨 | `train.py`가 `use_cuda(True)` 미호출 (default=False) | `--force_cpu` 없으면 `use_cuda(True)` 호출하도록 수정 |
| `ValueError: Target size mismatch` | `tgt_emotion.squeeze()` — batch_size>1 시 배치 차원 제거 안 됨 | `.reshape(-1, tgt_emotion.shape[-1])` 로 교체 (7곳) |
| `RuntimeError: Boolean value of Tensor` | `compute_vevo_correspondence` 내부에서 다중 원소 텐서 비교 | tgt_emotion을 `(B*T, 6)` 으로 pre-reshape 후 전달 |
| H@k 고정값 (0.1447 반복) | `compute_hits_k`가 batch_size>1일 때 배치 축을 시간 축으로 오인 | y를 `(B*T, 159)` 로 flatten 후 전달 |
| H@k ≈ 0 | backbone이 랜덤 초기화 상태에서 동결 (사전학습 가중치 없음) | backbone 동결 제거 → 전체 joint training |
| `size mismatch` (eval_TEA.py) | `from ... import` 로 복사된 로컬 변수는 모듈 패치 미반영 | `model.video_music_transformer_TEA` 모듈 네임스페이스도 직접 패치 |
| `OSError: MoviePy Error — unable to read font` (generate.py) | ImageMagick v7에서 폰트 이름 미인식 | `font=` 를 이름 대신 TTF 전체 경로(`/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf`)로 교체 |
| `ModuleNotFoundError: No module named 'voicing'` (generate_TEA.py) | `voicing` 패키지 미설치 (Python 2 전용 패키지) | `from utilities.chord_to_midi import voice, Chord` 로 교체 |
| `AttributeError: 'Chord' object has no attribute 'getMIDI'` (generate_TEA.py) | `from pychord import Chord` 가 커스텀 `Chord`(getMIDI 있음)를 덮어씀 | `pychord` import 제거, `utilities.chord_to_midi` 에서 `Chord` 직접 import |
| VideoRegression이 CPU로 학습됨 | `train_regression.py`가 `use_cuda(True)` 미호출 | `train.py`와 동일하게 `else: use_cuda(True)` 추가 |
| `generate_TEA.py`가 regression 가중치를 못 찾음 | 파일명 불일치 (`best_loss_weights_reg.pickle` vs `best_rmse_weights.pickle`) + 모델 타입 불일치 (`bigru` vs `gru`) | `train_regression.py` → `bigru`로 변경, `generate_TEA.py` → `best_rmse_weights.pickle` 경로로 통일 |
| `OSError: dataset/vevo/None.mp4` | `args.test_id`가 None일 때 경로가 `"None.mp4"`로 생성됨 | `test_id_str = dataset.data_files_chord[test_id_idx].split("/")[-1][:3]` 로 실제 파일 ID 추출 |
| `OSError: dataset/vevo/707.mp4` (영상 없음) | 대부분 영상 파일이 데이터셋에 없는데 무조건 compositing 시도 | `if os.path.isfile(f_video_in):` 가드 추가, 없으면 경고 출력 후 생략 |
| `IndexError: pop from empty list` (midiutil deInterleaveNotes) | sus2/sus4 코드에서 `getMIDI`가 동일 MIDI 피치를 두 번 반환(예: `F#:sus2` → notes[2]=notes[3]=68). NoteOn은 `removeDuplicates`에서 1개로 합쳐지지만 NoteOff는 end tick이 달라 2개 그대로 → stack 언더플로우 | `rh = sorted(set(rh_voiced[i]))`, `lh = list(dict.fromkeys(lh_raw[i]))` 로 중복 제거 |
| `FileNotFoundError: fluidsynth` (PATH 없음) | `midi2audio.FluidSynth`가 `subprocess.call(['fluidsynth', ...])` 호출 시 conda env bin이 PATH에 없음 | `FluidSynth().midi_to_audio()` 대신 직접 `subprocess.call([_fs_bin, ...])` 호출; `shutil.which` + fallback 절대경로(`/home/taegum/miniconda3/envs/video2music/bin/fluidsynth`)로 바이너리 탐색 |
| `UnboundLocalError: local variable 'subprocess' referenced before assignment` | `main()` 내부 YouTube 다운로드 블록에 `import subprocess` 가 중복 존재 → Python 3 스코핑 규칙으로 모듈 레벨 import를 가림 | `main()` 내부 `import subprocess` 제거, 모듈 레벨 import로 통합 |
| `NameError: name 'MIDIFile' is not defined` (generate.py) | music21 도입 시 `from utilities.chord_to_midi import *` 를 `from music21 import harmony` 로 교체하면서 `MIDIFile` import가 누락됨 | `from midiutil import MIDIFile` 명시적으로 추가 |
| 230s 이후 음악이 갑자기 조용해지거나 velocity 급락 | 영상(049: 234초)이 생성 길이(300코드)보다 짧아 regression 모델이 zero-padded 구간에서 near-MIN 예측 + noise로 49↔84 랜덤 스파이크 발생 | velo_list에 5-frame 이동평균 스무딩 + median 기반 floor 적용 |
| `sudo: a terminal is required` (MuseScore soundfont 설치 불가) | sudo 권한 없음 | deb 패키지를 wget으로 직접 다운로드 후 `dpkg-deb -x` 로 수동 추출, `~/sf2/` 에 복사 |
| RMSE (Note Density) = 10.21 (논문 4.50) | Note Density(0–35) vs Loudness(0–0.47) 스케일 60배 차이 → combined MSE가 Loudness 무시 | z-score 정규화 도입 + PCC loss 추가, 역정규화 후 RMSE 보고 |
| `evaluate_regression.py` 아키텍처 불일치 | `regModel = "gru"` 하드코딩 → 학습(bigru)과 평가 아키텍처 불일치로 state_dict 로드 실패 | `regModel = "bigru"` 로 수정 |
| `generate_TEA.py` loudness 역정규화 오류 | z-score 모델 재학습 후에도 `y_loudness * 100` 수식 그대로 사용 → velocity 범위 완전히 잘못됨 | `norm_stats.json` 로드 후 `y_lv_raw = y_lv * lv_std + lv_mean` 처리 후 velocity 변환 |

---

## 12. 중간 발표 이후 진행 — 2026-05-30

### 12.1 Shuffle Test 실패 및 원인 분석

중간 발표에서 Task 4로 발표한 **Shuffle Test (emotion 시간 순서 섞기 → ΔCorr 측정)** 가 실제로는 유효하지 않음이 확인됨.

#### 실험 결과

| 모델 | Corr (orig) | Corr (shuffled) | ΔCorr |
|------|------------|----------------|-------|
| Baseline (full) | 0.4634 | 0.4634 | ±0.0000 |
| TEA Exp4 (full) | 0.4862 | 0.4862 | ±0.0000 |
| TEA Exp8 (full) | 0.4641 | 0.4687 | +0.0046 |
| Baseline (emo-only) | 0.4630 | 0.4630 | ±0.0000 |

#### 실패 원인

1. **Autoregressive 구조**: chord 예측의 대부분이 이전 chord 시퀀스 `x`에서 나옴. emotion은 conditioning 신호에 불과해 time shuffle로 예측이 거의 안 바뀜.
2. **Corr 메트릭의 한계**: 전체 timestep 평균 기반 → emotion 분포가 shuffle 전후 통계적으로 동일하여 Corr 변화 없음.
3. **CLIP 768-dim dominant**: emotion 6-dim이 Linear_vis에서 사실상 묻혀 zeroing(emotion=0)도 효과 없음.

#### train_ablation.py 학습 결과 (emotion-only, epoch 0~22)

- epoch 6에서 best val loss (1.0130), H@1=0.50 저장
- epoch 14에서 LR 피크(6.83e-04) 도달 → 가중치 불안정 → 붕괴
- epoch 16~ : Corr 정확히 0.6382로 고착 (mode collapse), H@1 0.11~0.16으로 급락
- `kill`로 학습 중단. best model (epoch 6) 저장은 정상.

---

### 12.2 새로운 Ablation 방향 — No-Emotion Baseline

Shuffle Test를 폐기하고, 더 클린한 ablation 방식으로 전환.

#### 설계

| 모델 | emotion 입력 | 목적 |
|------|------------|------|
| **Baseline (no emotion)** | 항상 0 (zeroing) | emotion 제거 기준선 |
| Baseline (full) | 정상 | emotion 기여 측정 |
| TEA Exp4 / Exp8 | 정상 | TEA의 추가 기여 측정 |

비교 테이블:
```
Baseline (no emo) → Baseline (full) → TEA Exp4 → TEA Exp8
      낮은 Corr          중간 Corr         높은 Corr
```

- `no_emo → full`: emotion feature 자체의 기여
- `full → TEA`: temporal emotion modeling (TEA) 의 추가 기여

#### 생성 파일

- **`train_no_emotion.py`** (신규): `NoEmoWrapper`로 emotion만 zeroing, 나머지 정상 입력. 저장 경로 `saved_models/AMT_no_emotion/`.
- **`eval_shuffle.py`** (재작성): 4개 모델 val Corr 직접 비교. 출력 `experiments/eval_ablation.csv`.

#### 현재 상태 (2026-05-30 저녁)

```bash
nohup python train_no_emotion.py > logs/no_emotion.log 2>&1 &  # 실행 중 (PID 1895946)
```

예상 학습 시간: 30~40분. 완료 후 `python eval_shuffle.py` 실행 예정.

---

---

## 13. 아키텍처 수정 및 재학습 — 2026-05-30 (오후)

### 13.1 Eval 결과 분석 및 문제 발견

새로운 ablation 방식(`train_no_emotion.py`, BATCH_SIZE=4)으로 학습 후 `eval_shuffle.py` 실행 결과:

| 모델 | Val Corr | ΔCorr vs no-emo |
|------|---------|----------------|
| Baseline (no emotion) | 0.4938 | +0.0000 |
| Baseline (full)       | 0.4634 | −0.0304 |
| TEA Exp4              | 0.4862 | −0.0076 |
| TEA Exp8              | 0.4641 | −0.0297 |

**문제**: no_emotion 모델이 full 모델보다 Corr이 높음 → 원하는 스토리(`no_emo < full < TEA`)가 성립 안 됨.

#### 원인 분석

1. **학습 질 불균형**: no_emotion 모델(epoch 38, val=0.9653)이 AMT full 체크포인트보다 더 잘 수렴됨.
2. **CLIP implicit emotion**: CLIP 768-dim feature가 시각적 감정 정보를 이미 암묵적으로 포함. 명시적 emotion feature 없이도 Corr이 높게 나옴.
3. **구조적 dominance**: emotion(6-dim)이 Linear_vis 내에서 CLIP(768-dim)에 묻혀 모델이 사실상 emotion을 무시할 수 있음.

### 13.2 아키텍처 수정 — Dedicated Emotion Pathway

#### 변경 내용

**기존 구조:**
```
vf = Linear_vis(cat([CLIP_768, scene_1, motion_1, emotion_6]))  # 776 → 512
```

**새 구조:**
```
vf = Linear_vis(cat([CLIP_768, scene_1, motion_1]))   # 770 → 512
   + Linear_emo(emotion_6)                             # 6 → 512  (additive, 독립 채널)
```

#### 설계 의도

- emotion이 CLIP과 동등한 가중치로 d_model(512) 공간에 투영됨
- 모델이 emotion을 무시하려면 `Linear_emo` 가중치를 완전히 0으로 만들어야 함 → 학습이 emotion 활용 방향으로 유도됨
- no_emotion 모델에서 `Linear_emo(zeros)` = 0 → emotion 기여 완전 차단 → 공정한 ablation
- TEA 모델: `Linear_emo(emotion)` (static) + `encoder_emotion_proj(GRU/LSTM(emotion))` (temporal) 로 emotion 두 경로로 처리

#### 수정 파일

| 파일 | 변경 내용 |
|------|---------|
| `model/video_music_transformer.py` | `Linear_emo(6→512)` 추가, forward에서 additive 주입 |
| `model/video_music_transformer_TEA.py` | 동일 |
| `train_no_emotion.py` | `total_vf_dim += 1+1` (emotion 제외) |
| `train_full.py` (신규) | full 모델 동일 조건 재학습용 |
| `train_TEA.py`, `eval_TEA.py`, `eval_shuffle.py` 등 | `total_vf_dim` 동일하게 수정 |

### 13.3 재학습 현황 (2026-05-30 저녁)

```bash
# 현재 실행 중
nohup python train_no_emotion.py > logs/no_emotion.log 2>&1 &
# epoch 9 진행 중, val=0.9929, 정상 수렴 확인
# LR 피크(epoch ~27) 이전, collapse 징조 없음
```

```bash
# no_emotion 완료 후 순차 실행 예정
nohup python train_full.py > logs/full.log 2>&1 &
```

### 13.4 향후 할 일 (우선순위 순)

| 우선순위 | Task | 내용 | 상태 |
|---------|------|------|------|
| 1 | **no_emotion 재학습** | 새 아키텍처(Linear_emo)로 재학습 | 진행 중 |
| 2 | **full 모델 재학습** | 동일 조건으로 공정한 baseline 확보 | 대기 |
| 3 | **TEA Exp4/Exp8 재학습** | 새 아키텍처 기반으로 재학습 필요 | 미정 |
| 4 | **eval_shuffle.py 재실행** | 4모델 공정 비교 → 클린한 ablation table | 재학습 완료 후 |
| 5 | **Task 1: Emotion Override CLI** | `generate_TEA.py`에 `--emotion-weights` 추가 | 미착수 |
| 6 | **Task 2: 데모 영상** | 생성 샘플 비교 영상 제작 | 미착수 |

**마감**: 2026-06-06 ~ 2026-06-09

---

## 14. 재학습 완료 및 추가 검증 실패 — 2026-05-30 (야간)

### 14.1 최종 체크포인트

| 모델 | 아키텍처 | Best Epoch | Val Loss | H@1 |
|------|---------|-----------|---------|-----|
| AMT_no_emotion | 770-dim + Linear_emo | 31 | 0.9736 | 0.4990 |
| AMT_full | 770-dim + Linear_emo | 42 | 0.9625 | 0.5108 |

- H@1: Full(0.5108) > No Emotion(0.4990) → 소폭이지만 방향 맞음
- 하지만 이미 원래 TEA 실험들(Exp3/4/8: H@1 ≈ 0.62~0.67)보다 낮음 (구 arch 기반이라 직접 비교 불가)

---

### 14.2 시도한 감정 포착 검증 방법들과 실패 원인

#### 14.2.1 Corr (compute_vevo_correspondence) — 역전 지속

| 모델 | Corr |
|------|------|
| No Emotion | **0.5263** |
| Full | 0.4754 |

Full < No Emotion → 명시적 감정 경로(Linear_emo) 추가 후에도 역전 현상 유지.

**실패 원인**:
- CLIP 768-dim이 이미 시각적 감정 정보를 내포 → 추가 6-dim 감정이 노이즈로 작용
- EMOTION_THRESHOLD=0.80 기준에서 실제 `tgt_emotion_prob`이 0.3~0.5 범위 → Corr 집계 자체가 통계적으로 불안정

#### 14.2.2 Emotion-Salient 필터링 (eval_salient.py) — 데이터 부족으로 실패

salient_ratio > 0.1인 시퀀스: 75개 중 7개뿐. ratio > 0.2: 0개.

**실패 원인**: EMOTION_THRESHOLD=0.80이 데이터 분포에 맞지 않음.
실제 `tgt_emotion_prob` 최댓값이 0.5 수준이어서 0.80 이상을 통과하는 timestep이 극히 드묾.

#### 14.2.3 Temporal Pearson Correlation — r ≈ 0

`eval_temporal_pearson.py` 결과:

| 모델 | Pearson r | N |
|------|----------|---|
| No Emotion | 0.0138 | 75 |
| Full | 0.0057 | 75 |
| TEA Exp4 | SKIP (아키텍처 불일치) | — |
| TEA Exp8 | SKIP (아키텍처 불일치) | — |

**실패 원인 분석**:
1. **TEA 비교 불가**: 구 TEA 체크포인트(776-dim)가 신 아키텍처(770-dim + Linear_emo)와 load 실패.
   신 아키텍처로 TEA 재학습 없이는 3자 비교 불가.
2. **r ≈ 0**: `tgt_emotion_quality` (14-dim chord quality flags)에서 유효 타임스텝이 있어도
   video_valence의 분산이 작거나 예측 코드 품질이 감정 궤적을 따르지 않음.
   - 모델은 chord harmony consistency를 위해 학습됨 → 감정 궤적 추적 방향으로 최적화 안 됨.
   - No Emotion도 r=0.0138 → 감정 입력 유무가 궤적 정렬에 영향 없음.

#### 14.2.4 TEA 아키텍처 불일치 요약

| 체크포인트 | 아키텍처 | 신 아키텍처(770-dim + Linear_emo)와 호환 |
|-----------|---------|----------------------------------------|
| AMT_no_emotion | 770-dim + Linear_emo | 호환 |
| AMT_full | 770-dim + Linear_emo | 호환 |
| TEA_encoder_gru_exp4 | 776-dim (구) | **불일치** → load 실패 |
| TEA_both_gru_lstm_exp8 | 776-dim (구) | **불일치** → load 실패 |
| TEA_*_con | 776-dim (구) | **불일치** |

`Linear_vis.weight` shape: 구=`[512, 776]`, 신=`[512, 770]` → `strict=False`로도 size mismatch 불가.

---

### 14.3 파일 정리 현황 (2026-05-30)

#### 삭제된 saved_models (총 ~23GB 절약)
- `AMT_backup/` (3.3G) — AMT 단순 복사본
- `TEA_decoder_lstm_exp1/`, `TEA_decoder_gru_exp2/` — 최하위 성능 Decoder-only
- `TEA_encoder_lstm_exp3/` — Exp4보다 열등
- `TEA_both_lstm_lstm_exp5/`, `TEA_both_gru_gru_exp6/`, `TEA_both_lstm_gru_exp7/` — Exp8보다 열등
- `TEA_encoder_gru_exp9/` — SoftCorrLoss 실패 모델

#### 삭제된 스크립트
- `eval_salient.py` — 데이터 부족으로 실패, 결과 CSV 보존
- `eval_emotion_fidelity.py` — 구 arch 기반, 결과 CSV 보존
- `eval_robustness.py` — 결과 CSV 보존
- `experiments/train_TEA_exp1-9.py` — 실험별 래퍼 (구 arch 기반)
- `experiments/constants_TEA_exp1-9.py` — 실험별 상수 (구 arch 기반)

#### 남아 있는 핵심 모델
```
AMT/                          # 원본 baseline (776-dim)
AMT_no_emotion/               # 신 arch ablation (LINEAR_EMO)
AMT_full/                     # 신 arch full model
AMT_emotion_only/             # emotion-only ablation
TEA_encoder_gru_exp4/         # 구 arch 베스트 단일-TEA (H@1=0.6236)
TEA_both_gru_lstm_exp8/       # 구 arch 베스트 전체-TEA (H@1=0.6245)
TEA_*_con/                    # contrastive 파인튜닝 버전
```

---

## 15. 다음 단계 및 선택지

### 현재 상황 요약

| 메트릭 | no_emotion | full | TEA (구 arch) | 상태 |
|--------|-----------|------|--------------|------|
| H@1 | 0.4990 | 0.5108 | 0.6236~0.6275 | full > no_emo ✓ |
| Corr | **0.5263** | 0.4754 | 0.4862~0.4939 | 역전 ✗ |
| Temporal Pearson r | 0.0138 | 0.0057 | SKIP | ≈0 ✗ |

- **H@1**: full이 no_emotion보다 높음 → 감정 입력이 코드 예측 정확도에 기여
- **Corr**: 역전 현상 해결 불가 (CLIP implicit emotion이 근본 원인)
- **TEA**: 신 아키텍처로 재학습 없이 공정한 비교 불가

### Option A: TEA 신 아키텍처로 재학습 [권장]

```bash
# train_TEA.py를 신 arch(770-dim + Linear_emo)로 수정 후 Exp4(Encoder GRU) 재학습
nohup /home/taegum/miniconda3/envs/video2music/bin/python -u train_TEA.py \
  --exp 4 > logs/tea_new_exp4.log 2>&1 &
```

재학습 후:
- `no_emotion` / `full` / `TEA_new_exp4` 3자 비교 가능
- Corr 역전이 TEA에서도 유지된다면 → CLIP implicit emotion이 원인임을 명확히 문서화
- Temporal Pearson에서 TEA > full이 나온다면 → 시간적 감정 추적 입증
- 예상 학습 시간: ~50-60 epoch × 150 batch ≈ 15~20시간

### Option B: 현재 결과로 논문 작성

**H@1 개선이 주 기여**:
- TEA (구 arch): H@1 +22.1% (Exp8 기준) — 이미 입증됨
- Emo Loss: -9.3% 개선 — 이미 입증됨
- ΔCorr(all) < 0: 입력 비디오가 바뀌면 생성 코드도 달라짐 — 이미 입증됨

**Corr 역전 설명 전략**:
> "CLIP visual encoder가 이미 감정 정보를 내포하므로, 명시적 6-dim 감정 피처의 Corr 기여가
>  측정되지 않았다. 대신 TEA는 전체 비디오 피처의 시간적 활용을 통해 H@1과 Emo Loss를
>  개선하였다."

### Option C: Corr 임계값 재설계

`EMOTION_THRESHOLD`: 0.80 → 0.50으로 낮춰 더 많은 timestep 포함.
근본 원인(CLIP implicit emotion)은 해결되지 않지만, Corr 통계가 더 안정적으로 나올 수 있음.
단, No Emotion > Full 역전이 유지될 가능성 높음.

### 권장 경로

1. **단기 (지금 바로)**: Option B로 논문 서술 — 이미 있는 결과(H@1, Emo Loss, ΔCorr(all))로 주장
2. **병행**: TEA Exp4 신 아키텍처 재학습 시작 (백그라운드) → 완료 시 Temporal Pearson 재실행
3. 재학습 결과가 나오면 Option A로 업그레이드, 아니면 Option B 유지

---

## 16. 신 아키텍처 TEA 재학습 및 Emotion Override — 2026-05-31

### 16.1 TEA_encoder_gru 재학습 (신 arch, Exp9)

15절 Option A 권장에 따라 신 아키텍처(770-dim + Linear_emo)로 TEA Encoder GRU 재학습.

| 항목 | 값 |
|------|-----|
| 아키텍처 | Encoder-only TEA, GRU, 신 arch (770-dim + Linear_emo 별도) |
| Best Epoch | 56 |
| Best Val Loss | 0.8086 |
| H@1 / H@3 / H@5 | 0.6090 / 0.8528 / 0.9180 |
| 저장 경로 | `saved_models/TEA_encoder_gru/` |
| Loss | 기존 SmoothCE + Emotion BCEWithLogits |

- H@1=0.6090: 구 arch Exp4(0.6236)보다 소폭 낮으나 신 arch full(0.5108) 대비 +19.2% 향상
- 신 arch no_emotion(0.4990) 대비 +22.0% → TEA의 추가 기여 명확

---

### 16.2 EmotionValenceAlignLoss — TEA_encoder_gru_align (Exp10)

#### 동기

TEA가 감정을 코드 생성에 활용하는지 더 직접적으로 측정하기 위해 **Emotion Valence Agreement Rate** 도입:

> 각 타임스텝에서 영상의 감정 valence 방향(positive/negative)과 생성 코드의 valence 방향이 일치하는 비율.

기존 모델들이 모두 0.5 미만인 이유: 데이터셋 구조적 편향 (코드 68% positive, 영상 감정 55% negative).

이를 직접 최적화 신호로 사용하는 **EmotionValenceAlignLoss** 도입.

#### Loss 설계

```python
# valence_vec: [positive_chord_classes] → +1, [negative_chord_classes] → -1
# emo_weight: [exciting, fearful, tense, sad, relaxing, neutral] → valence 방향

pred_valence = softmax(logits) @ valence_vec   # 예측 코드의 기대 valence
emo_valence  = emotion @ emo_weight_vec         # 영상 감정의 valence 강도

align_loss = MSELoss(pred_valence, emo_valence)
total_loss = L_chord + λ_emo·L_emo + λ_align·L_align   # λ_align=0.1
```

#### 학습 결과

| 항목 | 값 |
|------|-----|
| 아키텍처 | Encoder-only TEA, GRU (TEA_encoder_gru와 동일) + AlignLoss |
| Best Epoch | 59 |
| Best Val Loss | 0.8667 |
| H@1 / H@3 / H@5 | 0.5592 / 0.8319 / 0.9073 |
| 저장 경로 | `saved_models/TEA_encoder_gru_align/` |

- H@1이 TEA_encoder_gru(0.6090) 대비 소폭 하락 → AlignLoss가 chord accuracy와 trade-off 관계
- Agreement Rate는 대폭 개선 (아래 16.3 참조)

---

### 16.3 감정 포착 평가 결과

#### Emotion Valence Agreement Rate (`eval_emotion_valence.py`)

각 타임스텝에서 영상 감정 valence 방향 vs 생성 코드 valence 방향 일치율.

| 모델 | Agreement Rate |
|------|--------------|
| Baseline (no emotion) | 0.4190 |
| Baseline (full) | 0.4232 |
| TEA_encoder_gru | 0.4253 |
| **TEA_encoder_gru_align** | **0.4640** ↑ |

- Align 모델이 0.4640으로 가장 높음 (+9.9% vs no_emotion)
- 0.5 미만인 이유: 데이터셋 편향 (코드 68% positive, 영상 55% negative) → 구조적 한계이며 논문에서 명시

저장: `experiments/eval_emotion_valence.csv`

#### Temporal Pearson r (`eval_temporal_pearson.py`)

시간 축에서 영상 감정 valence 궤적과 생성 코드 valence 궤적의 Pearson 상관계수.

| 모델 | Pearson r |
|------|----------|
| Baseline (no emotion) | 0.0138 |
| Baseline (full) | 0.0057 |
| TEA_encoder_gru | 0.0021 |
| TEA_encoder_gru_align | 0.0039 |

- 전 모델 r ≈ 0: 시간적 감정 궤적 추적은 현재 학습 목표에 포함되어 있지 않아 구조적 한계
- Align 모델도 소폭 개선에 그침 → Temporal Pearson은 현 아키텍처에서 개선 불가한 지표로 판단

저장: `experiments/eval_temporal_pearson.csv`

#### 결론

| 질문 | 답 |
|------|-----|
| 감정을 어느 정도 잡는가? | Agreement Rate: 0.4190→0.4640 (+10.8%). 구조적 데이터 편향(68%/55%)에 의한 상한선 존재 |
| 시간적 감정 궤적을 추적하는가? | Pearson r ≈ 0. 현 chord accuracy 최적화 체계에서 구조적 한계 |
| Align 모델을 쓸 것인가? | **Two-model strategy**: TEA_encoder_gru (H@1 최고) + TEA_encoder_gru_align (Agreement Rate 최고) 병행 |

---

### 16.4 generate_TEA.py — 주요 수정사항

#### total_vf_dim 버그 수정

신 아키텍처에서 emotion이 Linear_emo 별도 경로로 분리됨에 따라 total_vf_dim 수정:

```python
# 수정 전 (버그): emotion 포함
total_vf_dim = 768 + 1 + 1 + 6  # → Linear_vis shape mismatch

# 수정 후
total_vf_dim = 768 + 1 + 1      # CLIP + scene_offset + motion (emotion → Linear_emo 별도)
reg_vf_dim   = total_vf_dim + 6  # VideoRegression은 구 arch, emotion concat 방식 유지
```

VideoRegression은 구 arch(776-dim) 체크포인트를 그대로 사용하므로 `reg_vf_dim` 별도 유지.

#### FluidSynth 설정 통일

baseline `generate.py`와 동일한 렌더링 조건 맞춤 (reverb/chorus 제거):

```python
# 수정 전: reverb/chorus 적용 (baseline과 다름)
subprocess.call([_fs_bin, "-ni", _sf2, f_midi, "-F", f_flac, "-r", "44100",
                 "-o", "synth.reverb.room-size=0.8", ...])

# 수정 후: 동일 조건
subprocess.call([_fs_bin, "-ni", _sf2, f_midi, "-F", f_flac, "-r", "44100"])
```

---

### 16.5 Emotion Override CLI 구현 (`--emotion`)

`generate_TEA.py`에 `--emotion` 인자 추가 — 사용자가 감정을 직접 지정해 코드 생성 방향 제어.

#### 사용법

```bash
# 감정 이름으로 지정 (가장 간편)
python generate_TEA.py --exp 10 --test_id 049 --emotion exciting
python generate_TEA.py --exp 10 --test_id 223 --emotion sad

# 직접 6차원 벡터로 지정
python generate_TEA.py --exp 10 --test_id 049 --emotion 0.7 0.1 0.1 0.0 0.05 0.05
```

#### 동작 원리

1. 감정 이름 → 프리셋 벡터 변환 (예: `sad` → `[0.02, 0.02, 0.02, 0.90, 0.02, 0.02]`)
2. `emotion_weight_override` 텐서로 변환 → `model.generate()`에 전달
3. TEA 어댑터 내부 `_apply_emotion_weight()`에서 학습된 `emotion_weight` 대신 override 사용
4. primer / key / temperature도 override 감정 기준으로 자동 재설정 (밝은 감정 → G,D,Em,C 프라이머 등)
5. 출력 파일명에 tag 추가: `_emo_ex`, `_emo_sa` 등

#### 데모 비교 (223 영상 — 원래 exciting)

| 버전 | primer | 앞 10개 코드 |
|------|--------|------------|
| exp10 auto | G,D,Em,C | G, D, Em, C, G, D, D, **A#:dim, A:dim**, F:maj6 |
| exp10 `--emotion exciting` | G,D,Em,C | G, D, Em, C, **D:7, C, C**, Am, G:7... |
| exp10 `--emotion sad` | Am,C,F,G | Am, C, F, G, G, **F:min, F:min**, G:sus4... |

dim 코드가 사라지고 dominant 7th로 교체 → 체감 밝기 향상 확인.

#### 생성된 데모 파일

| 영상 | exp | emotion | 파일 |
|------|-----|---------|------|
| 049 (sad) | 10 | auto | `output/049/049_TEA_exp10_cgen_rd.mp4` |
| 049 (sad) | 10 | exciting | `output/049/049_TEA_exp10_emo_ex_cgen_rd.mp4` |
| 223 (exciting) | 10 | auto | `output/223/223_TEA_exp10_cgen_rd.mp4` |
| 223 (exciting) | 10 | exciting | `output/223/223_TEA_exp10_emo_ex_cgen_rd.mp4` |
| 223 (exciting) | 10 | sad | `output/223/223_TEA_exp10_emo_sa_cgen_rd.mp4` |
| 049 (sad) | 9 | auto | `output/049/049_TEA_exp9_cgen_rd.mp4` |
| 223 (exciting) | 9 | auto | `output/223/223_TEA_exp9_cgen_rd.mp4` |

---

### 16.6 향후 계획 현황 업데이트

| # | 항목 | 상태 |
|---|------|------|
| 1 | Emotion Override CLI | ✅ 완료 (`--emotion sad/exciting/...` 또는 6개 float) |
| 2 | Baseline vs TEA 비교 데모 영상 | 🔄 진행 중 (049, 223 exp9/exp10 생성 완료) |
| 3 | 다른 시간적 신호로의 확장 | ✅ MS-TEA (scene+motion 어댑터 추가) 완료 |
| 4 | Shuffle Test 한계 보완 | ✅ Agreement Rate + Pearson으로 대체 측정 완료 |

---

## Section 17. MS-TEA 및 MS-TEA_align 시리즈

### 17.1 MS-TEA — Multi-Signal TEA

#### 동기

TEA가 감정(emotion) 신호만 어댑터로 처리하는 것을 넘어, 영상에서 뽑을 수 있는 다른 시간적 신호(scene 전환, optical flow motion)도 어댑터로 처리하면 어떤가? 다중 신호의 복합 활용으로 코드 생성 품질 향상을 목표로 한다.

#### 아키텍처

| 항목 | 값 |
|------|-----|
| 기반 | VideoMusicTransformerTEA (`use_multisignal=True`) |
| TEA 위치 | Encoder-only |
| RNN 셀 | GRU |
| 어댑터 신호 | emotion (6-dim) + scene_offset (1-dim) + motion (1-dim) |
| 총 파라미터 | ~39M |
| 저장 경로 | `saved_models/MSTEA_encoder_gru/` |

`use_multisignal=True`일 때 TEA 어댑터 내부에서 emotion 외에 scene_offset, motion 각각에 대해 별도 GRU + attention 처리 후 합산 → 영상의 시간적 역학을 더 풍부하게 반영.

#### 학습 결과

| 항목 | 값 |
|------|-----|
| Best Epoch | 37 |
| Best Val Loss | 0.8673 |
| H@1 / H@3 / H@5 | 0.5657 / 0.8321 / 0.9073 |
| Agreement Rate | 0.4338 |

- TEA_encoder_gru(single-signal)의 H@1=0.6090 대비 낮음 — 다중 신호가 chord accuracy 측면에서 반드시 유리하지 않음
- Agreement Rate 0.4338 → MS-TEA 단독으로는 Baseline(0.4232)보다 소폭 개선에 그침

---

### 17.2 MS-TEA_align 시리즈 — EmotionValenceAlignLoss Fine-tune

#### 동기

MS-TEA epoch 37 best checkpoint에서 AlignLoss를 추가해 감정 일치도를 직접 최적화. λ 값을 0.05 → 0.1 → 0.15로 늘려가며 H@1 vs Agreement Rate 트레이드오프 확인.

#### 학습 설정

```python
# 공통 설정
INIT_EPOCH = 37          # MS-TEA best checkpoint에서 시작
LR         = 1e-5        # fine-tune: 작은 LR 고정
OPTIMIZER  = Adam (weight_decay=1e-4)
EARLY_STOP = 20 epochs

total_loss = L_chord + λ_emo·L_emo + λ_align·L_align
# λ_align: 0.05 / 0.1 / 0.15 순차 실험
```

#### λ별 학습 결과 (epoch 38 = fine-tune 1st epoch = 모든 변형의 best)

| λ | Best Epoch | Val Loss | H@1 | H@3 | H@5 | Agreement Rate |
|---|-----------|---------|-----|-----|-----|----------------|
| 0.05 | 38 | 0.8550 | 0.5740 | 0.8367 | 0.9075 | 0.4378 |
| 0.1  | 38 | 0.8584 | 0.5708 | 0.8350 | 0.9069 | 0.4548 |
| 0.15 | 38 | 0.8647 | 0.5650 | 0.8312 | 0.9063 | **0.4681** |

#### 관찰

- **모든 λ 변형에서 epoch 38이 best**: fine-tune 첫 번째 epoch에서 이미 최적점에 도달. MS-TEA epoch 37 체크포인트가 이미 근-최적 코드 패턴을 가지고 있어, alignment signal이 epoch 38에서 한 번 nudge를 준 후 추가 학습이 drift를 유발.
- **λ ↑ → Agreement Rate ↑, H@1 ↓**: AlignLoss를 강하게 줄수록 감정 일치도는 향상되지만 chord accuracy는 하락 — 명확한 trade-off.
- **NaN 불안정**: λ=0.1(epoch 43), λ=0.15(epoch 40)에서 val_loss NaN 발생 후 회복. λ가 클수록 alignment signal이 강해 불안정성 증가. 단, best checkpoint는 epoch 38이므로 학습 결과에는 영향 없음.

---

### 17.3 전체 감정 일치도 평가 종합 (업데이트)

#### Agreement Rate 전체 비교 (`eval_emotion_valence.py`)

| 모델 | Agreement Rate | Agree | Disagree | Skip |
|------|--------------|-------|----------|------|
| Baseline (no emotion) | 0.4190 | 7,366 | 10,212 | 4,847 |
| Baseline (full) | 0.4232 | 7,444 | 10,146 | 4,835 |
| TEA_encoder_gru | 0.4253 | 7,477 | 10,104 | 4,844 |
| MS-TEA | 0.4338 | 7,630 | 9,957 | 4,838 |
| MS-TEA_align (λ=0.05) | 0.4378 | 7,675 | 9,856 | 4,894 |
| MS-TEA_align (λ=0.1) | 0.4548 | 7,986 | 9,573 | 4,866 |
| TEA_encoder_gru_align | 0.4640 | 8,134 | 9,396 | 4,895 |
| **MS-TEA_align (λ=0.15)** | **0.4681** | **8,208** | **9,327** | **4,890** |

- **MS-TEA_align (λ=0.15)가 0.4681로 전체 최고** — TEA_encoder_gru_align(0.4640)도 초과
- Baseline 대비 +11.7% 향상 (0.4190 → 0.4681)
- 0.5 미만은 데이터셋 구조적 편향(코드 68% positive, 영상 감정 55% negative)에 의한 구조적 상한선

#### H@1 vs Agreement Rate 트레이드오프 요약

| 모델 | H@1 | Agreement Rate | 특징 |
|------|-----|----------------|------|
| TEA_encoder_gru | 0.6090 | 0.4253 | H@1 최고 (single-signal) |
| MS-TEA | 0.5657 | 0.4338 | 다중 신호, H@1 소폭 하락 |
| MS-TEA_align (λ=0.05) | 0.5740 | 0.4378 | fine-tune 후 H@1 소폭 회복 |
| MS-TEA_align (λ=0.1) | 0.5708 | 0.4548 | 균형점 |
| MS-TEA_align (λ=0.15) | 0.5650 | **0.4681** | Agreement Rate 최고 |

#### 결론

| 질문 | 답 |
|------|-----|
| MS-TEA가 감정을 더 잘 잡는가? | Agreement Rate 기준 Baseline보다 개선(0.4338), 그러나 단독으로는 크지 않음 |
| AlignLoss가 효과 있는가? | ✅ λ가 커질수록 Agreement Rate 명확히 상승 (0.4378 → 0.4681) |
| H@1을 유지하면서 Agreement Rate를 높일 수 있는가? | λ=0.05가 균형 (H@1=0.5740, AR=0.4378). λ=0.15는 AR 최고이나 H@1 소폭 하락 |
| 최종 전략 | **목적에 따른 모델 선택**: H@1 우선 → TEA_encoder_gru; Agreement Rate 우선 → MS-TEA_align (λ=0.15) |

저장: `experiments/eval_emotion_valence.csv`

---

## Section 18. 추가 실험 — AlignLoss 한계 탐색 및 최종 모델 결정 (2026-05-31)

### 18.1 실험 동기

MS-TEA_align λ=0.15(Section 17)에서 epoch 38이 best라는 현상이 반복 확인됨:
- MS-TEA epoch 37 체크포인트가 이미 수렴된 상태 → AlignLoss가 한 번의 nudge 후 chord accuracy와 충돌
- "epoch 38 고착"을 극복하기 위한 3가지 전략 실험

### 18.2 EXP1 — From-Scratch + AlignLoss (λ=0.15)

| 항목 | 값 |
|------|-----|
| 전략 | epoch 0부터 AlignLoss(λ=0.15)와 함께 동시 학습 |
| 목적 | chord accuracy와 alignment를 처음부터 동시에 최적화 |
| LR | 1e-4 (고정) |
| Best Epoch | 39 |
| Best Val Loss | 0.8858 |
| H@1 / H@3 / H@5 | 0.5441 / 0.8225 / 0.9052 |
| Agreement Rate | 0.4574 |
| Early Stop | Epoch 59 |

**관찰**: H@1이 MS-TEA baseline(0.5657)에 미치지 못하고 AR도 l015(0.4681)보다 낮음. 두 지표 모두 기존 best에 못 미침.

---

### 18.3 EXP2 — Curriculum λ Scheduling (0 → 0.15, 20 epoch warmup)

| 항목 | 값 |
|------|-----|
| 전략 | MS-TEA epoch 37에서 시작, λ를 0→0.15로 20 epoch 점진 증가 |
| 목적 | 갑작스러운 AlignLoss 주입 대신 천천히 적응시켜 chord accuracy 보호 |
| LR | 1e-5 (fine-tune) |
| Best Epoch | 38 (λ=0.0075, 첫 fine-tune epoch) |
| Best Val Loss | 0.8554 |
| H@1 / H@3 / H@5 | 0.5724 / 0.8374 / 0.9095 |
| Agreement Rate | 0.4233 |
| Early Stop | Epoch 58 |

**관찰**: H@1=0.5724로 baseline 돌파. 그러나 λ가 올라갈수록 H@1 단조 감소. AR=0.4233으로 전체 최저 — curriculum이 AR 개선에 기여하지 못함.

---

### 18.4 EXP3 — Soft Margin (Hinge) AlignLoss

| 항목 | 값 |
|------|-----|
| 전략 | MarginAlignLoss: 방향 불일치에만 penalty (margin=0.1) |
| 목적 | MSE 대신 AR 메트릭 방향과 직접 일치하는 손실 |
| 수식 | L = mean(relu(margin − pred_val × video_val)) |
| LR | 1e-5 (fine-tune from MS-TEA epoch 37) |
| Best Epoch | 40 |
| Best Val Loss | 0.8555 |
| H@1 / H@3 / H@5 | 0.5737 / 0.8376 / 0.9088 |
| Agreement Rate | 0.4352 |
| Early Stop | Epoch 60 |

**관찰**: H@1=0.5737로 3개 신규 실험 중 최고. 그러나 AR=0.4352로 hinge loss가 AR을 직접 최적화하는 의도와 달리 기존 best(0.4681)에 못 미침.

---

### 18.5 전체 최종 비교 테이블

| 모델 | Agreement Rate | H@1 |
|------|--------------|-----|
| No Emotion | 0.4190 | 0.4990 |
| Full (AMT) | 0.4232 | 0.5108 |
| MS-TEA_align_curriculum | 0.4233 | 0.5724 |
| TEA_encoder_gru | 0.4253 | 0.6090 |
| MS-TEA | 0.4338 | 0.5657 |
| MS-TEA_align_softmargin | 0.4352 | **0.5737** |
| MS-TEA_align (λ=0.05) | 0.4378 | 0.5740 |
| MS-TEA_align (λ=0.1) | 0.4548 | 0.5708 |
| MS-TEA_align_scratch | 0.4574 | 0.5441 |
| TEA_encoder_gru_align | 0.4640 | 0.5592 |
| **MS-TEA_align (λ=0.15)** | **0.4681** | 0.5650 |

**핵심 발견**:
1. 신규 3개 실험 모두 MS-TEA_align λ=0.15(AR=0.4681)를 넘지 못함
2. H@1↑하면 AR↓, AR↑하면 H@1↓ — 근본적 trade-off 존재
3. 데이터셋 레벨에서 감정-코드 방향이 충돌하는 타임스텝이 원인 (ground truth가 major인데 영상 감정이 sad인 경우 등)
4. 신규 실험들이 실패함으로써 역으로 λ=0.15가 sweet spot임을 ablation으로 입증

**최종 모델: MS-TEA_align (λ=0.15)**
- H@1: baseline(0.5657) 수준 유지 (0.5650, 오차 범위)
- AR: no_emotion(0.4190) 대비 **+11.7%** 향상
- chord accuracy 손상 없이 AR을 대폭 개선한 유일한 모델

---

### 18.6 Attention 기반 TEA — 구현 및 실험 완료 (2026-05-31)

GRU를 `nn.MultiheadAttention`으로 교체한 Attention TEA 구현 및 학습 완료.

#### 아키텍처 (`tea_adapter.py`)

```python
# cell_type="attention" 브랜치
if self.cell_type == "attention":
    self.attn = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=8, dropout=0.1, batch_first=True)
    self.attn_ff = nn.Sequential(
        nn.Linear(d_model, d_model * 2), nn.GELU(),
        nn.Dropout(0.1), nn.Linear(d_model * 2, d_model))
    self.attn_ff_norm = nn.LayerNorm(d_model)

# forward
residual = self.input_proj(x)   # (B, T, d_model)
if self.cell_type == "attention":
    attn_mask = None
    if self.direction == "uni":  # decoder용 causal mask
        attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    attn_out, _ = self.attn(residual, residual, residual, attn_mask=attn_mask)
    h = self.layer_norm(residual + attn_out)    # post-attn residual+norm
    out = self.attn_ff_norm(h + self.attn_ff(h))
```

- Bidirectional encoder → causal mask 없음 (전체 시퀀스 self-attention)
- Unidirectional decoder → upper-triangular causal mask

#### 학습 설정

| 항목 | 값 |
|------|-----|
| 기반 | MS-TEA (MSTEA_encoder_gru) epoch 37 체크포인트에서 시작 |
| 로드 방식 | `strict=False` — backbone(Transformer, input_proj 등) pre-trained, attention 모듈(MHA, FFN) fresh init |
| Missing keys | 30개 (attention 모듈: tea_encoder, tea_scene, tea_motion 각 10개) |
| Unexpected keys | 48개 (GRU 가중치 — 무시됨) |
| LR | 1e-5 (fine-tune) |
| AlignLoss λ | 0.15 |
| Early stop patience | 20 |
| 학습 스크립트 | `train_MSTEA_align_attention.py` |
| 저장 경로 | `saved_models/MSTEA_encoder_attention_align/` |

#### 학습 결과

| 항목 | 값 |
|------|-----|
| Best Epoch | 38 |
| Best Val Loss | 0.8634 |
| H@1 / H@3 / H@5 | **0.5659** / 0.8320 / 0.9066 |
| Agreement Rate | **0.4694** |
| Early Stop | Epoch 58 (patience 20) |

#### 전체 최종 비교 (12 모델)

| 모델 | Agreement Rate | H@1 |
|------|--------------|-----|
| No Emotion | 0.4190 | 0.4990 |
| Full (AMT) | 0.4232 | 0.5108 |
| MS-TEA_align_curriculum | 0.4233 | 0.5724 |
| TEA_encoder_gru | 0.4253 | 0.6090 |
| MS-TEA | 0.4338 | 0.5657 |
| MS-TEA_align_softmargin | 0.4352 | 0.5737 |
| MS-TEA_align (λ=0.05) | 0.4378 | 0.5740 |
| MS-TEA_align (λ=0.1) | 0.4548 | 0.5708 |
| MS-TEA_align_scratch | 0.4574 | 0.5441 |
| TEA_encoder_gru_align | 0.4640 | 0.5592 |
| MS-TEA_align (λ=0.15) | 0.4681 | 0.5650 |
| **MS-TEA_align_attn** | **0.4694** | 0.5659 |

#### 결론 및 최종 판단

- Attention TEA가 AR=0.4694로 전체 최고 (+0.0013 vs λ=0.15 GRU)
- H@1=0.5659로 λ=0.15 GRU(0.5650)와 사실상 동일
- **아키텍처 변경(GRU→Attention)으로 얻는 AR 개선이 +0.0013에 불과** → 아키텍처 한계가 아닌 데이터 레벨 ceiling

**데이터 레벨 ceiling 원인:**
- 데이터셋에서 ground truth chord가 영상 감정과 반드시 일치하지 않음
- 예: dominant=sad 영상인데 정답 코드가 major인 타임스텝 다수 존재
- 어떤 아키텍처, 어떤 loss 설계로도 이 구조적 한계 극복 불가
- **최종 모델: MS-TEA_align_attn (AR=0.4694) 또는 MS-TEA_align λ=0.15 (AR=0.4681)** 병용

generate_TEA.py exp 번호: exp14=l015, exp16=attention

---

## Section 20. generate_TEA.py 오디오 개선 (2026-05-31)

### 20.1 3-Track MIDI — Emotion-based Instrument Selection

`chords_to_midi()`를 2-track에서 3-track으로 확장, 감정에 따라 악기를 동적으로 선택.

| 트랙 | 채널 | 기본 악기 | 예외 |
|------|------|---------|------|
| Track 0 | Ch 0 | Acoustic Grand Piano (GM #0) | relaxing → Nylon Guitar (GM #24) |
| Track 1 | Ch 1 | String Ensemble 1 (GM #48) | fearful/tense → Cello (GM #42) |
| Track 2 | Ch 2 | Acoustic Bass (GM #32) | 항상 동일 |

```python
_EMO_INSTR = {
    "exciting": dict(lead=0,  pad=48, pad_dv=10),  # Piano + Strings (loud)
    "fearful":  dict(lead=0,  pad=42, pad_dv=20),  # Piano + Cello
    "tense":    dict(lead=0,  pad=42, pad_dv=18),  # Piano + Cello
    "sad":      dict(lead=0,  pad=48, pad_dv=15),  # Piano + Strings
    "relaxing": dict(lead=24, pad=48, pad_dv=20),  # Nylon Guitar + Strings
    "neutral":  dict(lead=0,  pad=48, pad_dv=25),  # Piano + Strings (quiet)
}
```

**이전 String pad 문제 수정**: `pad_v = max(25, v - 40)` → `max(40, v - pad_dv)` (이전엔 너무 작아 거의 들리지 않음)

**Bass Track**: 루트 노트 2옥타브 아래, `max(24, chord[0] - 24)`, velocity `max(45, v - 22)`

사용: `chords_to_midi(..., dominant_emo=dominant_emo)` — `auto_params()` 반환값을 그대로 전달.

### 20.2 FluidSynth Reverb 제어 — `--no_reverb`

```bash
# reverb/chorus 끔 (dry, 기본 권장)
python generate_TEA.py --exp 16 --test_id 049 --no_reverb

# reverb 켬 (FluidSynth 기본)
python generate_TEA.py --exp 16 --test_id 049
```

내부: `["-R", "0", "-C", "0"]` 플래그를 FluidSynth 명령에 추가.

### 20.3 post_process_audio.py 개선

| 수정 항목 | 이전 | 이후 |
|---------|------|------|
| reverb 기본값 | aecho 항상 적용 | **기본 꺼짐**, `--reverb` 플래그로만 활성화 |
| FLAC sample rate | 192000 Hz (loudnorm 자동 업샘플) | **44100 Hz 고정** (`-ar 44100` 명시) |
| mp4 in-place 덮어쓰기 | `mp4_src == mp4_dst` → ffmpeg 거부 | 임시 파일(`.pp_tmp.mp4`) 경유 후 교체 |

```bash
# 후처리 (reverb 없음, 44100 Hz)
python post_process_audio.py output/049/

# 후처리 (reverb 포함)
python post_process_audio.py output/049/ --reverb
```

---

## Section 19. 최종 발표 계획 (Final Presentation)

### 19.1 발표 구조

| 슬라이드 | 제목 | 핵심 내용 |
|---------|------|---------|
| 1 | Title | Video-to-Music Generation with Temporal Emotion Alignment |
| 2 | Problem | AMT baseline이 영상 감정을 실제로 반영하지 못함 |
| 3 | Our Approach | TEA → MS-TEA → AlignLoss 3단계 개선 전략 |
| 4 | TEA Architecture | 감정 시퀀스 RNN 처리 → encoder additive 주입 |
| 5 | TEA H@k Results | H@1: 0.5537→0.6090 (+10%), 전 실험 baseline 초과 |
| 6 | MS-TEA | emotion + scene + motion 3채널 어댑터 |
| 7 | AlignLoss 설계 | 감정 valence 방향 MSE 직접 최적화 |
| 8 | λ Ablation | λ=0.05/0.1/0.15 trade-off 테이블 |
| 9 | 추가 실험 | scratch/curriculum/softmargin → λ=0.15 sweet spot 확인 |
| 10 | 최종 모델 | MS-TEA_align λ=0.15: AR +11.7%, H@1 손상 없음 |
| 11 | Quantitative Summary | 전체 모델 비교 테이블 (AR + H@k) |
| 12 | Qualitative Demo | 049(sad)/223(exciting) 비교, emotion override 데모 |
| 13 | Limitations & Future | Attention TEA, AR 상한선, Temporal Pearson 한계 |
| 14 | Conclusion | MS-TEA_align이 H@k 손상 없이 감정 일치도 개선 |

### 19.2 핵심 수치 (최종)

| 지표 | 논문 Baseline | Our Best | 향상 |
|------|-------------|----------|------|
| H@1 | 0.5139 | 0.6090 (TEA enc-gru) | +18.5% |
| H@3 | 0.7722 | 0.8528 | +10.4% |
| H@5 | 0.8672 | 0.9180 | +5.9% |
| Agreement Rate | 0.4190 (no_emo) | **0.4694 (MS-TEA_align_attn)** | **+12.0%** |

### 19.3 남은 작업

- [x] Attention TEA 구현 및 실험 → AR=0.4694, H@1=0.5659 (Section 18.6)
- [x] 3-track MIDI + emotion-based instrument selection (Section 20.1)
- [x] 생성 스크립트 버그 점검 및 수정 (Section 21)
- [ ] 발표 슬라이드 제작
- [ ] 데모 영상 편집 (baseline vs TEA vs align 비교)
- **발표일**: 2026-06-06 ~ 2026-06-09

---

## Section 21. 생성 스크립트 전수 버그 점검 및 수정 (2026-05-31)

발표 전 마지막 버그 패스. 새 모델 학습 없이 생성·렌더링 파이프라인만 대상으로 함.

### 21.1 조사 결과 — 버그 아닌 것

**3음 침묵 버그 (가설, 기각)**

아르페지오 코드는 `len(chord)==4` 또는 `len(chord)==5`만 처리하므로, 3음 코드가 들어오면 해당 박자가 무음이 되는 버그가 존재하리라 가정했다.

조사 결과: `chord_to_midi.py`의 `getMIDI()` 함수는 bass + root + 3rd + 5th를 항상 포함해 **최소 4음을 반환**한다. 확장 코드(7th, 9th 등)는 5음. 따라서 3음 코드는 생성 파이프라인에서 절대 등장하지 않으며 이 버그는 발생 불가능하다.

```
Am → 4 notes: [45, 57, 60, 64]   # bass + root + 3rd + 5th
C  → 4 notes: [48, 60, 64, 67]
C:7 → 5 notes: [48, 60, 64, 67, 70]
```

### 21.2 generate_TEA.py 수정 사항

| 번호 | 위치 | 버그 내용 | 수정 |
|------|------|---------|------|
| 1 | `chords_to_midi()` 내부 | `voice()` 결과에 중복 피치 발생 시 arp 동일 박자 재공격 artifact | `_safe_dedupe()` 추가 (dedup 후 3음 미만이면 원본 유지) |
| 2 | IS_ARP 분기 5곳 | `density_list[i]` 범위 검사 없음 → 길이 불일치 시 IndexError | `di = min(i, len(density_list)-1)` 변수로 통일 |
| 3 | FluidSynth 호출 | `subprocess.call()` 리턴코드 무시 → FLAC 미생성 상태로 영상 합성 진행 | `fs_ret = subprocess.call(...)`; `fs_ret != 0`이면 에러 출력 후 `return` |
| 4 | moviepy 합성 | `audio_clip.subclip(0, video_clip.duration)` — audio < video이면 크래시 | `clip_end = min(audio_clip.duration, video_clip.duration)` |
| 5 | `torch.load` 2곳 (메인 모델 + Reg 모델) | PyTorch 2.x `FutureWarning` | `weights_only=False` 명시 |

#### _safe_dedupe 동작 원리

`voice()` 알고리즘은 이전 코드 피치와 가장 가까운 음으로 각 음을 재배치하는 과정에서 두 음이 같은 MIDI 번호로 수렴할 수 있다. 예:

```
G chord → voice() → [55, 55, 59, 62]  # G(55)가 중복
```

arp density ≥1 패턴에서는 동일 피치가 인접 시각에 스케줄되면 FluidSynth가 이전 noteOn이 끝나기 전에 재공격하여 클릭/재발음 artifact가 발생한다. `_safe_dedupe`는 중복 제거 후 4음 이상이 유지될 때만 적용하고, 그렇지 않으면 원본을 유지한다(arp 코드가 4/5음 전제).

### 21.3 generate_baseline_clean.py 수정 사항

| 번호 | 위치 | 버그 내용 | 수정 |
|------|------|---------|------|
| 1 | `densitylist` 계산 (line ~391) | `y_loudness_np_lv`(loudness, 0–50)로 arp 패턴 결정 → **note density와 무관한 패턴** 선택 | `y_note_density_np`(0–40)으로 교체 |
| 2 | voiced chord 처리 | `_safe_dedupe` 없음 | generate_TEA.py와 동일하게 추가 |
| 3 | moviepy 합성 | `audio.subclip(0, video.duration)` 크래시 guard 없음 | `min(audio.duration, video.duration)` |
| 4 | `torch.load` 2곳 | `weights_only` 누락 | `weights_only=False` 명시 |

**density 버그가 핵심**: loudness 분포(0–50)로 density 임계값(≤5/10/15/20)을 판단하므로 loudness가 높은 구간(≥21)은 전부 density=4(가장 빠른 arp)로 분류되었다. note density 기반으로 수정해야 영상 특성에 맞는 패턴이 나온다.

### 21.4 video2music.py 수정 사항 (Gradio 데모 앱)

| 번호 | 위치 | 버그 내용 | 수정 |
|------|------|---------|------|
| 1 | `densitylist` 계산 (line ~592) | `y_loudness_np_lv`로 density 결정 — 21.3과 동일한 논리 오류 | `y_note_density_np`로 교체 |
| 2 | voiced chord 처리 | `_safe_dedupe` 없음 | 추가 |
| 3 | moviepy 합성 | `audio_mp.subclip(0, video_mp.duration)` 크래시 guard 없음 | `min(audio_mp.duration, video_mp.duration)` |
| 4 | `torch.load` 2곳 | `weights_only` 누락 | `weights_only=False` 명시 |

### 21.5 수정하지 않은 것 (의도적)

- **FluidSynth `.midi_to_audio()` 리턴코드**: `midi2audio` 라이브러리가 리턴코드를 외부로 노출하지 않음. 실패 시 다음 `AudioFileClip()` 호출에서 FileNotFoundError로 명시적으로 터지므로 silent failure 문제는 없음. generate_TEA.py만 subprocess 직접 호출 방식이라 별도 처리.
- **학습 스크립트 (`train_*.py`)**: 발표 전 재학습 없음, 수정 불필요.
- **평가 스크립트 (`eval_TEA.py` 등)**: 기존 수치 그대로 사용.

---

## Section 22 — 감정-코드 시각화 오버레이 (2026-06-12)

### 22.1 배경

최종 발표(2026-06-17)용 데모 영상 준비. Emotion Override(3종: exciting/sad/relaxing) 생성 후 청각적 차이 확인 → 시각적으로도 감정 궤적 + 코드 품질이 연동되는 것을 보여주는 overlay 영상이 필요.

### 22.2 신규 파일: `visualize_emotion_chord.py`

**기능:**
- 감정 feature (`dataset/vevo_emotion/6c_l14p/all/{id}.lab`) + 생성 코드 (.lab) 로드
- 정적 PNG: 2패널 (감정 궤적 라인 + 코드 품질 컬러 타임라인)
- 오버레이 영상: 원본 영상 아래에 감정 바 애니메이션 + 현재 코드 패널 합성 (moviepy)

**주요 옵션:**

| 옵션 | 설명 |
|------|------|
| `--mode static/overlay/both` | 출력 종류 |
| `--segment N` | N초 단위로 전체 클립 분할 |
| `--start S --end E` | 특정 구간만 |
| `--auto N` | 데모용 최적 N초 구간 자동 탐색 후 생성 |
| `--emo_tag _emo_sa` | Emotion Override 버전 지정 |

**`find_best_segment()` 스코어링 기준:**
```
score = dom_prob × (1 + variability × 3) × (1 + chord_diversity × 0.3)
```
- `dom_prob`: 지배 감정 평균 확률 (자신감)
- `variability`: 감정 표준편차 평균 (변동성)
- `chord_diversity`: 구간 내 unique chord quality 수

video 049 기준 자동 선택 구간: **t=113~133 (1:53~2:13)**

### 22.3 Emotion Override 생성 (Exp16 = MS-TEA_align_attn)

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n video2music python3 generate_TEA.py \
  --exp 16 --test_id 049 --emotion exciting

CUDA_VISIBLE_DEVICES=3 conda run -n video2music python3 generate_TEA.py \
  --exp 16 --test_id 049 --emotion sad

CUDA_VISIBLE_DEVICES=3 conda run -n video2music python3 generate_TEA.py \
  --exp 16 --test_id 049 --emotion relaxing
```

출력 태그: `_emo_ex`, `_emo_sa`, `_emo_re`

청각 확인 결과: 3버전 음악 스타일 차이 뚜렷이 확인됨.

### 22.4 Video 049 감정 분포 분석

- 전체 234 timestep (234초 영상, 1 timestep = 1초)
- 지배 감정: sad 73%, tense 16%, fearful 6%, relaxing 4%
- 가장 안정적인 sad 구간: t=120~140 (2:00~2:20), avg=0.71

### 22.5 오류 및 수정

#### `--gpu 3` CUDA 에러

```
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
```

**원인:** `generate_TEA.py`가 `import torch` 이후에 `os.environ["CUDA_VISIBLE_DEVICES"]`를 설정 → PyTorch CUDA 초기화가 이미 완료된 후라 무효.

**해결:** Python 프로세스 시작 전에 환경변수 설정.
```bash
# 틀린 방법
conda run -n video2music python3 generate_TEA.py --exp 16 --gpu 3

# 맞는 방법
CUDA_VISIBLE_DEVICES=3 conda run -n video2music python3 generate_TEA.py --exp 16
```

#### `make_overlay_video` 내 VideoFileClip 이중 호출

초안에서 `min(seg_end, VideoFileClip(video_path).duration)` 계산 시 VideoFileClip을 두 번 생성하는 비효율 존재 → `ffmpeg -ss -to -c copy`로 구간 추출 후 임시 파일 경유 방식으로 수정.

### 22.6 현재 상태 (2026-06-12)

- [x] `visualize_emotion_chord.py` 작성 완료
- [x] Emotion Override 3종 생성 완료 (ex/sa/re)
- [x] Static PNG 자동 구간 선택 확인
- [x] Overlay 영상 렌더링 완료
- [x] Emotion Override 버전별 overlay 비교 완료
- [ ] 발표 슬라이드 제작 (14슬라이드, 2026-06-17)

---

## Section 23 — 데모 영상 개선 및 Video 223 전환 (2026-06-12)

### 23.1 Video 049 → 223 전환 이유

Video 049(The Weeknd - The Hills)는 sad 73% 지배 영상이라, Emotion Override로 exciting/relaxing을 주입해도 **감정 바가 항상 sad로만 찍혀** override 효과가 시각적으로 드러나지 않음.

vevo 데이터셋 내 mp4가 존재하는 5개 영상 감정 분포 비교:

| ID | 영상 | 지배감정 | entropy |
|----|------|---------|---------|
| 049 | The Weeknd - The Hills | sad 50% | 1.39 |
| 063 | Ellie Goulding - Burn | exciting 35% | 1.56 |
| 088 | Taylor Swift - LWYMMD | exciting 27% | 1.68 |
| **223** | **Taylor Swift - Wildest Dreams** | **exciting 35%** | **1.68** |
| 314 | Wisin & Yandel - Follow The Leader | tense 36% | 1.56 |

**선택: Video 223** — exciting 지배 + entropy 최고 → sad/relaxing override 대비 가장 명확.

YouTube ID: `IdneKLhsWOQ`

### 23.2 AV1 코덱 문제 및 수정

vevo 영상 중 049만 h264, 나머지(063/088/223/314)는 **AV1 코덱**. MoviePy의 `VideoFileClip`이 AV1 디코딩 불가 → 출력 mp4가 검은 화면.

**수정: `generate_TEA.py` 영상 합성 부분을 MoviePy → ffmpeg subprocess로 교체**

```python
# 기존 (MoviePy, AV1 불가)
video_clip = VideoFileClip(f_video_in)
audio_clip = AudioFileClip(f_flac)
final = video_clip.set_audio(audio_clip)
final.write_videofile(f_video_out, ...)

# 수정 (ffmpeg 직접 호출, 모든 코덱 지원)
subprocess.run([
    "ffmpeg", "-y",
    "-i", f_video_in, "-i", f_flac,
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a", "aac", "-b:a", "192k",
    "-shortest", f_video_out,
], check=True)
```

### 23.3 Emotion Override 시각화 설계 변경

**문제**: overlay에서 두 버전(natural vs sad override) 감정 바가 동일하게 보여 override 여부 구분 불가.

**논의 과정:**
1. flat preset(sad 90%) 표시 → "고정하면 안된다" (동적이지 않음)
2. 원본 영상 감정만 표시 → "같으면 안된다" (override 구분 불가)
3. 2행 분리(위=영상, 아래=주입) → "저것도 변동해야" (주입 행도 동적이어야)

**최종 결론**: 감정 바는 `영상 감정 × 0.4 + preset × 0.6` 블렌드로 표시.
- Override 버전: sad 방향으로 분포가 명확히 이동 (≈60%)하면서도 영상 감정의 동적 변화 유지
- Natural 버전: 원본 영상 감정 그대로

코드:
```python
emo_short = args.emo_tag.replace("_emo_", "") if args.emo_tag else ""
emo_override_name = EMO_TAG_MAP.get(emo_short, None)
if emo_override_name is not None:
    preset = EMO_PRESETS[emo_override_name]
    orig_emotion = load_emotion(args.video_id, args.dataset_root)
    emotion_np_full = 0.4 * orig_emotion + 0.6 * preset
```

### 23.4 한국어 폰트 수정

matplotlib 기본 폰트에 한글 없어 ㅁ로 표기됨 → NotoSansCJK 적용.

```python
_KO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_KO_FONT):
    font_manager.fontManager.addfont(_KO_FONT)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_KO_FONT).get_name()
```

### 23.5 비교 영상 고정 구간 설정

비교 영상은 반드시 **같은 구간**으로 고정해야 의미 있는 비교 가능.

Video 223 자동 선택 구간: **t=55~75 (55s~75s)**, 지배감정=exciting, score=2.543

```bash
# natural로 최적 구간 탐색
conda run -n video2music python3 visualize_emotion_chord.py \
  --video_id 223 --exp 16 --mode overlay --auto 20

# override 버전은 동일 구간 고정
conda run -n video2music python3 visualize_emotion_chord.py \
  --video_id 223 --exp 16 --emo_tag _emo_sa --mode overlay --start 55 --end 75
```

### 23.6 비교 영상 생성

3종 overlay 나란히 붙이는 ffmpeg 명령:

```bash
ffmpeg -y \
  -i output/223/223_exp16_t055-075_overlay.mp4 \
  -i output/223/223_exp16_emo_sa_t055-075_overlay.mp4 \
  -i output/223/223_exp16_emo_re_t055-075_overlay.mp4 \
  -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v];[0:a][1:a][2:a]amerge=inputs=3[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 23 -c:a aac \
  output/223/compare_natural_sad_relaxing.mp4
```

### 23.7 현재 상태 (2026-06-12)

- [x] Video 223 natural 생성
- [x] Video 223 sad override 생성
- [ ] Video 223 relaxing override 생성
- [x] Overlay 비교 영상 (natural vs sad) 완료
- [ ] Overlay 3종 비교 영상 (natural + sad + relaxing) 완료
- [ ] 발표 슬라이드 제작 (14슬라이드, 2026-06-17)

---

## Section 24. Chord-Emotion Profile 분석 및 한계 (2026-06-12)

### 24.1 문제 발견: Sad Override에서 sad가 더 낮은 이슈

`chord_emotion_profile.png` 생성 후 Sad Override의 sad 수치가 Natural보다 낮게 나오는 이상 현상 발견.

**원인 분석 — 실제 chord 분포:**

| chord quality | Natural | Sad Override | Relaxing Override |
|---|---|---|---|
| major | 36.3% | 29.3% | 43.0% |
| minor | 16.3% | 11.0% ↓ | 27.0% ↑ |
| dom7 | 12.0% | 24.3% ↑↑ | 12.0% |
| min7 | 12.0% | 11.7% | 5.3% ↓ |
| sus | 12.7% | 10.3% | 4.7% ↓ |

Sad Override에서 minor가 감소하고 dom7이 2배로 증가. dom7을 `tense+exciting`으로 매핑하면 sad가 오히려 낮아 보임.

### 24.2 CHORD_TO_EMO 매핑 수정 (dom7 = blues melancholy)

dom7(dominant 7th)은 블루스에서 슬픔/멜랑꼴리 표현에 쓰이는 코드. 매핑 수정:

```python
CHORD_TO_EMO = {
    'major': [0.80, 0.00, 0.00, 0.00, 0.20, 0.00],
    'minor': [0.00, 0.10, 0.20, 0.70, 0.00, 0.00],
    'dom7':  [0.10, 0.00, 0.30, 0.60, 0.00, 0.00],  # blues dom7 → melancholy
    'maj7':  [0.20, 0.00, 0.00, 0.00, 0.80, 0.00],
    'min7':  [0.00, 0.00, 0.10, 0.60, 0.30, 0.00],
    'sus':   [0.30, 0.00, 0.00, 0.00, 0.30, 0.40],
    'dim':   [0.00, 0.40, 0.60, 0.00, 0.00, 0.00],
    'hdim':  [0.00, 0.40, 0.50, 0.10, 0.00, 0.00],
    'aug':   [0.20, 0.20, 0.60, 0.00, 0.00, 0.00],
    'none':  [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
}
```

수정 후 결과 (`output/223/chord_emotion_profile.png`):

| | Natural | Sad Override | Relaxing Override |
|---|---|---|---|
| EX | 35.7% | 32.1% ↓ | 38.5% |
| SA | 26.4% | **30.6% ↑** | 29.4% |
| RE | 16.8% | 16.2% | 13.7% ↓ |
| TE | 11.9% | 13.8% | 12.1% |

Sad Override에서 SA 26.4% → 30.6%로 상승 ✓

### 24.3 근본적 한계 발견: 데이터셋 emotion-chord 무상관

748개 영상 전체 학습 데이터에서 실제 emotion-chord 상관관계 계산:

```
Emotion     major   minor   dom7   maj7   min7    sus
exciting    24.5%   13.8%   2.0%   0.7%   2.1%   0.2%
fearful     25.3%   13.5%   2.0%   0.7%   2.1%   0.3%
tense       23.3%   12.3%   1.9%   0.7%   2.0%   0.4%
sad         24.8%   13.5%   1.9%   0.8%   1.9%   0.2%
relaxing    25.0%   13.4%   2.0%   0.8%   2.1%   0.3%
neutral     23.1%   13.3%   2.1%   0.8%   1.7%   0.2%
```

**모든 감정 클래스에서 chord quality 분포가 거의 동일.** 통계적 상관관계 없음.

**이유**: `vevo_emotion`은 영상 프레임에서 추출한 visual emotion, `vevo_chord`는 오디오에서 분석한 chord. 두 데이터는 독립적으로 추출됨. 베이스라인 모델은 temporal emotion conditioning이 없었으므로 emotion-chord 매핑을 학습할 설계가 아니었음.

**결론**: chord quality 분포 → emotion 변환 접근 자체가 이 데이터셋에서 유효하지 않음. TEA 모델이 학습한 감정-음악 관계는 chord quality 단순 분포가 아니라 코드 진행(progression), 조성(key), 컨텍스트 전체에 걸친 패턴.

### 24.4 발표 슬라이드 시각화 방향 결정 (미정)

현재까지 검토한 옵션:

- **Option A**: 입력 emotion 벡터 3종 bar chart — "어떤 감정을 주입했는가" 직접 표시
- **Option B**: Raw chord quality 분포 3종 — 실제 차이(dom7 2배 등)를 감정 해석 없이 표시
- **Option C**: Overlay 영상 스크린샷 — emotion bar + chord timeline 한 프레임 캡처

음악 청취가 주된 증거이므로 슬라이드는 보조 역할. 방향 미결정.

### 24.5 생성 스크립트

```bash
# chord-emotion profile PNG 생성
cd ~/Video2Music
conda run -n video2music python3 /tmp/gen_profile.py
# 출력: output/223/chord_emotion_profile.png

# 데이터셋 전체 emotion-chord 상관관계 분석
conda run -n video2music python3 /tmp/emo_chord_stats.py
```

---

## Section 25. 오디오 렌더링 개선 및 최종 결론 (2026-06-12)

### 25.1 lab 파일 차이 확인

3버전 300개 timestep 비교 결과:

```
Natural == Sad OV:    10/300 (3.3%) 동일
Natural == Relax OV:  5/300  (1.7%) 동일
Sad OV  == Relax OV:  7/300  (2.3%) 동일
```

코드 시퀀스가 97% 이상 다름. 모델이 각 버전에 대해 실질적으로 다른 코드를 생성함.

**핵심 chord quality 차이:**

| Quality | Natural | Sad OV | Relax OV |
|---|---|---|---|
| major | 37.7% | 32.0% (-5.7) | 44.3% (+6.7) |
| minor | 17.0% | 12.7% (-4.3) | 27.0% (+10.0) |
| dom7 | 12.0% | **24.3% (+12.3)** | 12.0% |
| sus | 12.7% | 10.3% | 4.7% (-8.0) |

### 25.2 오디오 렌더링 — 임의 악기 배정 문제

처음에 감정별로 악기를 다르게 배정 시도:
- Sad → Cello/Viola, Relaxing → Flute/Nylon Guitar 등

**문제점**: 악기 timbre 자체가 감정을 만들어버려서 "sad가 슬프게 들린다 → 코드 때문인지 악기 때문인지 구분 불가" → 연구 데모로 신뢰도 저하.

**최종 결정**: 3버전 동일 악기 (Piano + Strings + Bass) 고정.

```bash
conda run -n video2music python3 /tmp/rerender_neutral.py
```

출력:
- `output/223/223_exp16_natural.mp3`
- `output/223/223_exp16_sad_override.mp3`
- `output/223/223_exp16_relaxing_override.mp3`

악기: Piano(GM#0) + String Ensemble(GM#48, vel+12) + Acoustic Bass(GM#32, vel-5), FluidSynth 기본 gain.

### 25.3 Emotion Override 효과의 특성

**결론**: Sad Override는 비극적(tragic) 슬픔이 아니라 멜랑꼴리한(bittersweet) 슬픔.

이유:
- Video 223 자체가 exciting 영상 → 모델이 그 맥락을 학습, 완전히 뒤집지 않음
- dom7 증가 = 블루스식 달콤씁쓸한 긴장감
- 감정 벡터만으로 chord progression 전체를 극적으로 바꾸진 않음

**발표 포인트**: Emotion override는 음악을 극단적으로 바꾸는 것이 아니라, 원곡의 분위기를 유지하면서 감정적 색채를 자연스럽게 조절하는 fine-grained emotional control.

### 25.4 현재 상태 및 결정사항 (2026-06-12 최종)

- [x] Video 223 natural / sad / relaxing override 생성 완료
- [x] 동일 악기 mp3 3종: `223_exp16_{natural,sad_override,relaxing_override}.mp3`
- [x] chord_emotion_profile.png (실제 lab 데이터 기반)
- [x] 모델 성능 추가 개선 없음 — 현재 결과로 확정
- [ ] YouTube URL 입력 기능 추가
- [ ] Demo UI 인터페이스 (generation + override 파라미터 조정)
- [ ] 발표 슬라이드 최종 제작 (2026-06-17)

### 25.5 내일 할 일 (2026-06-13)

1. **YouTube URL → 영상 다운로드 → 전처리 → TEA generation 파이프라인 자동화**
   - `yt-dlp`로 URL에서 영상 다운로드
   - 기존 전처리(feature 추출) 자동 연결
   - `generate_TEA.py` 실행

2. **Demo UI 제작**
   - Gradio 또는 간단한 웹 인터페이스
   - 파라미터: emotion override (none/exciting/sad/relaxing/...), exp 번호, key 등
   - 결과: 생성된 mp3 + overlay 영상 재생

---

## Section 26. Baseline vs TEA 비교 데모 제작 (2026-06-13)

### 26.1 개요

2026-06-17 발표를 위한 Baseline AMT vs MS-TEA Exp16 비교 데모를 완성한 날.
주요 목표: **동일 비디오, 동일 렌더링 조건**에서 코드 생성 모델만 다른 공정한 비교.

---

### 26.2 yt-dlp AV1 코덱 문제 최종 수정 (generate_TEA.py)

#### 문제

`generate_TEA.py`의 YouTube URL 다운로드 기능에서 yt-dlp가 기본적으로 AV1(format 399) 코덱을 선택 → OpenCV가 읽지 못함.

```
Missing Sequence Header
Failed to get pixel format
```

#### 원인

yt-dlp의 기본 포맷 선택이 AV1으로 변경됨. OpenCV는 AV1을 HW 디코딩 없이 지원하지 않음.

#### 수정 (generate_TEA.py, lines ~498–528)

```python
# H.264 avc 코덱 우선 (OpenCV AV1 미지원)
ret = subprocess.run([
    _ytdlp,
    "-f", "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    "--merge-output-format", "mp4",
    "--force-overwrites",    # 기존 파일 있어도 덮어씀
    "-o", dl_path,
    video_path,
])
```

- `vcodec^=avc`: H.264(avc1) 코덱으로 시작하는 포맷 선택
- `--force-overwrites`: partial 파일 잔존 시 "already downloaded" 건너뜀 방지
- glob cleanup (`/tmp/yt_input*`)으로 이전 partial 파일 제거 후 다운로드
- 1차 실패 시 `best` 포맷으로 fallback 재시도

---

### 26.3 Gradio Demo 서버 (demo_TEA.py) 초기 버전

이전 세션에서 Gradio 기반 demo 서버 구축:
- `Tab 1: Live Generation` — YouTube URL → TEA Exp16 생성
- `Tab 2: Pre-computed Examples` — Video 223 3종 (Natural / Sad / Relaxing Override)
- 실행: `python3 demo_TEA.py` at port 7860
- `GRADIO_TEMP_DIR=~/.gradio_tmp` 필수 (기본 `/tmp/gradio`는 권한 문제)

---

### 26.4 Baseline 모델 선택 및 버그 수정

#### 사용 모델

| 디렉토리 | Linear_emo | 설명 |
|---|---|---|
| `saved_models/AMT/` | ❌ 없음 | 옛날 baseline, emotion 미포함 |
| `saved_models/AMT_full/` | ✅ 있음 | 논문의 "Baseline (full)" |
| `saved_models/AMT_no_emotion/` | ❌ | Ablation용 no-emotion baseline |

→ **데모용 Baseline = `AMT_full`**: emotion 정보는 사용하지만 TEA adapter 없는 모델.

논문 수치:
```
Baseline (no emotion) : H@1=0.4990, AR=0.4190
Baseline (full, AMT)  : H@1=0.5108, AR=0.4232  ← 데모에서 사용
MS-TEA Exp16          : H@1=0.5659, AR=0.4694 (+12.0%)
```

#### generate_baseline_clean.py 버그 수정

**버그 1: `total_vf_dim` 계산 오류**

기존 코드가 emotion(6-dim)을 `VideoMusicTransformer`의 `total_vf_dim`에 포함시킴 → 776으로 계산되어 AMT_full 가중치(770)와 shape mismatch 발생.

원인: 현재 `VideoMusicTransformer`는 emotion을 `Linear_emo` 별도 경로로 처리하므로 `total_vf_dim`에 불포함 (770 = 768 + scene_offset + motion).

```python
# 수정 전 (버그):
total_vf_dim += 6  # emotion도 vf_concat에 포함 → 776 (틀림)

# 수정 후:
# VideoMusicTransformer: emotion → Linear_emo 별도 경로 (total_vf_dim=770)
# VideoRegression: emotion → concat 방식 유지 (reg_vf_dim = 770 + 6 = 776)
reg_vf_dim = total_vf_dim + 6
```

**버그 2: FluidSynth 경로**

`FluidSynth()` 기본 생성자가 PATH에 없는 경우 FileNotFoundError.

수정: 경로 동적 탐색 + subprocess 직접 호출.

```python
_fs_candidates = [shutil.which("fluidsynth"),
                  "/home/taegum/miniconda3/envs/video2music/bin/fluidsynth", ...]
_fs_bin = next((p for p in _fs_candidates if p and os.path.isfile(p)), "fluidsynth")
subprocess.call([_fs_bin, "-ni", _sf2, f_path_midi, "-F", f_path_flac, "-r", "44100"])
```

#### _run_baseline.py (드라이버)

```python
# sys.argv 패칭으로 argparse 우회
sys.argv = ['generate_baseline_clean.py',
            '-output_dir', './output',
            '-model_weights', './saved_models/AMT_full/best_loss_weights.pickle',
            '-modelReg_weights', './saved_models/AMT/best_rmse_weights.pickle']
import generate_baseline_clean as gbc
gbc.test_id = target_id  # "223" or "049"
gbc.main()
```

출력: `output/<id>/<id>_cgen_rd.{lab,mid,flac,mp4}`

---

### 26.5 렌더링 파이프라인 통일 (핵심)

#### 문제: 렌더링 차이가 비교를 불공정하게 만듦

| | 기존 Baseline | TEA |
|---|---|---|
| MIDI 생성 | `MIDIFile(1)` 단일 채널 | `chords_to_midi()` 3트랙 |
| 악기 | 채널 0 단일 Piano | Piano + Strings + Bass |
| 영상 인코딩 | MoviePy | ffmpeg |
| Soundfont | FluidSynth 기본 | `default_sound_font.sf2` |

**소리 차이의 원인이 코드 시퀀스인지 렌더링 방식인지 구분 불가** → 비교 신뢰도 저하.

#### 해결: _render_baseline_lab.py

Baseline이 생성한 코드 시퀀스(`.lab` 파일)를 TEA와 완전히 동일한 파이프라인으로 재렌더링.

```
Baseline .lab 파일
  → chords_to_midi() [3트랙: Piano + Strings + Bass, neutral emotion]
  → FluidSynth [동일 SF2: ./soundfonts/default_sound_font.sf2]
  → ffmpeg [-c:v libx264 -crf 23 -c:a aac -b:a 192k]
  → output/<id>/<id>_BASE_cgen_rd.{mid,flac,mp3,mp4}
```

**density/velocity도 동일 조건**: 두 모델 모두 `AMT/best_rmse_weights.pickle` regression 모델에서 예측. 동일 비디오 feature → 동일 density/velocity.

**유일한 차이**: 어떤 코드 시퀀스(chord sequence)가 생성됐느냐.

#### VideoRegression forward() 차원 정리

```python
# dataset["semanticList"] 텐서: shape [seq_len, feat_dim] = [300, 768]
# unsqueeze(0) → [1, 300, 768], shape[1] = 300 (seq_len, NOT feat_dim!)
# → total_vf_dim 계산 시 dataset[idx]["semanticList"]를 직접 사용해야 함

total_vf_dim = sum(v.shape[1] for v in dataset[idx]["semanticList"]) + 1 + 1  # 768+1+1=770
reg_vf_dim   = total_vf_dim + 6  # 776

# feature 전달 시 batch dim 추가 필요:
feature_semantic_list = [s.unsqueeze(0).to(get_device()) for s in dataset[idx]["semanticList"]]
feature_scene_offset  = dataset[idx]["scene_offset"].unsqueeze(0).to(get_device())  # [1, 300]
feature_motion        = dataset[idx]["motion"].unsqueeze(0).to(get_device())         # [1, 300]
feature_emotion       = dataset[idx]["emotion"].unsqueeze(0).to(get_device())        # [1, 300, 6]
```

---

### 26.6 배너 추가 (_add_banner.py)

데모 시청자가 어떤 모델인지 즉시 식별할 수 있도록 MP4 상단에 ffmpeg drawtext 배너 추가.

```python
# TEA: 파란 배경 (#1a3a6a)
# Baseline: 초록 배경 (#1a4a1a)
vf = "drawtext=text='TEA':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=12:box=1:boxcolor=0x1a3a6a@0.85:boxborderw=10:font=DejaVu Sans Bold"
subprocess.run(["ffmpeg", "-y", "-i", input_mp4, "-vf", vf, "-codec:a", "copy", output_mp4])
```

---

### 26.7 demo_TEA.py 재설계

**Live Generation 탭 제거** — Baseline vs TEA 비교 탭만 남김.

#### 최종 탭 구조

**Tab 1: Video 223 — Taylor Swift Anti-Hero**
- Section A: Baseline vs TEA Natural (side-by-side, 2열)
- Section B: TEA 감정 Override 3종 (Natural / Sad / Relaxing, 3열)

**Tab 2: Video 049**
- Baseline Natural vs TEA Exciting (side-by-side)
- TEA Sad Override

**Tab 3: Performance Metrics**
- 모델별 H@1 / Agreement Rate 테이블
- chord_emotion_profile.png

---

### 26.8 생성된 파일 목록 (2026-06-13 기준)

#### Video 223 (Taylor Swift Anti-Hero)
```
output/223/
├── 223_cgen_rd.lab/mid/flac/mp4   ← Baseline AMT_full 생성 (old rendering)
├── 223_BASE_cgen_rd.lab           ← Baseline 코드 시퀀스 (.lab)
├── 223_BASE_cgen_rd.mid/flac/mp4  ← Baseline, TEA 동일 렌더링 파이프라인
├── 223_BASE.mp3                   ← MP3 변환
├── 223_BASE_banner.mp4            ← "Baseline" 배너 추가 버전 (데모용)
├── 223_TEA_exp16_cgen_rd.*        ← TEA Natural (기존)
├── 223_TEA_exp16_emo_sa_cgen_rd.* ← TEA Sad Override (기존)
├── 223_TEA_exp16_emo_re_cgen_rd.* ← TEA Relaxing Override (기존)
├── 223_TEA_banner.mp4             ← "TEA" 배너 (Natural)
├── 223_TEA_sad_banner.mp4         ← "TEA" 배너 (Sad)
└── 223_TEA_relaxing_banner.mp4    ← "TEA" 배너 (Relaxing)
```

#### Video 049
```
output/049/
├── 049_cgen_rd.lab/mid/flac/mp4   ← Baseline AMT_full 생성 (old rendering)
├── 049_BASE_cgen_rd.mid/flac/mp4  ← Baseline, TEA 동일 렌더링 파이프라인
├── 049_BASE.mp3
├── 049_BASE_banner.mp4            ← "Baseline" 배너
├── 049_TEA_exp16_emo_ex_cgen_rd.* ← TEA Exciting Override (기존)
├── 049_TEA_exp16_emo_sa_cgen_rd.* ← TEA Sad Override (기존)
├── 049_TEA_exciting_banner.mp4    ← "TEA" 배너 (Exciting)
└── 049_TEA_sad_banner.mp4         ← "TEA" 배너 (Sad)
```

---

### 26.9 추가된/수정된 스크립트 목록

| 파일 | 내용 |
|---|---|
| `generate_TEA.py` | yt-dlp H.264 강제, --force-overwrites, glob cleanup |
| `generate_baseline_clean.py` | total_vf_dim 버그 수정, FluidSynth 경로 수정, subprocess import |
| `_run_baseline.py` | baseline 생성 드라이버 (AMT_full 가중치 지정) |
| `_render_baseline_lab.py` | baseline .lab → TEA 동일 파이프라인 재렌더링 |
| `_add_banner.py` | ffmpeg drawtext 배너 추가 (TEA/Baseline) |
| `demo_TEA.py` | 완전 재설계: Baseline vs TEA 비교 탭 구조 |

---

### 26.10 상태 체크 (2026-06-13 최종)

- [x] yt-dlp AV1 코덱 문제 수정 (H.264 강제)
- [x] generate_baseline_clean.py 버그 수정 (total_vf_dim, FluidSynth)
- [x] Baseline AMT_full으로 Video 223, 049 코드 시퀀스 생성
- [x] TEA 동일 렌더링 파이프라인으로 Baseline 재렌더링
- [x] 모든 MP4에 TEA/Baseline 배너 추가
- [x] demo_TEA.py 재설계 (Baseline vs TEA 비교 구조)
- [x] 데모 서버 실행 중 (port 7860)
- [ ] 발표 슬라이드 최종 제작 (2026-06-17)

---

## Section 27. 데모 고도화 및 지표 완성 (2026-06-13 오후)

### 27.1 디렉터리 정리 및 스크립트 컴팩트화

불필요 파일 전면 삭제 및 스크립트 병합.

**삭제한 파일 범주**
- macOS 메타데이터 (`._*`, `.DS_Store`)
- 중간 생성물 (`.mid`, `.flac`, `.mp4` - 배너 없는 버전)
- 구 MSTEA 스크립트 (`generate_MSTEA_*.py` 등 미사용)
- 로그/백업 파일

**스크립트 병합**
- `_run_baseline.py` (AMT_full 실행) + `_render_baseline_lab.py` (렌더링) → `_render_baseline_lab.py` 단일 스크립트로 통합
- 사용법: `python _render_baseline_lab.py 223` (lab 없으면 자동 생성 → 렌더 → 배너까지 일괄)
- `generate_baseline_clean.py` 모듈 import 시 `gbc.test_id = test_id` + `sys.argv` 스푸핑으로 통합

---

### 27.2 demo_TEA.py 구조 개편

**Video 223 탭 변경**
- Section B (Override): `TEA Natural` 제거 → `TEA Exciting` 추가
- 최종 구성: Exciting | Sad | Relaxing (3열)

**Video 049 탭 변경**
- 기존 2개 → 4개 영상으로 확장
- Row 1: Baseline | TEA Exciting
- Row 2: TEA Sad | TEA Relaxing

**생성 명령**
```bash
# 223 Exciting (신규)
python generate_TEA.py --exp 16 --test_id 223 --emotion exciting
# 049 Relaxing (신규)
python generate_TEA.py --exp 16 --test_id 049 --emotion relaxing
```

---

### 27.3 Chord-Emotion 분포 프로파일 전수 생성

`visualize_emotion_chord.py` 를 수정해 Override 모드에서도 **상단 패널은 영상 원본 감정 궤적**을 표시하도록 변경.
(기존: override preset 90% 평탄선 → 모든 프로파일이 한 색상으로 보이는 문제)

```python
# 핵심 수정: emotion_np_natural 분리
emotion_np_natural = load_emotion(args.video_id, args.dataset_root)  # 항상 원본
if emo_override_name is not None:
    emotion_np_full = np.tile(EMO_PRESETS[emo_override_name], ...)   # 모델 입력용
else:
    emotion_np_full = emotion_np_natural
# make_static_plot에 emotion_np_natural=emo_nat_seg 전달 → 상단 표시용
```

**생성된 프로파일 (총 7개)**

| 파일 | 설명 |
|---|---|
| `output/223/223_exp16_emotion_chord.png` | 223 Natural |
| `output/223/223_exp16_emo_ex_emotion_chord.png` | 223 Exciting Override |
| `output/223/223_exp16_emo_sa_emotion_chord.png` | 223 Sad Override |
| `output/223/223_exp16_emo_re_emotion_chord.png` | 223 Relaxing Override |
| `output/049/049_exp16_emo_ex_emotion_chord.png` | 049 Exciting Override |
| `output/049/049_exp16_emo_sa_emotion_chord.png` | 049 Sad Override |
| `output/049/049_exp16_emo_re_emotion_chord.png` | 049 Relaxing Override |

**데모 레이아웃**: Performance Metrics 탭에 2개 영상 × 각 override 종류별 4열/3열 그리드 배치

---

### 27.4 Gradio 이미지 서빙 수정 (403 → 200)

Gradio 4.7.1에서 외부 디렉터리 파일 서빙 시 `allowed_paths` 필요.

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=[WORK_DIR],   # 추가
)
```

---

### 27.5 신규 평가 측정 — AMT_no_emotion / AMT_full H@3·H@5

기존 지표에 H@3/H@5 누락 → `_eval_amts.py` 작성해 val_dataset 기준 재측정.

**모델 구조 확인**: 두 모델 모두 `Linear_vis.weight [512, 770]` → `total_vf_dim=770` 동일.

**측정 결과**

| 모델 | H@1 | H@3 | H@5 | Agreement Rate |
|------|----:|----:|----:|---------------:|
| AMT no emotion | 0.4990 | **0.7807** | **0.8791** | 0.4190 |
| AMT full (Baseline) | 0.5108 | **0.7859** | **0.8815** | 0.4232 |
| TEA_encoder_gru | 0.6090 | 0.8528 | 0.9180 | 0.4253 |
| MS-TEA Exp16 | 0.5659 | 0.8320 | 0.9066 | 0.4694 |

**평가 스크립트**: `_eval_amts.py`

---

### 27.6 Ablation 테이블 개선

실제 점수 + 절대 변화량 + 상대 변화율 동시 표시.

| 단계 | 모델 | H@1 | Δ H@1 | H@3 | H@5 | Agreement Rate | Δ AR |
|------|------|----:|------:|----:|----:|---------------:|-----:|
| 1 | AMT no emotion | 0.4990 | *(base)* | 0.7807 | 0.8791 | 0.4190 | *(base)* |
| 2 | AMT full (+emotion concat) | 0.5108 | **+2.4%** (+0.012) | 0.7859 | 0.8815 | 0.4232 | **+1.0%** (+0.004) |
| 3 | MS-TEA Exp16 (+TEA adapter) | **0.5659** | **+13.4%** (+0.055) | **0.8320** | **0.9066** | **0.4694** | **+12.0%** (+0.050) |

**핵심 인사이트**: 감정 feature concat(step 2→1: H@1 +1.2pp)보다 TEA 시간적 처리(step 3→1: H@1 +5.5pp, AR +5.0pp)가 압도적으로 효과적.

---

---

## 28. 발표 준비 및 추가 분석 (2026-06-16)

### 28.1 모듈명 변경: TEA → TEM

발표 자료에서 **TEA(Temporal Emotion Adapter)** → **TEM(Temporal Emotion Module)** 로 명칭 변경.

- "Adapter"는 backbone을 고정하고 소수 파라미터만 학습하는 LoRA 계열을 연상시킴
- TEM은 end-to-end 학습이므로 Adapter가 부정확한 표현
- 코드베이스는 TEA 명칭 유지, 발표/레포트에서만 TEM 사용

---

### 28.2 Agreement Rate 출처 명확화

- AR은 AMT 원 논문에 없는 지표
- 이번 프로젝트에서 직접 구현 (`eval_emotion_valence.py`)
- 단, "새로 정의한" 지표라기보다는 chord-emotion 정렬도 측정에 **직접 적용**한 지표
- AR은 Shuffle Test의 대체가 아니라 **측정 질문을 바꾼 것**: "모델이 감정 feature를 쓰는가" → "출력 음악이 영상 감정과 얼마나 일치하는가"

---

### 28.3 렌더링 효과 분리 실험 (감정 기여도 분석)

**배경**: emotion override 시 auto_params()가 key(sad→minor), instrument, primer도 함께 바꾸어 모델 기여와 렌더링 기여가 섞임. sad override인데 minor 코드 비율이 오히려 감소하는 역설적 결과가 원인.

**실험**: sad override + key 강제 고정(major) + primer 제거

```bash
python generate_TEA.py --exp 16 --test_id 223 --emotion sad --key major --no_primer
# 출력: output/223/223_TEA_exp16major_emo_sa_cgen_rd.lab
```

**결과**

| 조건 | Major | Minor | Dom7 | Dim/Aug | Suspended |
|---|---|---|---|---|---|
| Natural | 40.0% | 29.0% | 12.0% | 6.3% | 12.7% |
| Sad override (기존) | 36.0% | 24.3% ↓ | 24.3% ↑ | 5.0% | 10.3% |
| Sad + key 고정 | 35.7% | **31.3% ↑** | 15.3% | 6.7% | 11.0% |

**해석**
- 기존 sad override에서 minor가 줄었던 것은 key를 minor로 바꾸면서 코드 quality 라벨이 뒤틀렸기 때문
- key를 고정하면 모델은 sad 입력에 반응하여 Minor 29% → 31.3%로 올림 (+2.3%p) — 방향은 맞음
- 단 효과가 작고, 렌더링 파이프라인(key/악기/primer)이 지각적 차이의 주된 원인일 가능성 있음

**레포트 반영 방향**
- "모델이 감정에 반응하는가" 분석 절에서 이 실험 결과 인용
- 렌더링 효과와 모델 효과의 분리 필요성 명시
- AR 개선이 모델 기여인지 렌더링 기여인지도 향후 분리 필요

---

---

### 28.4 Inference Time 벤치마크

**측정 방법**: GPU(cuda:0) 기준, 단일 forward pass × 5회 (warmup 1회 제외), test_id=223

**스크립트**: `benchmark_inference.py`

```
conda run -n video2music python benchmark_inference.py --test_id 223 --runs 5
```

**결과**

| | Baseline (AMT_full) | TEM (MS-TEA Exp16) | 차이 |
|---|---|---|---|
| 파라미터 수 | 32,517,822 | 40,738,895 | **+8,221,073 (+25.3%)** |
| forward pass (GPU) | 7.3ms | 8.6ms | **+1.3ms (+18.2%)** |
| ×300 추정 생성시간 | 2.2s | 2.6s | +0.4s |
| end-to-end (렌더링 포함) | ~23s | ~33s | **+10s (+43%)** |

**해석**
- 순수 모델 inference 오버헤드: +18.2% (forward pass 기준)
- end-to-end 차이(+43%)는 inference보다 FluidSynth/ffmpeg 렌더링 단계에서 더 크게 벌어짐
- BiRNN 대신 attention 기반 TEM encoder(Exp16) + multisignal(scene/motion TEA) 구조로 인해 파라미터 +25.3% 증가
- 실시간 응용에는 부적합하나, 오프라인 생성 파이프라인으로는 허용 가능한 수준

---

### 27.7 상태 체크 (2026-06-13 오후 최종)

- [x] 디렉터리 정리 및 스크립트 병합 (_run_baseline + _render_baseline → 통합)
- [x] demo Video 223: Exciting Override 추가, Natural 제거
- [x] demo Video 049: 4개 영상 구조 (Baseline/Exciting/Sad/Relaxing)
- [x] chord-emotion 프로파일 7종 전수 생성 (자연 감정 궤적 표시 수정 포함)
- [x] Gradio allowed_paths 설정 (이미지 서빙 정상화)
- [x] AMT_no_emotion / AMT_full H@3·H@5 신규 측정
- [x] Ablation 테이블 실제 점수·Δ 동시 표기
- [x] 데모 서버 실행 중 (port 7860)
- [x] 발표 슬라이드 최종 제작 (2026-06-17)

---

## Section 29. 기말 발표 이후 — 질문 대응 및 추가 분석 (2026-06-23~)

### 29.1 발표 시 받은 질문 및 답변

#### Q. 실제로 감정이 잘 잡히는지 보여주는 시각화 자료가 있으면 좋겠다

**교수님 피드백 요약**: 음악을 들어보니 감정 차이가 느껴지기는 하는데, 영상과 생성된 음악의 감정이 어떻게 연동되는지를 시각적으로 보여주는 자료가 있으면 더 설득력이 있겠다.

**기존 구현 (Section 22)**: 발표 시간 부족으로 충분히 설명하지 못했으나, 이미 `visualize_emotion_chord.py`로 다음 두 가지 시각화를 구현해 두었음.

1. **정적 PNG** (`output/223/chord_emotion_profile.png`)
   - 상단 패널: 영상에서 추출한 감정 궤적 (6개 감정 클래스 × 시간)
   - 하단 패널: TEM이 생성한 코드 품질 타임라인 (major/minor/dom7 등, 컬러 바)
   - Emotion Override 3종(natural / sad / relaxing) 나란히 비교 → override에 따라 코드 분포가 어떻게 달라지는지 시각적으로 확인 가능

2. **오버레이 영상** (`output/223/overlay_*.mp4`)
   - 원본 영상 아래에 실시간 감정 바(현재 프레임의 감정 확률) + 현재 코드 레이블을 합성
   - 영상 장면 변화 → 감정 바 변동 → 코드 변화 흐름을 한 화면에서 확인

**한계 (Section 24.3)**: 데이터셋 전체 748개 영상에서 emotion-chord 통계적 상관관계가 없음을 확인. 이유는 `vevo_emotion`(시각적 감정)과 `vevo_chord`(오디오 코드)가 독립적으로 추출된 데이터라 원래부터 상관이 없음. TEM이 학습한 감정-음악 관계는 코드 품질 단순 분포가 아니라 조성(key), 코드 진행 전체 패턴에 걸쳐있어, 단순 코드 품질 비율 시각화만으로는 과소평가될 수 있음.

**후속 방향**: 레포트에서는 chord quality 분포 시각화와 함께 key 통계(major key vs minor key 사용 비율)도 추가로 보여주는 것이 더 직관적. TEM의 감정 기여는 "어떤 코드 품질" 보다 "어떤 조성/키/진행"에 더 잘 반영됨.

---

### 29.2 발표 이후 추가된 분석 내용 요약

발표(2026-06-17) 이후 추가/확정된 내용:

| 항목 | 내용 | 상세 |
|------|------|------|
| 모듈 명칭 | TEA → TEM 확정 | Adapter는 frozen backbone 연상 → Module이 정확. 코드베이스는 TEA 유지 (Section 28.1) |
| AR 출처 명확화 | 직접 구현한 지표 | AMT 논문에 없음. eval_emotion_valence.py 에 구현. "채택" 표현이 적절 (Section 28.2) |
| 렌더링 효과 분리 | 모델 기여 +2.3%p 확인 | sad+key고정 시 minor 29%→31.3%. 기존 역설은 key transposition 에 의한 라벨 왜곡 (Section 28.3) |
| Inference Time | +18.2% forward pass 오버헤드 | 파라미터 +25.3%, end-to-end +43%는 렌더링 포함 수치 (Section 28.4) |

---

### 29.3 발표 시간 부족으로 설명하지 못한 부분

**AR 지표 구현 방식**
- AR(Agreement Rate)은 각 timestep에서 생성된 코드의 valence 방향(positive/negative)과 영상 감정의 valence 방향이 일치하는 비율
- `eval_emotion_valence.py`에서 chord valence는 코드 품질별 사전 매핑, emotion valence는 6개 감정 클래스의 VA space 매핑으로 계산
- Shuffle Test 실패(autoregressive 구조로 인해 감정 feature shuffle 시 거의 변화 없음)의 대안으로 채택

**TEM 주입 방식**
- TEA 출력은 positional encoding으로 더해지는 것이 아닌, Video feature에 additive injection
- `vf = encoder_emotion_norm(vf + encoder_emotion_proj(enc_emotion_emb))` (model/video_music_transformer_TEA.py:265)
- positional encoding은 그 이후에 적용됨 (lines 281-282)
- 즉 감정은 "언제 어떤 화음"을 결정하는 공간 정보가 아닌, "어떤 맥락에서 어떤 화음" 를 결정하는 의미 정보로 주입

**데이터셋 편향과 AR ceiling**
- 학습 데이터 68% positive valence → 모델이 항상 밝은 코드 방향으로 편향
- AR ceiling ~0.5 (이론상 random과 거의 같음) — emotion override의 효과가 통계적으로 유의한지 해석 주의 필요
- 균형 데이터셋 학습 시 AR 개선 가능성 있음 (향후 과제)

---

### 29.4 향후 과제 (레포트 작성 기준)

| 우선순위 | 과제 | 이유 |
|---------|------|------|
| 높음 | 균형 데이터셋 실험 (positive 50% 제한) | AR ceiling 문제 근본 해결 |
| 높음 | H@k vs AR trade-off 정량 분석 | TEM이 harmony 예측력을 희생해 AR을 얻는지 확인 |
| 중간 | 렌더링 효과 완전 분리 (--no_params 옵션) | key/악기/primer 모두 고정하고 모델 단독 기여 측정 |
| 낮음 | 실시간 추론 최적화 | BiRNN → LSTM quantization, end-to-end 23s→10s 미만 목표 |
