# Genre Confidence UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 RandomForest 예측 pipeline을 유지하면서 Top-1 확률과 Top-1·Top-2 margin을 이용한 상대적 신뢰도 UI를 Streamlit 앱에 추가한다.

**Architecture:** 신뢰도 판단은 Streamlit과 분리된 순수 함수 `assess_confidence()`로 구현해 deterministic unit test가 가능하게 한다. `app.py`는 이 함수의 결과를 화면에 표시하고, 기존 sidebar·사용자 수정 문구·Top-3·멜스펙트로그램을 유지하며 임시 WAV를 `finally`에서 정리한다.

**Tech Stack:** Python 3, Streamlit 1.59.1, pytest 9.1.1, librosa, scikit-learn, matplotlib

## Global Constraints

- `model_rf.joblib`, `label_encoder.joblib`, `scaler.joblib`을 재학습하거나 변경하지 않는다.
- 57개 feature의 이름, 순서, 추출 방식은 변경하지 않는다.
- 사용자가 수정한 `서정현 · 장르예측앱 — WAV 파일을 올리면 장르를 예측합니다` 문구를 보존한다.
- sidebar의 지원 장르 10종과 `RandomForest (재학습, 57피처)` 문구를 보존한다.
- 신뢰도 단계는 calibration된 실제 확률이 아니라 UI 설명용 heuristic임을 명시한다.
- 현재 작업 폴더는 Git 저장소가 아니며 사용자가 commit을 요청하지 않았으므로 commit 단계는 수행하지 않는다.

## File Structure

- Create `confidence.py`: 상대적 신뢰도 규칙만 담당하는 순수 함수.
- Create `tests/test_confidence.py`: threshold, margin, 잘못된 입력을 검증하는 단위 테스트.
- Modify `app.py`: 신뢰도 결과 카드, sidebar 한계 안내, 안전한 임시파일 정리.
- Modify `README.md`: 실행법, heuristic 의미, 모델 한계 기록.

---

### Task 1: 상대적 신뢰도 판단 함수

**Files:**
- Create: `confidence.py`
- Create: `tests/test_confidence.py`

**Interfaces:**
- Consumes: `top3: Sequence[tuple[str, float]]`, 확률 내림차순으로 정렬된 예측 결과.
- Produces: `assess_confidence(top3) -> dict[str, str | float]`.
- Return keys: `level`, `top1_prob`, `margin`, `message`, `color`.

- [ ] **Step 1: 실패하는 threshold 단위 테스트 작성**

```python
import pytest

from confidence import assess_confidence


@pytest.mark.parametrize(
    ("top3", "expected_level", "expected_margin"),
    [
        ([('rock', 0.70), ('metal', 0.20), ('pop', 0.10)], '높음', 0.50),
        ([('rock', 0.50), ('metal', 0.35), ('pop', 0.15)], '높음', 0.15),
        ([('pop', 0.42), ('disco', 0.30), ('rock', 0.10)], '보통', 0.12),
        ([('pop', 0.30), ('disco', 0.22), ('rock', 0.12)], '보통', 0.08),
        ([('reggae', 0.14), ('pop', 0.13), ('jazz', 0.13)], '낮음', 0.01),
    ],
)
def test_assess_confidence_levels(top3, expected_level, expected_margin):
    result = assess_confidence(top3)

    assert result['level'] == expected_level
    assert result['margin'] == expected_margin
    assert result['top1_prob'] == top3[0][1]


def test_assess_confidence_requires_two_predictions():
    with pytest.raises(ValueError, match='최소 2개'):
        assess_confidence([('rock', 0.70)])
```

- [ ] **Step 2: 테스트를 실행해 정의되지 않은 module 실패 확인**

Run:

```powershell
python -m pytest tests/test_confidence.py -v
```

Expected: collection 단계에서 `ModuleNotFoundError: No module named 'confidence'`.

- [ ] **Step 3: 최소 신뢰도 판단 함수 구현**

```python
from collections.abc import Sequence


def assess_confidence(
    top3: Sequence[tuple[str, float]],
) -> dict[str, str | float]:
    """Top-3 예측을 UI 설명용 상대적 신뢰도로 변환한다."""
    if len(top3) < 2:
        raise ValueError('신뢰도 판단에는 최소 2개 예측이 필요합니다.')

    top1_prob = float(top3[0][1])
    top2_prob = float(top3[1][1])
    margin = round(top1_prob - top2_prob, 4)

    if top1_prob >= 0.50 and margin >= 0.15:
        level = '높음'
        message = '현재 지원 장르 안에서 비교적 뚜렷한 결과입니다.'
        color = 'green'
    elif top1_prob >= 0.30 and margin >= 0.08:
        level = '보통'
        message = '가능성은 있지만 다른 후보와 함께 확인해야 합니다.'
        color = 'orange'
    else:
        level = '낮음'
        message = '결과를 확정하기 어렵습니다.'
        color = 'red'

    return {
        'level': level,
        'top1_prob': top1_prob,
        'margin': margin,
        'message': message,
        'color': color,
    }
```

- [ ] **Step 4: 단위 테스트 통과 확인**

Run:

```powershell
python -m pytest tests/test_confidence.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: 입력 sequence를 변경하지 않는지 확인하는 테스트 추가**

```python
def test_assess_confidence_does_not_mutate_predictions():
    top3 = [('reggae', 0.14), ('pop', 0.13), ('jazz', 0.13)]
    original = list(top3)

    assess_confidence(top3)

    assert top3 == original
```

- [ ] **Step 6: 전체 confidence 테스트 재실행**

Run:

```powershell
python -m pytest tests/test_confidence.py -v
```

Expected: `7 passed`.

---

### Task 2: Streamlit 결과 카드와 안전한 임시파일 수명

**Files:**
- Modify: `app.py:1-15`
- Modify: `app.py:130-141`
- Modify: `app.py:154-224`

**Interfaces:**
- Consumes: `assess_confidence(top3)` from Task 1.
- Preserves: `predict_genre(...) -> list[tuple[str, float]]`, `plot_melspectrogram(...) -> Figure`.
- Produces: confidence badge, Top-1/margin 근거, 학습 범위 안내가 포함된 Streamlit 화면.

- [ ] **Step 1: import와 Python syntax의 현재 baseline 확인**

Run:

```powershell
python -m py_compile app.py
```

Expected: exit code `0`.

- [ ] **Step 2: 순수 신뢰도 함수 import 추가**

`app.py`의 project import 영역에 다음을 추가한다.

```python
from confidence import assess_confidence
```

- [ ] **Step 3: sidebar에 모델 범위 안내 추가**

기존 장르 목록과 사용자 문구를 유지하고 divider 아래에 다음 caption을 추가한다.

```python
st.caption('이 모델은 위 10개 장르만 구분합니다.')
```

- [ ] **Step 4: 예측 직후 신뢰도 계산 연결**

`top3 = predict_genre(...)` 다음에 다음을 추가한다.

```python
confidence = assess_confidence(top3)
```

- [ ] **Step 5: 기존 Top-1 metric을 신뢰도 중심 카드로 교체**

왼쪽 column 안에서 기존 사용자 caption과 Top-3 막대그래프는 유지한다. Top-1 영역을 다음 구조로 바꾼다.

```python
with st.container(border=True):
    st.caption('MOST LIKELY')
    st.markdown(f'## {emoji} {top_genre.upper()}')
    st.markdown(
        f":{confidence['color']}[● {confidence['level']} 신뢰도]"
    )
    st.progress(top_prob)
    st.caption(
        f"1위 {top_prob:.1%} · "
        f"2위와 차이 {confidence['margin']:.1%}p"
    )

    if confidence['level'] == '높음':
        st.success(confidence['message'])
    elif confidence['level'] == '보통':
        st.warning(confidence['message'])
    else:
        st.warning(confidence['message'])
```

- [ ] **Step 6: 멜스펙트로그램 column에 학습 범위 안내 추가**

기존 `st.pyplot(fig_mel)` 아래에 다음을 추가한다.

```python
st.info(
    '학습 범위 밖의 곡은 가장 가까운 기존 장르로 '
    '분류될 수 있습니다.'
)
```

- [ ] **Step 7: 풍선 효과 기준을 신뢰도 단계와 일치시키기**

기존 `if top_prob >= 0.50:` 조건을 다음으로 교체한다.

```python
if confidence['level'] == '높음':
    st.balloons()
```

낮음·보통 결과에서 장르를 확정하는 성공 문구를 출력하지 않는다. 단계별 설명은 결과 카드 내부에서만 표시한다.

- [ ] **Step 8: 임시 WAV 삭제를 finally로 이동**

`tmp_path = None`으로 시작하고, 업로드 저장부터 예측·그래프 표시까지 하나의 `try`로 감싼다. 기존 중간 `os.unlink(tmp_path)` 호출을 제거하고 다음 `finally`를 사용한다.

```python
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

예측 또는 시각화 오류의 `except`에서는 다음을 유지한다.

```python
except Exception as error:
    st.error(f'분석 실패: {error}')
    st.stop()
```

- [ ] **Step 9: Python syntax와 confidence unit test 확인**

Run:

```powershell
python -m py_compile app.py confidence.py
python -m pytest tests/test_confidence.py -v
```

Expected: compile exit code `0`, pytest `7 passed`.

- [ ] **Step 10: Streamlit headless startup smoke test**

Run:

```powershell
python -m streamlit run app.py --server.headless true --server.port 8502
```

Expected: `Local URL: http://localhost:8502`, startup traceback 없음. 확인 후 `Ctrl+C`로 종료한다.

- [ ] **Step 11: 브라우저 수동 검증**

Run:

```powershell
python -m streamlit run app.py
```

다음을 확인한다.

```text
1. 제목과 `서정현 · 장르예측앱` caption 유지
2. sidebar에 지원 장르 10종 유지
3. sidebar에 모델 범위 안내 추가
4. NewJeans WAV 업로드 성공
5. 14% / 13%와 유사한 결과에서 `낮은 신뢰도` 표시
6. Top-1 확률과 2위 margin 표시
7. Top-3 막대그래프와 멜스펙트로그램 유지
8. 낮은 신뢰도에서 풍선 효과 없음
```

---

### Task 3: README에 실행법과 한계 기록

**Files:**
- Modify: `README.md:54` 앞

**Interfaces:**
- Consumes: Task 1의 threshold와 Task 2의 UI 동작.
- Produces: 다른 사용자가 실행·해석·재현할 수 있는 문서.

- [ ] **Step 1: README에 장르 예측 앱 섹션 추가**

`## 참고` 앞에 다음 내용을 추가한다.

```markdown
## 장르 예측 앱

```powershell
python -m streamlit run app.py
```

WAV 파일을 업로드하면 RandomForest가 지원 장르 10종의 확률을 계산하고 Top-3와 멜스펙트로그램을 표시합니다.

### 상대적 신뢰도

화면의 `높음·보통·낮음`은 calibration된 실제 정답 확률이 아니라 Top-1 확률과 Top-1·Top-2 차이를 이용한 UI 설명용 heuristic입니다.

- 높음: Top-1 ≥ 50%, margin ≥ 15%p
- 보통: Top-1 ≥ 30%, margin ≥ 8%p
- 낮음: 나머지

### 알려진 한계

- 모델은 GTZAN의 10개 장르만 구분합니다.
- K-pop처럼 학습 범위 밖의 입력도 기존 장르 중 하나로 분류합니다.
- 첫 3초만 분석하므로 도입부가 곡 전체를 대표하지 않을 수 있습니다.
- 신뢰도 UI 개선은 모델 정확도 향상을 의미하지 않습니다.
```

- [ ] **Step 2: 문서 threshold와 코드 threshold 일치 확인**

Run:

```powershell
rg -n "0\.50|0\.30|0\.15|0\.08|50%|30%|15%p|8%p" confidence.py README.md
```

Expected: `confidence.py`와 `README.md`에서 같은 네 threshold 확인.

- [ ] **Step 3: 최종 자동 검증**

Run:

```powershell
python -m py_compile app.py confidence.py
python -m pytest -v
```

Expected: compile exit code `0`, 전체 pytest 실패 `0`.

- [ ] **Step 4: 변경 범위 최종 확인**

현재 폴더는 Git 저장소가 아니므로 다음 파일을 직접 확인한다.

```powershell
Get-Item app.py, confidence.py, README.md, tests\test_confidence.py |
    Select-Object FullName, Length, LastWriteTime
```

Expected: 네 파일만 기능 구현과 문서화 범위에 포함된다.

## Completion Evidence

완료 보고에는 다음 근거를 포함한다.

```text
- pytest 통과 개수
- py_compile 성공
- Streamlit startup URL과 traceback 없음
- 수동 UI 체크 결과
- 모델·feature pipeline을 변경하지 않았다는 확인
- 사용자 수정 caption과 sidebar 장르 목록 유지 확인
```
