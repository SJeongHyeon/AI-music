# 6강 실습 — Streamlit 장르 예측 앱
# 사용법: streamlit run app.py
# 의존 파일: model_rf.joblib, label_encoder.joblib, scaler.joblib

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Streamlit Cloud: 디스플레이 없는 환경
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
import tempfile, os, platform

from confidence import assess_confidence

# ── 한글 폰트 ──────────────────────────────────────────
_os = platform.system()
if _os == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
elif _os == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
else:
    matplotlib.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

def inject_app_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 44%, #EFF6FF 100%);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        .hero-shell {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(148, 163, 184, 0.10));
            border: 1px solid rgba(37, 99, 235, 0.16);
            border-radius: 24px;
            padding: 1.35rem 1.45rem;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.1rem;
        }
        .hero-kicker {
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #2563EB;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
        }
        .hero-subtitle {
            margin-top: 0.45rem;
            color: #475569;
            font-size: 0.98rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }
        .pill {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(37, 99, 235, 0.14);
            border-radius: 999px;
            padding: 0.35rem 0.78rem;
            font-size: 0.82rem;
            font-weight: 700;
            color: #1E3A8A;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
        }
        .result-card {
            background: white;
            border: 1px solid #DBEAFE;
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
        }
        .section-label {
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 800;
            color: #2563EB;
            margin-bottom: 0.3rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EEF2FF 0%, #F8FAFC 100%);
        }
        .sidebar-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 20px;
            padding: 1rem 0.95rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }
        .sidebar-title {
            font-size: 1rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.75rem;
        }
        .genre-chip-list {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
        .genre-chip {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(255, 255, 255, 0.9));
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 14px;
            padding: 0.45rem 0.7rem;
            color: #0F172A;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .genre-chip span {
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── 피처 컬럼 순서 (features_3_sec.csv 기준 57개) ──────
FEATURE_COLS = (
    ["chroma_stft_mean","chroma_stft_var",
     "rms_mean","rms_var",
     "spectral_centroid_mean","spectral_centroid_var",
     "spectral_bandwidth_mean","spectral_bandwidth_var",
     "rolloff_mean","rolloff_var",
     "zero_crossing_rate_mean","zero_crossing_rate_var",
     "harmony_mean","harmony_var",
     "perceptr_mean","perceptr_var",
     "tempo"]
    + [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")]
)

GENRES = ["blues","classical","country","disco",
          "hiphop","jazz","metal","pop","reggae","rock"]

GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠",
    "disco": "🪩",  "hiphop": "🎤",   "jazz": "🎷",
    "metal": "🤘",  "pop": "🎵",       "reggae": "🌴", "rock": "🎸",
}

# ── 모델 로드 (1회만) ───────────────────────────────────
@st.cache_resource
def load_models():
    """서버 시작 시 1번만 실행 — 이후 모든 요청은 캐시 반환"""
    base = os.path.dirname(os.path.abspath(__file__))
    rf = joblib.load(os.path.join(base, "model_rf.joblib"))
    le = joblib.load(os.path.join(base, "label_encoder.joblib"))
    sc = joblib.load(os.path.join(base, "scaler.joblib"))
    return rf, le, sc

# ── 피처 추출 함수 ──────────────────────────────────────
def extract_features(wav_path: str) -> np.ndarray:
    y_audio, sr = librosa.load(wav_path, sr=22050, mono=True, duration=3.0)
    feats = {}
    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
    feats["chroma_stft_mean"] = float(np.mean(chroma))
    feats["chroma_stft_var"]  = float(np.var(chroma))
    rms = librosa.feature.rms(y=y_audio)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_var"]  = float(np.var(rms))
    sc_f = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(sc_f))
    feats["spectral_centroid_var"]  = float(np.var(sc_f))
    bw = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)
    feats["spectral_bandwidth_mean"] = float(np.mean(bw))
    feats["spectral_bandwidth_var"]  = float(np.var(bw))
    ro = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)
    feats["rolloff_mean"] = float(np.mean(ro))
    feats["rolloff_var"]  = float(np.var(ro))
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    feats["zero_crossing_rate_mean"] = float(np.mean(zcr))
    feats["zero_crossing_rate_var"]  = float(np.var(zcr))
    harm, perc = librosa.effects.hpss(y_audio)
    feats["harmony_mean"]  = float(np.mean(harm))
    feats["harmony_var"]   = float(np.var(harm))
    feats["perceptr_mean"] = float(np.mean(perc))
    feats["perceptr_var"]  = float(np.var(perc))
    tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
    feats["tempo"] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    for i in range(20):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_var"]  = float(np.var(mfcc[i]))
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)

# ── 예측 함수 ───────────────────────────────────────────
def predict_genre(wav_path, rf, le, sc):
    vec = extract_features(wav_path)
    vec_sc = sc.transform(vec)
    proba = rf.predict_proba(vec_sc)[0]
    top3 = np.argsort(proba)[::-1][:3]
    return [(le.classes_[i], float(proba[i])) for i in top3]

# ── 멜스펙트로그램 ──────────────────────────────────────
def plot_melspectrogram(wav_path, title=""):
    y_audio, sr = librosa.load(wav_path, sr=22050, mono=True, duration=10.0)
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel",
        fmax=8000, ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or "멜스펙트로그램", fontsize=11)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="장르 예측기",
    page_icon="🎵",
    layout="wide",
)

inject_app_style()

st.markdown(
    """
    <div class="hero-shell">
      <div class="hero-kicker">기본 장르 예측기</div>
      <div class="hero-title">음악을 숫자로 읽는 가장 단순한 대시보드</div>
      <div class="hero-subtitle">WAV 1개를 넣으면 57개 피처로 장르를 예측하고, Top-3 확률과 멜스펙트로그램을 함께 보여줍니다.</div>
      <div class="pill-row">
        <span class="pill">RandomForest</span>
        <span class="pill">57 features</span>
        <span class="pill">Top-3 confidence</span>
        <span class="pill">3 sec audio</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 사이드바: 지원 장르 목록 ────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-title">지원 장르 (10종)</div>
          <div class="genre-chip-list">
        """
        + "".join(
            f'<div class="genre-chip"><span>{GENRE_EMOJI.get(g, "")}</span><span>{g}</span></div>'
            for g in GENRES
        )
        + """
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("모델: RandomForest (재학습, 57피처)")
    st.caption("피처: librosa 57개")
    st.caption("이 모델은 위 10개 장르만 구분합니다.")

# ── 모델 로드 ────────────────────────────────────────────
with st.spinner("모델 로딩 중... (최초 1회)"):
    try:
        rf_m, le_m, sc_m = load_models()
        st.success("모델 로드 완료", icon="✅")
    except FileNotFoundError as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        st.stop()

# ── 파일 업로더 ──────────────────────────────────────────
uploaded = st.file_uploader(
    label="WAV 파일을 업로드하세요",
    type=["wav"],
    help="3초 이상의 WAV 파일 권장 (MP3 불가 — WAV만)",
)

if uploaded is not None:
    tmp_path = None
    try:
        # 임시 파일에 저장 (librosa는 파일 경로를 요구)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # 오디오 미리듣기
        st.audio(tmp_path, format="audio/wav")

        # 예측 실행
        with st.spinner("분석 중..."):
            top3 = predict_genre(tmp_path, rf_m, le_m, sc_m)

        confidence = assess_confidence(top3)
        col1, col2 = st.columns([1, 1])

        # ── col1: 예측 결과 ────────────────────────────
        with col1:
            st.subheader("예측 결과")
            top_genre, top_prob = top3[0]
            emoji = GENRE_EMOJI.get(top_genre, "")

            with st.container(border=True):
                st.caption("MOST LIKELY")
                st.markdown(f"## {emoji} {top_genre.upper()}")
                st.markdown(
                    f":{confidence['color']}[● {confidence['level']} 신뢰도]"
                )
                st.progress(top_prob)
                st.caption(
                    f"1위 {top_prob:.1%} · "
                    f"2위와 차이 {confidence['margin']:.1%}p"
                )

                if confidence["level"] == "높음":
                    st.success(confidence["message"])
                else:
                    st.warning(confidence["message"])

            # 확률 막대그래프
            df_prob = pd.DataFrame(top3, columns=["장르", "확률"])
            fig_bar, ax_bar = plt.subplots(figsize=(5, 2.5))
            colors_bar = ["#2563EB", "#7C3AED", "#059669"]
            ax_bar.barh(
                df_prob["장르"], df_prob["확률"],
                color=colors_bar, edgecolor="white"
            )
            ax_bar.set_xlim(0, 1)
            ax_bar.set_xlabel("확률")
            ax_bar.set_title("Top-3 예측")
            for i, (_, row) in enumerate(df_prob.iterrows()):
                ax_bar.text(
                    row["확률"] + 0.01,
                    i, f"{row['확률']:.1%}",
                    va="center", fontsize=9
                )
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        # ── col2: 멜스펙트로그램 ───────────────────────
        with col2:
            st.subheader("멜스펙트로그램")
            fig_mel = plot_melspectrogram(
                tmp_path,
                title=f"{uploaded.name}"
            )
            st.pyplot(fig_mel)
            plt.close(fig_mel)
            st.info(
                "학습 범위 밖의 곡은 가장 가까운 기존 장르로 "
                "분류될 수 있습니다."
            )

        # ── 풍선 효과: 상대적 신뢰도 높음일 때 ─────────
        if confidence["level"] == "높음":
            st.balloons()

    except Exception as error:
        st.error(f"분석 실패: {error}")
        st.stop()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("왼쪽에서 WAV 파일을 업로드하면 장르를 예측합니다.")
    # 사용 예시 이미지
    st.markdown("""
    ### 사용 방법
    1. `Browse files` 버튼 클릭
    2. WAV 파일 선택 (예: `jazz.00000.wav`)
    3. 자동으로 분석 → 장르 + 멜스펙트로그램 표시

    > 팁: `genres_original` 폴더의 샘플 WAV로 먼저 테스트해보세요.
    """)
