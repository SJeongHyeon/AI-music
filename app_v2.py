# ┌────────────────────────────────────────────────────────┐
# │ [셀 14/16] app_v2.py 저장 (섹션 1/9)                    │
# └────────────────────────────────────────────────────────┘
# 7강 실습 — Streamlit 장르 예측 앱 v2
# 신뢰도 bar chart + session_state 이력 + 멀티파일 비교
# 사용법: streamlit run app_v2.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
import tempfile, os, platform
# ── 한글 폰트 설정 ─────────────────────────────────────
from pathlib import Path
import matplotlib.font_manager as fm

def setup_korean_font():
    system = platform.system()
    preferred = {
        "Darwin": ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"],
        "Windows": ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Noto Sans CJK KR"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"],
    }
    font_files = {
        "Darwin": [
            "/System/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            str(Path.home() / "Library/Fonts/NanumGothic.ttf"),
        ],
        "Windows": ["C:/Windows/Fonts/malgun.ttf"],
        "Linux": [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        ],
    }
    for font_path in font_files.get(system, []):
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((font for font in preferred.get(system, []) if font in available), None)
    # Streamlit Cloud 배포(Linux) 환경엔 한글 폰트가 기본 설치돼 있지 않을 수 있어,
    # 없으면 apt-get으로 나눔고딕을 설치한 뒤 다시 탐색합니다.
    if chosen is None and system == "Linux":
        import os
        os.system("apt-get -qq -y install fonts-nanum > /dev/null 2>&1")
        for _fp in fm.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/nanum"]):
            fm.fontManager.addfont(_fp)
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = next((font for font in preferred.get(system, []) if font in available), None)
    if chosen is None:
        chosen = "DejaVu Sans"
    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["font.sans-serif"] = [chosen, "NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen

setup_korean_font()

def inject_app_style():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 24%),
                linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 42%, #EEF2FF 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.6rem;
            max-width: 1220px;
        }
        .hero-shell {
            background: linear-gradient(135deg, #1E293B 0%, #312E81 52%, #2563EB 100%);
            color: white;
            border-radius: 28px;
            padding: 1.45rem 1.6rem;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.1rem;
        }
        .hero-kicker {
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.86);
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .hero-title {
            font-size: 2.1rem;
            line-height: 1.14;
            font-weight: 850;
            color: white;
            margin: 0;
        }
        .hero-subtitle {
            margin-top: 0.5rem;
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.98rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }
        .pill {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 700;
            color: white;
            backdrop-filter: blur(6px);
        }
        .stat-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.1rem 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(99, 102, 241, 0.14);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .stat-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6366F1;
            font-weight: 800;
        }
        .stat-value {
            font-size: 1.08rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 0.25rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EEF2FF 0%, #F8FAFC 100%);
        }
        .sidebar-card {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(49, 46, 129, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 1rem 0.95rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.16);
        }
        .sidebar-title {
            font-size: 1rem;
            font-weight: 800;
            color: white;
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
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 14px;
            padding: 0.45rem 0.7rem;
            color: white;
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
COLORS = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "accent":    "#059669",
    "neutral":   "#6B7280",
    "warning":   "#F59E0B",
}
GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠",
    "disco": "🪩",  "hiphop": "🎤",   "jazz": "🎷",
    "metal": "🤘",  "pop": "🎵",      "reggae": "🌴", "rock": "🎸",
}
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    rf = joblib.load(os.path.join(base, "model_rf.joblib"))
    le = joblib.load(os.path.join(base, "label_encoder.joblib"))
    sc = joblib.load(os.path.join(base, "scaler.joblib"))
    return rf, le, sc

def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=3.0)
    feats = {}
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    feats["chroma_stft_mean"] = float(np.mean(ch)); feats["chroma_stft_var"] = float(np.var(ch))
    rm = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(np.mean(rm)); feats["rms_var"] = float(np.var(rm))
    sc2 = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(sc2)); feats["spectral_centroid_var"] = float(np.var(sc2))
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["spectral_bandwidth_mean"] = float(np.mean(bw)); feats["spectral_bandwidth_var"] = float(np.var(bw))
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats["rolloff_mean"] = float(np.mean(ro)); feats["rolloff_var"] = float(np.var(ro))
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zero_crossing_rate_mean"] = float(np.mean(zcr)); feats["zero_crossing_rate_var"] = float(np.var(zcr))
    harm, perc = librosa.effects.hpss(y)
    feats["harmony_mean"] = float(np.mean(harm)); feats["harmony_var"] = float(np.var(harm))
    feats["perceptr_mean"] = float(np.mean(perc)); feats["perceptr_var"] = float(np.var(perc))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mf[i]))
        feats[f"mfcc{i+1}_var"]  = float(np.var(mf[i]))
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
def predict_with_confidence(wav_path, rf, le, sc):
    vec    = extract_features(wav_path)
    vec_sc = sc.transform(vec)
    proba  = rf.predict_proba(vec_sc)[0]
    prob_dict = {le.classes_[i]: float(proba[i]) for i in range(len(proba))}
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    return list(prob_dict.keys())[0], prob_dict

def plot_confidence_bar(prob_dict, title=""):
    genres = list(prob_dict.keys())
    probs  = list(prob_dict.values())
    top_g  = genres[0]
    colors = [COLORS["primary"] if g == top_g else COLORS["neutral"] for g in genres]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(genres[::-1], probs[::-1], color=colors[::-1],
            edgecolor="white", height=0.55, alpha=0.88)
    for g, p in zip(genres[::-1], probs[::-1]):
        if p > 0.01:
            ax.text(p + 0.01, genres[::-1].index(g), f"{p:.1%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("확률")
    ax.set_title(title or f"1위: {top_g.upper()}", fontsize=11)
    ax.axvline(0.5, color=COLORS["warning"], lw=1.2, linestyle="--", label="50% 기준")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
def plot_melspectrogram(wav_path, title=""):
    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=10.0)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                    fmax=8000, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or "멜스펙트로그램", fontsize=10)
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
      <div class="hero-kicker">7강 · 고급 예측 대시보드</div>
      <div class="hero-title">신뢰도, 이력, 멀티파일 비교를 한 화면에</div>
      <div class="hero-subtitle">6강보다 더 정보 밀도가 높은 버전으로, 여러 곡의 확률 분포와 예측 기록을 함께 확인합니다.</div>
      <div class="pill-row">
        <span class="pill">Confidence chart</span>
        <span class="pill">Session history</span>
        <span class="pill">Multi-file compare</span>
        <span class="pill">Streamlit dashboard</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="stat-strip">
      <div class="stat-card">
        <div class="stat-label">Input</div>
        <div class="stat-value">WAV 업로드 + 멀티파일</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Output</div>
        <div class="stat-value">Top-1, 확률 분포, 멜스펙트로그램</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Focus</div>
        <div class="stat-value">신뢰도 + 이력 + 비교 시각화</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 이력 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

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
    st.caption("모델: RandomForest (57피처)")
    st.caption("피처: librosa 57개")
    st.caption("이 앱은 위 10개 장르만 구분합니다.")
    st.caption("신뢰도 + 이력 + 멀티파일 비교")

with st.spinner("모델 로딩 중... (최초 1회)"):
    try:
        rf_m, le_m, sc_m = load_models()
        st.success("모델 로드 완료", icon="✅")
    except FileNotFoundError as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        st.stop()

uploaded_files = st.file_uploader(
    label="WAV 파일을 업로드하세요",
    type=["wav"],
    accept_multiple_files=True,
    help="3초 이상의 WAV 파일 권장 (여러 개를 업로드하면 비교 차트가 표시됩니다)",
)

def _record_history(fname, probs):
    st.session_state["history"].append({
        "파일명": fname,
        "1위 장르": list(probs.keys())[0],
        "1위 확률": f"{list(probs.values())[0]:.1%}",
        "2위 장르": list(probs.keys())[1],
        "고신뢰도": "✓" if list(probs.values())[0] >= 0.5 else "△",
    })

if uploaded_files:
    tmp_paths = []
    try:
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uf.read())
                tmp_paths.append((uf.name, tmp.name))

        if len(tmp_paths) == 1:
            fname, fpath = tmp_paths[0]
            st.audio(fpath, format="audio/wav")
            with st.spinner("분석 중..."):
                top_g, probs = predict_with_confidence(fpath, rf_m, le_m, sc_m)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("예측 결과")
                st.markdown(f"## {GENRE_EMOJI.get(top_g, '')} {top_g.upper()}")
                st.caption(f"1위 {probs[top_g]:.1%} · 2위 {list(probs.keys())[1]} {list(probs.values())[1]:.1%}")
                fig_bar = plot_confidence_bar(probs, title=fname)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            with col2:
                st.subheader("멜스펙트로그램")
                fig_mel = plot_melspectrogram(fpath, title=fname)
                st.pyplot(fig_mel)
                plt.close(fig_mel)

            _record_history(fname, probs)
        else:
            st.subheader(f"{len(tmp_paths)}개 곡 비교")
            compare_rows = []
            all_probs_list = []
            for fname, fpath in tmp_paths:
                top_g, probs = predict_with_confidence(fpath, rf_m, le_m, sc_m)
                compare_rows.append({
                    "파일명": fname,
                    "1위 장르": top_g,
                    "1위 확률": f"{probs[top_g]:.1%}",
                })
                all_probs_list.append([probs.get(g, 0.0) for g in GENRES])
                _record_history(fname, probs)

            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
            x = np.arange(len(GENRES))
            width = 0.8 / len(tmp_paths)
            palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["warning"]]
            fig_cmp, ax_cmp = plt.subplots(figsize=(12, 4))
            for i, (probs_row, (fname, _)) in enumerate(zip(all_probs_list, tmp_paths)):
                offset = (i - len(tmp_paths) / 2 + 0.5) * width
                ax_cmp.bar(
                    x + offset,
                    probs_row,
                    width,
                    label=fname,
                    color=palette[i % len(palette)],
                    alpha=0.82,
                    edgecolor="white",
                )
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(GENRES, rotation=30, ha="right")
            ax_cmp.set_ylabel("확률")
            ax_cmp.set_ylim(0, 1.05)
            ax_cmp.set_title("멀티파일 장르 확률 비교", fontsize=11, fontweight="bold")
            ax_cmp.legend(fontsize=9)
            ax_cmp.grid(axis="y", alpha=0.3)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)
    except Exception as error:
        st.error(f"분석 실패: {error}")
        st.stop()
    finally:
        for _, fpath in tmp_paths:
            try:
                os.unlink(fpath)
            except OSError:
                pass

if st.session_state["history"]:
    st.divider()
    st.subheader("예측 이력")
    df_h = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_h, use_container_width=True)
    if st.button("이력 초기화"):
        st.session_state["history"] = []
        st.rerun()
