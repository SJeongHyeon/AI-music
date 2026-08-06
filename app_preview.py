# 6강 실습 — Streamlit 장르 예측 앱
# 사용법: streamlit run app.py

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

GENRES = ["blues","classical","country","disco","hiphop",
          "jazz","metal","pop","reggae","rock"]

st.set_page_config(page_title="장르 분류 AI", page_icon="music")
st.title("장르 분류 AI")
st.markdown("WAV 파일을 업로드하면 AI가 장르를 예측합니다.")

@st.cache_resource
def load_model():
    """5강에서 학습한 모델 또는 RandomForest 로드"""
    # TODO 6강에서 실제 모델로 교체
    return None

def extract_features(y, sr):
    """librosa로 멜스펙트로그램 + MFCC 추출"""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mel_db, mfcc

uploaded = st.file_uploader("WAV 파일 업로드", type=["wav", "mp3"])

if uploaded:
    audio_bytes = uploaded.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=30)

    st.audio(audio_bytes)

    mel_db, mfcc = extract_features(y, sr)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("멜스펙트로그램")
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                  ax=ax, cmap="magma")
        ax.set_title("멜스펙트로그램")
        st.pyplot(fig)

    with col2:
        st.subheader("장르 예측")
        # TODO: 실제 모델 예측으로 교체
        import random
        random.seed(42)
        probs = np.random.dirichlet(np.ones(10))
        top_idx = probs.argsort()[::-1][:3]
        for i, idx in enumerate(top_idx):
            emoji = ["trophy", "2nd_place_medal", "3rd_place_medal"][i]
            st.metric(f"{i+1}위 장르", GENRES[idx], f"{probs[idx]*100:.1f}%")

    st.subheader("MFCC 히트맵")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm")
    ax2.set_title("MFCC (13차원)")
    ax2.set_xlabel("시간 프레임")
    ax2.set_ylabel("MFCC 계수")
    st.pyplot(fig2)

    st.info("6강에서 실제 모델 연결 및 배포를 완성합니다!")