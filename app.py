import configparser
import numpy as np
import streamlit as st
import wandb
from PIL import Image
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_theme import st_theme
import pandas as pd
import neurokit2 as nk
import scipy.io as sio
from model import DenseNet1dModule
from dataset_utils import Cinc2017DataModule
from dataset_utils import Cinc2017Dataset
import torch
import plotly.graph_objects as go
import plotly.express as px
from neurokit2.hrv.hrv_time import _hrv_time_show, _hrv_format_input


# ===== wandb settings =====

WANDB_RUN_PATH = "devilhyt/ecg-federated/durm1360"
api = wandb.Api()
run = api.run(WANDB_RUN_PATH)

# ===== config =====
config = configparser.ConfigParser()
config.read("config.ini")
src_freq = config["data_preprocessing"].getint("src_freq")
dst_freq = config["data_preprocessing"].getint("dst_freq")
dst_time = config["data_preprocessing"].getint("dst_time")
dst_length = dst_time * dst_freq

# ===== streamlit settings =====
st.set_page_config(layout="wide")

st.html(
    """
    <style>
        .stMainBlockContainer {
            max-width:80rem;
        }
    </style>
    """
)

theme = st_theme()
theme_base = theme["base"] if theme else "dark"
if theme_base == "dark":
    style_metric_cards(
        background_color="#262730",
        border_left_color=None,
        border_color=None,
        box_shadow=False,
    )
else:
    style_metric_cards(
        background_color="#f0f2f6",
        border_left_color=None,
        border_color=None,
        box_shadow=False,
    )


# ===== functions =====
def file_uploader_on_change():
    """callback function to set the signal_updated flag"""
    st.session_state["signal_updated"] = True


@st.cache_resource
def load_model():
    artifact = api.artifact("devilhyt/ecg-federated/model-durm1360:v0")
    model = DenseNet1dModule.load_from_checkpoint(
        checkpoint_path="artifacts/model-durm1360:v0/best.ckpt"
    )
    model.eval()
    return model


def preprocess_signal(signal, freq, dst_freq, dst_length):
    # denoising
    signal = nk.signal_filter(
        signal,
        sampling_rate=freq,
        lowcut=0.5,
        highcut=40,
        method="butterworth",
        order=6,
    )

    # inversion correction
    signal, _ = nk.ecg_invert(signal, sampling_rate=freq)

    # downsampling
    signal = nk.signal_resample(
        signal, sampling_rate=freq, desired_sampling_rate=dst_freq
    )

    # data length standardization
    signal = np.resize(signal, dst_length)

    return signal


# ===== main =====
# init
# initilaize the app
if "signal_updated" not in st.session_state:
    st.session_state["signal_updated"] = True
model = load_model()
dm = Cinc2017DataModule()
classes = {"N": "Normal", "A": "AF", "O": "Other"}

# title
st.title("ECG Arrhythmia Detection")
st.divider()

# file uploader
st.header("Choose a ECG signal file")
uploaded_file = st.file_uploader(
    "", type=["csv", "mat"], on_change=file_uploader_on_change
)
st.divider()

# ECG plot
if uploaded_file is not None:
    # load signal file and preprocess
    if uploaded_file.name.endswith(".csv"):
        signal = np.loadtxt(uploaded_file)
        preprocessed_signal = signal
    elif uploaded_file.name.endswith(".mat"):
        signal = sio.loadmat(uploaded_file)["val"].squeeze()
        signal = signal * 0.001  # Convert to mV
        preprocessed_signal = preprocess_signal(signal, src_freq, dst_freq, dst_length)

    # calculate ECG features
    signals, info = nk.ecg_process(preprocessed_signal, sampling_rate=dst_freq)
    intervalrelated = nk.ecg_intervalrelated(signals, sampling_rate=dst_freq)
    rpeaks = info[
        "ECG_R_Peaks_Uncorrected"
    ]  # Use uncorrected R-peaks. Because we already corrected the signal using custom preprocessing function.
    # heartrate = np.mean(nk.signal_rate(rpeaks, dst_freq, desired_length=len(preprocessed_signal)))
    preprocessed_rpeaks = rpeaks / dst_freq
    time = np.arange(len(preprocessed_signal)) / dst_freq

    # prepare signal chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=preprocessed_signal, mode="lines", name="ECG signal")
    )
    fig.add_trace(
        go.Scatter(
            x=preprocessed_rpeaks,
            y=preprocessed_signal[rpeaks],
            mode="markers",
            marker=dict(color="red"),
            name="R-peaks",
        )
    )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark" if theme_base == "dark" else "plotly_white",
    )

    # arrhythmia prediction
    transformed_signal = dm.transforms(preprocessed_signal).unsqueeze(0)
    with torch.no_grad():
        y_hat = model(transformed_signal)
        pred = torch.argmax(y_hat, dim=1)[0]
        pred_label = Cinc2017Dataset.classes[pred]
        pred_class = classes[pred_label]

    # display results
    st.header("Results")
    col = st.columns(4)
    col[0].metric("Prediction", pred_class)
    col[1].metric(
        "Heart Rate",
        f"{intervalrelated['ECG_Rate_Mean'][0]:0.2f} bpm",
    )
    col[2].metric("HRV MeanNN", f'{intervalrelated["HRV_MeanNN"][0][0][0]:.2f} ms')
    col[3].metric("HRV SDNN", f'{intervalrelated["HRV_SDNN"][0][0][0]:.2f} ms')

    st.subheader("ECG Signal")
    st.plotly_chart(fig)

    fig_col = st.columns(2)
    seg_ax = nk.ecg_segment(
        preprocessed_signal, rpeaks, sampling_rate=dst_freq, show="return"
    )
    seg_ax.set_title("")
    seg_fig = seg_ax.figure
    
    rri, rri_time, rri_missing = _hrv_format_input(rpeaks, sampling_rate=dst_freq)
    hrv_time_fig = _hrv_time_show(rri)
    hrv_time_fig.suptitle("")
    hrv_time_axes = hrv_time_fig.get_axes()
    hrv_time_axes[0].set_ylabel("Count")
    
    fig_col[0].subheader("Individual Heart Beats")
    fig_col[0].pyplot(seg_fig)
    fig_col[1].subheader("Distribution of R-R intervals")
    fig_col[1].pyplot(hrv_time_fig)
    # st.write(signals)
    # st.write(info)
    # st.write(intervalrelated.to_dict())