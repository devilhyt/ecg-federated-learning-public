import streamlit as st
import wandb
from PIL import Image
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_theme import st_theme
import pandas as pd

# ===== wandb settings =====
WANDB_RUN_PATH = "devilhyt/ecg-federated/durm1360"
api = wandb.Api()
run = api.run(WANDB_RUN_PATH)


# ===== streamlit settings =====
st.set_page_config(layout="wide")

st.html(
    """
    <style>
        .stMainBlockContainer {
            max-width:70rem;
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


# ===== main =====
# title
st.title("ECG Federated Learning Dashboard")
st.divider()

summary: dict = run.summary._json_dict



train_summary = {
    key: summary.get(key) for key in ["train_acc", "train_f1", "train_loss"]
}
valid_summary = {
    key: summary.get(key) for key in ["valid_acc", "valid_f1", "valid_loss"]
}
test_summary = {key: summary.get(key) for key in ["test_acc", "test_f1", "test_loss"]}

st.write(run.config)
st.write(summary)
st.write(train_summary)
st.write(valid_summary)
st.write(test_summary)

def create_federated_setting_card(run_config):
    COLUMNS = 3
    cols = st.columns(COLUMNS)

    for i, (key, value) in enumerate(run_config.items()):
        with cols[i % COLUMNS]:
            st.metric(label=key, value=value)

def create_metric_card(summary_dict, summary_dict_base=None):
    COLUMNS = 3
    cols = st.columns(COLUMNS)

    def display_metrics(summary_dict):
        for i, (key, value) in enumerate(summary_dict.items()):
            if key.endswith("f1") or key.endswith("acc"):
                value = f"{value:.2%}"
            elif key.endswith("loss"):
                value = f"{value:.4f}"
            with cols[i]:
                st.metric(label=key, value=value)

    def display_metrics_delta(summary_dict, summary_dict_base):
        for i, ((key, value), (key_base, value_base)) in enumerate(
            zip(summary_dict.items(), summary_dict_base.items())
        ):
            if key.endswith("f1") or key.endswith("acc"):
                value_delta = f"{value - value_base:.2%}"
                value = f"{value:.2%}"
            elif key.endswith("loss"):
                value_delta = f"{value - value_base:.4f}"
                value = f"{value:.4f}"
            with cols[i]:
                st.metric(label=key, value=value, delta=value_delta)

    # for i, title in enumerate(["Accuray", "F1 score", "Loss"]):
    #     with cols[i]:
    #         st.text(title)

    if summary_dict_base is None:
        display_metrics(summary_dict)
    else:
        display_metrics_delta(summary_dict, summary_dict_base)


# Summary
st.markdown("## Fedrated Setting")
create_federated_setting_card(run.config)

st.markdown("## Hospitals")


st.markdown("## Metric")
st.markdown("### Train")
create_metric_card(train_summary)
st.markdown("### Validation")
create_metric_card(valid_summary, train_summary)
st.markdown("### Test")
create_metric_card(test_summary, valid_summary)
