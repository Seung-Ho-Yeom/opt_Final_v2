# opt_Final.py  (final, tabs: Overview + Simulation, subplots in one figure, sidebar inputs, no border)
# ---------------------------------------------------------------
# Bayesian Optimization UI (GPR .pkl + Torch contour)
# - Tab 1: Overview (figure descriptions & usage notes)
# - Tab 2: Simulation (combined 1x3 subplots + live history table)
# - Sidebar: initial_design_run, total_design_run, w1, w2
# - Image axes without black border
# ---------------------------------------------------------------
import os
import time
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from sklearn.pipeline import Pipeline  # type hint only
import torch
import torch.nn as nn


# ------------------ Defaults & Styles ------------------
DEFAULT_INITIAL_DESIGN = 15
DEFAULT_TOTAL_DESIGN   = 30
DEFAULT_W1 = 1.0   # weight for dp
DEFAULT_W2 = 5.0   # weight for N2_sd

FIGSIZE = (7, 6)                # base per-plot size reference
FIGSIZE_COMBINED = (FIGSIZE[0] * 3.3, FIGSIZE[1])  # one row, 3 columns
DPI = 120
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
TICK_FONTSIZE  = 10
TITLE_PAD = 6

mpl.rcParams.update({
    "figure.dpi": DPI,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
})

# ------------------ Search space & Models ------------------
search_space = [
    Real(25.0, 65.0, name='FinAngle'),
    Real(0.2, 0.75, name='FinDepth'),
    Real(0.5, 1.5, name='FinLength'),
    Real(0.01, 0.2, name='FinWidth'),
    Integer(3, 6,   name='NumFins'),
]
baseline_params = [45, 0.5, 1.0, 0.02, 4]


# ------------------ 모델 경로 (자동 로드) ------------------
MODEL_FILES = {
    "Outlet_N2_SD": "krr_rbf_Outlet_N2_SD.pkl",
    "Outlet_dp":    "krr_rbf_Outlet_dp.pkl",
}
IMG_MODEL_PATH = "shape2contour.pth"

COVER_IMAGE_CANDIDATES = [
    r"Static-Mixer-Tool.png"
]
COVER_IMAGE_PATH = next((p for p in COVER_IMAGE_CANDIDATES if os.path.exists(p)), None)


# ------------------ Torch image model ------------------
class Shape2Contour(nn.Module):
    def __init__(self, var_dim=5, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(var_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, vars):
        z = self.fc(vars)
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(z)



# ---- 업로드 대신 고정 경로에서 자동 로드 ----


def load_models():
    for key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            st.error(f"필요한 모델 파일이 없습니다: {path}")
            st.stop()
    with open(MODEL_FILES["Outlet_N2_SD"], "rb") as f:
        gpr_sd = pickle.load(f)
    with open(MODEL_FILES["Outlet_dp"], "rb") as f:
        gpr_dp = pickle.load(f)

    if not os.path.exists(IMG_MODEL_PATH):
        st.error(f"이미지 예측 모델 파일이 없습니다: {IMG_MODEL_PATH}")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model = Shape2Contour().to(device)
    img_model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=device))
    img_model.eval()

    return gpr_sd, gpr_dp, img_model, device

# ---------------------------------------------------
def show_best_in_streamlit_simple(res):
    """최종 Best case 이미지만 Streamlit에 출력"""
    best_idx = int(np.argmin(res.func_vals))
    best_row = st.session_state.history[best_idx]

    test_vars = [
        best_row["FinAngle"],
        best_row["FinDepth"],
        best_row["FinLength"],
        best_row["FinWidth"],
        best_row["NumFins"]
    ]
    test_tensor = torch.tensor(test_vars, dtype=torch.float32, device=st.session_state.device).unsqueeze(0)

    with torch.no_grad():
        out = st.session_state.img_model(test_tensor)
    pred = out.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    # 새로운 figure 생성
    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    ax.imshow(pred, interpolation="nearest")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Outlet N2 Mass Fraction (Best Case)", fontsize=6, pad=3)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    



# ------------------ Streamlit page ------------------
st.set_page_config(
    page_title="Bayesian Optimization (Fixed)",
    page_icon=COVER_IMAGE_PATH if COVER_IMAGE_PATH else None,
    layout="wide"
)

#if COVER_IMAGE_PATH:
#    st.image(COVER_IMAGE_PATH, caption="", use_container_width=True)
st.title("Static Mixer Shape Optimizer")

# Sidebar inputs
with st.sidebar:
   # st.header("Models (auto-load from current folder)")
   # st.caption(f"SD : {MODEL_FILES['Outlet_N2_SD']}")
   # st.caption(f"DP : {MODEL_FILES['Outlet_dp']}")
   # st.caption(f"IMG: {IMG_MODEL_PATH}")

    st.subheader("Optimization Settings")
    initial_design_run = st.number_input(
        "Initial design runs",
        min_value=1, max_value=1000, value=DEFAULT_INITIAL_DESIGN, step=1
    )
    total_design_run = st.number_input(
        "Total design runs",
        min_value=int(initial_design_run), max_value=5000,
        value=max(DEFAULT_TOTAL_DESIGN, int(initial_design_run)), step=1
    )
    w1 = st.number_input("Weight for dp (w1)",   min_value=0.0, max_value=100.0, value=DEFAULT_W1, step=0.1, format="%.1f")
    w2 = st.number_input("Weight for N2_sd (w2)",min_value=0.0, max_value=100.0, value=DEFAULT_W2, step=0.1, format="%.1f")

    if total_design_run < initial_design_run:
        st.warning("Total design runs 는 Initial design runs 이상이어야 합니다.")

    run_btn = st.button("▶ Run Optimization")


# Session state
if "history" not in st.session_state:  st.session_state.history = []
if "run_counter" not in st.session_state: st.session_state.run_counter = 0
if "base_dp" not in st.session_state:  st.session_state.base_dp = None
if "base_sd" not in st.session_state:  st.session_state.base_sd = None


# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["Overview", "Simulation"])

with tab1:

    st.subheader("How to use")
    st.markdown(
        """
1. 왼쪽 Sidebar에서 탐색 설정값(초기 실험 횟수(Initial design runs), 총 실험 횟수(Total design runs))과 가중치(w1, w2)를 입력합니다.   
     [w1: 입출구 압력 편차에 대한 가중치,  w2: 출구에서의 N2 농도 표준편차에 대한 가중치]

2. **Run Optimization** 버튼을 누르면, 진행 상황과 결과가 **Simulation** 탭에 표시됩니다.  

3. 계산이 완료되면 **Simulation** 탭 하단의 History/Best 표와 CSV 다운로드 버튼으로 결과를 확인/저장할 수 있습니다.
        """
    )

    if COVER_IMAGE_PATH:
        st.image(COVER_IMAGE_PATH, caption="", use_container_width=True)

with tab2:

    st.subheader("What the three plots show?")
    st.markdown(
        """
- **(Left) Optimization Progress**  
  매 반복(iteration)마다의 Objective value 추이를 표시합니다. 이 값은 목적함수에 가중치를 각각 곱하여 더한 값으로 낮을수록 좋습니다.


- **(Middle) Pareto (colored by Objective Value)**  
  Iteration마다 계산된 각 설계 케이스에서의 출구 N2 농도 표준편차(Outlet_N2_SD)와 입출구 압력편차 (Outlet_dp)를 산점도로 표시하고, 색으로 Objective value 크기 값을 나타냅니다.  
  ★ = Best(최적의 케이스),  ✕ = Baseline (초기 설계 형상),  ◯ = Current (현재 계산된 케이스)

- **(Right) Outlet N2 Mass Fraction (출구 N2 농도 표준편차)**  
  현재 계산된 케이스의 출구 단면에서의 N2 농도 Contour를 보여줍니다. mixing이 잘 일어날수록 색의 편차가 줄어듭니다. (파란색은 농도가 낮은 영역, 빨간색은 농도가 높은 영역)
        """
    )

    # Placeholders in Simulation tab
    st.subheader("Real Time Simulation Results")
    progress_bar_ph = st.empty()        # 실제 진행률 표시용
    progress_ph = st.empty()
    plots_ph    = st.empty()
    live_table_title_ph = st.empty()
    live_table_ph       = st.empty()
    final_subject       = st.empty()
    best_ph             = st.empty()
    final_best_table_ph = st.empty()
    Last_figure         = st.empty()
    download_ph         = st.empty()

    # -------- Live plotter (1x3 subplots in one figure) --------
    class LivePlotter:
        def __init__(self, container):
            self.container = container

        def update(self, res, img_model, device):
            hist = st.session_state.history

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGSIZE_COMBINED,
                                                gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
                                                #constrained_layout=True)
            fig.subplots_adjust(wspace=0.2, top=0.95)

            # 1) Progress
            y_vals = np.asarray(res.func_vals)
            x_vals = np.arange(1, len(y_vals) + 1)
            ax1.plot(x_vals, y_vals, 'o-')
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Objective Value")
            ax1.set_title("Optimization Progress", pad=TITLE_PAD)

            # 2) Pareto
            if hist and len(res.func_vals) == len(hist):
                n2 = np.array([row["Outlet_N2_SD"] for row in hist], dtype=float)
                dp = np.array([row["Outlet_dp"] for row in hist], dtype=float)
                objs = np.array(list(res.func_vals), dtype=float)

                sc = ax2.scatter(n2, dp, c=objs,cmap='hot', s=60, edgecolor='k')
                best_idx = int(np.argmin(objs))
                ax2.plot(n2[best_idx], dp[best_idx], 'r*', markersize=14, label="Best")
                ax2.plot(n2[0], dp[0], 'bX', markersize=12, label="Baseline")
                ax2.plot(n2[-1], dp[-1], 'o', markerfacecolor='none',
                         markeredgecolor='darkviolet', markeredgewidth=2, markersize=12, label="Current")

                ax2.set_xlabel("Outlet_N2_SD")
                ax2.set_ylabel("Outlet_dp")
                ax2.set_title("Pareto (colored by Objective)", pad=TITLE_PAD)
                ax2.grid(True)
                ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax2.legend()
                cbar = fig.colorbar(sc, ax=ax2, shrink=0.92)
                cbar.set_label("Objective Value")

            # 3) Image (no border)
            if hist:
                last = hist[-1]
                vars_vec = [
                    float(last["FinAngle"]),
                    float(last["FinDepth"]),
                    float(last["FinLength"]),
                    float(last["FinWidth"]),
                    float(last["NumFins"]),
                ]
                with torch.no_grad():
                    t = torch.tensor(vars_vec, dtype=torch.float32, device=device).unsqueeze(0)
                    out = img_model(t).squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

                ax3.imshow(out, interpolation="nearest")
                ax3.set_aspect("equal", adjustable="box")
                for spine in ax3.spines.values():
                    spine.set_visible(False)
                ax3.set_xticks([]); ax3.set_yticks([])
                ax3.set_xlabel(""); ax3.set_ylabel("")
                ax3.set_title(f"Outlet N2 Mass Fraction (Iter {len(res.func_vals)})", pad=TITLE_PAD)

            self.container.pyplot(fig)
            plt.close(fig)      

    # -------- Optimization runner (uses tab2 placeholders) --------
    def run_optimization():
        gpr_sd, gpr_dp, img_model, device = load_models()

        st.session_state.img_model = img_model
        st.session_state.device = device

        st.session_state.history = []
        st.session_state.run_counter = 0
        st.session_state.base_dp = None
        st.session_state.base_sd = None

        live_table_title_ph.subheader("Optimization History (live) — last 10 rows")

        @use_named_args(search_space)
        def objective(**params):
            st.session_state.run_counter += 1
            t0 = time.time()

            clean_params = {
                "FinAngle":  float(params["FinAngle"]),
                "FinDepth":  float(params["FinDepth"]),
                "FinLength": float(params["FinLength"]),
                "FinWidth":  float(params["FinWidth"]),
                "NumFins":   int(params["NumFins"]),
            }
            x = np.array(list(clean_params.values()), dtype=float).reshape(1, -1)

            outlet_n2_sd = float(gpr_sd.predict(x)[0])
            outlet_dp    = float(gpr_dp.predict(x)[0])

            if outlet_n2_sd <= 0:
                outlet_n2_sd = abs(outlet_n2_sd - 0.005)
                outlet_dp = outlet_dp + 0.5*outlet_dp

            row = OrderedDict()
            row["Case"] = int(st.session_state.run_counter)
            row.update(clean_params)
            row["Outlet_N2_SD"] = outlet_n2_sd
            row["Outlet_dp"]    = outlet_dp
            st.session_state.history.append(row)

            if st.session_state.run_counter == 1:
                st.session_state.base_dp = outlet_dp
                st.session_state.base_sd = outlet_n2_sd

            dp_norm = outlet_dp    / st.session_state.base_dp
            sd_norm = outlet_n2_sd / st.session_state.base_sd
            obj_val = w1 * dp_norm + w2 * sd_norm

            df_live = pd.DataFrame(st.session_state.history)
            live_table_ph.dataframe(df_live.tail(10), use_container_width=True)

            # --- progress bar 업데이트 ---
            progress = st.session_state.run_counter / total_design_run
            progress_bar_ph.progress(progress)   # 0.0 ~ 1.0

            dt = time.time() - t0
            progress_ph.info(f"#{st.session_state.run_counter}/{total_design_run} done in {dt:.2f}s | Objective={obj_val:.6f}")
            return float(obj_val)

        live_plotter = LivePlotter(plots_ph)

        def _cb(res):
            live_plotter.update(res, st.session_state.img_model, st.session_state.device)

        res = gp_minimize(
            func=objective,
            dimensions=search_space,
            acq_func="EI",
            n_calls=total_design_run,
            n_initial_points=initial_design_run,
            x0=[baseline_params],
            callback=[_cb],
            initial_point_generator='lhs',
            random_state=42
        )

        # Final results
        df_hist = pd.DataFrame(st.session_state.history)
        best_idx = int(np.argmin(res.func_vals))
        best_value = float(res.fun)
        best_row = df_hist.iloc[[best_idx]]

        final_subject.subheader("Final Result")
        best_ph.success(f"**최적의 케이스** Best Objective value: {best_value:.6f} at iteration #{best_idx+1}")
        final_best_table_ph.subheader("Best design (params and outputs)")
        final_best_table_ph.dataframe(best_row, use_container_width=True)

        # Final redraw to ensure last state is shown
        #live_plotter.update(res, st.session_state.img_model, st.session_state.device)
        with Last_figure:
            show_best_in_streamlit_simple(res)
        #Last_figure = show_best_in_streamlit_simple(res)

        # Download
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        download_ph.download_button(
            "Download optimization_history.csv",
            data=csv_bytes,
            file_name="optimization_history.csv",
            mime="text/csv",
        )

    # Trigger only after tab2 placeholders exist
    if run_btn:
        run_optimization()
