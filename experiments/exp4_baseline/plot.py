import pandas as pd
import matplotlib.pyplot as plt
import os, re

# ── 데이터 로드 & 정제 ──────────────────────────────────
df = pd.read_csv("results/summary_Leiden-LPA_Hybrid.csv")

def extract_num(v):
    m = re.match(r"([0-9.]+)", v) if isinstance(v, str) else None
    return float(m.group(1)) if m else v

for col in ["Time (s)", "Modularity", "NMI"]:
    df[col] = df[col].apply(extract_num)

# ── 출력 폴더 ───────────────────────────────────────────
out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# ── 팔레트 정의 ────────────────────────────────────────
PALETTES = {
    "bw":    {"Leiden": "black",  "Leiden-LPA": "gray"},
    "color": {"Leiden": "#1f77b4","Leiden-LPA": "#ff7f0e"},
}

# ── 그래프 그룹 & y범위 ────────────────────────────────
time_groups = {
    "large": ["com-dblp", "com-youtube"],   # 0–max+100
    "small": ["email-Eu-core", "karate"],   # 0–0.1
}
pivot_time = df.pivot(index="Graph", columns="Algorithm", values="Time (s)")
pivot_mod  = df.pivot(index="Graph", columns="Algorithm", values="Modularity")
pivot_nmi  = df.pivot(index="Graph", columns="Algorithm", values="NMI")

# ── 공통 막대그래프 함수 ───────────────────────────────
def save_bar(pivot, ylabel, ylim, colors, fname):
    ax = pivot.plot(
        kind="bar", figsize=(7, 4), rot=0,
        color=[colors[c] for c in pivot.columns],
        edgecolor="black", linewidth=0.5,
    )
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname}", dpi=300)
    plt.close()

# ── 그래프 생성: 팔레트 × 지표 ──────────────────────────
for scheme, palette in PALETTES.items():
    # Time (s)
    # 1) large
    slice_large = pivot_time.loc[time_groups["large"]]
    ymax = slice_large.to_numpy().max() + 100
    save_bar(slice_large, "Running time (sec)", (0, ymax), palette,
             f"time_large_{scheme}.png")
    # 2) small
    slice_small = pivot_time.loc[time_groups["small"]]
    save_bar(slice_small, "Running time (sec)", (0, 0.1), palette,
             f"time_small_{scheme}.png")

    # Modularity
    save_bar(pivot_mod, "Modularity", (0,1), palette,
             f"modularity_{scheme}.png")

    # NMI
    save_bar(pivot_nmi, "NMI", (0,1), palette,
             f"nmi_{scheme}.png")

print("✓ 8 graph SAVED!")
