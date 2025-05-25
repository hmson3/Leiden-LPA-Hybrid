import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

# ───────────────── CSV 로드 & 숫자 추출 ──────────────────────────────
df = pd.read_csv("results/summary_ratio.csv")

def extract_mean(s: str) -> float:
    return float(re.match(r"([0-9.]+)", s).group(1))

df["Time"] = df["Time (s)"].apply(extract_mean)
df["Mod"]  = df["Modularity"].apply(extract_mean)
df["NMI"]  = df["NMI"].apply(extract_mean)

# ───────────────── 디렉터리 준비 ────────────────────────────────────
os.makedirs("graphs", exist_ok=True)

LABEL_SIZE = 20     # 축 라벨 글꼴
PADDING    = 0.10     # 여유 비율 (5%)
X_TICKS  = [i/10 for i in range(0,11,2)] 


metrics = [
    ("Time", "Running time (sec)"),
    ("Mod",  "Modularity"),
    ("NMI",  "NMI"),
]

# ───────────────── 그래프별(각 Graph) 저장 ───────────────────────────
for g in df["Graph"].unique():
    sub = df[df["Graph"] == g]
    g_dir = f"graphs/{g}"; os.makedirs(g_dir, exist_ok=True)

    for col, ylab in metrics:
        ymin, ymax = sub[col].min(), sub[col].max()
        if ymin == ymax:                      # 값이 모두 같을 때
            ymin = 0
        pad = (ymax - ymin) * PADDING
        ymin, ymax = ymin - pad, ymax + pad

        ax = sns.lineplot(
            data=sub, x="Core Ratio", y=col,
            marker="o", color="black"
        )
        ax.set_title("")
        ax.set_xlabel("Core Ratio", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylab, fontsize=LABEL_SIZE)
        ax.set_xticks(X_TICKS)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_ylim(ymin, ymax)
        plt.tight_layout()

        plt.savefig(f"{g_dir}/{col.lower()}.png")
        plt.close()
        print("✅ saved:", f"{g_dir}/{col.lower()}.png")

# ───────────────── 전체(overall) 그래프 ─────────────────────────────
for col, ylab in metrics:
    ymin, ymax = df[col].min(), df[col].max()
    pad = (ymax - ymin) * PADDING
    ymin, ymax = ymin - pad, ymax + pad

    ax = sns.lineplot(
        data=df, x="Core Ratio", y=col,
        hue="Graph", marker="o"
    )
    ax.set_title(f"{col} vs Core Ratio (All Graphs)")
    ax.set_xlabel("Core Ratio", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylab, fontsize=LABEL_SIZE)
    ax.set_ylim(ymin, ymax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    fname = f"graphs/overall_{col.lower()}.png"
    plt.savefig(fname); plt.close()
    print("📊 saved overall:", fname)
