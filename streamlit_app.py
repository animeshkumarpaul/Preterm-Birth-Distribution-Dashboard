"""
Preterm Birth Distribution Dashboard (Alberta Pregnancy Cohort)

This Streamlit app provides population-level summaries of prescription medication use during pregnancy
and associated preterm birth outcomes, stratified by ATC drug groups.

Repository notes
----------------
- This app expects two pickle files (not included in the public repo):
  1) Aggregated cohort summaries:  pregnancy_aggregated_data_all_ATC_CODES.pkl
  2) ATC name → ATC code mapping: mapping_atc_codes.pkl

Configure their locations via environment variables:
- AGGREGATED_PICKLE_PATH (default: data/pregnancy_aggregated_data_all_ATC_CODES.pkl)
- ATC_MAPPING_PICKLE_PATH (default: data/mapping_atc_codes.pkl)

Privacy/disclosure control:
- By default, the UI only shows drug groups with >= MIN_CELL_COUNT exposed pregnancies
  (default MIN_CELL_COUNT=20), consistent with common disclosure control practices.

Author: (Animesh Kumar Paul / Dept. of CS, University of Alberta)

"""

from __future__ import annotations

import pickle
from pathlib import Path
from textwrap import wrap
from typing import Any, List, Mapping, Sequence, Tuple

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "Preterm Birth Distribution Dashboard"
APP_ICON = "📊"

def _secrets_available() -> bool:
    """Return True if Streamlit secrets appear to be configured.

    Some Streamlit versions raise an exception as soon as `st.secrets` is accessed when no secrets file
    exists. To remain compatible across Streamlit versions, we check for the canonical secrets paths
    before touching `st.secrets`.
    """
    candidates = [
        Path("/app/.streamlit/secrets.toml"),
        Path("/app/.streamlit/secrets"),
        Path("/root/.streamlit/secrets.toml"),
        Path("/root/.streamlit/secrets"),
        Path.home() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets",
    ]
    for p in candidates:
        try:
            if p.is_file() and p.name == "secrets.toml":
                return True
            if p.is_dir() and p.name == "secrets":
                return True
        except Exception:
            continue
    return False


def _get_setting(key: str, default: str) -> str:
    """Return a configuration value from env vars, then (optionally) Streamlit secrets.
    """
    env_val = os.environ.get(key)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val)

    if hasattr(st, "secrets") and _secrets_available():
        try:
            return str(st.secrets.get(key, default))
        except Exception:
            return str(default)

    return str(default)


def _get_int_setting(key: str, default: int) -> int:
    val = _get_setting(key, str(default))
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


DEFAULT_AGGREGATED_PICKLE_PATH = Path(
    _get_setting("AGGREGATED_PICKLE_PATH", "data/pregnancy_aggregated_data_all_ATC_CODES.pkl")
)

DEFAULT_MAPPING_PICKLE_PATH = Path(
    _get_setting("ATC_MAPPING_PICKLE_PATH", "data/mapping_atc_codes.pkl")
)

MIN_CELL_COUNT = _get_int_setting("MIN_CELL_COUNT", 20)
PLOT_STYLE = {
    "theme": "whitegrid",
    "font_size": 12,
}

PALETTE_MAIN = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#085396",
    "#b2df8a",
    "#8da0cb",
    "#fc8d62",
    "#66c2a5",
    "#1f78b4",
]
PALETTE_EXPOSURE = ["#085396", "#b2df8a"]  # exposed, unexposed


# -----------------------------
# Utilities
# -----------------------------
def _safe_load_pickle(path: Path) -> Any:
    """Load a pickle file with a clear Streamlit error message on failure."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Missing required data file: `{path}`.\n\n"
            "If you are running from a public GitHub repo, you likely need to provide the "
            "private pickle files (see README) or set AGGREGATED_PICKLE_PATH / ATC_MAPPING_PICKLE_PATH."
        )
        st.stop()
    except Exception as e:  # noqa: BLE001 (we want a user-facing error)
        st.error(f"Failed to load pickle `{path}`: {type(e).__name__}: {e}")
        st.stop()


def _capitalize_dict_keys(obj: Any) -> Any:
    """Recursively capitalize dict keys (used to standardize ATC names)."""
    if isinstance(obj, dict):
        return {
            (k.capitalize() if isinstance(k, str) else k): _capitalize_dict_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_capitalize_dict_keys(v) for v in obj]
    return obj


def _set_plot_defaults() -> None:
    """Apply consistent plotting defaults for the app."""
    sns.set_theme(style=PLOT_STYLE["theme"])
    plt.rcParams.update(
        {
            "font.size": PLOT_STYLE["font_size"],
            "axes.titlesize": PLOT_STYLE["font_size"] + 1,
            "axes.labelsize": PLOT_STYLE["font_size"],
        }
    )


def _format_percent_axis_0_1(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))


def _format_percent_axis_0_100(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y / 100)))


def _annotate_bar_values(ax, values: Sequence[int]) -> None:
    """Annotate bar heights with integer counts."""
    for rect, v in zip(ax.patches, values):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            f"{int(v):,}",
            ha="center",
            va="bottom",
        )


def _annotate_two_group_counts(ax, group1: Sequence[int], group2: Sequence[int]) -> None:
    """Annotate two-group barplots where bars are ordered group1 then group2."""
    labels: List[str] = [f"{int(v):,}" for v in group1] + [f"{int(v):,}" for v in group2]
    for rect, label in zip(ax.patches, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
        )


def _wrap_text(s: str, width: int) -> str:
    return "\n".join(wrap(s, width=width))


def _get_exposed_total(history_dict: Sequence[Any]) -> int:
    """
    Return total *exposed* pregnancies for a drug group.

    Expected structure (as in the provided pickles):
        history_dict[0] = preterm_info
        preterm_info[0] = exposed_group
        exposed_group[2] = [preterm_count, term_count]
    """
    try:
        counts = history_dict[0][0][2]
        return int(np.sum(counts))
    except Exception:
        return 0


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(
    aggregated_path: Path = DEFAULT_AGGREGATED_PICKLE_PATH,
    mapping_path: Path = DEFAULT_MAPPING_PICKLE_PATH,
) -> Tuple[Mapping[str, Any], Mapping[str, str]]:
    """
    Load aggregated data and ATC mapping.

    Returns
    -------
    data : Mapping[str, Any]
        Dict keyed by ATC code → history_dict
    mapping : Mapping[str, str]
        Dict keyed by ATC name (capitalized) → ATC code
    """
    data = _safe_load_pickle(aggregated_path)
    mapping = _capitalize_dict_keys(_safe_load_pickle(mapping_path))
    if not isinstance(mapping, dict):
        st.error("ATC mapping pickle must be a dictionary.")
        st.stop()
    return data, mapping


# -----------------------------
# Plot builders
# -----------------------------
def build_preterm_distribution(history_dict: Sequence[Any], drug_label: str) -> plt.Figure:
    preterm_info = history_dict[0]

    df = pd.DataFrame(
        [
            {
                "Birth Status": preterm_info[0][0][0],
                "Percentage of Pregnant women": preterm_info[0][1][0],
                "Drug Usage": "Received ≥1 prescription",
            },
            {
                "Birth Status": preterm_info[0][0][1],
                "Percentage of Pregnant women": preterm_info[0][1][1],
                "Drug Usage": "Received ≥1 prescription",
            },
            {
                "Birth Status": preterm_info[1][0][0],
                "Percentage of Pregnant women": preterm_info[1][1][0],
                "Drug Usage": "No prescription",
            },
            {
                "Birth Status": preterm_info[1][0][1],
                "Percentage of Pregnant women": preterm_info[1][1][1],
                "Drug Usage": "No prescription",
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(
        x="Birth Status",
        y="Percentage of Pregnant women",
        hue="Drug Usage",
        data=df,
        ax=ax,
        palette=PALETTE_EXPOSURE,
    )
    ax.set_xlabel("Birth status")
    ax.set_ylabel("Percentage of pregnancies")
    ax.set_xticklabels(["Preterm", "Term"])
    ax.set_ylim([0, 1])
    _format_percent_axis_0_1(ax)

    # annotate counts
    _annotate_two_group_counts(ax, preterm_info[0][2], preterm_info[1][2])

    fig.tight_layout()
    return fig


def build_age_distribution(history_dict: Sequence[Any], drug_label: str) -> plt.Figure:
    age_info = history_dict[1]

    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(
        x="Age",
        y="Percentage of Pregnant women",
        hue="Birth Status",
        data=age_info[0][0],
        ax=ax,
        palette=PALETTE_MAIN,
    )
    ax.set_xlabel("Maternal age")
    ax.set_ylabel("Percentage of pregnancies")

    max_pct = float(np.max(age_info[0][0]["Percentage of Pregnant women"]))
    y_limit = min(max_pct + 10, 100)

    _annotate_two_group_counts(ax, age_info[0][1], age_info[0][2])
    _format_percent_axis_0_100(ax)
    ax.set_ylim([0, y_limit])
    ax.set_title(f"Received ≥1 prescription of\n{drug_label}")
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def build_trimester_distribution(history_dict: Sequence[Any]) -> plt.Figure:
    trimester_info = history_dict[2]

    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(
        x="Trimester",
        y="Percentage of Pregnant women",
        hue="Birth Status",
        data=trimester_info[0][0],
        ax=ax,
        palette=PALETTE_MAIN,
    )
    ax.set_xlabel("Trimester of first prescription")
    ax.set_ylabel("Percentage of pregnancies")
    ax.set_ylim([0, 100])
    _format_percent_axis_0_100(ax)

    _annotate_two_group_counts(ax, trimester_info[0][1], trimester_info[0][2])

    fig.tight_layout()
    return fig


def build_diagnosis_plot(history_dict: Sequence[Any]) -> plt.Figure:
    diagnosis_code = history_dict[3]
    temp_icd = diagnosis_code[0]
    icd_labels = diagnosis_code[1]

    # top 10 codes by frequency (as in original script)
    top_codes = list(temp_icd.codes.value_counts().iloc[:10].index)

    fig, ax = plt.subplots(figsize=(20, 12))
    sns.countplot(
        y="codes",
        data=temp_icd,
        ax=ax,
        order=top_codes,
        palette="Greens_d",
    )
    ax.set_xlabel("Number of pregnancies")
    ax.set_ylabel("ICD-9 code")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Robust ytick labels:
    # - If icd_labels is a dict, map code->label
    # - If icd_labels is a list of length 10, use it as-is
    # - Otherwise, fallback to codes
    tick_labels: List[str]
    if isinstance(icd_labels, dict):
        tick_labels = [str(icd_labels.get(c, c)) for c in top_codes]
    elif isinstance(icd_labels, list) and len(icd_labels) == len(top_codes):
        tick_labels = [str(l) for l in icd_labels]
    else:
        tick_labels = [str(c) for c in top_codes]

    tick_labels = [_wrap_text(l, width=60) for l in tick_labels]
    ax.set_yticks(range(len(top_codes)), labels=tick_labels, fontsize=12)

    fig.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
def render_overview() -> None:
    with st.expander("Study overview", expanded=True):
        st.markdown(
            """
This dashboard presents population-level data on prescription medication use during pregnancy and its association with preterm birth outcomes, using data from the **Alberta Pregnancy Cohort**.

**Study Population**  
- The dataset includes **238,676 mothers** and **328,834 singleton live births** in Alberta, Canada, between **2009 and 2018**.  
- Only **singleton live births** were included; **stillbirths and multiple gestations were excluded**.  
- Inclusion criteria required maternal **residency in Alberta for at least one year** before and throughout pregnancy.  
- Records with **missing gestational age or birth weight** were excluded.

**Data Sources**  
- The cohort links **pharmaceutical claims**, **birth outcomes**, **gestational age**, **maternal demographic information**, and **physician diagnostic codes**.

**Medication Exposure Definition**  
- Exposure was defined as **at least one pharmaceutical claim** for a medication between the estimated **first day of the last menstrual period (LMP)** and the **date of delivery**.  
- **Claims prior to LMP were not considered exposure during pregnancy**.

**Medication Classification**  
- Medications were grouped using the **Anatomical Therapeutic Chemical (ATC)** classification system at the **3-character level**, corresponding to therapeutic drug classes.  
- To ensure compliance with privacy regulations and standard disclosure control practices, only medications used by **at least 20 pregnant individuals** are displayed.  
- This threshold is consistent with Canadian disclosure standards to reduce the risk of re-identification in health data reporting.

**Visualizations Provided in the Tool**  
- **Preterm birth distribution** by medication exposure (exposed vs. unexposed)  
- **Maternal age distributions** stratified by birth outcome (preterm vs. term)  
- **Trimester of first prescription use** (first, second, or third trimester)  
- **Top 10 physician diagnosis codes** recorded within one year prior to drug dispensation  

This tool is intended as an exploratory, real-world data resource to support clinical awareness and public health insight, rather than causal inference.
""".format(
                min_cell=MIN_CELL_COUNT
            )
        )

    with st.expander("⚠️ Limitations & disclaimers", expanded=False):
        st.markdown(
            """
- **Observational Cohort Data**: This tool summarizes associations from real-world data and does not imply causation.
- **Confounding**: Associations may be influenced by maternal age, comorbidities, and other unadjusted factors.
- **Confounding by Indication**: Underlying medical conditions may themselves be associated with preterm birth, independent of medication exposure.
- **Population-Specific**: Data are specific to **Alberta** and may not be generalizable to other regions.
- **Exclusion of Stillbirths**: The cohort includes only **singleton live births**; stillbirths are not captured.
- **Route of Administration Not Distinguished**: Medication claims are grouped by ATC category regardless of administration method.
"""
        )
    with st.expander("👩‍👧 Population Demographics (Alberta Cohort)", expanded=False):
        st.markdown(
            """
To support interpretation, here is a summary of key demographics from the Alberta Pregnancy Cohort (2009–2018):

- **Mean Maternal Age**: ~30 years
- **Urban vs. Rural Residence**: ~91% urban, ~9% rural
- **Preterm Birth Rate**: ~6.5% of singleton live births
- **Married & Husband is the biological father of the child**: ~70%
"""
        )
    with st.expander("📄Citation", expanded=False):
        st.markdown(
            """
Paul, A.K., Kalmady, S.V., Greiner, R. et al. Developing point-of-care tools to inform decisions regarding prescription medication use in pregnancy. npj Womens Health 3, 43 (2025). https://doi.org/10.1038/s44294-025-00093-9
"""
        )


def render_sidebar(
    mapping: Mapping[str, str],
    data: Mapping[str, Any],
) -> Tuple[str, str]:
    st.sidebar.header("Controls")

    # Filter ATC groups based on disclosure threshold
    eligible_names: List[str] = []
    for atc_name, atc_code in mapping.items():
        history = data.get(atc_code)
        if history is None:
            continue
        if _get_exposed_total(history) >= MIN_CELL_COUNT:
            eligible_names.append(atc_name)

    eligible_names = sorted(eligible_names)
    if not eligible_names:
        st.sidebar.warning(
            "No drug groups meet the disclosure threshold (MIN_CELL_COUNT). "
            "Check your data files or lower MIN_CELL_COUNT."
        )
        st.stop()

    default_index = min(3, len(eligible_names) - 1)
    selected_name = st.sidebar.selectbox("Drug group (ATC level-3)", eligible_names, index=default_index)
    selected_code = mapping[selected_name]

    st.sidebar.caption(f"ATC code: {selected_code}")

    return selected_name, selected_code


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    _set_plot_defaults()

    st.title(APP_TITLE)

    data, mapping = load_data()

    render_overview()
    selected_name, selected_code = render_sidebar(mapping, data)

    history_dict = data[selected_code]
    wrapped_label = _wrap_text(selected_name, width=30)

    st.subheader(f"{selected_name}")

    # -------- Figure 1 --------
    fig1 = build_preterm_distribution(history_dict, wrapped_label)
    st.pyplot(fig1)
    st.markdown(
        """
**Figure 1. Preterm distribution by drug exposure.**  
This figure shows the proportion of preterm versus term births among pregnant women who received at least one prescription of the selected drug compared to those who did not. The percentages on the bars indicate the relative distribution of preterm (Preterm) and term (Term) births.
"""
    )

    # -------- Figure 2 --------
    fig2 = build_age_distribution(history_dict, wrapped_label)
    st.pyplot(fig2)
    st.markdown(
        """
**Figure 2. Age Distribution and Birth Outcomes.**  
This figure presents the distribution of maternal ages among pregnant women, stratified by birth status (Preterm vs. Term). The bar chart highlights how the percentage of pregnancies in each age group relates to birth outcomes.
"""
    )

    # -------- Figure 3 --------
    fig3 = build_trimester_distribution(history_dict)
    st.pyplot(fig3)
    st.markdown(
        """
**Figure 3. Trimester of first prescription use.**  
This figure illustrates the distribution of the trimester in which pregnant women first received the selected drug. The data is stratified by birth status, showing the percentage of women in the first, second, and third trimesters.
"""
    )

    # -------- Figure 4 --------
    fig4 = build_diagnosis_plot(history_dict)
    st.pyplot(fig4)
    st.markdown(
        """
**Figure 4. Diagnosis Codes Prior to Drug Dispensation Date.**  
This figure displays the top 10 ICD-9 diagnosis codes recorded within one year prior to the drug dispensation date. The horizontal bar chart represents the number of pregnant women with each diagnosis code, with large numbers formatted with comma separators for clarity.
"""
    )


if __name__ == "__main__":
    main()
