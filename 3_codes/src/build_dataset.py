"""Assemble the ML table from printed tables + digitized figure points."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from digitized_data import (
    ETA_SOLVENT_PA_S,
    FIG05,
    FIG06_300,
    FIG06_400,
    FIG07,
    FIG08,
    FIG12A,
    FIG12B,
    FIG13,
    FIG14A,
    FIG14B,
    FIG15B,
    FIG16,
    FIG17,
    RHO_KG_M3,
    TABLE1_PAAM,
    TABLE2_PEO,
    f_laminar,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def blasius_fanning(Re: float) -> float:
    """Newtonian turbulent Fanning line used as the paper's Blasius reference."""
    return 0.079 * Re ** (-0.25)


def virk_mdr_fanning(Re: float) -> float:
    """Virk maximum-drag-reduction Fanning friction factor (iterative)."""
    f = 0.01
    for _ in range(40):
        rhs = 19.0 * np.log10(Re * np.sqrt(f)) - 32.4
        f_new = 1.0 / (rhs**2)
        if abs(f_new - f) < 1e-10:
            break
        f = 0.5 * f + 0.5 * f_new
    return float(f)


def interp_props(polymer: str, Cp: float) -> tuple[float, float]:
    """Interpolate eta0 (Pa s) and lambda (s) vs concentration from Tables 1–2."""
    if polymer == "PAAm":
        rows = [(c, e * 1e-3, lam * 1e-3) for c, _, _, e, lam in TABLE1_PAAM if _ == 0.49 or c == 0]
        # unique Cp from the 0.49 mm series (has the full concentration sweep)
        seen = {}
        for c, e, lam in rows:
            seen[c] = (e, lam)
        cps = np.array(sorted(seen))
        etas = np.array([seen[c][0] for c in cps])
        lams = np.array([seen[c][1] for c in cps])
    else:
        seen = {}
        for c, _, _, e, lam in TABLE2_PEO:
            seen[c] = (e * 1e-3, lam * 1e-3)
        # add a dilute end-member so 0.5–10 ppm can be interpolated
        seen[0.0] = (ETA_SOLVENT_PA_S, 0.0)
        cps = np.array(sorted(seen))
        etas = np.array([seen[c][0] for c in cps])
        lams = np.array([seen[c][1] for c in cps])

    eta = float(np.interp(Cp, cps, etas))
    lam = float(np.interp(Cp, cps, lams))
    return eta, lam


def derived(Re, Cp, D_mm, polymer):
    eta0, lam = interp_props(polymer, Cp)
    D = D_mm * 1e-3
    V = Re * eta0 / (RHO_KG_M3 * D)
    gamma_avg = 8.0 * V / D
    Wi = lam * gamma_avg
    beta = min(ETA_SOLVENT_PA_S / eta0, 1.0)
    E = lam * eta0 / (RHO_KG_M3 * (D / 2.0) ** 2)
    return {
        "eta0_Pa_s": eta0,
        "lambda_s": lam,
        "V_m_s": V,
        "Wi": Wi,
        "beta": beta,
        "Wi_1m_beta": Wi * (1.0 - beta),
        "E": E,
        "E_1m_beta": E * (1.0 - beta),
        "f_16Re": f_laminar(Re),
        "f_Blasius": blasius_fanning(Re),
        "f_Virk": virk_mdr_fanning(Re),
    }


def add_row(rows, *, polymer, Cp, D_mm, Re, f, source, series):
    d = derived(Re, Cp, D_mm, polymer)
    regime = "laminar_like"
    if f > 1.15 * d["f_16Re"] and Re > 1800:
        regime = "newtonian_turbulent"
    elif d["Wi_1m_beta"] > 0.5 and f < 0.7 * d["f_Blasius"]:
        regime = "drag_reduced"
    elif Re > 1800 and f < 1.08 * d["f_16Re"]:
        regime = "relaminarized_or_MDR"
    rows.append(
        {
            "polymer": polymer,
            "Cp_ppm": Cp,
            "D_mm": D_mm,
            "Re": Re,
            "f": f,
            "source": source,
            "series": series,
            "regime": regime,
            **d,
        }
    )


def expand_split_figure(rows, fig, polymer, D_mm, source):
    for Cp, f_list in fig["f_split"].items():
        for Re in fig["Re_laminar"]:
            add_row(
                rows,
                polymer=polymer,
                Cp=Cp,
                D_mm=D_mm,
                Re=Re,
                f=f_laminar(Re),
                source=source,
                series=f"{polymer}_{D_mm}mm_{Cp}ppm",
            )
        for Re, f in zip(fig["Re_split"], f_list, strict=True):
            add_row(
                rows,
                polymer=polymer,
                Cp=Cp,
                D_mm=D_mm,
                Re=Re,
                f=f,
                source=source,
                series=f"{polymer}_{D_mm}mm_{Cp}ppm",
            )


def build() -> pd.DataFrame:
    rows: list[dict] = []

    expand_split_figure(rows, FIG12A, "PAAm", 0.49, "Fig. 12a")
    expand_split_figure(rows, FIG12B, "PEO", 0.49, "Fig. 12b")
    expand_split_figure(rows, FIG13, "PAAm", 2.84, "Fig. 13")

    for block in (FIG05, FIG06_300, FIG06_400, FIG07, FIG08):
        for Re, f in block["points"]:
            add_row(
                rows,
                polymer=block["polymer"],
                Cp=block["Cp_ppm"],
                D_mm=block["D_mm"],
                Re=Re,
                f=f,
                source=block["source"],
                series=f"{block['polymer']}_{block['D_mm']}mm_{block['Cp_ppm']}ppm",
            )

    for Cp, f in FIG14A:
        add_row(
            rows,
            polymer="PAAm",
            Cp=Cp,
            D_mm=0.49,
            Re=3150,
            f=f,
            source="Fig. 14a",
            series="PAAm_0.49mm_Re3150_sweep",
        )

    for D_mm, Re, Cp, ratio in FIG14B:
        add_row(
            rows,
            polymer="PAAm",
            Cp=Cp,
            D_mm=D_mm,
            Re=Re,
            f=ratio * f_laminar(Re),
            source="Fig. 14b",
            series=f"PAAm_{D_mm}mm_relaminar_Re{int(Re)}",
        )

    for D_mm, Re, Cp, f, src in FIG15B + FIG16 + FIG17:
        add_row(
            rows,
            polymer="PAAm",
            Cp=Cp,
            D_mm=D_mm,
            Re=Re,
            f=f,
            source=src,
            series=f"PAAm_{D_mm}mm_Re{int(Re)}_sweep",
        )

    df = pd.DataFrame(rows)
    # Near-duplicate sweeps (Figs. 15b/16/17) at the same (Re, C, D): keep mean f.
    keys = ["polymer", "Cp_ppm", "D_mm", "Re", "source"]
    df = (
        df.groupby(keys, as_index=False)
        .agg(
            {
                "f": "mean",
                "series": "first",
                "regime": "first",
                "eta0_Pa_s": "first",
                "lambda_s": "first",
                "V_m_s": "first",
                "Wi": "first",
                "beta": "first",
                "Wi_1m_beta": "first",
                "E": "first",
                "E_1m_beta": "first",
                "f_16Re": "first",
                "f_Blasius": "first",
                "f_Virk": "first",
            }
        )
    )
    return df.sort_values(["source", "polymer", "D_mm", "Cp_ppm", "Re"]).reset_index(drop=True)


def transition_table() -> pd.DataFrame:
    rows = []
    for Cp, D, Ret, eta, lam in TABLE1_PAAM:
        d = derived(Ret, Cp, D, "PAAm")
        rows.append(
            {
                "polymer": "PAAm",
                "Cp_ppm": Cp,
                "D_mm": D,
                "Ret": Ret,
                "eta0_mPa_s": eta,
                "lambda_ms": lam,
                "source": "Table 1",
                **{k: d[k] for k in ("beta", "Wi", "Wi_1m_beta", "E", "E_1m_beta")},
            }
        )
    for Cp, D, Ret, eta, lam in TABLE2_PEO:
        d = derived(Ret, Cp, D, "PEO")
        rows.append(
            {
                "polymer": "PEO",
                "Cp_ppm": Cp,
                "D_mm": D,
                "Ret": Ret,
                "eta0_mPa_s": eta,
                "lambda_ms": lam,
                "source": "Table 2",
                **{k: d[k] for k in ("beta", "Wi", "Wi_1m_beta", "E", "E_1m_beta")},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    df = build()
    trans = transition_table()
    df.to_csv(DATA / "friction_factor.csv", index=False)
    trans.to_csv(DATA / "transition_Ret.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"friction_factor.csv  {len(df):3d} rows  from {sorted(df.source.unique())}")
    print(f"transition_Ret.csv   {len(trans):3d} rows  (Tables 1–2, copied)")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
