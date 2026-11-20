"""
Level-2 extension. Not part of reproducing Appendix A.

The paper plots Re_t against E(1-beta) using the zero-shear relaxation time
and reports that the 0.49 mm and 2.84 mm tubes do not collapse. The discussion
says lambda and viscosity should be evaluated at the shear rate of the
experiment, and that those high-shear values were not measured.

This script supplies that evaluation with a closure. At each measured Re_t the
wall shear rate is taken from the Carreau pipe solution. The relaxation time
and the polymer contribution to the viscosity are then reduced by the Carreau
factor [1+(lambda*gamma)^2]^((n-1)/2). n = 0.8 is the value used in their
Figure 23, not a fit to the collapse.

The output is the same Re_t, replotted against the corrected elasticity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from carreau_pipe import _gamma_star, f_carreau, wi_pipe
from digitized_data import ETA_SOLVENT_PA_S, RHO_KG_M3, TABLE1_PAAM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
N_CARREAU = 0.8


def corrected(Cp: float, D_mm: float, Ret: float, eta_mPa: float, lam_ms: float, n: float = N_CARREAU) -> dict:
    eta0 = eta_mPa * 1e-3
    lam = lam_ms * 1e-3
    D = D_mm * 1e-3
    R = 0.5 * D
    beta = float(np.clip(ETA_SOLVENT_PA_S / eta0, 1e-3, 0.999))
    E_zero = lam * (eta0 - ETA_SOLVENT_PA_S) / (RHO_KG_M3 * R * R)
    Wi = wi_pipe(Ret, lam, eta0, D)
    f = f_carreau(Ret, n, beta, Wi)
    # Gamma* = gamma_dot / (8V/D), so the wall value is 1 for a Newtonian fluid.
    gamma_star = float(_gamma_star(np.array([1.0]), f, Ret, n, beta, Wi)[0])
    V = Ret * eta0 / (RHO_KG_M3 * D)
    gamma = gamma_star * (8.0 * V / D)
    thin = float((1.0 + (lam * gamma) ** 2) ** (0.5 * (n - 1.0)))
    lam_eff = lam * thin
    eta_poly = (eta0 - ETA_SOLVENT_PA_S) * thin
    E_corr = lam_eff * eta_poly / (RHO_KG_M3 * R * R)
    return {
        "polymer": "PAAm",
        "Cp_ppm": Cp,
        "D_mm": D_mm,
        "Ret": Ret,
        "n": n,
        "E_1m_beta_zero_shear": E_zero,
        "E_1m_beta_shear_corrected": E_corr,
        "wall_gamma_star": gamma_star,
        "lambda_gamma": lam * gamma,
        "thinning_factor": thin,
    }


def main() -> None:
    rows = [corrected(*rec) for rec in TABLE1_PAAM if rec[0] >= 100]
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    with (DATA / "shear_elasticity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    small = [r for r in rows if abs(r["D_mm"] - 0.49) < 1e-6 and r["Cp_ppm"] >= 300]
    large = [r for r in rows if abs(r["D_mm"] - 2.84) < 1e-6 and r["Cp_ppm"] >= 300]
    ratio_zero = np.mean([r["E_1m_beta_zero_shear"] for r in small]) / np.mean([r["E_1m_beta_zero_shear"] for r in large])
    ratio_corr = np.mean([r["E_1m_beta_shear_corrected"] for r in small]) / np.mean(
        [r["E_1m_beta_shear_corrected"] for r in large]
    )
    print(f"mean E(1-beta), 0.49 mm / 2.84 mm, Cp>=300 ppm")
    print(f"  zero-shear lambda: {ratio_zero:.1f}")
    print(f"  shear-corrected:   {ratio_corr:.1f}")
    large = [r for r in rows if abs(r["D_mm"] - 2.84) < 1e-6]
    print("same Ret, E(0.49 mm)/E(2.84 mm):")
    for r in rows:
        if abs(r["D_mm"] - 0.49) > 1e-6 or r["Cp_ppm"] < 100:
            continue
        match = min(large, key=lambda q: abs(q["Ret"] - r["Ret"]))
        if abs(match["Ret"] - r["Ret"]) > 250:
            continue
        print(
            f"  {r['Cp_ppm']:.0f} ppm Ret={r['Ret']:.0f} vs {match['Cp_ppm']:.0f} ppm Ret={match['Ret']:.0f}: "
            f"{r['E_1m_beta_zero_shear']/match['E_1m_beta_zero_shear']:.1f} -> "
            f"{r['E_1m_beta_shear_corrected']/match['E_1m_beta_shear_corrected']:.1f}"
        )
    _draw(rows)
    print("wrote data/shear_elasticity.csv and figures/shear_elasticity.png")


def _draw(rows: list[dict]) -> None:
    img = Image.new("RGB", (980, 460), "white")
    draw = ImageDraw.Draw(img)
    panels = (
        (40, 50, 470, 420, "E_1m_beta_zero_shear", "Zero-shear lambda, as in the paper"),
        (510, 50, 940, 420, "E_1m_beta_shear_corrected", "Lambda and viscosity at the wall shear rate"),
    )
    xmin, xmax = np.log10(1e-5), np.log10(1e-1)
    ymin, ymax = np.log10(800), np.log10(2600)
    for x0, y0, x1, y1, key, title in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0))
        draw.text((x0 + 8, y0 - 28), title, fill=(0, 0, 0))
        for decade in range(-5, 0):
            x = x0 + (decade - xmin) / (xmax - xmin) * (x1 - x0)
            draw.line([(x, y0), (x, y1)], fill=(230, 230, 230))
        for Ret in (1000, 1500, 2000, 2500):
            y = y1 - (np.log10(Ret) - ymin) / (ymax - ymin) * (y1 - y0)
            draw.line([(x0, y), (x1, y)], fill=(230, 230, 230))

        def px(E):
            return x0 + (np.log10(E) - xmin) / (xmax - xmin) * (x1 - x0)

        def py(Ret):
            return y1 - (np.log10(Ret) - ymin) / (ymax - ymin) * (y1 - y0)

        for r in rows:
            color = (20, 70, 140) if abs(r["D_mm"] - 0.49) < 1e-6 else (170, 50, 40)
            x, y = px(r[key]), py(r["Ret"])
            if abs(r["D_mm"] - 0.49) < 1e-6:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            else:
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x0 + 8, y1 - 36), "log E(1-beta)", fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 8), "log Ret", fill=(0, 0, 0))
    draw.text((40, 8), "Blue circles: 0.49 mm    Red squares: 2.84 mm    n = 0.8", fill=(0, 0, 0))
    img.save(FIG / "shear_elasticity.png")


if __name__ == "__main__":
    main()
