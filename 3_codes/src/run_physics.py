"""
Physics + numerics for Chandra, Shankar & Das, JFM 885, A47 (2020).

Appendix A is solved by Newton + bisection (carreau_pipe.py). No PINN.
A linear model is then fit only to (f - f_Carreau)/f_Carreau.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from carreau_pipe import ETA_SOLVENT, f_carreau, f_carreau_curve, wi_pipe
from digitized_data import TABLE1_PAAM, TABLE2_PEO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
HOLD_CP = {10.0, 400.0}


def rheology(polymer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if polymer == "PAAm":
        raw = [(c, e * 1e-3, lam * 1e-3) for c, D, _, e, lam in TABLE1_PAAM if abs(D - 0.49) < 1e-6]
    else:
        raw = [(0.0, ETA_SOLVENT, 0.0)]
        raw += [(c, e * 1e-3, lam * 1e-3) for c, _, _, e, lam in TABLE2_PEO]
    raw = sorted(set(raw))
    c = np.array([r[0] for r in raw])
    _, idx = np.unique(c, return_index=True)
    return c[idx], np.array([r[1] for r in raw])[idx], np.array([r[2] for r in raw])[idx]


def fluid_props(polymer: str, Cp: float) -> tuple[float, float, float]:
    c, eta, lam = rheology(polymer)
    eta0 = float(np.interp(Cp, c, eta))
    lam_s = float(np.interp(Cp, c, lam))
    beta = float(np.clip(ETA_SOLVENT / max(eta0, 1e-8), 1e-3, 1.0))
    return eta0, lam_s, beta


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_n(Re, f, beta, lam_s, eta0, D_m) -> float:
    if len(Re) < 3 or lam_s <= 0.0 or beta > 0.995:
        return 1.0

    def cost(n: float) -> float:
        pred = f_carreau_curve(Re, n, beta, lam_s, eta0, D_m)
        return float(np.mean(((pred - f) / f) ** 2))

    a, b = 0.35, 1.0
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(14):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = cost(d)
    n_best = c if fc < fd else d
    if cost(1.0) <= cost(n_best) * 1.02:
        return 1.0
    return float(n_best)


def detect_ret(Re, f, f_lam) -> float | None:
    order = np.argsort(Re)
    Re, f, f_lam = Re[order], f[order], f_lam[order]
    ratio = f / f_lam
    for i in range(1, len(Re)):
        if ratio[i] > 1.05 and ratio[i - 1] <= 1.05:
            t = (1.05 - ratio[i - 1]) / (ratio[i] - ratio[i - 1] + 1e-15)
            return float(Re[i - 1] + t * (Re[i] - Re[i - 1]))
    return None


def check_newtonian() -> None:
    err = max(abs(f_carreau(Re, 1.0, 0.5, 10.0) - 16.0 / Re) / (16.0 / Re) for Re in (500, 1500, 4000))
    thin = f_carreau(1000.0, 0.6, 0.4, 20.0)
    print(f"Newtonian check: max relative error {err:.2e}", flush=True)
    print(f"Shear-thinning check: f(1000)={thin:.5f} < 16/Re={0.016:.5f}", flush=True)
    if thin >= 0.016:
        raise SystemExit("Carreau solver failed the shear-thinning check")


def features(Re, Cp, D, wib) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(Re)),
            np.log10(Re),
            np.log10(1.0 + wib),
            Cp / 1000.0,
            D,
        ]
    )


def r2_mae(y, yhat) -> tuple[float, float]:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(r2), float(np.mean(np.abs(y - yhat)))


def main() -> None:
    FIG.mkdir(exist_ok=True)
    check_newtonian()
    rows = read_csv(DATA / "friction_factor.csv")
    trans = read_csv(DATA / "transition_Ret.csv")
    ret_lookup = {
        (r["polymer"], round(float(r["Cp_ppm"]), 1), round(float(r["D_mm"]), 2)): float(r["Ret"])
        for r in trans
    }

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["polymer"], float(row["Cp_ppm"]), float(row["D_mm"]), row["series"])
        groups.setdefault(key, []).append(row)
    print(f"fitting {len(groups)} series", flush=True)

    enriched = []
    summary = []
    for (polymer, Cp, D, series), group in groups.items():
        Re = np.array([float(r["Re"]) for r in group])
        f = np.array([float(r["f"]) for r in group])
        order = np.argsort(Re)
        Re, f = Re[order], f[order]
        eta0, lam_s, beta = fluid_props(polymer, Cp)
        D_m = D * 1e-3
        ret_paper = ret_lookup.get((polymer, round(Cp, 1), round(D, 2)))
        if ret_paper is None:
            mask = Re < 1200
        else:
            mask = Re < 0.85 * ret_paper
        if mask.sum() < 3:
            mask = Re <= np.quantile(Re, 0.4)
        n = fit_n(Re[mask], f[mask], beta, lam_s, eta0, D_m)
        f_lam = f_carreau_curve(Re, n, beta, lam_s, eta0, D_m)
        ret_num = detect_ret(Re, f, f_lam)
        Wi = np.array([wi_pipe(float(R), lam_s, eta0, D_m) for R in Re])
        for i in range(len(Re)):
            enriched.append(
                {
                    "polymer": polymer,
                    "Cp_ppm": Cp,
                    "D_mm": D,
                    "series": series,
                    "source": group[order[i]]["source"],
                    "Re": Re[i],
                    "f": f[i],
                    "n_carreau": n,
                    "beta_fluid": beta,
                    "lambda_s_fluid": lam_s,
                    "eta0_fluid": eta0,
                    "Wi_fluid": Wi[i],
                    "Wi_1m_beta_fluid": Wi[i] * (1.0 - beta),
                    "f_carreau": f_lam[i],
                    "f_excess": (f[i] - f_lam[i]) / f_lam[i],
                    "Ret_paper": "" if ret_paper is None else ret_paper,
                    "Ret_numerical": "" if ret_num is None else ret_num,
                }
            )
        summary.append(
            {
                "polymer": polymer,
                "Cp_ppm": Cp,
                "D_mm": D,
                "series": series,
                "n": n,
                "beta": beta,
                "lambda_ms": lam_s * 1e3,
                "Ret_paper": "" if ret_paper is None else ret_paper,
                "Ret_numerical": "" if ret_num is None else ret_num,
            }
        )
        print(f"  {polymer} {Cp:g} ppm D={D} n={n:.3f} Ret_num={ret_num}", flush=True)

    write_csv(DATA / "with_carreau.csv", enriched)
    write_csv(DATA / "carreau_fit.csv", summary)

    both = [s for s in summary if s["Ret_paper"] != "" and s["Ret_numerical"] != ""]
    if both:
        rel = [abs(float(s["Ret_numerical"]) - float(s["Ret_paper"])) / float(s["Ret_paper"]) for s in both]
        print(f"Ret on {len(both)} series: median |error| {100*np.median(rel):.1f}%", flush=True)

    scores = fit_excess(enriched)
    print(scores, flush=True)
    draw_plots(enriched, summary)


def f_blasius(Re: np.ndarray) -> np.ndarray:
    return 0.079 * np.asarray(Re, dtype=float) ** (-0.25)


def f_virk(Re: np.ndarray) -> np.ndarray:
    """Virk MDR Fanning friction factor, solved from 1/sqrt(f) = 19 log10(Re sqrt(f)) - 32.4."""
    Re = np.asarray(Re, dtype=float)
    f = np.full(len(Re), 0.008)
    for _ in range(40):
        rhs = 19.0 * np.log10(Re * np.sqrt(f)) - 32.4
        f = 1.0 / np.maximum(rhs, 1.0) ** 2
    return f


def f_after_transition(Re: np.ndarray, wib: np.ndarray) -> np.ndarray:
    """Paper closure: blend Blasius to Virk as Wi(1-beta) goes from 0 to ~0.87."""
    t = np.clip(wib / 0.87, 0.0, 1.0)
    return (1.0 - t) * f_blasius(Re) + t * f_virk(Re)


def _past_transition(row: dict) -> bool:
    ret = row["Ret_numerical"]
    return ret != "" and float(row["Re"]) > float(ret)


def fit_excess(rows: list[dict]) -> str:
    """
    The Carreau solution is kept for every laminar point.
    The linear model is trained only on points past Re_t, and it predicts
    the fractional departure (f - f_Carreau)/f_Carreau.
    """
    train = [r for r in rows if r["Cp_ppm"] not in HOLD_CP and _past_transition(r)]
    test = [r for r in rows if r["Cp_ppm"] in HOLD_CP]
    Xtr = features(
        np.array([r["Re"] for r in train]),
        np.array([r["Cp_ppm"] for r in train]),
        np.array([r["D_mm"] for r in train]),
        np.array([r["Wi_1m_beta_fluid"] for r in train]),
    )
    ytr = np.array([r["f_excess"] for r in train])
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    fte = np.array([r["f"] for r in test])
    flam = np.array([r["f_carreau"] for r in test])
    Xte = features(
        np.array([r["Re"] for r in test]),
        np.array([r["Cp_ppm"] for r in test]),
        np.array([r["D_mm"] for r in test]),
        np.array([r["Wi_1m_beta_fluid"] for r in test]),
    )
    past = np.array([_past_transition(r) for r in test])
    f_hybrid = flam.copy()
    f_hybrid[past] = flam[past] * (1.0 + Xte[past] @ coef)
    Re_te = np.array([r["Re"] for r in test])
    wib_te = np.array([r["Wi_1m_beta_fluid"] for r in test])
    f_closed = flam.copy()
    f_closed[past] = f_after_transition(Re_te[past], wib_te[past])

    # A fit that never sees the equation, trained on all non-held-out points.
    raw_train = [r for r in rows if r["Cp_ppm"] not in HOLD_CP]
    Xraw = features(
        np.array([r["Re"] for r in raw_train]),
        np.array([r["Cp_ppm"] for r in raw_train]),
        np.array([r["D_mm"] for r in raw_train]),
        np.array([r["Wi_1m_beta_fluid"] for r in raw_train]),
    )
    coef_raw, *_ = np.linalg.lstsq(Xraw, np.array([r["f"] for r in raw_train]), rcond=None)
    f_raw = Xte @ coef_raw

    rows_out = []
    for name, pred in (
        ("Carreau equation only", flam),
        ("Carreau, then Blasius-Virk after Re_t", f_closed),
        ("Carreau, then linear excess after Re_t", f_hybrid),
        ("linear fit of f, no equation", f_raw),
    ):
        r2, mae = r2_mae(fte, pred)
        rows_out.append({"model": name, "R2": r2, "MAE": mae, "n_test": len(fte)})
    if past.any():
        for name, pred in (
            ("post-transition, equation only", flam[past]),
            ("post-transition, Blasius-Virk closure", f_closed[past]),
            ("post-transition, linear excess", f_hybrid[past]),
        ):
            r2, mae = r2_mae(fte[past], pred)
            rows_out.append({"model": name, "R2": r2, "MAE": mae, "n_test": int(past.sum())})
    write_csv(DATA / "physics_ml_scores.csv", rows_out)
    names = ["intercept", "log10_Re", "log10_1_plus_Wi1mb", "Cp_per_1000", "D_mm"]
    write_csv(DATA / "excess_coefficients.csv", [{"term": n, "coef": float(c)} for n, c in zip(names, coef)])
    return "\n".join(f"{r['model']}: R2={float(r['R2']):.3f}  MAE={float(r['MAE']):.5f}  n={r['n_test']}" for r in rows_out)


def _panel(draw, box, x, y, color, radius=3):
    x0, y0, x1, y1 = box
    if len(x) == 0:
        return
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1e-4
    pad_x = 0.08 * (xmax - xmin)
    pad_y = 0.12 * (ymax - ymin)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    def px(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    def py(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    draw.rectangle(box, outline=(40, 40, 40))
    pts = [(px(a), py(b)) for a, b in zip(x, y)]
    if len(pts) > 1 and radius == 0:
        draw.line(pts, fill=color, width=2)
    else:
        for p in pts:
            draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=color)
    return px, py


def draw_plots(rows: list[dict], summary: list[dict]) -> None:
    # Three concentrated series: 16/Re, Carreau, data.
    img = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(img)
    picks = [("PAAm", 300.0, 0.49, "Fig. 6a"), ("PAAm", 800.0, 0.49, "Fig. 7"), ("PAAm", 600.0, 2.84, "Fig. 8")]
    for i, (poly, Cp, D, src) in enumerate(picks):
        g = [r for r in rows if r["polymer"] == poly and r["Cp_ppm"] == Cp and abs(r["D_mm"] - D) < 1e-6 and r["source"] == src]
        g = sorted(g, key=lambda r: r["Re"])
        box = (40 + i * 390, 40, 380 + i * 390, 360)
        if not g:
            continue
        Re = np.array([r["Re"] for r in g])
        f = np.array([r["f"] for r in g])
        n = g[0]["n_carreau"]
        line = np.linspace(Re.min(), Re.max(), 40)
        flam = f_carreau_curve(line, n, g[0]["beta_fluid"], g[0]["lambda_s_fluid"], g[0]["eta0_fluid"], D * 1e-3)
        # scale using all y
        ys = np.concatenate([f, 16 / line, flam])
        xs = np.concatenate([Re, line])
        # draw frame via a dummy then lines in the same box
        x0, y0, x1, y1 = box
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        pad_x, pad_y = 0.06 * (xmax - xmin), 0.1 * (ymax - ymin)
        xmin, xmax, ymin, ymax = xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

        def px(v):
            return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

        def py(v):
            return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

        draw.rectangle(box, outline=(30, 30, 30))
        draw.line([(px(a), py(b)) for a, b in zip(line, 16 / line)], fill=(80, 80, 80), width=2)
        draw.line([(px(a), py(b)) for a, b in zip(line, flam)], fill=(180, 50, 50), width=2)
        for a, b in zip(Re, f):
            draw.ellipse((px(a) - 3, py(b) - 3, px(a) + 3, py(b) + 3), fill=(20, 60, 120))
        draw.text((x0 + 8, y0 + 6), f"{Cp:.0f} ppm, {D} mm, n={n:.2f}", fill=(0, 0, 0))
    draw.text((40, 8), "Grey: 16/Re    Red: Carreau solution of Appendix A    Blue: paper", fill=(0, 0, 0))
    img.save(FIG / "carreau_vs_data.png")

    comp = [s for s in summary if s["Ret_paper"] != "" and s["Ret_numerical"] != ""]
    img = Image.new("RGB", (480, 480), "white")
    draw = ImageDraw.Draw(img)
    if comp:
        xp = np.array([float(s["Ret_paper"]) for s in comp])
        yp = np.array([float(s["Ret_numerical"]) for s in comp])
        lim = max(xp.max(), yp.max()) * 1.1
        box = (50, 40, 450, 440)

        def px(v):
            return box[0] + v / lim * (box[2] - box[0])

        def py(v):
            return box[3] - v / lim * (box[3] - box[1])

        draw.rectangle(box, outline=(0, 0, 0))
        draw.line([(px(0), py(0)), (px(lim), py(lim))], fill=(120, 120, 120), width=2)
        for a, b in zip(xp, yp):
            draw.ellipse((px(a) - 4, py(b) - 4, px(a) + 4, py(b) + 4), fill=(20, 60, 120))
        draw.text((60, 12), "Paper Ret (x) vs numerical Ret (y)", fill=(0, 0, 0))
    img.save(FIG / "ret_comparison.png")
    print("wrote figures", flush=True)


if __name__ == "__main__":
    main()
