"""Report figures drawn from the code output. No journal figure is copied."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from carreau_pipe import f_carreau_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "report" / "figures"


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open() as handle:
        return list(csv.DictReader(handle))


def _ticks(lo: float, hi: float, log: bool) -> list[float]:
    if log:
        decades = range(int(np.floor(np.log10(max(lo, 1e-12)))), int(np.ceil(np.log10(hi))) + 1)
        return [10.0 ** d for d in decades if lo <= 10.0 ** d <= hi]
    span = max(hi - lo, 1e-9)
    raw = span / 5.0
    mag = 10 ** np.floor(np.log10(raw))
    step = mag
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * mag
        if span / step <= 6.0:
            break
    start = np.ceil(lo / step) * step
    values = []
    v = start
    while v <= hi + 0.01 * step:
        values.append(float(v))
        v += step
    return values


def panel(draw, box, x, y, xlabel, ylabel, logx, logy, title):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(0, 0, 0), width=2)
    xa = np.log10(x) if logx else np.asarray(x, dtype=float)
    ya = np.log10(y) if logy else np.asarray(y, dtype=float)
    xmin, xmax = float(np.min(xa)), float(np.max(xa))
    ymin, ymax = float(np.min(ya)), float(np.max(ya))
    padx = 0.08 * (xmax - xmin)
    pady = 0.10 * (ymax - ymin)
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady

    def px(v):
        q = np.log10(v) if logx else v
        return x0 + (q - xmin) / (xmax - xmin) * (x1 - x0)

    def py(v):
        q = np.log10(v) if logy else v
        return y1 - (q - ymin) / (ymax - ymin) * (y1 - y0)

    fnt = font(16)
    small = font(14)
    for t in _ticks(10**xmin if logx else xmin, 10**xmax if logx else xmax, logx):
        try:
            xx = px(t)
        except Exception:
            continue
        if x0 + 8 < xx < x1 - 8:
            draw.line([(xx, y1), (xx, y1 - 6)], fill=(0, 0, 0))
            label = f"{t:.0e}" if logx else (f"{t:.0f}" if abs(t) >= 100 else f"{t:.3g}")
            draw.text((xx - 16, y1 + 6), label, fill=(0, 0, 0), font=small)
    for t in _ticks(10**ymin if logy else ymin, 10**ymax if logy else ymax, logy):
        try:
            yy = py(t)
        except Exception:
            continue
        if y0 + 8 < yy < y1 - 8:
            draw.line([(x0, yy), (x0 + 6, yy)], fill=(0, 0, 0))
            label = f"{t:.0e}" if logy else (f"{t:.0f}" if abs(t) >= 100 else f"{t:.3g}")
            draw.text((x0 - 52, yy - 8), label, fill=(0, 0, 0), font=small)
    draw.text((x0 + 8, y0 + 6), title, fill=(0, 0, 0), font=fnt)
    draw.text((x0 + 8, y0 + 26), ylabel, fill=(0, 0, 0), font=small)
    draw.text(((x0 + x1) / 2 - 50, y1 + 28), xlabel, fill=(0, 0, 0), font=fnt)
    return px, py


def friction_figure() -> None:
    rows = read_csv("with_carreau.csv")
    picks = [
        ("PAAm", 300.0, 0.49, "Fig. 6a", "300 ppm, 0.49 mm"),
        ("PAAm", 800.0, 0.49, "Fig. 7", "800 ppm, 0.49 mm"),
        ("PAAm", 600.0, 2.84, "Fig. 8", "600 ppm, 2.84 mm"),
    ]
    img = Image.new("RGB", (1500, 560), "white")
    draw = ImageDraw.Draw(img)
    fnt = font(16)
    for i, (poly, Cp, D, src, title) in enumerate(picks):
        g = [
            r
            for r in rows
            if r["polymer"] == poly
            and abs(float(r["Cp_ppm"]) - Cp) < 1e-6
            and abs(float(r["D_mm"]) - D) < 1e-6
            and r["source"] == src
        ]
        g = sorted(g, key=lambda r: float(r["Re"]))
        Re = np.array([float(r["Re"]) for r in g])
        f = np.array([float(r["f"]) for r in g])
        n = float(g[0]["n_carreau"])
        line = np.logspace(np.log10(Re.min()), np.log10(Re.max()), 40)
        flam = f_carreau_curve(
            line,
            n,
            float(g[0]["beta_fluid"]),
            float(g[0]["lambda_s_fluid"]),
            float(g[0]["eta0_fluid"]),
            D * 1e-3,
        )
        box = (90 + i * 470, 20, 520 + i * 470, 470)
        # limits from all three series of y
        xs = np.concatenate([Re, line])
        ys = np.concatenate([f, 16 / line, flam])
        # local panel limits: pass the cloud by drawing after computing maps from this panel only
        px, py = panel(draw, box, xs, ys, "Re", "f", True, True, f"{title}, n = {n:.2f}")
        draw.line([(px(a), py(b)) for a, b in zip(line, 16 / line)], fill=(90, 90, 90), width=2)
        draw.line([(px(a), py(b)) for a, b in zip(line, flam)], fill=(180, 40, 40), width=3)
        for a, b in zip(Re, f):
            x, y = px(a), py(b)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 60, 140))
    draw.text((70, 520), "Grey: 16/Re     Red: computed Carreau friction factor     Blue: input values read from published symbols", fill=(0, 0, 0), font=fnt)
    img.save(OUT / "friction_curves.png")


def ret_figure() -> None:
    rows = [
        r
        for r in read_csv("carreau_fit.csv")
        if r["Ret_paper"] and r["Ret_numerical"] and "sweep" not in r["series"] and "relaminar" not in r["series"]
    ]
    xp = np.array([float(r["Ret_paper"]) for r in rows])
    yp = np.array([float(r["Ret_numerical"]) for r in rows])
    img = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(img)
    lo = min(xp.min(), yp.min()) * 0.85
    hi = max(xp.max(), yp.max()) * 1.08
    grid = np.array([lo, hi])
    px, py = panel(draw, (90, 30, 580, 540), np.concatenate([xp, grid]), np.concatenate([yp, grid]), "published Re_t", "computed Re_t", False, False, "Transition Reynolds number")
    draw.line([(px(lo), py(lo)), (px(hi), py(hi))], fill=(140, 140, 140), width=2)
    for a, b in zip(xp, yp):
        x, y = px(a), py(b)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 60, 140))
    img.save(OUT / "ret_parity.png")


def elasticity_figure() -> None:
    rows = read_csv("shear_elasticity.csv")
    img = Image.new("RGB", (1200, 560), "white")
    draw = ImageDraw.Draw(img)
    fnt = font(16)
    keys = (
        ("E_1m_beta_zero_shear", "Zero-shear relaxation time"),
        ("E_1m_beta_shear_corrected", "Relaxation time at the wall shear rate"),
    )
    for i, (key, title) in enumerate(keys):
        E = np.array([float(r[key]) for r in rows])
        Ret = np.array([float(r["Ret"]) for r in rows])
        box = (80 + i * 580, 20, 560 + i * 580, 470)
        px, py = panel(draw, box, E, Ret, "E(1-beta)", "Re_t", True, True, title)
        for r, e, rt in zip(rows, E, Ret):
            x, y = px(e), py(rt)
            if abs(float(r["D_mm"]) - 0.49) < 1e-6:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 60, 140))
            else:
                draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=(170, 40, 40))
    draw.text((80, 520), "Blue circles: 0.49 mm     Red squares: 2.84 mm     n = 0.8 fixed", fill=(0, 0, 0), font=fnt)
    img.save(OUT / "shear_elasticity.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    friction_figure()
    ret_figure()
    elasticity_figure()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
