"""
Implied Volatility Surface Generator
=====================================
Pulls the live option chain for a ticker via yfinance, builds an implied
volatility surface (moneyness x time-to-expiration x IV), renders it with
matplotlib, and exports a slowly-rotating, lightweight animated GIF.

Usage
-----
    python iv_surface.py --ticker AAPL
    python iv_surface.py --ticker TSLA --out tsla_iv.gif --option-type put
    python iv_surface.py --ticker NVDA --min-days 5 --max-days 180 --frames 48

Requires: yfinance, pandas, numpy, scipy, matplotlib, pillow
    pip install yfinance pandas numpy scipy matplotlib pillow
"""

import argparse
import io

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.interpolate import griddata
from PIL import Image


# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------
BG_COLOR = "#0d1117"
GRAD_START = "#ff5f56"
GRAD_END = "#27c93f"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"

IV_CMAP = LinearSegmentedColormap.from_list("iv_cmap", [GRAD_START, GRAD_END])


# ----------------------------------------------------------------------
# 1. Pull & clean option chain data
# ----------------------------------------------------------------------
def fetch_iv_data(ticker_symbol, min_days=7, max_days=365, option_type="call"):
    """
    Download every available expiration's option chain for ticker_symbol
    and return a tidy DataFrame with one row per contract:
    strike, expiration, days_to_exp, T (years), moneyness,
    impliedVolatility, volume, openInterest, spot.
    """
    tk = yf.Ticker(ticker_symbol)

    spot = None
    try:
        spot = tk.fast_info["lastPrice"]
    except Exception:
        pass
    if not spot:
        hist = tk.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not retrieve a spot price for '{ticker_symbol}'.")
        spot = float(hist["Close"].iloc[-1])

    expirations = tk.options
    if not expirations:
        raise ValueError(f"No listed options found for '{ticker_symbol}'.")

    today = pd.Timestamp.today().normalize()
    frames = []
    for exp in expirations:
        exp_date = pd.Timestamp(exp)
        days_to_exp = (exp_date - today).days
        if days_to_exp < min_days or days_to_exp > max_days:
            continue
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        data = chain.calls if option_type == "call" else chain.puts
        if data.empty:
            continue

        cols = ["strike", "impliedVolatility", "volume", "openInterest"]
        df = data[cols].copy()
        df["expiration"] = exp_date
        df["days_to_exp"] = days_to_exp
        df["T"] = days_to_exp / 365.0
        df["moneyness"] = df["strike"] / spot
        frames.append(df)

    if not frames:
        raise ValueError(
            "No option data collected in the requested expiration window "
            f"({min_days}-{max_days} days)."
        )

    result = pd.concat(frames, ignore_index=True)
    result["spot"] = spot
    return clean_iv_data(result)


def clean_iv_data(df, min_iv=0.01, max_iv=3.0, min_volume=0, min_open_interest=0):
    """Drop illiquid quotes and obviously bad implied-vol values."""
    out = df.copy()
    out = out[(out["impliedVolatility"] > min_iv) & (out["impliedVolatility"] < max_iv)]
    out = out[out["openInterest"].fillna(0) >= min_open_interest]
    out = out[out["volume"].fillna(0) >= min_volume]
    out = out.dropna(subset=["impliedVolatility", "strike", "T"])
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Interpolate onto a regular grid
# ----------------------------------------------------------------------
def build_surface_grid(df, x_col="moneyness", y_col="T", z_col="impliedVolatility",
                        grid_size=60, method="cubic"):
    """Interpolate scattered (moneyness, T, IV) points onto a regular grid."""
    x, y, z = df[x_col].values, df[y_col].values, df[z_col].values

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method=method)
    if np.isnan(Z).any():
        # patch interpolation holes (e.g. near the edges) with nearest-neighbor
        Z_nearest = griddata((x, y), z, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    return X, Y, Z


# ----------------------------------------------------------------------
# 3. Plot
# ----------------------------------------------------------------------
def plot_iv_surface(X, Y, Z, ticker_symbol, figsize=(7, 5.2), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot_surface(
        X, Y, Z * 100,
        cmap=IV_CMAP,
        linewidth=0,
        antialiased=True,
        edgecolor="none",
        alpha=0.95,
    )

    ax.set_xlabel("Moneyness (K / Spot)", color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Time to Expiration (yrs)", color=TEXT_COLOR, labelpad=10)
    ax.set_zlabel("Implied Vol (%)", color=TEXT_COLOR, labelpad=10)
    ax.set_title(f"{ticker_symbol} \u2014 Implied Volatility Surface",
                 color=TEXT_COLOR, fontsize=13, pad=20)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.pane.set_facecolor(BG_COLOR)
        axis.pane.set_edgecolor(GRID_COLOR)

    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# 4. Rotate & export a lightweight GIF
# ----------------------------------------------------------------------
def render_rotating_gif(fig, ax, out_path, n_frames=36, elev=25,
                         duration_ms=1000, dpi=65, max_colors=40):
    """
    Spin the camera azimuth a full 360 degrees, capturing one frame per
    step, then write a palette-quantized GIF. A single shared palette
    (built from the first frame, via an octree quantizer which handles
    smooth gradients better than median-cut) is reused across all frames
    so colors stay stable frame-to-frame and the file stays small.
    """
    raw_frames = []
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    for az in azimuths:
        ax.view_init(elev=elev, azim=az)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        buf.seek(0)
        raw_frames.append(Image.open(buf).convert("RGB"))
        buf.close()

    base_palette = raw_frames[0].quantize(colors=max_colors, method=Image.FASTOCTREE)
    frames = [f.quantize(palette=base_palette, dither=Image.FLOYDSTEINBERG) for f in raw_frames]

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate an option IV surface rotating GIF for a ticker.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--out", default=None, help="Output GIF path (default: <TICKER>_iv_surface.gif)")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--min-days", type=int, default=7, help="Skip expirations closer than this")
    parser.add_argument("--max-days", type=int, default=365, help="Skip expirations farther than this")
    parser.add_argument("--grid-size", type=int, default=60, help="Interpolation grid resolution")
    parser.add_argument("--frames", type=int, default=30, help="Frames per full rotation")
    parser.add_argument("--dpi", type=int, default=65, help="Render DPI (lower = lighter file)")
    args = parser.parse_args()

    out_path = args.out or f"{args.ticker.upper()}_iv_surface.gif"

    print(f"Fetching option chain for {args.ticker}...")
    df = fetch_iv_data(args.ticker, min_days=args.min_days, max_days=args.max_days,
                        option_type=args.option_type)
    print(f"Collected {len(df)} quotes across {df['expiration'].nunique()} expirations.")

    print("Interpolating surface...")
    X, Y, Z = build_surface_grid(df, grid_size=args.grid_size)

    print("Rendering plot...")
    fig, ax = plot_iv_surface(X, Y, Z, args.ticker.upper())

    print(f"Rendering {args.frames}-frame rotation and saving GIF...")
    render_rotating_gif(fig, ax, out_path, n_frames=args.frames, dpi=args.dpi)
    plt.close(fig)

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()