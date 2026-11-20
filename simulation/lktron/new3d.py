"""
plot_pierce3D.py
================
Reads pierce3D.txt and produces interactive 3D surface plots
(one per dataset) using Plotly.

File format
-----------
Line 1  : <idx_x>  <n_rows>  <idx_y>  <n_cols>   e.g.  1  201  1  176
Lines 2+ : N_DATASETS × n_rows lines, each with n_cols whitespace-separated
           values.  No sub-headers between datasets.

Usage
-----
    python plot_pierce3D.py                     # uses pierce3D.txt in cwd
    python plot_pierce3D.py path/to/file.txt    # custom path

Each plot opens in the default browser as a self-contained HTML file and is
also saved as  plot_index<N>_<Title>.html  alongside the script.
"""
#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DATASETS      = 3
ANOMALY_THRESH  = 120.0
ANOMALY_PLOTVAL = 100.0
CLIM            = (0.0, 100.0)      # fixed colour scale across all plots

TITLES = {
    1: "Normalized Potential",
    2: "Normalized Axial Electric Field",
    3: "Normalized Radial Electric Field",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Holds one parsed 2-D field slice and its metadata."""

    index  : int
    n_rows : int          # z-direction (201)
    n_cols : int          # r-direction (176)
    data   : np.ndarray   # shape (n_rows, n_cols)

    @property
    def title(self) -> str:
        return TITLES.get(self.index, f"Dataset Index {self.index}")

    @property
    def anomaly_mask(self) -> np.ndarray:
        return self.data > ANOMALY_THRESH

    @property
    def clamped(self) -> np.ndarray:
        """Data with anomalies clamped to ANOMALY_PLOTVAL."""
        z = self.data.copy()
        z[self.anomaly_mask] = ANOMALY_PLOTVAL
        return z


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PierceDataParser:
    """
    Parses the pierce3D.txt format.

    Responsibilities
    ----------------
    - Read the single header line to extract grid dimensions.
    - Read all remaining values and split into N_DATASETS equal slices.
    """

    def __init__(self, filepath: str | Path, n_datasets: int = N_DATASETS):
        self.filepath   = Path(filepath)
        self.n_datasets = n_datasets

    def parse(self) -> list[Dataset]:
        text = self.filepath.read_text()
        tokens = text.split()

        # Header: first four tokens are idx_x, n_rows, idx_y, n_cols
        n_rows, n_cols = int(tokens[1]), int(tokens[3])
        expected       = self.n_datasets * n_rows * n_cols

        values = np.array(tokens[4:], dtype=np.float64)

        if values.size < expected:
            raise ValueError(
                f"Expected {expected} data values, found {values.size}."
            )

        values = values[:expected].reshape(self.n_datasets, n_rows, n_cols)

        datasets = [
            Dataset(index=i + 1, n_rows=n_rows, n_cols=n_cols, data=values[i])
            for i in range(self.n_datasets)
        ]

        print(f"Parsed {self.n_datasets} datasets  "
              f"(grid {n_rows} z × {n_cols} r)  "
              f"from '{self.filepath.name}'")
        return datasets


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class Pierce3DPlotter:
    """
    Builds and saves one interactive Plotly figure per Dataset.

    Parameters
    ----------
    output_dir  : folder where HTML files are written (default: cwd)
    auto_open   : open each plot in the browser immediately
    """

    def __init__(
        self,
        output_dir : str | Path = ".",
        auto_open  : bool       = True,
    ):
        self.output_dir = Path(output_dir)
        self.auto_open  = auto_open
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_all(self, datasets: list[Dataset]) -> None:
        for ds in datasets:
            fig  = self._build_figure(ds)
            path = self._save(fig, ds)
            print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    def _build_figure(self, ds: Dataset) -> go.Figure:
        z_grid = ds.clamped                          # (n_rows, n_cols)
        z_axis = np.arange(1, ds.n_rows + 1)        # z-axis indices
        r_axis = np.arange(1, ds.n_cols + 1)        # r-axis indices
        Z, R   = np.meshgrid(z_axis, r_axis)        # both (n_cols, n_rows)

        fig = go.Figure()

        # ---- surface --------------------------------------------------
        fig.add_trace(go.Surface(
            x          = Z,                         # z-axis → X in Plotly
            y          = R,                         # r-axis → Y in Plotly
            z          = z_grid.T,                  # transpose: (n_cols, n_rows)
            colorscale = "Jet",
            cmin       = CLIM[0],
            cmax       = CLIM[1],
            colorbar   = dict(
                title      = dict(text="Normalised Value", side="right"),
                thickness  = 18,
                len        = 0.75,
            ),
            hovertemplate = (
                "z-index: %{x}<br>"
                "r-index: %{y}<br>"
                "Value: %{z:.4f}<extra></extra>"
            ),
            name = ds.title,
        ))

        # ---- anomaly scatter ------------------------------------------
        mask        = ds.anomaly_mask               # (n_rows, n_cols)
        row_idx, col_idx = np.where(mask)           # z-idx, r-idx (0-based)
        n_anom      = int(mask.sum())

        if n_anom > 0:
            fig.add_trace(go.Scatter3d(
                x    = row_idx + 1,                 # z-axis (1-based)
                y    = col_idx + 1,                 # r-axis (1-based)
                z    = np.full(n_anom, ANOMALY_PLOTVAL + 1),
                mode = "markers",
                marker = dict(size=4, color="black", symbol="circle"),
                name = f"Anomaly > {ANOMALY_THRESH:.0f}  [{n_anom} pts]",
                hovertemplate = (
                    "z-index: %{x}<br>"
                    "r-index: %{y}<br>"
                    f"Original value > {ANOMALY_THRESH:.0f}"
                    "<extra>Anomaly</extra>"
                ),
            ))

        # ---- layout ---------------------------------------------------
        footnote = (
            f"Run No: {ds.index}  |  "
            f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"File: {Path(sys.argv[1] if len(sys.argv) > 1 else 'pierce.3D').name}"
        )

        fig.update_layout(
            title = dict(
                text     = ds.title,
                x        = 0.5,
                xanchor  = "center",
                font     = dict(size=18, family="Arial"),
            ),
            scene = dict(
                xaxis = dict(title=dict(text="z-axis (grid index)", font=dict(size=13))),
                yaxis = dict(title=dict(text="r-axis (grid index)", font=dict(size=13))),
                zaxis = dict(title=dict(text="Normalised Value",    font=dict(size=12)),
                             range=list(CLIM)),
                aspectmode  = "manual",
                aspectratio = dict(
                    x = ds.n_rows / max(ds.n_rows, ds.n_cols),
                    y = ds.n_cols / max(ds.n_rows, ds.n_cols),
                    z = 0.5,
                ),
                camera = dict(eye=dict(x=-1.4, y=-1.4, z=0.9)),
            ),
            legend = dict(
                x=0.01, y=0.99,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="#ccc",
                borderwidth=1,
            ),
            margin       = dict(l=20, r=20, t=70, b=60),
            width        = 1200,
            height       = 700,
            paper_bgcolor= "white",
            annotations  = [dict(
                text       = footnote,
                xref       = "paper", yref="paper",
                x=0.5, y=-0.04,
                showarrow  = False,
                font       = dict(size=9, color="#777"),
                xanchor    = "center",
            )],
        )

        return fig

    # ------------------------------------------------------------------
    def _save(self, fig: go.Figure, ds: Dataset) -> Path:
        safe  = re.sub(r"\s+", "_", ds.title)
        fname = f"plot_index{ds.index}_{safe}.html"
        path  = self.output_dir / fname
        fig.write_html(str(path), auto_open=self.auto_open)
        return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    filepath   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pierce.3D")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    if not filepath.exists():
        sys.exit(f"Error: file not found — {filepath}")

    parser   = PierceDataParser(filepath)
    datasets = parser.parse()

    plotter  = Pierce3DPlotter(output_dir=output_dir, auto_open=True)
    plotter.plot_all(datasets)

    print(f"\nDone. {len(datasets)} interactive plots saved.")


if __name__ == "__main__":
    main()
