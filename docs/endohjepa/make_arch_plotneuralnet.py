"""PlotNeuralNet rendering of the audited Endo-HJEPA architecture.

The diagram is laid out as three short, left-aligned rows instead of one very
wide row, which keeps the aspect ratio close to square:

    row 1  shared encoder      clip -> tubelet embed -> ViT-L x24 -> pool -> tokens
    row 2  forecast heads      independent L1 causal and L2 coarse heads
    row 3  grounded branch     twist -> deterministic dynamics -> latent -> CEM diagnostic

Both branch rows are fed from the dense-token block of row 1 by axis-aligned
connectors that travel in the clear band between rows and enter the next row
through its top (north) anchor, so no connector crosses a block or a caption.

The colour key and the reading instructions live in the LaTeX caption rather
than inside the drawing, so the figure carries no free-floating legend text.

    python docs/endohjepa/make_arch_plotneuralnet.py
    pdflatex -output-directory docs/endohjepa/figures \
        docs/endohjepa/figures/figure1_network.tex
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "PLOTNEURALNET_ROOT",
        Path(__file__).resolve().parents[3] / "PlotNeuralNet-master",
    )
)
if not (ROOT / "pycore" / "tikzeng.py").is_file():
    raise FileNotFoundError(
        "PlotNeuralNet was not found. Set PLOTNEURALNET_ROOT to its checkout."
    )
sys.path.insert(0, str(ROOT))
from pycore.tikzeng import (  # noqa: E402
    to_begin,
    to_connection,
    to_Conv,
    to_ConvConvRelu,
    to_end,
    to_generate,
    to_head,
    to_Pool,
)

# Row baselines in TikZ units (y grows upwards).
Y_ENCODER, Y_FORECAST, Y_GROUNDED = 0.0, -6.0, -11.6

PREDICTOR_TEAL = r"\def\ConvColor{rgb:blue,1.0;green,2.5;white,7.8}"
BLANK = " "


def row_label(text: str, y: float) -> str:
    return (
        r"\node[anchor=east, align=right, font=\bfseries\footnotesize,"
        r" text=black!72] at (-1.5," + f"{y}" + r",0) {" + text + r"};" + "\n"
    )


def bridge(source: str, target: str, run: float, drop: float) -> str:
    """Right, down, left, then down into ``target``-north; all axis-aligned."""
    return (
        r"\path ("
        + target
        + r"-north) ++(0,"
        + f"{drop}"
        + r",0) coordinate ("
        + target
        + r"-entry);"
        + "\n"
        + r"\draw [connection] ("
        + source
        + r"-east) -- ++("
        + f"{run}"
        + r",0,0) |- ("
        + target
        + r"-entry) -- node {\midarrow} ("
        + target
        + r"-north);"
        + "\n"
    )


def main():
    arch = [
        to_head(str(ROOT).replace("\\", "/")),
        r"""
\def\ConvColor{rgb:blue,2.4;green,1.1;white,7.6}
\def\ConvReluColor{rgb:blue,1.0;green,2.4;white,7.4}
\def\PoolColor{rgb:black,0.9;white,9.1}
\def\SoftmaxColor{rgb:red,1.3;yellow,1.6;white,8.0}
""",
        to_begin(),
        # ---------------- row 1: shared encoder ------------------------------
        row_label(r"Shared\\encoder\\(frozen)", Y_ENCODER),
        to_Conv(
            "inclip",
            16,
            3,
            offset="(0,0,0)",
            to=f"(0,{Y_ENCODER},0)",
            height=18,
            depth=18,
            width=2,
            caption="Input clip",
        ),
        to_Conv(
            "tubelet",
            BLANK,
            1024,
            offset="(1.8,0,0)",
            to="(inclip-east)",
            height=16,
            depth=16,
            width=2.4,
            caption="Tubelet embed",
        ),
        to_connection("inclip", "tubelet"),
        to_Conv(
            "vit1",
            BLANK,
            "",
            offset="(1.8,0,0)",
            to="(tubelet-east)",
            height=15,
            depth=15,
            width=2.8,
            caption="",
        ),
        to_connection("tubelet", "vit1"),
        to_Conv(
            "vit2",
            BLANK,
            "",
            offset="(0.36,0,0)",
            to="(vit1-east)",
            height=15,
            depth=15,
            width=2.8,
            caption=r"ViT-L $\times$24",
        ),
        to_Conv(
            "vit3",
            BLANK,
            "",
            offset="(0.36,0,0)",
            to="(vit2-east)",
            height=15,
            depth=15,
            width=2.8,
            caption="",
        ),
        to_Pool(
            "tok",
            offset="(2.0,0,0)",
            to="(vit3-east)",
            height=10,
            depth=10,
            width=2.4,
            opacity=0.8,
            caption=r"Dense tokens $z$",
        ),
        to_connection("vit3", "tok"),
        to_Pool(
            "pooled",
            offset="(1.9,0,0)",
            to="(tok-east)",
            height=8,
            depth=8,
            width=2.2,
            opacity=0.8,
            caption=r"Pooled $\bar z$",
        ),
        to_connection("tok", "pooled"),
        # ---------------- row 2: forecast heads ------------------------------
        PREDICTOR_TEAL,
        row_label(r"Forecast\\heads", Y_FORECAST),
        to_ConvConvRelu(
            "l1",
            4,
            (512, ""),
            offset="(0,0,0)",
            to=f"(0,{Y_FORECAST},0)",
            height=12,
            depth=12,
            width=(2.4, 2.4),
            caption="L1 causal",
        ),
        to_ConvConvRelu(
            "l2",
            2,
            (512, ""),
            offset="(3.4,0,0)",
            to="(l1-east)",
            height=12,
            depth=12,
            width=(2.4, 2.4),
            caption="L2 coarse",
        ),
        bridge("pooled", "l1", run=0.9, drop=0.9),
        bridge("pooled", "l2", run=2.8, drop=0.9),
        # ---------------- row 3: grounded branch ------------------------------
        row_label(r"Grounded\\dynamics\\and decision", Y_GROUNDED),
        to_Conv(
            "twist",
            4,
            6,
            offset="(0,0,0)",
            to=f"(0,{Y_GROUNDED},0)",
            height=6,
            depth=6,
            width=1.8,
            caption=r"SE(3) twist $u$",
        ),
        to_ConvConvRelu(
            "dyn",
            4,
            (256, ""),
            offset="(2.1,0,0)",
            to="(twist-east)",
            height=16,
            depth=16,
            width=(2.6, 2.6),
            caption="Block-causal dynamics",
        ),
        to_connection("twist", "dyn"),
        to_Pool(
            "dist",
            offset="(2.1,0,0)",
            to="(dyn-east)",
            height=11,
            depth=11,
            width=2.4,
            opacity=0.8,
            caption=r"{$\hat{\bar z}_{1:H}$}",
        ),
        to_connection("dyn", "dist"),
        to_ConvConvRelu(
            "cem",
            1,
            (256, ""),
            offset="(2.1,0,0)",
            to="(dist-east)",
            height=11,
            depth=11,
            width=(2.2, 2.2),
            caption="CEM proxy",
        ),
        to_connection("dist", "cem"),
        bridge("pooled", "dyn", run=1.9, drop=0.7),
        to_end(),
    ]
    out = Path(__file__).resolve().parent / "figures" / "figure1_network.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    to_generate([chunk for chunk in arch if chunk], str(out))
    print(f"[arch] wrote {out}")


if __name__ == "__main__":
    main()
