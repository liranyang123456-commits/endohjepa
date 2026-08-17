"""PlotNeuralNet rendering of the audited Endo-HJEPA architecture.

The main path is horizontal and all branches use orthogonal connectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\PlotNeuralNet-master")
sys.path.insert(0, str(ROOT))
from pycore.tikzeng import (  # noqa: E402
    to_begin, to_connection, to_Conv, to_ConvConvRelu, to_end, to_generate,
    to_head, to_Pool, to_SoftMax,
)


def main():
    arch = [
        to_head(str(ROOT).replace("\\", "/")),
        r"""
\def\ConvColor{rgb:blue,2.2;green,1.2;white,7.5}
\def\ConvReluColor{rgb:blue,1.2;green,2.6;white,7.5}
\def\PoolColor{rgb:black,0.7;white,9.3}
\def\SoftmaxColor{rgb:red,1.2;yellow,1.3;white,8.2}
""",
        to_begin(),
        to_Conv("inclip", 16, 3, offset="(0,0,0)", to="(0,0,0)",
                height=30, depth=30, width=2, caption="Endoscopic clip"),
        to_Conv("vit1", 256, 1024, offset="(1.5,0,0)", to="(inclip-east)",
                height=28, depth=28, width=3.2, caption="V-JEPA2"),
        to_connection("inclip", "vit1"),
        to_Conv("vit2", 256, 1024, offset="(0.25,0,0)", to="(vit1-east)",
                height=26, depth=26, width=3.2, caption=""),
        to_connection("vit1", "vit2"),
        to_Conv("vit3", 256, 1024, offset="(0.25,0,0)", to="(vit2-east)",
                height=24, depth=24, width=3.2, caption="frozen ViT-L"),
        to_connection("vit2", "vit3"),
        to_Pool("tok", offset="(1.25,0,0)", to="(vit3-east)",
                height=18, depth=18, width=2.2, opacity=0.75,
                caption="dense tokens $z$"),
        to_connection("vit3", "tok"),
        # Validated forecast heads above the physical branch.
        to_ConvConvRelu("forecast", 4, (512, 512),
                        offset="(1.4,4.2,0)", to="(tok-east)",
                        height=14, depth=14, width=(2.2, 2.2),
                        caption="L1 causal / L2 coarse"),
        r"\draw [connection] (tok-east) -| node[pos=0.85] {\midarrow} (forecast-west);",
        # Grounded state path.
        to_ConvConvRelu("slots", 4, (256, 256),
                        offset="(1.4,-2.8,0)", to="(tok-east)",
                        height=18, depth=18, width=(2.4, 2.4),
                        caption="geometry / tool / semantic slots"),
        r"\draw [connection] (tok-east) -| node[pos=0.85] {\midarrow} (slots-west);",
        to_Conv("action", 4, 6, offset="(1.3,-4.2,0)", to="(slots-east)",
                height=8, depth=8, width=1.8, caption="SE(3) action"),
        to_ConvConvRelu("dyn", 4, (512, 512),
                        offset="(1.8,0,0)", to="(slots-east)",
                        height=18, depth=18, width=(2.6, 2.6),
                        caption="block-causal ensemble"),
        to_connection("slots", "dyn"),
        r"\draw [connection] (action-east) -| node[pos=0.8] {\midarrow} (dyn-south);",
        to_Pool("dist", offset="(1.3,0,0)", to="(dyn-east)",
                height=13, depth=13, width=2.2, opacity=0.75,
                caption=r"{$\mu\,/\,\Sigma_{\mathrm{alea}}\,/\,\Sigma_{\mathrm{epi}}$}"),
        to_connection("dyn", "dist"),
        to_SoftMax("risk", 1, offset="(1.2,0,0)", to="(dist-east)",
                   height=10, depth=10, width=2.0, opacity=0.75,
                   caption="calibrated risk"),
        to_connection("dist", "risk"),
        to_ConvConvRelu("mpc", 1, (256, 256),
                        offset="(1.25,0,0)", to="(risk-east)",
                        height=12, depth=12, width=(2.2, 2.2),
                        caption="continuous CEM / MPPI"),
        to_connection("risk", "mpc"),
        to_SoftMax("gate", 1, offset="(1.15,0,0)", to="(mpc-east)",
                   height=9, depth=9, width=2.0, opacity=0.75,
                   caption="hard safety gate"),
        to_connection("mpc", "gate"),
        r"""
\node[anchor=north, align=center, text width=18cm, font=\scriptsize] at
([yshift=-14mm]current bounding box.south)
{Blue: visual encoder \quad Teal: learned prediction modules \quad
Grey: latent tensors \quad Ochre: calibrated decision outputs.
The physical branch is evaluated only with aligned pose/depth; no pixel decoder is used.};
""",
        to_end(),
    ]
    out = Path(__file__).resolve().parent / "figures" / "figure1_network.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    arch = [c for c in arch if c]
    to_generate(arch, str(out))
    print(f"[arch] wrote {out}")


if __name__ == "__main__":
    main()
