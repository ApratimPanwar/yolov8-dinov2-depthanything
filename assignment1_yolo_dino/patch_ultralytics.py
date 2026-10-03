"""Add the ConvDummy and DINOv2 blocks from the AI4DM page
into a local editable Ultralytics checkout, and register them with the YAML parser.

Usage:  python patch_ultralytics.py [--repo ./ultralytics]

The web page tells you to (a) paste the classes into conv.py / block.py and
(b) import them in tasks.py. That is necessary but NOT sufficient -- see
DEBUG NOTES in README.md. This script applies the fixes as well.
"""

import argparse
import re
import sys
from pathlib import Path

CONV_DUMMY = '''

class ConvDummy(nn.Module):
    """Identity pass-through so a YAML config can route the raw input image to a later layer.

    Args are optional: parse_model() calls unknown modules as m(*args), and the YAML
    entry `[-1, 1, ConvDummy, []]` passes no args at all.
    """

    def __init__(self, c1=None, c2=None, *args, **kwargs):
        """Initialize the pass-through layer, ignoring any parser-supplied arguments."""
        super().__init__()

    def forward(self, x):
        """Return the input unchanged."""
        return x
'''

DINOV2 = '''

def _ensure_dinov2_importable():
    """Put the torch.hub DINOv2 checkout on sys.path.

    Saved checkpoints pickle references to `dinov2.models.vision_transformer`. Unpickling
    happens before any DINOv2.__init__ runs, so without this a trained .pt cannot be
    reloaded at all: ModuleNotFoundError: No module named 'dinov2'.
    """
    import sys

    from torch.hub import get_dir

    hub = os.path.join(get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(hub) and hub not in sys.path:
        sys.path.insert(0, hub)


_ensure_dinov2_importable()


class DINOv2(nn.Module):
    """Frozen DINOv2 trunk that broadcasts a global image embedding to a YOLO feature map.

    The DINOv2 backbone is frozen; only `projector` is trained. The 1D CLS embedding is
    projected to `c2` channels and broadcast to the P5 (stride-32) spatial grid, so the
    detection head receives a "what is in this scene" summary alongside spatial features.

    Args:
        c1 (int): Input channels (3, the raw image).
        c2 (int): Output channels, set from the YAML.
        variant (str): DINOv2 hub entrypoint.
        imgsz (int): Side length fed to DINOv2 (rounded to a multiple of the patch size).
        stride (int): YOLO stride of the feature map this output is concatenated with.
    """

    def __init__(self, c1=3, c2=1024, variant="dinov2_vits14", imgsz=518, stride=32):
        """Load a frozen DINOv2 trunk and build the trainable projection head."""
        super().__init__()
        _ensure_dinov2_importable()
        self.stride_ = stride
        patch = 14
        self.imgsz = max(patch, round(imgsz / patch) * patch)  # must be divisible by the patch size

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)

        # Freeze for real: no_grad in forward() alone still leaves requires_grad=True, which
        # hands the optimizer 21M dead parameters and trips DDP's unused-parameter check.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        embed = getattr(self.model, "embed_dim", 384)
        self.projector = nn.Conv2d(embed, c2, kernel_size=1)  # the only trained part
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        """Keep the frozen trunk in eval mode so its norm layers never update."""
        super().train(mode)
        self.model.eval()
        return self

    def preprocess(self, x):
        """Resize to a patch-divisible square and normalize to DINOv2's expected range."""
        x = nn.functional.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x):
        """Return the projected global embedding broadcast to the stride-32 grid."""
        with torch.no_grad():
            embedding = self.model(self.preprocess(x))  # (B, embed)
        embedding = embedding.unsqueeze(-1).unsqueeze(-1).to(self.projector.weight.dtype)  # (B, embed, 1, 1)
        h, w = x.shape[2] // self.stride_, x.shape[3] // self.stride_
        return self.projector(embedding).expand(-1, -1, h, w)  # broadcast, cheaper than interpolate
'''

# parse_model() branch. Without this, DINOv2 is called as DINOv2(1024) -> TypeError, and
# its output channel count is recorded as ch[f]=3 instead of c2, so the following Concat
# reports 3 extra channels and the head is silently built with the wrong input width.
PARSE_BRANCH = """        elif m is DINOv2:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)  # scale with the model width like every other layer
            args = [c1, c2, *args[1:]]
        elif m is ConvDummy:
            c1 = c2 = ch[f]
            args = [c1, c2]
"""

ANCHOR = "        elif m in frozenset({TorchVision, Index}):"
IMPORTS = "from ultralytics.nn.modules.block import DINOv2\nfrom ultralytics.nn.modules.conv import ConvDummy\n"


def patch(path: Path, marker: str, apply, force=False):
    text = path.read_text(encoding="utf-8")
    if force and marker in text:
        text = text[: text.index(marker)].rstrip("\n") + "\n"
        text = text.replace('    "ConvDummy",\n', "").replace('    "DINOv2",\n', "")
        path.write_text(text, encoding="utf-8")
        text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"  = {path.name}: already patched")
        return
    path.write_text(apply(text), encoding="utf-8")
    print(f"  + {path.name}: patched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ultralytics", help="path to the cloned ultralytics repo")
    ap.add_argument("--force", action="store_true", help="re-apply an updated patch")
    args = ap.parse_args()
    nn_dir = Path(args.repo).resolve() / "ultralytics" / "nn"
    if not nn_dir.is_dir():
        sys.exit(f"ERROR: {nn_dir} not found -- clone ultralytics first and pass --repo")

    print(f"Patching {nn_dir}")

    # 1. conv.py -- ConvDummy + __all__ entry
    def do_conv(t):
        t = t.replace('__all__ = (\n    "CBAM",', '__all__ = (\n    "CBAM",\n    "ConvDummy",', 1)
        return t + CONV_DUMMY

    patch(nn_dir / "modules" / "conv.py", "class ConvDummy", do_conv, args.force)

    # 2. block.py -- DINOv2 + __all__ entry + the stdlib imports the class needs
    def do_block(t):
        t = t.replace(
            "from __future__ import annotations\n\nimport math\n",
            "from __future__ import annotations\n\nimport contextlib\nimport math\nimport os\nimport warnings\n",
            1,
        )
        t = t.replace('__all__ = (\n    "C1",', '__all__ = (\n    "C1",\n    "DINOv2",', 1)
        return t + DINOV2

    patch(nn_dir / "modules" / "block.py", "\ndef _ensure_dinov2_importable", do_block, args.force)

    # 3. tasks.py -- import the classes so globals()[m] resolves them, and register
    #    their channel bookkeeping in parse_model()
    def do_tasks(t):
        t = t.replace("from ultralytics.utils.ops import make_divisible", IMPORTS + "from ultralytics.utils.ops import make_divisible", 1)
        if ANCHOR not in t:
            sys.exit("ERROR: parse_model anchor not found -- ultralytics layout changed, patch by hand")
        return t.replace(ANCHOR, PARSE_BRANCH + ANCHOR, 1)

    patch(nn_dir / "tasks.py", "from ultralytics.nn.modules.block import DINOv2", do_tasks)

    # sanity check
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from ultralytics.nn.modules.conv import ConvDummy  # noqa
    from ultralytics.nn.tasks import DINOv2, parse_model  # noqa

    print("OK: ConvDummy and DINOv2 are importable and registered in tasks.py")


if __name__ == "__main__":
    main()
