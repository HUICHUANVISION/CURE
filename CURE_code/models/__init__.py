from .tca import TCA
from .coral import CORAL
from .bda import BDA
from .jda import JDA
from .hdpks import HDPKS
from .cca import CCAplus

MODELS = {
    "TCA": TCA,
    "CORAL": CORAL,
    "BDA": BDA,
    "JDA": JDA,
    "HDPKS": HDPKS,
    "CCAplus": CCAplus
}