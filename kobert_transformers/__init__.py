import os

from .load_model import get_distilkobert_lm, get_distilkobert_model, get_kobert_lm, get_kobert_model  # noqa
from .utils import get_tokenizer  # noqa

version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_txt) as f:
    __version__ = f.read().strip()
