from .tokenization_kobert import KoBertTokenizer


def get_tokenizer(cache_dir=None):
    if cache_dir is not None:
        return KoBertTokenizer.from_pretrained(cache_dir)
    else:
        return KoBertTokenizer.from_pretrained('monologg/kobert')
