from .tokenization_kobert import KoBertTokenizer


def get_tokenizer():
    return KoBertTokenizer.from_pretrained("monologg/kobert")
