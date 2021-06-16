import transformers
from packaging import version

from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()


def test_transformers_version():
    assert version.parse("3.0") <= version.parse(transformers.__version__) < version.parse("5.0")


def test_tokenization():
    sample_text = "[CLS] 한국어 모델을 공유합니다. [SEP]"

    tokens = tokenizer.tokenize(sample_text)
    assert tokens == ["[CLS]", "▁한국", "어", "▁모델", "을", "▁공유", "합니다", ".", "[SEP]"]

    encoded_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert encoded_ids == [2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]


def test_tokenizer_attribute():
    assert tokenizer.unk_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.cls_token_id == 2
    assert tokenizer.sep_token_id == 3
    assert tokenizer.mask_token_id == 4

    assert tokenizer.unk_token == "[UNK]"
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.cls_token == "[CLS]"
    assert tokenizer.sep_token == "[SEP]"
    assert tokenizer.mask_token == "[MASK]"

    assert tokenizer.model_max_length == 512
    assert tokenizer.max_len_single_sentence == 510
    assert tokenizer.max_len_sentences_pair == 509

    assert sorted(tokenizer.all_special_tokens) == [
        "[CLS]",
        "[MASK]",
        "[PAD]",
        "[SEP]",
        "[UNK]",
    ]
    assert sorted(tokenizer.all_special_ids) == [0, 1, 2, 3, 4]

    assert tokenizer.vocab_size == 8002
