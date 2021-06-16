# KoBERT-Transformers

`KoBERT` & `DistilKoBERT` on 🤗 Huggingface Transformers 🤗

KoBERT 모델은 [공식 레포](https://github.com/SKTBrain/KoBERT)의 것과 동일합니다. 본 레포는 **Huggingface tokenizer의 모든 API를 지원**하기 위해서 제작되었습니다.

## 🚨 중요! 🚨

### 🙏 TL;DR

1. `transformers` 는 `v3.0` 이상을 반드시 설치!
2. `tokenizer`는 본 레포의 `kobert_transformers/tokenization_kobert.py`를 사용!

### 1. Tokenizer 호환

`Huggingface Transformers`가 `v2.9.0`부터 tokenization 관련 API가 일부 변경되었습니다. 이에 맞춰 기존의 `tokenization_kobert.py`를 상위 버전에 맞게 수정하였습니다.

### 2. Embedding의 padding_idx 이슈

이전부터 `BertModel`의 `BertEmbeddings`에서 `padding_idx=0`으로 **Hard-coding**되어 있었습니다. (아래 코드 참고)

```python
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
```

그러나 Sentencepiece의 경우 기본값으로 `pad_token_id=1`, `unk_token_id=0`으로 설정이 되어 있고 (이는 KoBERT도 동일), 이를 그대로 사용하는 BertModel의 경우 원치 않은 결과를 가져올 수 있습니다.

Huggingface에서도 최근에 해당 이슈를 인지하여 이를 수정하여 `v2.9.0`에 반영하였습니다. ([관련 PR #3793](https://github.com/huggingface/transformers/pull/3793)) config에 `pad_token_id=1` 을 추가 가능하여 이를 해결할 수 있게 하였습니다.

```python
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
```

그러나 `v.2.9.0`에서 `DistilBERT`, `ALBERT` 등에는 이 이슈가 해결되지 않아 직접 PR을 올려 처리하였고 ([관련 PR #3965](https://github.com/huggingface/transformers/pull/3965)), **`v2.9.1`에 최종적으로 반영되어 배포되었습니다.**

아래는 이전과 현재 버전의 차이점을 보여주는 코드입니다.

```python
# Transformers v2.7.0
>>> from transformers import BertModel, DistilBertModel
>>> model = BertModel.from_pretrained("monologg/kobert")
>>> model.embeddings.word_embeddings
Embedding(8002, 768, padding_idx=0)
>>> model = DistilBertModel.from_pretrained("monologg/distilkobert")
>>> model.embeddings.word_embeddings
Embedding(8002, 768, padding_idx=0)


### Transformers v2.9.1
>>> from transformers import BertModel, DistilBertModel
>>> model = BertModel.from_pretrained("monologg/kobert")
>>> model.embeddings.word_embeddings
Embedding(8002, 768, padding_idx=1)
>>> model = DistilBertModel.from_pretrained("monologg/distilkobert")
>>> model.embeddings.word_embeddings
Embedding(8002, 768, padding_idx=1)
```

## KoBERT / DistilKoBERT on 🤗 Transformers 🤗

### Dependencies

- torch>=1.1.0
- transformers>=3,<5

### How to Use

```python
>>> from transformers import BertModel, DistilBertModel
>>> bert_model = BertModel.from_pretrained('monologg/kobert')
>>> distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
```

**Tokenizer를 사용하려면, [`kobert_transformers/tokenization_kobert.py`](https://github.com/monologg/KoBERT-Transformers/blob/master/kobert_transformers/tokenization_kobert.py) 파일을 복사한 후, `KoBertTokenizer`를 임포트하면 됩니다.**

- KoBERT와 DistilKoBERT 모두 동일한 토크나이저를 사용합니다.
- **기존 KoBERT의 경우 Special Token이 제대로 분리되지 않는 이슈**가 있어서 해당 부분을 수정하여 반영하였습니다. ([Issue link](https://github.com/SKTBrain/KoBERT/issues/11))

```python
>>> from tokenization_kobert import KoBertTokenizer
>>> tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
>>> ['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
>>> [2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Kobert-Transformers (Pip library)

[![PyPI](https://img.shields.io/pypi/v/kobert-transformers)](https://pypi.org/project/kobert-transformers/)
[![license](https://img.shields.io/badge/license-Apache%202.0-red)](https://github.com/monologg/DistilKoBERT/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/kobert-transformers)](https://pepy.tech/project/kobert-transformers)

- `tokenization_kobert.py`를 랩핑한 파이썬 라이브러리
- KoBERT, DistilKoBERT를 Huggingface Transformers 라이브러리 형태로 제공
- `v0.5.0`에서는 `transformers v3.0` 이상으로 기본 설치합니다. (`transformers v4.0` 까지는 이슈 없이 사용 가능)

### Install Kobert-Transformers

```bash
pip3 install kobert-transformers
```

### How to Use

```python
>>> import torch
>>> from kobert_transformers import get_kobert_model, get_distilkobert_model
>>> model = get_kobert_model()
>>> model.eval()
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
>>> sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
>>> sequence_output[0]
tensor([[-0.2461,  0.2428,  0.2590,  ..., -0.4861, -0.0731,  0.0756],
        [-0.2478,  0.2420,  0.2552,  ..., -0.4877, -0.0727,  0.0754],
        [-0.2472,  0.2420,  0.2561,  ..., -0.4874, -0.0733,  0.0765]],
       grad_fn=<SelectBackward>)
```

```python
>>> from kobert_transformers import get_tokenizer
>>> tokenizer = get_tokenizer()
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [DistilKoBERT](https://github.com/monologg/DistilKoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
