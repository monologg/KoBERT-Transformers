from transformers import BertModel, BertForMaskedLM, DistilBertModel, DistilBertForMaskedLM


def get_kobert_model(cache_dir=None):
    """ Return BertModel for Kobert """
    if cache_dir is not None:
        model = BertModel.from_pretrained(cache_dir)
    else:
        model = BertModel.from_pretrained('monologg/kobert')
    return model


def get_kobert_lm(cache_dir=None):
    """ Return BertForMaskedLM for Kobert """
    if cache_dir is not None:
        model = BertModel.from_pretrained(cache_dir)
    else:
        model = BertForMaskedLM.from_pretrained('monologg/kobert-lm')
    return model


def get_distilkobert_model(cache_dir=None):
    """ Return DistilBertModel for DistilKobert """
    if cache_dir is not None:
        model = BertModel.from_pretrained(cache_dir)
    else:
        model = DistilBertModel.from_pretrained('monologg/distilkobert')
    return model


def get_distilkobert_lm(cache_dir=None):
    """ Return DistilBertForMaskedLM for DistilKobert """
    if cache_dir is not None:
        model = BertModel.from_pretrained(cache_dir)
    else:
        model = DistilBertForMaskedLM.from_pretrained('monologg/distilkobert')
    return model
