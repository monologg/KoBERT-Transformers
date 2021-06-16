from transformers import BertForMaskedLM, BertModel, DistilBertForMaskedLM, DistilBertModel


def get_kobert_model():
    """ Return BertModel for Kobert """
    model = BertModel.from_pretrained("monologg/kobert")
    return model


def get_kobert_lm():
    """ Return BertForMaskedLM for Kobert """
    model = BertForMaskedLM.from_pretrained("monologg/kobert-lm")
    return model


def get_distilkobert_model():
    """ Return DistilBertModel for DistilKobert """
    model = DistilBertModel.from_pretrained("monologg/distilkobert")
    return model


def get_distilkobert_lm():
    """ Return DistilBertForMaskedLM for DistilKobert """
    model = DistilBertForMaskedLM.from_pretrained("monologg/distilkobert")
    return model
