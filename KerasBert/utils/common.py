import re
import os
import pickle
import numpy as np
import keras
from ..models import bert, transformer


class ModelExporter:
    encoder_layer_prefix = "encoder_layer_"
    decoder_layer_prefix = "decoder_layer_"
    mha_layer_postfix = "_mha_layer"
    pwff_layer_postfix = "_pwff_layer"

    def to_bert_encoder(self, model):
        params_dic = self._extract_bert_hyperparams(model)
        bert_encoder = bert.BertEncoder(**params_dic)
        bert_encoder = self._transfer_model(from_model=model, to_model=bert_encoder)
        return bert_encoder

    def to_bert_mlm(self, model):
        params_dic = self._extract_bert_hyperparams(model)
        bert_mlm = bert.BertMlm(**params_dic)
        bert_mlm = self._transfer_model(from_model=model, to_model=bert_mlm)
        return bert_mlm

    def to_bert_token_cls(self, model, num_class):
        params_dic = self._extract_bert_hyperparams(model)
        params_dic["num_class"] = num_class
        bert_token_cls = bert.BertTokenCls(**params_dic)
        bert_token_cls = self._transfer_model(from_model=model, to_model=bert_token_cls)
        return bert_token_cls

    def _extract_layer_dic(self, model):
        layer_dic = dict()
        layers = model.layers.copy()
        while len(layers) > 0:
            layer = layers.pop()
            if isinstance(layer, keras.models.Model):
                for _layer in layer.layers: layers.append(_layer)
            else:
                layer_name = layer.name
                layer_dic[layer_name] = layer
        return layer_dic

    def _extract_weights_dic(self, model):
        weights_dic = dict()
        layers = model.layers.copy()
        while len(layers) > 0:
            layer = layers.pop()
            if isinstance(layer, keras.models.Model):
                for _layer in layer.layers: layers.append(_layer)
            else:
                layer_name = layer.name
                if layer_name.startswith(self.encoder_layer_prefix) or layer_name.startswith(self.decoder_layer_prefix):
                    # encoder/decoder layer
                    # extract mha sublayer weights from encoder/decoder layer
                    mha_layer = layer.mha_layer
                    mha_layer_weights = mha_layer.get_weights()
                    mha_layer_name = layer_name + self.mha_layer_postfix
                    if mha_layer_name not in weights_dic or len(mha_layer_weights) != 0: weights_dic[mha_layer_name] = mha_layer_weights
                    # extract pwff sublayer weights from encoder/decoder layer
                    pwff_layer = layer.pwff_layer
                    pwff_layer_weights = pwff_layer.get_weights()
                    pwff_layer_name = layer_name + self.pwff_layer_postfix
                    if pwff_layer_name not in weights_dic or len(pwff_layer_weights) != 0: weights_dic[pwff_layer_name] = pwff_layer_weights
                else:
                    # ~ encoder/decoder layer
                    weights = layer.get_weights()
                    if layer_name not in weights_dic or len(weights) != 0: weights_dic[layer_name] = weights
        return weights_dic

    def _extract_transformer_hyperparams(self, model, seperate_corpus):
        # TODO: encoder_layer_1와 decoder_layer_1를 mha와 pwff로 나누기
        transformer_layer_dic = self._extract_layer_dic(model)
        # assert if model does not contain all required layers
        required_layers = None
        if seperate_corpus: required_layers = ["enc_token_embedding", "dec_token_embedding", "encoder_layer_1", "decoder_layer_1"]
        else: required_layers = ["token_embedding", "encoder_layer_1", "decoder_layer_1"]
        diff_layers = list(set(required_layers).difference(set(transformer_layer_dic.keys())))
        assert_statement = "model does not contain all required layers; {diff_layers}".format(diff_layers=diff_layers)
        assert len(diff_layers) < 1, assert_statement

        enc_vocab_size = None
        dec_vocab_size = None
        if seperate_corpus:
            enc_vocab_size = transformer_layer_dic["enc_token_embedding"].vocab_size
            dec_vocab_size = transformer_layer_dic["dec_token_embedding"].vocab_size
        else:
            enc_vocab_size = dec_vocab_size = transformer_layer_dic["token_embedding"].token_vocab_size

        encode_pos = False
        num_pos = None
        if "pos_embedding" in transformer_layer_dic:
            encode_pos = True
            num_pos = transformer_layer_dic["pos_embedding"].num_pos
        d_model = transformer_layer_dic["encoder_layer_1"].pwff_layer.d_model
        d_ff = transformer_layer_dic["encoder_layer_1"].pwff_layer.d_ff
        dropout = transformer_layer_dic["encoder_layer_1"].pwff_layer.dropout
        num_layers = max([self._get_layer_idx(layer_name) for layer_name in transformer_layer_dic.keys() if layer_name.startswith(self.encoder_layer_prefix)])
        num_heads = transformer_layer_dic["encoder_layer_1"].mha_layer.num_heads
        timesteps = transformer_layer_dic["encoder_layer_1"].mha_layer.timesteps

        params_dic = dict()
        params_dic["seperate_corpus"] = seperate_corpus
        params_dic["enc_vocab_size"] = enc_vocab_size
        params_dic["dec_vocab_size"] = dec_vocab_size
        params_dic["encode_pos"] = encode_pos
        params_dic["num_pos"] = num_pos
        params_dic["d_model"] = d_model
        params_dic["d_ff"] = d_ff
        params_dic["dropout"] = dropout
        params_dic["num_layers"] = num_layers
        params_dic["num_heads"] = num_heads
        params_dic["timesteps"] = timesteps
        return params_dic

    def _extract_bert_hyperparams(self, model, seperate_corpus):
        bert_layer_dic = self._extract_layer_dic(model)
        # assert if model does not contain all required layers
        required_layers = None
        if seperate_corpus: required_layers = ["enc_token_embedding", "dec_token_embedding", "mha_layer_1", "pwff_layer_1"]
        else: required_layers = ["token_embedding", "encoder_layer_1", "decoder_layer_1"]
        diff_layers = list(set(required_layers).difference(set(bert_layer_dic.keys())))
        assert_statement = "model does not contain all required layers; {diff_layers}".format(diff_layers=diff_layers)
        assert len(diff_layers) < 1, assert_statement

        vocab_size = bert_layer_dic["token_embedding"].vocab_size
        num_segments = bert_layer_dic["segment_embedding"].num_segments - 1
        encode_pos = False
        num_pos = None
        if "pos_embedding" in bert_layer_dic:
            encode_pos = True
            num_pos = bert_layer_dic["pos_embedding"].num_pos
        d_model = bert_layer_dic["pwff_layer_1"].pwff_layer.d_model
        d_ff = bert_layer_dic["pwff_layer_1"].pwff_layer.d_ff
        dropout = bert_layer_dic["pwff_layer_1"].pwff_layer.dropout
        num_layers = max([self._get_layer_idx(layer_name) for layer_name in bert_layer_dic.keys() if layer_name.startswith("pwff_layer_")])
        num_heads = bert_layer_dic["mha_layer_1"].num_heads
        timesteps = bert_layer_dic["mha_layer_1"].timesteps

        params_dic = dict()
        params_dic["vocab_size"] = vocab_size
        params_dic["num_segments"] = num_segments
        params_dic["encode_pos"] = encode_pos
        params_dic["num_pos"] = num_pos
        params_dic["d_model"] = d_model
        params_dic["d_ff"] = d_ff
        params_dic["dropout"] = dropout
        params_dic["num_layers"] = num_layers
        params_dic["num_heads"] = num_heads
        params_dic["timesteps"] = timesteps
        return params_dic

    def _transfer_model(self, from_model, to_model, target_layers=None):
        from_model_weights_dic = self._extract_weights_dic(model=from_model)
        # load weights of from_model
        format_template = "{0:^25}|{1:^10}|{2:^10}"
        print(format_template.format("layer_name", "loaded", "updated"))
        print("-"*45)
        for layer in to_model.layers:
            layer_name = layer.name
            loaded = False
            updated = False
            if layer_name in from_model_weights_dic:
                weights = from_model_weights_dic[layer_name]
                loaded = True
                loaded_weights_sum = np.mean([np.mean(weight) for weight in weights])
                layer_weights_sum = np.mean([np.mean(weight) for weight in layer.get_weights()])
                if len(weights) > 0 and loaded_weights_sum != layer_weights_sum: updated = True
                layer.set_weights(weights)
            print(format_template.format(layer_name, str(loaded), str(updated)))
        return to_model

    def _get_layer_idx(self, layer_name):
        layer_idx_matcher = re.search("[0-9]+", layer_name)
        layer_idx = int(layer_name[layer_idx_matcher.start():layer_idx_matcher.end()])
        return layer_idx

def save_model(model_dir, model):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    json_path = model_dir + "model.json"
    weight_path = model_dir + "weights.h5"
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight_path)
    print("saved model into '{path}'".format(path=model_dir))
    return True


def load_model(model_dir, custom_objects):
    json_path = model_dir + "model.json"
    weight_path = model_dir + "weights.h5"
    model = None
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(weight_path)
    print("loaded model from '{path}'".format(path=model_dir))
    return model


def save_pickle(path, data):
    with open(path, "wb") as fp:
        pickle.dump(data, path)
    return True


def load_pickle(path):
    data = None
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data


def get_trained_batch(model_dir, history_filename):
    trained_batch = 0
    if os.path.isdir(model_dir) and history_filename in os.listdir(model_dir):
        history = load_pickle(model_dir + history_filename)
        if history is not None and "loss" in history:
            trained_batch = len(history["loss"])
    return trained_batch