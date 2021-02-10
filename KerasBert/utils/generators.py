import os
import pickle
import numpy as np
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K

class TransformerGenerator(keras.utils.Sequence):
    def __init__(self, rows, preprocessor, batch_size, encode_pos, shuffle=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        self.data_size = len(rows)
        self.data = self._encode(rows)
        self.indices = np.arange(0, self.data_size)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        enc_token_id_inputs_batch = self.data["enc_token_id_inputs"][begin_idx:end_idx]
        dec_token_id_inputs_batch = self.data["dec_token_id_inputs"][begin_idx:end_idx]
        dec_token_id_outputs_batch = self.data["dec_token_id_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            enc_pos_id_inputs_batch = self.data["enc_pos_id_inputs"][begin_idx:end_idx]
            dec_pos_id_inputs_batch = self.data["dec_pos_id_inputs"][begin_idx:end_idx]
            enc_pos_id_inputs_batch = self.preprocessor.pad_sequence(enc_pos_id_inputs_batch)
            dec_pos_id_inputs_batch = self.preprocessor.pad_sequence(dec_pos_id_inputs_batch)
            x_batch["enc_pos_id_inputs"] = enc_pos_id_inputs_batch
            y_batch["dec_pos_id_inputs"] = dec_pos_id_inputs_batch

        # padding
        enc_token_id_inputs_batch = self.preprocessor.pad_sequence(enc_token_id_inputs_batch)
        dec_token_id_inputs_batch = self.preprocessor.pad_sequence(dec_token_id_inputs_batch)
        dec_token_id_outputs_batch = self.preprocessor.pad_sequence(dec_token_id_outputs_batch)
        # onehot encoding
        dec_token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(dec_token_id_outputs_batch)

        x_batch["enc_token_id_inputs"] = enc_token_id_inputs_batch
        x_batch["dec_token_id_inputs"] = dec_token_id_inputs_batch
        y_batch["dec_token_id_outputs"] = dec_token_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, rows):
        data = dict()
        input_data_dict = self.preprocessor.train_encode(rows, encode_pos=self.encode_pos)
        data["enc_token_id_inputs"] = input_data_dict["encoder_input"][0]
        data["dec_token_id_inputs"] = input_data_dict["decoder_input"][0]
        data["dec_token_id_outputs"] = input_data_dict["decoder_output"][0]

        if self.encode_pos:
            data["enc_pos_id_inputs"] = input_data_dict["encoder_input"][1]
            data["dec_pos_id_outputs"] = input_data_dict["decoder_input"][1]
        return data


class BertGenerator(keras.utils.Sequence):
    def __init__(self, sequences, labels, preprocessor, batch_size, encode_pos, shuffle=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        error_statement = "Data size incompatible; {len1} vs {len2}".format(len1=len(sequences), len2=len(labels))
        assert len(sequences) == len(labels), error_statement
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)
        self.data["nsp_id_outputs"] = labels

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        num_segments = 2
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        token_id_inputs_batch = self.data["token_id_inputs"][begin_idx:end_idx]
        seq_id_inputs_batch = self.data["seq_id_inputs"][begin_idx:end_idx]
        token_id_outputs_batch = self.data["token_id_outputs"][begin_idx:end_idx]
        nsp_id_outputs_batch = self.data["nsp_id_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            pos_id_inputs_batch = self.data["pos_id_inputs"][begin_idx:end_idx]
            pos_id_inputs_batch = self.preprocessor.pad_sequence(pos_id_inputs_batch)
            x_batch["pos_id_inputs"] = pos_id_inputs_batch

        # padding
        token_id_inputs_batch = self.preprocessor.pad_sequence(token_id_inputs_batch)
        seq_id_inputs_batch = self.preprocessor.pad_sequence(seq_id_inputs_batch)
        token_id_outputs_batch = self.preprocessor.pad_sequence(token_id_outputs_batch)
        # onehot encoding
        token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(token_id_outputs_batch)
        nsp_id_outputs_batch = self.preprocessor.onehot_dec_sentence(nsp_id_outputs_batch, num_class=num_segments)

        x_batch["token_id_inputs"] = token_id_inputs_batch
        x_batch["seq_id_inputs"] = seq_id_inputs_batch
        y_batch["mlm_output"] = token_id_outputs_batch
        y_batch["nsp_output"] = nsp_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        token_id_inputs, seq_id_inputs, token_id_outputs = None, None, None
        if self.encode_pos:
            token_id_inputs, seq_id_inputs, pos_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            data["pos_id_inputs"] = pos_id_inputs
        else:
            token_id_inputs, seq_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
        token_id_outputs, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=self.encode_pos)
        data["token_id_inputs"] = token_id_inputs
        data["seq_id_inputs"] = seq_id_inputs
        data["token_id_outputs"] = token_id_outputs
        return data


class BertMlmGenerator(keras.utils.Sequence):
    def __init__(self, sequences, preprocessor, batch_size, encode_pos, shuffle=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        token_id_inputs_batch = self.data["token_id_inputs"][begin_idx:end_idx]
        seq_id_inputs_batch = self.data["seq_id_inputs"][begin_idx:end_idx]
        token_id_outputs_batch = self.data["token_id_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            pos_id_inputs_batch = self.data["pos_id_inputs"][begin_idx:end_idx]
            pos_id_inputs_batch = self.preprocessor.pad_sequence(pos_id_inputs_batch)
            x_batch["pos_id_inputs"] = pos_id_inputs_batch

        # padding
        token_id_inputs_batch = self.preprocessor.pad_sequence(token_id_inputs_batch)
        seq_id_inputs_batch = self.preprocessor.pad_sequence(seq_id_inputs_batch)
        token_id_outputs_batch = self.preprocessor.pad_sequence(token_id_outputs_batch)
        # onehot encoding
        token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(token_id_outputs_batch)

        x_batch["token_id_inputs"] = token_id_inputs_batch
        x_batch["seq_id_inputs"] = seq_id_inputs_batch
        y_batch["mlm_output"] = token_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        token_id_inputs, seq_id_inputs, token_id_outputs = None, None, None
        if self.encode_pos:
            token_id_inputs, seq_id_inputs, pos_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            data["pos_id_inputs"] = pos_id_inputs
        else:
            token_id_inputs, seq_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
        token_id_outputs, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=self.encode_pos)
        data["token_id_inputs"] = token_id_inputs
        data["seq_id_inputs"] = seq_id_inputs
        data["token_id_outputs"] = token_id_outputs
        return data


class BertTokenClsGenerator(keras.utils.Sequence):
    def __init__(self, sequences, labels, num_class, preprocessor, batch_size, encode_pos, shuffle=False):
        self.num_class = num_class
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        error_statement = "Data size incompatible; {len1} vs {len2}".format(len1=len(sequences), len2=len(labels))
        assert len(sequences) == len(labels), error_statement
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)
        self.data["cls_id_outputs"] = labels

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        token_id_inputs_batch = self.data["token_id_inputs"][begin_idx:end_idx]
        seq_id_inputs_batch = self.data["seq_id_inputs"][begin_idx:end_idx]
        cls_id_outputs_batch = self.data["cls_id_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            pos_id_inputs_batch = self.data["pos_id_inputs"][begin_idx:end_idx]
            pos_id_inputs_batch = self.preprocessor.pad_sequence(pos_id_inputs_batch)
            x_batch["pos_id_inputs"] = pos_id_inputs_batch

        # padding
        token_id_inputs_batch = self.preprocessor.pad_sequence(token_id_inputs_batch)
        seq_id_inputs_batch = self.preprocessor.pad_sequence(seq_id_inputs_batch)
        # onehot encoding
        cls_id_outputs_batch = self.preprocessor.onehot_dec_sentence(cls_id_outputs_batch, num_class=self.num_class)

        x_batch["token_id_inputs"] = token_id_inputs_batch
        x_batch["seq_id_inputs"] = seq_id_inputs_batch
        y_batch["cls_outputs"] = cls_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        token_id_inputs, seq_id_inputs = None, None
        if self.encode_pos:
            token_id_inputs, seq_id_inputs, pos_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            data["pos_id_inputs"] = pos_id_inputs
        else:
            token_id_inputs, seq_id_inputs = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
        data["token_id_inputs"] = token_id_inputs
        data["seq_id_inputs"] = seq_id_inputs
        return data


class SentenceBertClsGenerator(keras.utils.Sequence):
    def __init__(self, sequences, labels, num_class, preprocessor, batch_size, encode_pos, mask=False, shuffle=False):
        self.num_class = num_class
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.mask = mask
        self.shuffle = shuffle
        error_statement = "Data size incompatible; {len1} vs {len2}".format(len1=len(sequences), len2=len(labels))
        assert len(sequences) == len(labels), error_statement
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)
        self.data["cls_id_outputs"] = labels

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        left_token_id_inputs_batch = self.data["left_token_id_inputs"][begin_idx:end_idx]
        left_seq_id_inputs_batch = self.data["left_seq_id_inputs"][begin_idx:end_idx]
        right_token_id_inputs_batch = self.data["right_token_id_inputs"][begin_idx:end_idx]
        right_seq_id_inputs_batch = self.data["right_seq_id_inputs"][begin_idx:end_idx]
        cls_id_outputs_batch = self.data["cls_id_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            left_pos_id_inputs_batch = self.data["left_pos_id_inputs"][begin_idx:end_idx]
            left_pos_id_inputs_batch = self.preprocessor.pad_sequence(left_pos_id_inputs_batch)
            right_pos_id_inputs_batch = self.data["right_pos_id_inputs"][begin_idx:end_idx]
            right_pos_id_inputs_batch = self.preprocessor.pad_sequence(right_pos_id_inputs_batch)
            x_batch["left_pos_id_inputs"] = left_pos_id_inputs_batch
            x_batch["right_pos_id_inputs"] = right_pos_id_inputs_batch

        # padding
        left_token_id_inputs_batch = self.preprocessor.pad_sequence(left_token_id_inputs_batch)
        left_seq_id_inputs_batch = self.preprocessor.pad_sequence(left_seq_id_inputs_batch)
        right_token_id_inputs_batch = self.preprocessor.pad_sequence(right_token_id_inputs_batch)
        right_seq_id_inputs_batch = self.preprocessor.pad_sequence(right_seq_id_inputs_batch)
        # onehot encoding
        cls_id_outputs_batch = self.preprocessor.onehot_dec_sentence(cls_id_outputs_batch, num_class=self.num_class)

        x_batch["left_token_id_inputs"] = left_token_id_inputs_batch
        x_batch["left_seq_id_inputs"] = left_seq_id_inputs_batch
        x_batch["right_token_id_inputs"] = right_token_id_inputs_batch
        x_batch["right_seq_id_inputs"] = right_seq_id_inputs_batch
        x_batch["cls_outputs"] = cls_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        left_sentences = []
        right_sentences = []
        for left_sentence, right_sentence in sequences:
            left_sentences.append(left_sentence)
            right_sentences.append(right_sentence)

        left_token_id_inputs, left_seq_id_inputs = None, None
        right_token_id_inputs, right_seq_id_inputs = None, None
        if self.encode_pos:
            left_token_id_inputs, left_seq_id_inputs, left_pos_id_inputs = self.preprocessor.encode(left_sentences, mask=self.mask, encode_pos=self.encode_pos)
            right_token_id_inputs, right_seq_id_inputs, right_pos_id_inputs = self.preprocessor.encode(right_sentences, mask=self.mask, encode_pos=self.encode_pos)
            data["left_pos_id_inputs"] = left_pos_id_inputs
            data["right_pos_id_inputs"] = right_pos_id_inputs
        else:
            left_token_id_inputs, left_seq_id_inputs = self.preprocessor.encode(left_sentences, mask=self.mask, encode_pos=self.encode_pos)
            right_token_id_inputs, right_seq_id_inputs = self.preprocessor.encode(right_sentences, mask=self.mask, encode_pos=self.encode_pos)

        data["left_token_id_inputs"] = left_token_id_inputs
        data["left_seq_id_inputs"] = left_seq_id_inputs
        data["right_token_id_inputs"] = right_token_id_inputs
        data["right_seq_id_inputs"] = right_seq_id_inputs
        return data


class SentenceBertRegGenerator(keras.utils.Sequence):
    def __init__(self, sequences, scores, num_class, preprocessor, batch_size, encode_pos, mask=False, shuffle=False):
        self.num_class = num_class
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.mask = mask
        self.shuffle = shuffle
        error_statement = "Data size incompatible; {len1} vs {len2}".format(len1=len(sequences), len2=len(scores))
        assert len(sequences) == len(scores), error_statement
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)
        self.data["reg_value_outputs"] = scores

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        left_token_id_inputs_batch = self.data["left_token_id_inputs"][begin_idx:end_idx]
        left_seq_id_inputs_batch = self.data["left_seq_id_inputs"][begin_idx:end_idx]
        right_token_id_inputs_batch = self.data["right_token_id_inputs"][begin_idx:end_idx]
        right_seq_id_inputs_batch = self.data["right_seq_id_inputs"][begin_idx:end_idx]
        reg_value_outputs_batch = self.data["reg_value_outputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            left_pos_id_inputs_batch = self.data["left_pos_id_inputs"][begin_idx:end_idx]
            left_pos_id_inputs_batch = self.preprocessor.pad_sequence(left_pos_id_inputs_batch)
            right_pos_id_inputs_batch = self.data["right_pos_id_inputs"][begin_idx:end_idx]
            right_pos_id_inputs_batch = self.preprocessor.pad_sequence(right_pos_id_inputs_batch)
            x_batch["left_pos_id_inputs"] = left_pos_id_inputs_batch
            x_batch["right_pos_id_inputs"] = right_pos_id_inputs_batch

        # padding
        left_token_id_inputs_batch = self.preprocessor.pad_sequence(left_token_id_inputs_batch)
        left_seq_id_inputs_batch = self.preprocessor.pad_sequence(left_seq_id_inputs_batch)
        right_token_id_inputs_batch = self.preprocessor.pad_sequence(right_token_id_inputs_batch)
        right_seq_id_inputs_batch = self.preprocessor.pad_sequence(right_seq_id_inputs_batch)

        x_batch["left_token_id_inputs"] = left_token_id_inputs_batch
        x_batch["left_seq_id_inputs"] = left_seq_id_inputs_batch
        x_batch["right_token_id_inputs"] = right_token_id_inputs_batch
        x_batch["right_seq_id_inputs"] = right_seq_id_inputs_batch
        x_batch["reg_output"] = reg_value_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        left_sentences = []
        right_sentences = []
        for left_sentence, right_sentence in sequences:
            left_sentences.append(left_sentence)
            right_sentences.append(right_sentence)

        left_token_id_inputs, left_seq_id_inputs = None, None
        right_token_id_inputs, right_seq_id_inputs = None, None
        if self.encode_pos:
            left_token_id_inputs, left_seq_id_inputs, left_pos_id_inputs = self.preprocessor.encode(left_sentences, mask=self.mask, encode_pos=self.encode_pos)
            right_token_id_inputs, right_seq_id_inputs, right_pos_id_inputs = self.preprocessor.encode(right_sentences, mask=self.mask, encode_pos=self.encode_pos)
            data["left_pos_id_inputs"] = left_pos_id_inputs
            data["right_pos_id_inputs"] = right_pos_id_inputs
        else:
            left_token_id_inputs, left_seq_id_inputs = self.preprocessor.encode(left_sentences, mask=self.mask, encode_pos=self.encode_pos)
            right_token_id_inputs, right_seq_id_inputs = self.preprocessor.encode(right_sentences, mask=self.mask, encode_pos=self.encode_pos)

        data["left_token_id_inputs"] = left_token_id_inputs
        data["left_seq_id_inputs"] = left_seq_id_inputs
        data["right_token_id_inputs"] = right_token_id_inputs
        data["right_seq_id_inputs"] = right_seq_id_inputs
        return data


class SentenceBertTripletGenerator(keras.utils.Sequence):
    def __init__(self, sequences, d_model, preprocessor, batch_size, encode_pos, mask=False, shuffle=False):
        self.d_model = d_model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.mask = mask
        self.shuffle = shuffle
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # split by batch_size
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        anchor_token_id_inputs_batch = self.data["anchor_token_id_inputs"][begin_idx:end_idx]
        anchor_seq_id_inputs_batch = self.data["anchor_seq_id_inputs"][begin_idx:end_idx]
        positive_token_id_inputs_batch = self.data["positive_token_id_inputs"][begin_idx:end_idx]
        positive_seq_id_inputs_batch = self.data["positive_seq_id_inputs"][begin_idx:end_idx]
        negative_token_id_inputs_batch = self.data["negative_token_id_inputs"][begin_idx:end_idx]
        negative_seq_id_inputs_batch = self.data["negative_seq_id_inputs"][begin_idx:end_idx]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            anchor_pos_id_inputs_batch = self.data["anchor_pos_id_inputs"][begin_idx:end_idx]
            anchor_pos_id_inputs_batch = self.preprocessor.pad_sequence(anchor_pos_id_inputs_batch)
            positive_pos_id_inputs_batch = self.data["positive_pos_id_inputs"][begin_idx:end_idx]
            positive_pos_id_inputs_batch = self.preprocessor.pad_sequence(positive_pos_id_inputs_batch)
            negative_pos_id_inputs_batch = self.data["negative_pos_id_inputs"][begin_idx:end_idx]
            negative_pos_id_inputs_batch = self.preprocessor.pad_sequence(negative_pos_id_inputs_batch)
            x_batch["anchor_pos_id_inputs"] = anchor_pos_id_inputs_batch
            x_batch["positive_pos_id_inputs"] = positive_pos_id_inputs_batch
            x_batch["negative_pos_id_inputs"] = negative_pos_id_inputs_batch

        # padding
        anchor_token_id_inputs_batch = self.preprocessor.pad_sequence(anchor_token_id_inputs_batch)
        anchor_seq_id_inputs_batch = self.preprocessor.pad_sequence(anchor_seq_id_inputs_batch)
        positive_token_id_inputs_batch = self.preprocessor.pad_sequence(positive_token_id_inputs_batch)
        positive_seq_id_inputs_batch = self.preprocessor.pad_sequence(positive_seq_id_inputs_batch)
        negative_token_id_inputs_batch = self.preprocessor.pad_sequence(negative_token_id_inputs_batch)
        negative_seq_id_inputs_batch = self.preprocessor.pad_sequence(negative_seq_id_inputs_batch)
        # onehot encoding
        dummy_output_batch = np.zeros((self.batch_size, 3, self.d_model))

        x_batch["anchor_token_id_inputs"] = anchor_token_id_inputs_batch
        x_batch["anchor_seq_id_inputs"] = anchor_seq_id_inputs_batch
        x_batch["positive_token_id_inputs"] = positive_token_id_inputs_batch
        x_batch["positive_seq_id_inputs"] = positive_seq_id_inputs_batch
        x_batch["negative_token_id_inputs"] = negative_token_id_inputs_batch
        x_batch["negative_seq_id_inputs"] = negative_seq_id_inputs_batch
        x_batch["dummy_output"] = dummy_output_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, sequences):
        data = dict()
        anchor_sentences = []
        positive_sentences = []
        negative_sentences = []
        for anchor_sentence, positive_sentence, negative_sentence in sequences:
            anchor_sentences.append(anchor_sentence)
            positive_sentences.append(positive_sentence)
            negative_sentences.append(negative_sentence)

        anchor_token_id_inputs, anchor_seq_id_inputs = None, None
        positive_token_id_inputs, positive_seq_id_inputs = None, None
        negative_token_id_inputs, negative_seq_id_inputs = None, None
        if self.encode_pos:
            anchor_token_id_inputs, anchor_seq_id_inputs, anchor_pos_id_inputs = self.preprocessor.encode(anchor_sentences, mask=self.mask, encode_pos=self.encode_pos)
            positive_token_id_inputs, positive_seq_id_inputs, positive_pos_id_inputs = self.preprocessor.encode(positive_sentences, mask=self.mask, encode_pos=self.encode_pos)
            negative_token_id_inputs, negative_seq_id_inputs, negative_pos_id_inputs = self.preprocessor.encode(negative_sentences, mask=self.mask, encode_pos=self.encode_pos)
            data["anchor_pos_id_inputs"] = anchor_token_id_inputs
            data["positive_pos_id_inputs"] = positive_pos_id_inputs
            data["right_pos_id_inputs"] = negative_pos_id_inputs
        else:
            anchor_token_id_inputs, anchor_seq_id_inputs = self.preprocessor.encode(anchor_sentences, mask=self.mask, encode_pos=self.encode_pos)
            positive_token_id_inputs, positive_seq_id_inputs = self.preprocessor.encode(positive_sentences, mask=self.mask, encode_pos=self.encode_pos)
            negative_token_id_inputs, negative_seq_id_inputs = self.preprocessor.encode(negative_sentences, mask=self.mask, encode_pos=self.encode_pos)

        data["anchor_token_id_inputs"] = anchor_token_id_inputs
        data["anchor_seq_id_inputs"] = anchor_seq_id_inputs
        data["positive_token_id_inputs"] = positive_token_id_inputs
        data["positive_seq_id_inputs"] = positive_seq_id_inputs
        data["negative_token_id_inputs"] = negative_token_id_inputs
        data["negative_seq_id_inputs"] = negative_seq_id_inputs
        return data


class FileStreamingIterator:
    def __init__(self, path, preprocessor, batch_size, encode_pos, encoding="UTF-8", shuffle=False):
        self.path = path
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encoding = encoding
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        self.batch_reader = self._read_batch_from_txt(path=self.path, batch_size=self.batch_size, encoding=self.encoding)
        self.data_size = self._get_data_size(path=self.path, encoding=self.encoding)
        self.indices = np.arange(0, self.data_size)

    def _read_batch_from_txt(self, path, batch_size, encoding):
        file_pointer = open(path, "r", encoding=encoding)
        batch = []
        batch_count = 0

        while True:
            row = file_pointer.readline()
            if row == "" or batch_count >= batch_size:
                yield batch
                batch = []
                batch_count = 0

            if row == "":
                file_pointer.close()
                file_pointer = open(path, "r", encoding=encoding)
            else:
                batch.append(row)
                batch_count += 1

    def _get_data_size(self, path, encoding):
        data_size = 0
        with open(path, "r", encoding=encoding) as fp:
            for row in fp: data_size += 1
        return data_size


class TransformerFileStreamingGenerator(keras.utils.Sequence, FileStreamingIterator):
    def __init__(self, path, preprocessor, batch_size, encode_pos, shuffle=False):
        FileStreamingIterator.__init__(self=self, path=path, preprocessor=preprocessor, batch_size=batch_size, encode_pos=encode_pos, shuffle=shuffle)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # get batch from text file
        batch_rows = self.batch_reader.__next__()
        # parse batch_ rows
        rows = None
        try:
            rows = self._parse_rows(batch_rows)
        except:
            invalid_row_error_statement = "Check if there is invalid row"
            raise AssertionError(invalid_row_error_statement)
        data = self._encode(rows=rows)
        enc_token_id_inputs_batch = data["enc_token_id_inputs"]
        dec_token_id_inputs_batch = data["dec_token_id_inputs"]
        dec_token_id_outputs_batch = data["dec_token_id_outputs"]

        x_batch = dict()
        y_batch = dict()
        if self.encode_pos:
            enc_pos_id_inputs_batch = self.data["enc_pos_id_inputs"]
            dec_pos_id_inputs_batch = self.data["dec_pos_id_inputs"]
            enc_pos_id_inputs_batch = self.preprocessor.pad_sequence(enc_pos_id_inputs_batch)
            dec_pos_id_inputs_batch = self.preprocessor.pad_sequence(dec_pos_id_inputs_batch)
            x_batch["enc_pos_id_inputs"] = enc_pos_id_inputs_batch
            y_batch["dec_pos_id_inputs"] = dec_pos_id_inputs_batch

        # padding
        enc_token_id_inputs_batch = self.preprocessor.pad_sequence(enc_token_id_inputs_batch)
        dec_token_id_inputs_batch = self.preprocessor.pad_sequence(dec_token_id_inputs_batch)
        dec_token_id_outputs_batch = self.preprocessor.pad_sequence(dec_token_id_outputs_batch)
        # onehot encoding
        dec_token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(dec_token_id_outputs_batch)

        x_batch["enc_token_id_inputs"] = enc_token_id_inputs_batch
        x_batch["dec_token_id_inputs"] = dec_token_id_inputs_batch
        y_batch["dec_token_id_outputs"] = dec_token_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _encode(self, rows):
        data = dict()
        input_data_dict = self.preprocessor.train_encode(rows, encode_pos=self.encode_pos)
        data["enc_token_id_inputs"] = input_data_dict["encoder_input"][0]
        data["dec_token_id_inputs"] = input_data_dict["decoder_input"][0]
        data["dec_token_id_outputs"] = input_data_dict["decoder_output"][0]

        if self.encode_pos:
            data["enc_pos_id_inputs"] = input_data_dict["encoder_input"][1]
            data["dec_pos_id_outputs"] = input_data_dict["decoder_input"][1]
        return data

    def _parse_rows(self, rows, delimeter="\t"):
        output = []
        for row in rows:
            row = row.strip().split(delimeter)
            output.append(row)
        return output


class BertFileStreamingGenerator(keras.utils.Sequence, FileStreamingIterator):
    def __init__(self, path, preprocessor, batch_size, encode_pos, shuffle=False):
        FileStreamingIterator.__init__(self=self, path=path, preprocessor=preprocessor, batch_size=batch_size, encode_pos=encode_pos, shuffle=shuffle)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        num_segments = 2
        # get batch from text file
        batch_rows = self.batch_reader.__next__()
        # parse batch_ rows
        sequences = None
        nsp_id_outputs = None
        try:
            sequences, nsp_id_outputs_batch = self._parse_rows(batch_rows)
        except:
            invalid_row_error_statement = "Check if there is invalid row"
            raise AssertionError(invalid_row_error_statement)

        x_batch = dict()
        y_batch = dict()
        # encode
        token_id_inputs_batch, seq_id_inputs_batch, token_id_outputs_batch = None, None, None
        if self.encode_pos:
            token_id_inputs_batch, seq_id_inputs_batch, pos_id_inputs_batch = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            token_id_outputs_batch, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=False)
            pos_id_inputs_batch = self.preprocessor.pad_sequence(pos_id_inputs_batch)
            x_batch["pos_id_inputs"] = pos_id_inputs_batch
        else:
            token_id_inputs_batch, seq_id_inputs_batch = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            token_id_outputs_batch, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=False)

        # padding
        token_id_inputs_batch = self.preprocessor.pad_sequence(token_id_inputs_batch)
        seq_id_inputs_batch = self.preprocessor.pad_sequence(seq_id_inputs_batch)
        token_id_outputs_batch = self.preprocessor.pad_sequence(token_id_outputs_batch)
        # onehot encoding
        token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(token_id_outputs_batch)
        nsp_id_outputs_batch = self.preprocessor.onehot_dec_sentence(nsp_id_outputs_batch, num_class=num_segments)

        x_batch["token_id_inputs"] = token_id_inputs_batch
        x_batch["seq_id_inputs"] = seq_id_inputs_batch
        y_batch["mlm_output"] = token_id_outputs_batch
        y_batch["nsp_output"] = nsp_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _parse_rows(self, rows, num_segments, delimeter="\t"):
        invalid_row_error_statement = "Some data is missing; Check if each row contains two sentences and one nsp label"
        sequences = []
        nsp_id_outputs = []

        for row in rows:
            row = row.strip().split(delimeter)
            if num_segments is not None:
                assert len(row) == (num_segments + 1), invalid_row_error_statement
                sequence = row[:-1]
            else:
                sequence = row[0:1]
            nsp_id_output = row[-1]
            sequences.append(sequence)
            nsp_id_outputs.append(nsp_id_output)

        return sequences, nsp_id_outputs


class BertMlmFileStreamingGenerator(keras.utils.Sequence):
    def __init__(self, sequences, preprocessor, batch_size, encode_pos, shuffle=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.encode_pos = encode_pos
        self.shuffle = shuffle
        self.data_size = len(sequences)
        self.indices = np.arange(0, self.data_size)
        self.data = self._encode(sequences)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        # get batch from text file
        batch_rows = self.batch_reader.__next__()
        # parse batch_ rows
        sequences = None
        nsp_id_outputs = None
        try:
            sequences = self._parse_rows(batch_rows)
        except:
            invalid_row_error_statement = "Check if there is invalid row"
            raise AssertionError(invalid_row_error_statement)

        x_batch = dict()
        y_batch = dict()
        # encode
        token_id_inputs_batch, seq_id_inputs_batch, token_id_outputs_batch = None, None, None
        if self.encode_pos:
            token_id_inputs_batch, seq_id_inputs_batch, pos_id_inputs_batch = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            token_id_outputs_batch, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=False)
            pos_id_inputs_batch = self.preprocessor.pad_sequence(pos_id_inputs_batch)
            x_batch["pos_id_inputs"] = pos_id_inputs_batch
        else:
            token_id_inputs_batch, seq_id_inputs_batch = self.preprocessor.encode(sequences, mask=True, encode_pos=self.encode_pos)
            token_id_outputs_batch, _ = self.preprocessor.encode(sequences, mask=False, encode_pos=False)

        # padding
        token_id_inputs_batch = self.preprocessor.pad_sequence(token_id_inputs_batch)
        seq_id_inputs_batch = self.preprocessor.pad_sequence(seq_id_inputs_batch)
        token_id_outputs_batch = self.preprocessor.pad_sequence(token_id_outputs_batch)
        # onehot encoding
        token_id_outputs_batch = self.preprocessor.onehot_dec_sentence(token_id_outputs_batch)

        x_batch["token_id_inputs"] = token_id_inputs_batch
        x_batch["seq_id_inputs"] = seq_id_inputs_batch
        y_batch["mlm_output"] = token_id_outputs_batch
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            # random sampling on each batch iteration
            self.indices = np.random.permutation(self.data_size)

    def _parse_rows(self, rows, num_segments, delimeter="\t"):
        invalid_row_error_statement = "Some data is missing; Check if each row contains two sentences and one nsp label"
        sequences = []

        for row in rows:
            row = row.strip().split(delimeter)
            sequence = None
            if num_segments is not None:
                assert len(row) >= num_segments, invalid_row_error_statement
                sequence = row[:num_segments]
            else:
                sequence = row[0:1]
            sequences.append(sequence)

        return sequences