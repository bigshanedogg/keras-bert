from tqdm import tqdm
import numpy as np
import keras
from .preprocessor import PreprocessorConfig, Preprocessor

class BertPreprocessor:
    timesteps = None
    seperate_corpus = None
    encoder_preprocessor = None
    decoder_preprocessor = None
    over_length_row = None

    def __init__(self, timesteps, seperate_corpus, over_length_row="truncate", konlpy_mecab=True, config=PreprocessorConfig):
        '''
        :param timesteps: int, max sequence length
        :param seperate_corpus: whether to use different vocabs for encoder and decoder
        :param over_length_row: string, how to cope with exception when sequence length is over than given timesteps
                                (available: ["ignore", "truncate", "stop"])
        :param konlpy_mecab: True if konlpy.Mecab(both available on Windows & Linux) has been installed
                              False if MeCab(usually available on Linux, not on Windows) has been installed
        :param config:
        '''
        self.timesteps = timesteps
        self.seperate_corpus = seperate_corpus
        self.over_length_row = over_length_row
        if seperate_corpus:
            self.encoder_preprocessor = Preprocessor(config=config, konlpy_mecab=konlpy_mecab)
            self.decoder_preprocessor = Preprocessor(config=config, konlpy_mecab=konlpy_mecab)
        else:
            self.encoder_preprocessor = self.decoder_preprocessor = Preprocessor(config=config, konlpy_mecab=konlpy_mecab)

    def train_enode(self, encoder_sentences, decoder_sentences, encode_pos, verbose=False):
        self.encoder_preprocessor._assert_length_match(encoder_sentences, decoder_sentences)

        encoder_input_token_ids = []
        decoder_input_token_ids = []
        decoder_output_token_ids = []
        encoder_input_pos_ids = []
        decoder_input_pos_ids = []
        decoder_output_pos_ids = []

        rows = zip(encoder_sentences, decoder_sentences)
        if verbose: rows = tqdm(rows)
        for encoder_sentence, decoder_sentence in rows:
            eti, epi = self.encoder_encode(sentences=encoder_sentence, encode_pos=encode_pos, verbose=verbose)
            dti, dpi = self.decoder_encode(sentences=decoder_sentence, encode_pos=encode_pos, verbose=verbose)
            dto, dpo = self.decoder_encode(sentences=decoder_sentence, encode_pos=encode_pos, verbose=verbose)
            dti.insert(0, self.encoder_preprocessor.config["bos_token_id"])
            dpi.insert(0, self.encoder_preprocessor.pos_dict["bos_token"])
            dto.append(self.encoder_preprocessor.config["eos_token_id"])
            dpo.appendvb(self.encoder_preprocessor.pos_dict["eos_token"])

            row = [eti, epi, dti, dpi, dto, dpo]
            if not self.is_feasible_row(row): continue
            encoder_input_token_ids.append(eti)
            decoder_input_token_ids.append(dti)
            decoder_output_token_ids.append(dto)

            if encode_pos:
                encoder_input_pos_ids.append(epi)
                decoder_input_pos_ids.append(dpi)
                decoder_output_pos_ids.append(dpo)

        output = dict()
        output["encoder_input"] = (encoder_input_token_ids, encoder_input_pos_ids)
        output["decoder_input"] = (decoder_input_token_ids, decoder_input_pos_ids)
        output["decoder_output"] = (decoder_output_token_ids, decoder_output_pos_ids)
        return output

    def encoder_encode(self, sentences, encode_pos, verbose=False):
        '''
        :param sentences: list of sentences to feed encoder
                            e.g.) "I ate an apple."
        :param encode_pos: whether to feed pos tag while training bert likewise seq_ids
        :param verbose:
        :return input_ids: list of sequence encoded with SentencePiece model
        :return pos_ids: ids of each token's POS(Part Of Speech) tag, return empty list when encode_pos is False
        '''
        self._assert_data_type_list(sentences)

        input_ids = []
        pos_ids = []
        if verbose: sentences = tqdm(sentences)
        for sentence in sentences:
            input_id, pos_id = self.encoder_preprocessor.sentence_to_ids(sentence, mask=False, encode_pos=encode_pos)
            input_ids.append(input_id)
            if encode_pos: pos_ids.append(pos_id)

        input_ids = self.trim_row_length(input_ids, over_length_row=self.over_length_row, margin=3)
        if encode_pos: pos_ids = self.trim_row_length(pos_ids, over_length_row=self.over_length_row, margin=3)
        input_ids = np.array(input_ids)
        pos_ids = np.array(pos_ids)
        return input_ids, pos_ids

    def decoder_encode(self, sentences, encode_pos, verbose=False):
        '''
        :param sentences: list of sentences to feed encoder
                            e.g.) "나는 사과를 먹는다."
        :param encode_pos: whether to feed pos tag while training bert likewise seq_ids
        :param verbose:
        :return input_ids: list of sequence encoded with SentencePiece model
        :return pos_ids: ids of each token's POS(Part Of Speech) tag, return empty list when encode_pos is False
        '''
        self._assert_data_type_list(sentences)

        input_ids = []
        pos_ids = []
        if verbose: sentences = tqdm(sentences)
        for sentence in sentences:
            input_id, pos_id = self.decoder_preprocessor.sentence_to_ids(sentence, mask=False, encode_pos=encode_pos)
            input_ids.append(input_id)
            if encode_pos: pos_ids.append(pos_id)

        input_ids = self.trim_row_length(input_ids, over_length_row=self.over_length_row, margin=3)
        if encode_pos: pos_ids = self.trim_row_length(pos_ids, over_length_row=self.over_length_row, margin=3)
        input_ids = np.array(input_ids)
        pos_ids = np.array(pos_ids)
        return input_ids, pos_ids

    def encoder_decode(self, ids):
        output = None
        if isinstance(ids, np.ndarray): ids = ids.tolist()
        if isinstance(ids[0], list):
            # list of sequence
            output = [self.encoder_preprocessor.spm_model.DecodeIds(_ids) for _ids in ids]
        else:
            # a sequence
            output = self.encoder_preprocessor.spm_model.DecodeIds(ids)
        return output

    def decoder_decode(self, ids):
        output = None
        if isinstance(ids, np.ndarray): ids = ids.tolist()
        if isinstance(ids[0], list):
            # list of sequence
            output = [self.decoder_preprocessor.spm_model.DecodeIds(_ids) for _ids in ids]
        else:
            # a sequence
            output = self.decoder_preprocessor.spm_model.DecodeIds(ids)
        return output

    def inference(self, sentences, transformer, encode_pos, beam_search=False):
        if isinstance(sentences, np.ndarray): sentences = sentences.tolist()
        self.encoder_preprocessor._assert_data_type_list(sentences)

        output = []
        if isinstance(sentences[0], list):
            if beam_search:
                for sentence in sentences:
                    output_row = self.inference_sentence_beam_search(self, sentence=sentences, transformer=transformer, encode_pos=encode_pos, beam_size=5, top_beams=3, return_all=False)
                    output.append(output_row)
            else:
                for sentence in sentences:
                    output_row = self.inference_sentence(self, sentence=sentences, transformer=transformer, encode_pos=encode_pos, post=True)
                    output.append(output_row)
        else:
            if beam_search:
                output = self.inference_sentence_beam_search(self, sentence=sentences, transformer=transformer, encode_pos=encode_pos, beam_size=5, top_beams=3, return_all=False)
            else:
                output = self.inference_sentence(self, sentence=sentences, transformer=transformer, encode_pos=encode_pos, post=True)
        return output

    def inference_sentence(self, sentence, transformer, encode_pos, post=True):
        # TODO
        output = []
        dec_bos_token_id = self.decoder_preprocessor.config["bos_token_id"]
        dec_eos_token_id = self.decoder_preprocessor.config["eos_token_id"]
        encoder_input_ids, encoder_pos_ids = self.encoder_encode([sentence], encode_pos=encode_pos)
        encoder_input_ids, encoder_pos_ids = self.pad_rows([encoder_input_ids, encoder_pos_ids], padding_value=self.encoder_preprocessor.config["pad_token_id"], post=post)
        encoder_input_ids, encoder_pos_ids = self.pad_rows([encoder_input_ids, encoder_pos_ids], padding_value=self.encoder_preprocessor.config["pad_token_id"], post=post)
        decoder_input_ids = [self.decoder_preprocessor.config["pad_token_id"]] * self.timesteps
        decoder_pos_ids = [self.decoder_preprocessor.pos_dict["pad_token"]] * self.timesteps
        # encoder_input_ids = np.expand_dims(encoder_input_ids, axis=0)
        # encoder_pos_ids = np.expand_dims(encoder_pos_ids, axis=0)

        timestep = 0
        next_timestep_token_id = dec_bos_token_id
        while timestep < self.timesteps:
            decoder_input_ids[timestep] = next_timestep_token_id
            decoder_pos_ids[timestep] = self.decoder_preprocessor.pos_dict[next_timestep_token_id]

            batch_data = dict()
            batch_data["encoder_token_ids"] = encoder_input_ids
            batch_data["decoder_token_ids"] = np.expand_dims(decoder_input_ids, axis=0)
            if encode_pos:
                batch_data["encoder_pos_ids"] = encoder_pos_ids
                batch_data["decoder_pos_ids"] = decoder_pos_ids
            prediction = transformer.predict(batch_data)
            prediction = np.squeeze(np.argmax(prediction, axis=-1))
            next_timestep_token_id = prediction[timestep]
            output.append(next_timestep_token_id)
            if next_timestep_token_id == dec_eos_token_id: break
            timestep += 1
        return output

    def inference_sentence_beam_search(self, sentence, transformer, encode_pos, beam_size=5, top_beams=3, return_all=False, post=True):
        # TODO
        output = []
        dec_bos_token_id = self.decoder_preprocessor.config["bos_token_id"]
        dec_eos_token_id = self.decoder_preprocessor.config["eos_token_id"]
        encoder_input_ids, encoder_pos_ids = self.encoder_encode([sentence], encode_pos=encode_pos)
        encoder_input_ids, encoder_pos_ids = self.pad_rows([encoder_input_ids, encoder_pos_ids], padding_value=self.encoder_preprocessor.config["pad_token_id"], post=post)
        encoder_input_ids, encoder_pos_ids = self.pad_rows([encoder_input_ids, encoder_pos_ids], padding_value=self.encoder_preprocessor.config["pad_token_id"], post=post)
        decoder_input_ids = [self.decoder_preprocessor.config["pad_token_id"]] * self.timesteps
        decoder_pos_ids = [self.decoder_preprocessor.pos_dict["pad_token"]] * self.timesteps
        # encoder_input_ids = np.expand_dims(encoder_input_ids, axis=0)
        # encoder_pos_ids = np.expand_dims(encoder_pos_ids, axis=0)

        timestep = 0
        next_timestep_token_id = dec_bos_token_id
        pass

    def trim_row_length(self, rows, over_length_row, margin):
        '''
        Trim rows longer than given timesteps
        # ignore: exclude the over_length_row
        # truncate: truncate tokens(ids) longer than timesteps
        # stop: raise AssertionError
        :param rows: list of encoded ids
        :return rows: list of trimmed encoded ids
        '''
        if over_length_row == "stop":
            error_statement = "There are rows longer than given timesteps"
            raise AssertionError(error_statement)
        elif over_length_row == "truncate":
            special_token_dict = self.preprocessor.config["special_token_dict"]
            if rows[0][-1] in special_token_dict.values():
                # row encodde including special tokens such as cls_token and sep_token
                rows = [row if len(row)<=self.timesteps else row[:self.timesteps-1]+[row[-1]] for row in rows]
            else:
                rows = [row if len(row) <= self.timesteps else row[:self.timesteps-margin] for row in rows]
        elif over_length_row == "ignore":
            rows = [row for row in rows if len(row) <= self.timesteps]
        return rows

    def pad_rows(self, rows, padding_value, post=True):
        '''
        :param rows: 2d list, list of integer arrays
        :param value: value to pad with
        :param post: whether to pad from head or tail
        :return:
        '''
        # output = keras.preprocessing.sequence.pad_sequences(ids, maxlen=self.timesteps, dtype="int32", padding="post", value=value)
        if isinstance(rows, np.ndarray): rows = rows.tolist()
        self.encoder_preprocessor._assert_data_type_list(data=rows, parameter_name="rows")
        self.encoder_preprocessor._assert_data_type_list(data=rows[0], parameter_name="row")

        output = []
        for row in rows:
            row_padded = self.encoder_preprocessor.pad_row(row=row, timesteps=self.timesteps, padding_value=padding_value, post=post)
            output.append(row_padded)
        return output

    def onehot_encoder_rows(self, rows, num_class=None):
        '''
        :param rows: 2d list, list of integer arrays
        :param num_class:
        :return:
        '''
        # output = keras.utils.to_categorical(rows, num_class)
        if num_class is None: num_class = self.encoder_preprocessor.vocab_size
        output = []
        for row in rows:
            row_onehot = self.encoder_preprocessor.onehot_row(row=row, num_class=num_class)
            output.append(row_onehot)
        return output

    def onehot_decoder_rows(self, rows, num_class=None):
        '''
        :param rows: 2d list, list of integer arrays
        :param num_class:
        :return:
        '''
        # output = keras.utils.to_categorical(rows, num_class)
        if num_class is None: num_class = self.decoder_preprocessor.vocab_size
        output = []
        for row in rows:
            row_onehot = self.decoder_preprocessor.onehot_row(row=row, num_class=num_class)
            output.append(row_onehot)
        return output

    def is_feasible_row(self, row, margin):
        '''
        Check if given row is valid according to predefined timesteps
        :param row:
        :param margin:
        :return:
        '''
        timesteps_upper = self.timesteps - margin
        flag = True
        for ids in row:
            if len(ids) > timesteps_upper or len(ids) < 1:
                flag = False
                break
        return flag