from tqdm import tqdm
import numpy as np
import keras
from .preprocessor import PreprocessorConfig, Preprocessor

class BertPreprocessor:
    timesteps = None
    preprocessor = None
    over_length_row = None
    is_next_label = 1
    not_next_label = 0

    def __init__(self, timesteps, over_length_row="truncate", konlpy_mecab=True, config=PreprocessorConfig):
        '''
        :param timesteps: int, max sequence length
        :param over_length_row: string, how to cope with exception when sequence length is over than given timesteps
                                (available: ["ignore", "truncate", "stop"])
        :param konlpy_mecab: True if konlpy.Mecab(both available on Windows & Linux) has been installed
                              False if MeCab(usually available on Linux, not on Windows) has been installed
        :param config:
        '''
        self.timesteps = timesteps
        self.over_length_row = over_length_row
        self.preprocessor = Preprocessor(config=config, konlpy_mecab=konlpy_mecab)

    def encode(self, sequences, mask, encode_pos, verbose=False):
        '''
        :param sequences: list of sequence which consists of sentences concatenated with "\n"
                            e.g.) "I ate an apple.\nbut, it was not juicy."
        :param mask: whether to mask or not, masking is only recommended when encoding train data
        :param encode_pos: whether to feed pos tag while training bert likewise seq_ids
        :param verbose:
        :return input_ids: list of sequence encoded with SentencePiece model
        :return seq_ids: ids indicating sentence's order
        :return pos_ids: ids of each token's POS(Part Of Speech) tag, return empty list when encode_pos is False
        '''
        self.preprocessor._assert_data_type_list(data=sequences, parameter_name="sequences")

        input_ids = []
        seq_ids = []
        pos_ids = []

        sequences_iter = sequences
        if verbose: sequences_iter = tqdm(sequences)
        for sequence in sequences_iter:
            input_id, seq_id, pos_id = self.sequence_to_ids(sequence, mask=mask, encode_pos=encode_pos)
            input_ids.append(input_id)
            seq_ids.append(seq_id)
            if encode_pos: pos_ids.append(pos_id)

        input_ids = self.trim_row_length(input_ids, over_length_row=self.over_length_row, margin=3)
        seq_ids = self.trim_row_length(seq_ids, over_length_row=self.over_length_row, margin=3)
        if encode_pos: pos_ids = self.trim_row_length(pos_ids, over_length_row=self.over_length_row, margin=3)

        input_ids = np.array(input_ids)
        seq_ids = np.array(seq_ids)
        pos_ids = np.array(pos_ids)
        return input_ids, seq_ids, pos_ids

    def decode(self, ids):
        output = None
        if isinstance(ids, np.ndarray): ids = ids.tolist()
        if isinstance(ids[0], list):
            # list of sequence
            output = [self.preprocessor.spm_model.DecodeIds(_ids) for _ids in ids]
        else:
            # a sequence
            output = self.preprocessor.spm_model.DecodeIds(ids)
        return output

    def sequence_to_ids(self, sequence, mask, encode_pos):
        '''
        :param sequence: sequence, list of sentences, e.g.) ["I ate an apple.", "but, It was not juicy."]
        :param mask: whether to mask or not, masking is only recommended when encoding train data
        :param encode_pos: whether to feed pos tag while training bert likewise seq_ids
        :return:
        '''
        sequence_ids = []
        seq_ids = []
        pos_ids = []

        sentence_end_idx = 0
        sequence_ids.append(self.preprocessor.config["cls_token_id"])
        pos_ids.append(self.preprocessor.pos_dict["cls_token"])
        for sentence_idx, sentence in enumerate(sequence):
            sentence_idx += 1 # begin with 1

            sentence_ids, sentence_pos_ids = self.preprocessor.sentence_to_ids(sentence, mask=mask, encode_pos=encode_pos)
            sequence_ids += sentence_ids
            sequence_ids.append(self.preprocessor.config["sep_token_id"])
            sentence_seq_ids = [sentence_idx] * (len(sequence_ids) - sentence_end_idx)
            seq_ids += sentence_seq_ids
            sentence_end_idx = len(sequence_ids)

            if encode_pos:
                pos_ids += sentence_pos_ids
                pos_ids.append(self.preprocessor.pos_dict["sep_token"])

        return sequence_ids, seq_ids, pos_ids

    def extract_sequence(self, path, documents, min_seq_len, append=False, shuffle=True, delimiter="\t", encoding="UTF-8"):
        '''
        extract sequences consist of sequential sentences and unrelated sentences from document.
        :param path: path to save output txt
        :param documents: list of document (document: list of sentence)
        :param min_seq_len: minimum tokens length of each sentence to use
        :param append: whether to allow appending to existing file if there is a file with given path
        :param shuffle: whether to shuffle extracted sequences
        :param delimiter: delimeter to join sequence (two sentences)
        :return output: boolean
        '''
        positive_sequences = []
        negative_sequences = []

        for cur_document_idx, document in tqdm(enumerate(documents)):
            for sentence_idx in range(0, len(document)-1):
                # positive samples
                cur_sentence = document[sentence_idx]
                next_sentence = document[sentence_idx+1]
                positive_sequence = [cur_sentence, next_sentence]
                if len(cur_sentence.split()) <= min_seq_len or len(next_sentence.split()) <= min_seq_len: continue
                if not self.is_feasible_sequence(positive_sequence): continue
                positive_sequence = delimiter.join(positive_sequence)
                positive_sequences.append(positive_sequence)

                # negative samples
                while True:
                    next_document_idx = np.random.randint(0, len(documents))
                    if cur_document_idx == next_document_idx: continue
                    next_document = documents[next_document_idx]
                    next_document_sentence = np.random.choice(next_document)
                    negative_sequence = [cur_sentence, next_document_sentence]
                    if len(cur_sentence.split()) <= min_seq_len or len(next_document_sentence.split()) <= min_seq_len: continue
                    if not self.is_feasible_sequence(negative_sequence): continue
                    negative_sequence = delimiter.join(negative_sequence)
                    negative_sequences.append(negative_sequence)

        positive_nsp_ids = [self.is_next_label] * len(positive_sequences)
        negative_nsp_ids = [self.not_next_label] * len(negative_sequences)
        total_sequences = positive_sequences + negative_sequences
        total_nsp_ids = positive_nsp_ids + negative_nsp_ids

        if shuffle:
            random_indice = np.arange(0, total_sequences)
            np.random.shuffle(random_indice)
            total_sequences = np.array(total_sequences)
            total_sequences = total_sequences[random_indice]
            total_sequences = total_sequences.tolist()
            total_nsp_ids = np.array(total_nsp_ids)
            total_nsp_ids = total_nsp_ids[random_indice]
            total_nsp_ids = total_nsp_ids.tolist()

        if append: fp = open(path, "a", encoding=encoding)
        else: fp = open(path, "w", encoding=encoding)

        row_template = "{sequence}{delimiter}{nsp_id}\n"
        for sequence, nsp_id in zip(total_sequences, total_nsp_ids):
            row = row_template.format(sequence=sequence, delimiter=delimiter, nsp_id=str(nsp_id))
            fp.write(row)
        fp.close()

        print("total sequences size:", len(total_sequences))
        print("positive_sequences size: {pos_len}\tnegative_sequences size:{neg_len}".format(pos_len=len(positive_sequences), neg_len=len(negative_sequences)))
        return True

    def is_feasible_sequence(self, sequence, margin):
        '''
        Check wheter given sequence is valid according to predefined parameters
        :param sequence: list of encoded ids
        :param margin: margin for special tokens (one cls_token, two sep_tokens)
        :return output: boolean, return True if given sequence is valid
        '''
        delimiter = " "
        timesteps_upper = self.timesteps - margin
        row = delimiter.join(sequence)
        _row = self.preprocessor.spm_model.EncodeAsIds(row)
        if len(_row) <= timesteps_upper: return True
        else: return False

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

    def pad_rows(self, rows, padding_value=None, post=True):
        '''
        :param rows: 2d list, list of integer arrays
        :param value: value to pad with
        :param post: whether to pad from head or tail
        :return:
        '''
        # output = keras.preprocessing.sequence.pad_sequences(ids, maxlen=self.timesteps, dtype="int32", padding="post", value=value)
        if padding_value is None: padding_value = int(self.preprocessor.config["pad_token_id"])
        if isinstance(rows, np.ndarray): rows = rows.tolist()
        self.preprocessor._assert_data_type_list(data=rows, parameter_name="rows")
        self.preprocessor._assert_data_type_list(data=rows[0], parameter_name="row")

        output = []
        for row in rows:
            row_padded = self.preprocessor.pad_row(row, timesteps=self.timesteps, padding_value=padding_value, post=post)
            output.append(row_padded)
        return output

    def onehot_rows(self, rows, num_class=None):
        '''
        :param rows: 2d list, list of integer arrays
        :param num_class:
        :return:
        '''
        # output = keras.utils.to_categorical(rows, num_class)
        if num_class is None: num_class = self.preprocessor.config["vocab_size"]

        output = []
        for row in rows:
            row_onehot = self.preprocessor.onehot_row(row=row, num_class=num_class)
            output.append(row_onehot)
        return output