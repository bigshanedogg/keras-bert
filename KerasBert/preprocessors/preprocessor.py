import re
import os
import pickle
from datetime import date
from tqdm import tqdm
import shutil
import nltk
import numpy as np
import sentencepiece as spm
try:
    from .mecab_preprocessor import AdvancedMeCab
except ImportError as e:
    print("AdvancedMeCab ImportError:", e)

PreprocessorConfig = {
    # fixed values
    "nltk_prefix": "nltk_{tag}",
    "mecab_prefix": "mecab_{tag}",
    "mecab_eng_tag": "SL",
    "default_remain_regex": "[^ㄱ-ㅣ가-힣a-zA-Z0-9\u4E00-\u9FA5 #.:;@%&/-`\'\"]",
    "not_eng_regex": "[ㄱ-ㅣ가-힣]",
    "white_space_regex": "\s+",
    "mecab_tag_delimiter": "+",

    # special tokens
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "cls_token": "<cls>",
    "sep_token": "<sep>",
    "turn_token": "<turn>",
    "mask_token": "<mask>",
    "num_token": "<num>",
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
    "cls_token_id": 4,
    "sep_token_id": 5,
    "turn_token_id": 6,
    "mask_token_id": 7,
    "num_token_id": 8,

    # special_token_dict
    "special_token_dict": {
        "pad_token": 0,
        "bos_token": 1,
        "eos_token": 2,
        "unk_token": 3,
        "cls_token": 4,
        "sep_token": 5,
        "turn_token": 6,
        "mask_token": 7,
        "num_token": 8
    },

    # mecaab_preprocessor parameter
    "remain_regex": "[^ㄱ-ㅣ가-힣a-zA-Z0-9\u4E00-\u9FA5 .,:;`!@#$%%\^&*\?~\(\)\{\}\[\]\_\-\+\=\"\']+",
    "bracket_regex": "\s?[\(\{\<\[][\s+,-_~]*[\]\>\}\)]\s?", # " (Peanut Farmer)" or "{element equation}"
    "bracket_replace": True,
    "interjection_normalize": True,
    "chinese_normalize": True,
    "arabia_normalize": False,

    # hyper parameter
    "mask_lm_ratio": 0.15,
    "random_mask_ratio": 0.1,
    "skip_mask_ratio": 0.1,
    "keep_mask_ratio": 0.8, # 1- random_mask_ratio - skip_mask_ratio
}

class Preprocessor:
    config = None
    vocab_size = None
    spm_model_path = None
    spm_model = None
    spm_tokens = None
    tokens = None
    tokens_dict = None
    pos_dict = None
    spm_model_type = None
    spm_model_prefix = "sentence_piece"
    default_spm_input_path = "./spm_input_file.txt"
    mecab = None
    is_advanced_mecab = False

    def __init__(self, konlpy_mecab, config=None):
        if config is None: self.config = PreprocessorConfig
        else: self.config = config

        try:
            self.mecab = AdvancedMeCab(konlpy_mecab=konlpy_mecab)
            is_advanced_mecab = True
        except:
            mecab_load_error_statemtn = "Cannot Import Mecab library."
            assert konlpy_mecab, mecab_load_error_statemtn
            from konlpy.tag import Mecab
            self.mecab =  Mecab()

    def tokenizer(self, sentence):
        # basic preprocessor
        # remain allowed characters only
        remain_regex = self.config["remain_regex"]
        if remain_regex is None: remain_regex = self.config["default_remain_regex"]
        remain_regex_pattern = re.compile(remain_regex)
        sentence = remain_regex_pattern.sub(" ", sentence)

        # replace braket to space
        if self.config["bracket_replace"]:
            bracket_regex = self.config["bracket_regex"]
            bracket_regex_pattern = re.compile(bracket_regex)
            sentence = bracket_regex_pattern.sub(" ", sentence)
        # white space preprocessor
        white_space_regex_pattern = re.compile(self.config["white_space_regex"])
        sentence = white_space_regex_pattern.sub(" ", sentence)

        sentence_token = None
        is_eng_sentence = re.search(self.config["not_eng_regex"], sentence) is None
        if is_eng_sentence:
            sentence_token = nltk.pos_tag(sentence.split())
            output = []
            for token, tag in sentence_token:
                output.append((token, self.config["nltk_prefix"].format(tag=tag)))
        else:
            if self.is_advanced_mecab: sentence_token = self.mecab.pos(sentence, interjection_normalize=self.config["interjection_normalize"], arabia_normalize=self.config["arabia_normalize"], chinese_normalize=self.config["chinese_normalize"])
            else: sentence_token = self.mecab_pos(sentence)
            output = []
            for token, tag in sentence_token:
                tag = self._convert_mecab_tag_to_pos(tag)
                if tag == "SL":
                    _, tag = nltk.pos(tag([token]))[0]
                    output.append(token, self.config["nltk_prefix"].format(tag=tag))
                else:
                    output.append(token, self.config["mecab_prefix"].format(tag=tag))

        return output

    def _get_mask_indice(self, tokens):
        mask_sample_num = int(len(tokens) * self.config["mask_lm_ratio"])
        mask_token_indice = np.random.choice(list(range(0, len(tokens))), mask_sample_num, replace=False)
        mask_token_indice.sort()
        mask_token_indice = mask_token_indice.tolist()
        return mask_token_indice

    def _get_mask_token(self, token_ids, tag_id):
        random_prob = np.random.rand()
        if len(token_ids) not in self.tokens_dict or random_prob < self.config["keep_mask_ratio"]:
            # keep mask
            token_ids = [self.config["mask_token_id"]] * len(token_ids)
            if tag_id is not None:
                tag_id = self.pos_dict["mask_token"]
        elif random_prob < self.config["keep_mask_ratio"] + self.config["random_mask_ratio"]:
            # random mask
            random_tokens = self.tokens_dict[len(token_ids)]
            random_token_idx = np.random.choice(list(range(0, len(random_tokens))))
            random_token, random_tag = random_tokens[random_token_idx]
            token_ids = self.spm_model.EncodeAsIds(random_token)
            if tag_id is not None:
                pos_list = random_tag.split(self.config["mecab_tag_delimiter"])
                for pos in pos_list:
                    self.update_pos_dict(pos)
                random_tag = pos_list[0]
                tag_id = self.pos_dict[random_tag]
        else:
            # skip mask
            pass
        return token_ids, tag_id

    def _convert_mecab_tag_to_pos(self, tag):
        pos_list = tag.split(self.config["mecab_tag_delimiter"])
        output = pos_list[0]
        return output

    def sentence_to_ids(self, sentence, mask, encode_pos):
        '''
        :param sentence: str, sentence  e.g.) "I ate an apple."
        :param mask: whether to mask or not, masking is only recommended when encoding train data
        :param encode_pos: whether to feed pos tag while training bert likewise seq_ids
        :return sentence_ids: encoded sentence with SentencePiece model
        '''
        sentence = self.tokenizer(sentence) # [(token_1, tag_1), (token_2, tag_2), ...]
        sentence_ids = []
        sentence_tag_ids = []

        sentence_tokens = []  # [token_1, token_2, ...]
        sentence_tags = []  # [tag_1, tag_2, ...]
        for token, tag in sentence:
            sentence_tokens.append(token)
            sentence_tags.append(tag)

        if encode_pos:
            if mask:
                # encode_pos:True, mask:True
                mask_token_indice = self._get_mask_indice(sentence_tokens)
                mask_iter_idx = 0
                for token_idx, (token, tag) in enumerate(zip(sentence_tokens, sentence_tags)):
                    pos_list = tag.split(self.config["mecab_tag_delimiter"])
                    for pos in pos_list:
                        self.update_pos_dict(pos)
                    tag = pos_list[0]
                    token_ids = self.spm_model.EncodeAsIds(token)
                    tag_id = self.pos_dict[tag]
                    if mask_iter_idx < len(mask_token_indice) and token_idx == mask_token_indice[mask_iter_idx]:
                        token_ids, tag_id = self._get_mask_token(token_ids=token_ids, tag_id=tag_id)
                        mask_iter_idx += 1
                    tag_ids = len(token_ids) * [tag_id]
                    sentence_ids += token_ids
                    sentence_tag_ids += tag_ids
            else:
                # encode_pos:True, mask:False
                for token_idx, (token, tag) in enumerate(zip(sentence_tokens, sentence_tags)):
                    pos_list = tag.split(self.config["mecab_tag_delimiter"])
                    for pos in pos_list:
                        self.update_pos_dict(pos)
                    tag = pos_list[0]
                    tag_id = self.pos_dict[tag]
                    token_ids = self.spm_model.EncodeAsIds(token)
                    tag_ids = len(token_ids) * [tag_id]
                    sentence_ids += token_ids
                    sentence_tag_ids += tag_ids
        else:
            if mask:
                # encode_pos:False, mask:True
                mask_token_indice = self._get_mask_indice(sentence_tokens)
                mask_iter_idx = 0
                for token_idx, (token, tag) in enumerate(zip(sentence_tokens, sentence_tags)):
                    token_ids = self.spm_model.EncodeAsIds(token)
                    if mask_iter_idx < len(mask_token_indice) and token_idx == mask_token_indice[mask_iter_idx]:
                        token_ids, _ = self._get_mask_token(token_ids=token_ids, tag_id=None)
                        mask_iter_idx += 1
                    sentence_ids += token_ids
            else:
                # encode_pos:False, mask:False
                sentence = " ".join(sentence_tokens)
                sentence_ids = self.spm_model.EncodeAsIds(sentence)
                sentence_ids = list(sentence_ids)

        return sentence_ids, sentence_tag_ids

    def update_pos_dict(self, pos):
        if pos not in self.pos_dict:
            self.pos_dict[pos] = len(self.pos_dict)

    def make_default_eng_tokens_dict(self):
        '''
        extract list of tokens, dictionary for pos2idx encoding,
        and dictionary for replace token with another token with equal length of subword while random masking
        from nltk.wordnet corpus
        :return:
        '''
        self._assert_spm_model_load(self.spm_model)

        nltk_wordest = set()
        try:
            nltk_wordest = nltk.corpus.wordnet.words()
        except LookupError:
            nltk.download("wordnet")
            nltk_wordest = nltk.corpus.wordnet.words()
        self._assert_nltk_download(nltk_wordest)

        tokens = list(set([word for word in nltk_wordest]))
        pos_dict = self.config["special_token_dict"].copy()
        tokens_dict = dict()
        for token in tqdm(tokens):
            token, tag = nltk.pos_tag([token])[0]
            if tag not in pos_dict: pos_dict[tag] = len(pos_dict)
            spm_token_list = self.spm_model.EncodeAsPiece(token)
            if len(spm_token_list) not in tokens_dict: tokens_dict[len(spm_token_list)] = []
            tokens_dict[len(spm_token_list)].append((token, tag))

        self.tokens = tokens
        self.tokens_dict = tokens_dict
        self.pos_dict = pos_dict

    def make_custom_tokens_dict(self, sentences):
        '''
        extract list of tokens, dictionary for pos2idx encoding,
        and dictionary for replace token with another token with equal length of subword while random masking
        from nltk.wordnet corpus
        :return:
        '''
        self._assert_spm_model_load(self.spm_model)

        tokens = []
        pos_dict = self.self.config["special_token_dict"].copy()
        tokens_dict = dict()
        for sentence in tqdm(sentences):
            for token, tag in self.tokenizer(sentence):
                if token in tokens: continue
                if tag not in pos_dict: pos_dict[tag] = len(pos_dict)
                tokens.append(token)
                spm_token_list = self.spm_model.EncodeAsIds(token)
                if len(spm_token_list) not in tokens_dict: tokens_dict[len(spm_token_list)] = []
                tokens_dict[len(spm_token_list)].append((token, tag))

        self.tokens = tokens
        self.tokens_dict = tokens_dict
        self.pos_dict = pos_dict

    def load_tokens_dict(self, path):
        with open(path, "rb") as fp:
            self.tokens, self.tokens_dict, self.pos_dict = pickle.load(fp)

    def save_tokens_dict(self, path):
        with open(path, "wb") as fp:
            pickle.dump([self.tokens, self.tokens_dict, self.pos_dict])

    def train_spm_model(self, sentences, vocab_size, spm_model_path=None, spm_model_type="bpe"):
        self.config["vocab_size"] = vocab_size
        if spm_model_path is None:
            today_date = date.today().strftime("%Y%m%d")[2:]
            spm_model_path = "./spm_v{vocab_size}_{today}/".format(vocab_size=vocab_size, today=today_date)
            print("default spm_model_path: '{path}'".format(path=spm_model_path))

        if not os.path.isdir(spm_model_path): os.mkdir(spm_model_path)

        with open(self.default_spm_input_path, "w", encoding="UTF-8") as fp:
            for sentence in sentences:
                fp.write(sentence + "\n")

        spm_cmd = self._get_spm_cmd(spm_model_path=spm_model_path, spm_model_type=spm_model_type, vocab_size=vocab_size)

        # train
        spm.SentencePieceTrainer.Train(spm_cmd)
        os.remove(self.default_spm_input_path)

        # load trained model
        spm_model_file = spm_model_path + self.spm_model_prefix + ".model"
        spm_vocab_file = spm_model_path + self.spm_model_prefix + ".vocab"
        spm_model = spm.SentencePiecePreocessor()
        spm_model.Load(spm_model_file)

        spm_tokens = []
        with open(spm_vocab_file, "r") as fp:
            for row in fp:
                token, idx = row.strip().split("\t")
                spm_tokens.append(token)

        self.vocab_size = self.spm_model.GetPieceSize()
        self.spm_model = spm_model
        self.spm_tokens = spm_tokens
        self.spm_model_path = spm_model_path
        self.spm_model_type = spm_model_type

    def _get_spm_cmd(self, spm_model_path, spm_model_type, vocab_size):
        # setting training command
        # e.g.) '--input=./spm_input_file.txt --model_prefix=test --vocab_size=10000 --model_type=bpe'
        spm_cmd_template = "--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}"
        spm_cmd = spm_cmd_template.format(input_path=self.default_spm_input_path, model_prefix=spm_model_path + self.spm_model_prefix, vocab_size=vocab_size, model_type=spm_model_type)

        # e.g.) '--bos_id=1 --bos_piece=<s> --unkid=3 --unk_piece=<unk> --eos_id=2 --eos_piece=</s> --pad_id=0 --pad_piece=<pad>'
        token_append_template = " --{token_type}_id={token_id} --{token_type}_piece={token}"
        token_append_cmd = \
            token_append_template.format(token_type="pad", token_id=self.config["pad_token_id"], token=self.config["pad_token"]) + \
            token_append_template.format(token_type="bos", token_id=self.config["bos_token_id"], token=self.config["bos_token"]) + \
            token_append_template.format(token_type="eos", token_id=self.config["eos_token_id"], token=self.config["eos_token"]) + \
            token_append_template.format(token_type="unk", token_id=self.config["unk_token_id"], token=self.config["unk_token"])

        # e.g.) ' --user_defined_symbols=<cls>,<sep>,<turn>,<mask>,<num>'
        uds_append_template = " --user_defined_symbols={user_defiend_symbols}"
        user_defined_symbols = ",".join([self.config["cls_token"], self.config["sep_token"], self.config["turn_token"], self.config["mask_token"], self.config["num_token"]])
        uds_cmd = uds_append_template.format(user_defined_symbols=user_defined_symbols)

        spm_cmd = spm_cmd + token_append_cmd + uds_cmd
        return spm_cmd

    def load_spm_model(self, path):
        if not path.endswith("/"): path = path + "/"
        self.spm_model_path = path
        spm_model_file = self.spm_model_path + self.spm_model_prefix + ".model"
        spm_vocab_file = self.spm_model_path + self.spm_model_prefix + ".vocab"

        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.load(spm_model_file)
        self.vocab_size = self.spm_model.GetPieceSize()

        spm_tokens = []
        with open(spm_vocab_file, "r") as fp:
            for row in fp:
                token, idx = row.strip().split("\t")
                spm_tokens.append(token)
        self.spm_tokens = spm_tokens
        print("loaded spm_model: '{path}'".format(path=path))

    def save_spm_model(self, path, copy=False):
        if os.path.isdir(path): shutil.rmtree(path)
        shutil.copytree(self.spm_model_path, path)
        if not copy:
            shutil.rmtree(self.spm_model_path)
            print("saved spm_model: '{path}'".format(path=path))
        else:
            print("copied spm_model: '{path}'".format(path=path))
        self.spm_model_path = path

    def pad_row(self, row, timesteps, padding_value, post):
        output = None
        padding_len = (timesteps - len(row))
        padding_vector = [padding_value] * padding_len
        if post: output = row + padding_vector
        else: output = padding_vector + row
        return output

    def onehot_row(self, row, num_class):
        self._assert_data_type_list(data=row, parameter_name="row")
        output = np.zeros((len(row), num_class))
        output = output.tolist()
        for row_idx, idx in enumerate(row):
            output[row_idx][idx] = 1.0
        return output

    def _assert_mecab_load(self):
        mecab_load_error_statemtn = "Cannot Import Mecab library."
        assert konlpy_mecab, mecab_load_error_statemtn

    def _assert_spm_model_load(self, spm_model):
        spm_model_error_statement = "SentencePiece model (spm_model) has not been loaded."
        assert spm_model is not None, spm_model_error_statement

    def _assert_nltk_download(self, nltk_wordest):
        nltk_download_error_statement = "Check if following command works:\n>> import nltk\n>> nltk.download('wordnet')"
        assert len(nltk_wordest) > 0, nltk_download_error_statement

    def _assert_data_type_list(self, data, parameter_name):
        error_statement = "The data type of parameter '{parameter}' must be list.".format(parameter=parameter_name)
        assert isinstance(data, list), error_statement

    def _assert_length_match(self, rows_1, rows_2):
        error_statement = "The length of given inputs must be equal: {len1} vs {len2}".format(len1=len(rows_1), len2=len(rows_2))
        assert len(rows_1) == len(rows_2), error_statement