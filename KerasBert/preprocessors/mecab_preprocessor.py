import json

AdvancedMeCabConfig = {
    "interjection_normalize_pos": ["IC"],
    "arabia_normalize_pos": ["NR", "MM"],
    "chinese_normalize_pos": ["NNG", "XPN", "XR", "XSN"],
    "keyword_pos": ["NNP", "NNBC", "NNG", "NNB" , "VCN", "VA", "VV", "VX", "VCP", "SL"],
    "a2h":{
        "1": ["하나", "한", "일", "一"],
        "2": ["둘", "두", "이", "二"],
        "3": ["셋", "세", "서", "삼", "三"],
        "4": ["넷", "네", "너", "사", "四"],
        "5": ["다섯", "다서", "대", "댓", "오", "五"],
        "6": ["여섯", "여서", "예", "육", "륙", "六"],
        "7": ["일곱", "일고", "닐곱", "칠", "七"],
        "8": ["여덟", "여더", "여덜", "팔", "八"],
        "9": ["아홉", "구", "九"],
        "10": ["열", "십", "十"],
        "20": ["스물", "이십"],
        "30": ["서른", "삼십"],
        "40": ["마흔", "사십"],
        "50": ["쉰", "오십"],
        "60": ["예순", "육십"],
        "70": ["일흔", "칠십"],
        "80": ["여든", "팔십"],
        "90": ["아흔", "구십"],
        "100": ["백", "百"],
        "1000": ["천", "千"],
        "10000": ["만", "萬"],
        "100000000": ["억", "億"],
        "1000000000000": ["조", "兆"]
    },
    "decimal_arabia": [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000]
}

class AdvancedMeCab:
    tagger = None
    konlpy_mecab = None
    interjection_normalize_pos = None
    arabia_normalize_pos = None
    chinese_normalize_pos = None
    keyword_pos = None
    a2h = None
    h2a = None
    decimal_arabia = None

    def __init__(self, config_path=None, konlpy_mecab=False):
        self.konlpy_mecab = konlpy_mecab
        if self.konlpy_mecab:
            from konlpy.tag import Mecab
            self.tagger = Mecab().tagger
        else:
            import MeCab
            self.tagger = MeCab.Tagger()
        self._load_config(config_path=config_path)

    def _load_config(self, config_path):
        json_data = None
        if config_path is not None:
            with open(config_path, encoding="UTF-8") as json_file:
                json_data = json.load(json_file)
        else:
            json_data = AdvancedMeCabConfig

        self.interjection_normalize_pos = json_data["interjection_normalize_pos"]
        self.arabia_normalize_pos = json_data["arabia_normalize_pos"]
        self.chinese_normalize_pos = json_data["chinese_normalize_pos"]
        self.keyword_pos = json_data["keyword_pos"]
        self.a2h = {int(a):h for a,h in json_data["a2h"].items()}
        self.h2a = {h:a for a,h_list in json_data["a2h"] for h in h_list}
        self.decimal_arabia = json_data["decimal_arabia"]

    def pos(self, token, interjection_normalize=False, arabia_normalize=False, chinese_normalize=False):
        parsed_token = self._parse_simplify(token, chinese_normalize=chinese_normalize)

        # interjection_normalize
        if interjection_normalize: parsed_token = self._interjection_normalize(parsed_token)
        # arabia_normalize
        if arabia_normalize: parsed_token = self._arabia_normalize(parsed_token)
        return parsed_token

    def extract_keywords(self, token):
        parsed_token = self._parse_simplify(token, chinese_normalize=False)
        keywords = [word for word,pos in parsed_token if pos in self.keyword_pos]
        return keywords

    def _parse_simplify(self, token, chinese_normalize=False):
        '''
        불필요한 parsing 정보 제외하고 필요한 정보만 추출
        e.g) '회사\tNNG,*,F,회사,*,*,*,*' => ('회사', 'NNG')
        :param token:
        :param chinese_normalize:
        :return:
        '''
        parsed_token = []
        parse_result = self.tagger.parse(token).split("\n")[:-2]
        for token_row in parse_result:
            word, _pos = token_row.split("\t")[0:2]
            pos_splited = _pos.split(",")
            pos = pos_splited[0]

            if chinese_normalize and pos_splited[0] in self.chinese_normalize_pos:
                # 한자어일 경우, 한글 득음으로 정규화
                word = pos_splited[3]
            row = (word, pos)
            parsed_token.append(row)
        return parsed_token

    def _interjection_normalize(self, parsed_token):
        idx = 0
        while True:
            if idx >= len(parsed_token) - 1: break
            cur_token_part = parsed_token.pop(idx)
            next_token_part = parsed_token.pop(idx)
            if cur_token_part[1] == next_token_part[1] and cur_token_part[1] in self.interjection_normalize_pos:
                if set(cur_token_part[0]) <= set(next_token_part[0]):
                    # "아아아아아" 와 "아아아" 인 경우
                    parsed_token.insert(idx, cur_token_part)
                else:
                    parsed_token.insert(idx, next_token_part)
                continue
            else:
                # "하하하하" 와 "하하하하핫" 인 경우
                if set(cur_token_part[0]).intersection(set(next_token_part[0])) == set(next_token_part[0]):
                    parsed_token.insert(idx, cur_token_part)
                    continue
                if set(cur_token_part[0]).intersection(set(next_token_part[0])) == set(cur_token_part[0]):
                    parsed_token.insert(idx, next_token_part)
                    continue

            parsed_token.insert(idx, next_token_part)
            parsed_token.insert(idx, cur_token_part)
            idx += 1
        return parsed_token

    def _arabia_normalize(self, parsed_token):
        '''
        서수, 기수, 한자형 모두 아라비아 숫자로 정규화
        :param parsed_token:
        :return:
        '''
        _result = []
        for idx, (word,pos) in enumerate(parsed_token):
            if pos in self.arabia_normalize_pos:
                _, arabia_eujeol, hangul_eujeol = self._arabia_split(word, [], [])
                word_list = []
                for eujeol in arabia_eujeol:
                    idx = word.index(eujeol)
                    word_list.append((idx, (eujeol, "NR")))
                for eujeol in hangul_eujeol:
                    idx = word.index(eujeol)
                    word_list.append((idx, (eujeol, "MM")))

                word_list.sort(key=lambda x: x[0])
                for element in word_list:
                    _result.append(element[1])
            else:
                _result.append((word, pos))
        parsed_token = _result

        arabia = []
        _result = []
        for idx, (word, pos) in enumerate(parsed_token):
            if pos not in self.arabia_normalize_pos or word not in self.h2a.keys():
                if len(arabia) > 0:
                    _result.append((str(sum(arabia)), "SN")) # "SN": 숫자
                    arabia = []
                row = (word, pos)
                _result.append(row)
                continue

            arabia_token = self.h2a[word]
            if len(arabia) > 0:
                if arabia_token in self.decimal_arabia:
                    prev_arabia_token = arabia.pop()
                    arabia.append(prev_arabia_token * arabia_token)
                    arabia = [sum(arabia)]
                else:
                    if arabia[-1] < 10:
                        _result.append((str(arabia[0]), "SN")) # "SN": 숫자
                        arabia = []
                    arabia.append(arabia_token)
            else:
                arabia.append(arabia_token)

            if len(arabia) > 0 and (idx == len(parsed_token) - 1):
                _result.append((str(sum(arabia)), "SN")) # "SN": 숫자
                arabia = []
        return _result


    def _arabia_split(self, token, split_list, unsplit_list):
        '''
        2개 이상의 숫자가 묶여있는 문자열을 나누는 메소드
        :param token:
        :param split_list:
        :param unsplit_list:
        :return:
        '''
        split_list_init, unsplit_list_init = split_list.copy(), unsplit_list.copy()
        for window_size in range(1, len(token)+1):
            split_list, unsplit_list = split_list_init.copy(), unsplit_list_init.copy()
            left = token[:window_size]
            right = token[window_size:]
            if left in self.h2a: split_list.append(left)
            else: unsplit_list.append(left)

            if len(right) < 1:
                return right, split_list, unsplit_list
            else:
                right, split_list, unsplit_list = self._arabia_split(right, split_list, unsplit_list)
                if len(split_list) > len(split_list_init): break
        return right, split_list, unsplit_list


if __name__ == "__main__":
    mecab = AdvancedMeCab()

    print("\n########### arabia normalize test ###########")
    test_sentences = [
        "351번째 문제와 삼천오백구번 문제는 해결했다.",
        "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
        "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
        "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
    ]

    for test_sentence in test_sentences:
        _test_result = mecab.pos(test_sentence, arabia_normalize=True, chinese_normalize=True)
        test_result = " ".join([word for word,pos in _test_result])
        print(test_sentence)
        print(test_result)
        print()

    print("\n########### chinese normalize test ###########")
    test_sentences = [
        "현상 관측 결과 이상 無",
        "利得을 본 경기였다.",
        "상태 체크 보고, 동작 不"
    ]

    for test_sentence in test_sentences:
        _test_result = mecab.pos(test_sentence, arabia_normalize=True, chinese_normalize=True)
        test_result = " ".join([word for word, pos in _test_result])
        print(test_sentence)
        print(test_result)
        print()

    print("\n########### interjection normalize test ###########")
    test_sentences = [
        "ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이렇게 많이 웃는, 경우에도 작동합니까?ㅋㅋㅋㅋㅋㅋ",
        "하하하하하하하핫 이게 잘 작동할까아아아아아아요?"
    ]

    for test_sentence in test_sentences:
        _test_result = mecab.pos(test_sentence, arabia_normalize=True, chinese_normalize=True)
        test_result = " ".join([word for word, pos in _test_result])
        print(test_sentence)
        print(test_result)
        print()