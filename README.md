# keras-bert
Transformer and BERT implemented in Keras

Hierarchical BERT: Contextualized Classifier for Manufacturing Text



Abstract



1. Introduction

엔지니어링 로그에서 알아내고자 하는 목적 중 가장 본질적인 것은 이후의 고장 예측과 설비 보전에 대한 인사이트를 추출하고, 자동화하는 것이다. 

가장 우선적인 태스크는 엔지니어링 로그에서 현상/원인/조치에 해당하는 문장을 찾고 이들간의 연관 관계를 규명해, 경고장은 자동 보전, 중고장에 필요한 인력의 판단 소요를 줄임으로서 결과적으로 고장률과 인건비 감소를 목표로 한다.

다만, 몇십년간 쌓아온 데이터는 모델 학습에 사용되기에 충분한 양의 비정형 데이터이지만, 데이터 자체의 몇가지 어려움이 있다.

첫째로, 대부분의 산업계 데이터와 마찬가지로 엔지니어링 로그의 경우 정돈이 잘되지 않은 경우가 대부분이다. 라인별, 공정별 공통화된 템플릿의 부재와 긴박한 상황, 다양한 구성원이 작성하는 문서이기 때문에 오탈자와 축약어의 비중이 높다. //TODO

둘째로, 충분한 양의 비정형 데이터를 가지고 있음에도 레이블링된 충분한 데이터를 확보하는 것은 어려운 일이다.

주어진 현상에 대해 조치방법은 다양하게 존재할 수 있고, 현상은 1가지 원인으로 인해 다양한 차원에서 발생할 수 있기 때문에 또한 여러가지일 수 있다. 

게다가 원인은 연속적인 현상들의 안쪽을 파고들어서 본질적인 고장원인을 파악해야하기 때문에 높은 경력의 엔지니어가 레이블링 작업에 참여해야 한다.

셋째로, 엔지니어링 로그는 일반적인 언어와 일정 부분 문법적 구조를 공유하지만, 일반적인 general text의 단어와 오버랩이 많이 다르다.

추가적으로, general 도메인과 다르게 manufacturing data의 경우, 현상/원인/조치가 상황마다 다르다. 

같은 단어 구성의 문장임에도 불구하고, 앞선 현상이나 원인의 조합에 따라 문단 내에서 조치로 분류하면 안되는 경우가 있기 때문에 substantial한 데이터의 확보는 중요한 이슈이다. 

첫째의 경우, 기본적인 시소러스를 구축함으로써 유의미한 데이터 품질을 기대할 수 있다. //TODO

최근의 BERT를 시작으로 한 pretrained Language Model은 labelled 데이터 대비 상대적으로 많은 양의 unlabelled data에서 유의미한 모델 성능을 이끌어내기 위한 시도이다.

또한, BERT를 general한 도메인 뿐 아니라, science / biomedical / legal 분야의 unannotated document에 적용하여 유의미한 성능 향상을 도출한 사례가 있다.

이번 작업에서, 

1) unlabelled data를 최대한 레버리지를 올리기 위해 BERT를 manufacturing document에 manuBERT를 finetunning하고 이 모델을 이용한 extensive experimentation을 수행한다.

2) 앞 뒤의 다른 문장의 정보를 반영하여 현상/원인/조치를 분류할 수 있도록 manuBERT를 계층적으로 쌓아 BERT의 단어간 contextualization과 문장 간 contextulization을 가능하게 하는 contextualized classifier 구조 Hierarchical BERT를 제안한다.



1. Background (다시쓰기 필요)
   The BERT model architecture (Devlin et al., 2019) is based on a multilayer bidirectional Transformer (Vaswani et al., 2017). Instead of the traditional left-to-right language modeling objective, BERT is trained on two tasks: predicting randomly masked tokens and predicting
   whether two sentences follow each other. manuBERT follows the same architecture as BERT but is
   instead pretrained on scientific text.
   1. 



1. Experimental Setup
   1. Tasks
   2. Datasets
   3. Architecture
       
2. Result



1. Discussion



1. Related Work



1. Conclusion and Future Work


BERT Vis

BERT는 어텐션 모델.

이전 large LM works well but, 이유를 특정하기 어렵

BERT 어텐션 헤드는 많은 경우, 의미를 지니지 않는 SEP 토큰에 주목. 

For example, we find

heads that find direct objects of verbs, determiners of nouns, objects of prepositions, and objects

of possessive pronouns with >75% accuracy. We

perform a similar analysis for coreference resolution, also finding a BERT head that performs quite

well. These results are intriguing because the behavior of the attention heads emerges purely from

self-supervised training on unlabeled data, without

explicit supervision for syntax or coreference



SEP, CLS는 반드시 모든 시퀀스에 존재하므로 다른 토큰들과 다른 성질을 띄는 것으로 보인다.

layer 5에서부터 SEP 토큰에 대한 어텐션이 많이 나타난다. 만약 SEP을 향한 어텐션은 높지만, SEP 자신은 0.9 이상 본인에게 어텐션을 주는 것으로 보아 segment 전체에 대한 representation이라고 보긴 어렵다. 또, 실제 MLM loss의 gradient는 적다. 즉, no-op이다. (어텐션은 높지만, 실제 효과는 미미한)

CLS 토큰의 경우는 마지막 레이어에서 high entropy를 보이며, 다른 토큰들에 대해 broad한 attention을 보인다. NSP 태스크에 직접 사용되는 만큼 전체 representation을 aggregate하는 것으로 보인다.



broad attention -> high entropy, focused attention -> low entropy

하층 layer의 경우 bag of word에 가깝게 broad한 어텐션을 보인다. 

Syntactic

BERT의 단어를 품사로 치환한 뒤 품사간 attention으로 단어 간의 의존성(dependency parsing)을 분석한 결과, 일정한 규칙이 있음을 발견했다. 그러나, 사람이 인식하는 품사 간의 관계가 아니었다. 그러므로, BERT가 사람이 생각하는 의존성을 따라하진 않지만, self-supervised의 by-product로 syntactic behavior를 학습하는 것으로 보인다. BERT의 attention 관계가 dependency를 잘 보이고는 있으나, 이는 수백개의 head 중 의미를 보이는 head의 이야기기 때문에, 각 attention head가 문장 구조의 의존성을 잘 파악한다고 보기는 어렵다. 

(이러한 특성이 영어가 아닌 다른 언어에도 유사하게 나타나는지 확인하는 것도 재미있을 것으로 보인다.)

Despite not being explicitly trained on these tasks, BERT’s attention heads perform

remarkably well, illustrating how syntax-sensitive behavior can emerge from self-supervised training alone.

Semantic

특정 어텐션의 헤드는 coreference resolution 태스크에서 rule-based에 근접할 정도의 성능을 보였다.

Since individual attention heads specialize to particular aspects of syntax, the model’s overall

“knowledge” about syntax is distributed across

multiple attention heads.

We find the Attn +

GloVe probing classifier substantially outperforms

our baselines and achieves a decent UAS of 77,

suggesting BERT’s attention maps have a fairly

thorough representation of English syntax.

 어텐션 맵을 이용한 coreference resolution 태스크 결과, BERT를 이용한 벡터 representation은 attention map 만큼의 문법적 의미를 지니진 않는다. 



Overall, our results from probing both individual and combinations of attention heads suggest

that BERT learns some aspects syntax purely as a

by-product of self-supervised training. Our findings are part of

a growing body of work indicating that indirect

supervision from rich pre-training tasks like language modeling can also produce models sensitive

to language’s hierarchical structure.

BERT의 어텐션들은 같은 레이어에 있거나, 같은 목적을 달하는 attention head끼리 유사한 분포를 보인다. 



bert는 explainable AI라는타이틀을 달고 나오진 않았지만, 그 방향을 제시해준 것 같다. 

버트의 설명성의 핵심은 구조는 인간의 결정방식을 모방하도록 학습시키는 것은 설명성을 높여줄 수 있다는 것이다. 언어모델이 언어에 대해 이해하는 방법은 무수히 많을 수 있으나, 그 중에서 인간의 결정방식을 선택했다는 것이다.

또 다른 핵심은 중요한 단어에 주의를 기울인다는 결정 방식은 인간의 것을 모방했지만, 그 세부적인 (자세히 정의되지 않은) 방법에 대해서는 자의적인 패턴을 보인다는 것이다. (문법적인 해석이 인간 기준과 다름, 어텐션의 주의 기준(의미론/구문론)적 분배가 경향성은 보이나, 명확히 일관되진 않음==각 층별로 기준이 나뉜것이 아니라 층 안에 흩뿌려져있음)



What Does BERT Look At? An Analysis of BERT’s Attention

Visualizing Attention in Transformer-Based Language Representation Models



This finding is a bit

surprising given that Tu et al. (2018) show that encouraging attention heads to have different behaviors can improve Transformer performance at machine translation.

Multi-Head Attention with Disagreement Regularization


