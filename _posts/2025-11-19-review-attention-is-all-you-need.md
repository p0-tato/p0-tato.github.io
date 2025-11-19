---
title: "[Review] Attention Is All You Need"
date: 2025-11-19 22:00:00 +0900
categories: [Review, AI]
tags: [nlp, transformer, paper-review]
---

Attention is All You Need라는 NeurIPS 2017에서 발표된 논문을 읽어보려고 합니다. 논문 원본은 아래에서 확인해보실 수 있습니다.
https://arxiv.org/abs/1706.03762

Google Brain 팀이 발표한 이 논문은 기존의 RNN(LSTM, GRU) 기반의 시퀀스 모델링이 가진 구조적 한계를 탈피하고, 오직 Attention 메커니즘만으로 구성된 Transformer 아키텍처를 제안했습니다. 현재 우리가 사용하는 GPT, Claude 등 모든 거대 언어 모델(LLM)의 모태가 되는 아키텍처이기에, 읽어볼 가치가 높은 논문입니다.

## Background

![RNN vs Transformer](/assets/img/2025-11-19-review-attention-is-all-you-need/RNNvsTransformer.png){: width="700" }

먼저 Attention Is All You Need 논문이 등장한 시점의 배경 상황부터 살펴봅시다. Transformer 모델이 등장하기 전, 자연어 처리(NLP) 분야를 지배하던 것은 **RNN(Recurrent Neural Networks)**과 그 파생형인 **LSTM, GRU**였습니다. 주로 **Encoder-Decoder** 구조를 띤 모델들이 번역과 같은 시퀀스 변환 작업에서 SOTA를 차지하고 있었습니다.

그러나, 이 모델들에게는 태생적인 구조적 한계가 존재했습니다.

### 1. 순차적 처리

RNN 계열 모델은 $t$ 시점의 Hidden State($h_t$)를 계산하기 위해, 반드시 직전 시점인 $t-1$의 결과($h_{t-1}$)를 필요로 합니다. 즉, 데이터가 입력되는 순서대로 연산이 진행되어야 한다는 **순차적 특성(Sequential Nature)**을 가집니다.

이는 **병렬 처리(Parallel Processing)**를 기본으로 하는 GPU 하드웨어의 성능을 제대로 활용하지 못하게 만듭니다. 문장의 길이가 길어질수록 연산 시간은 선형적으로 증가하며, 학습 속도를 높이는 명확한 한계점으로 작용했습니다.

### 2. 정보의 병목과 장기 의존성

또한, Encoder가 입력된 긴 문장의 정보를 고정된 크기의 벡터(Context Vector)로 압축해서 Decoder에 넘겨주는 구조였습니다. 문장이 길어질수록 초반의 정보가 뒤쪽까지 전달되지 못하고 소실되는 장기 의존성 문제가 발생했습니다.

## Transformer 아키텍처

이런 상황에서, Attention Is All You Need 논문이 발표되었습니다. 

논문에서 제시한 Transformer 아키텍처는 기존의 Recurrence와 Convolution을 완전히 배제하고, 오직 Attention 메커니즘만으로 Encoder와 Decoder를 연결하는 새로운 구조를 제안했습니다.

전체적인 구조는 크게 Encoder와 Decoder 스택으로 나뉩니다.

![Transformer Architecture](/assets/img/2025-11-19-review-attention-is-all-you-need/transformer_architecture.png){: width="500" }

- Encoder: 입력 시퀀스를 받아 문맥을 이해하고 압축된 표현(Representation)을 생성합니다.
- Decoder: Encoder가 만든 표현을 바탕으로 타겟 시퀀스를 생성합니다. Decoder에서는 **Masked Self-Attention**을 사용하여 미래의 정보를 참조하지 않습니다.

이 구조를 지탱하는 Attention에 대해서 살펴봅시다.

### 1. Scaled Dot-Product Attention

Transformer의 가장 기본이 되는 연산 단위입니다. 입력된 벡터들을 **Query(Q), Key(K), Value(V)**로 매핑하여 연관성을 계산합니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![Scaled Dot-Product Attention](/assets/img/2025-11-19-review-attention-is-all-you-need/scaled_dot_product_attention.png){: width="300" }

- Query ($Q$): 현재 시점의 단어 (검색어)
- Key ($K$): 문장 내의 다른 단어 (검색 대상)
- Value ($V$): 실제 단어의 정보 (검색된 내용)

이 과정은 데이터베이스에서 Query를 날려 Key와 매칭되는 Value를 찾아오는 과정(Lookup)과 유사합니다. 다만, 딱 떨어지는 매칭이 아니라 **'유사도(Similarity)'** 기반의 Soft 매칭이라는 점이 다릅니다.

1. Dot-Product ($QK^T$): Query와 Key 벡터를 내적하여 두 단어간의 유사도를 계산합니다. 내적 값이 클수록 유사도가 높다는 의미입니다.
2. Scaling ($\frac{1}{\sqrt{d_k}}$): 내적 값을 차원 수($d_k$)의 제곱근으로 나누어 스케일링합니다. 
3. Softmax & Weighted Sum: 스케일링된 값을 확률값으로 변환(Softmax)한 뒤, 이를 가중치로 사용하여 Value($V$) 벡터를 가중합한 결과를 반환합니다.

> [!TIP]
> **왜 Scaling이 필요한가요?**
>
>$d_k$의 크기가 커질수록 내적 값이 커지게 됩니다. 값이 너무 커지면 Softmax 함수의 기울기(Gradient)가 0에 가까운 구간(Saturation Region)으로 진입하게 됩니다.
이는 역전파 시 **Gradient Vanishing** 문제를 유발하여 학습 속도를 저하시키고, 모델의 성능을 떨어뜨릴 수 있습니다. 즉, $\sqrt{d_k}$로 나누어 대규모 모델 학습의 수치적 안정성을 보장하기 위함입니다.


### 2. Multi-Head Attention
단일 Attention을 사용하는 대신, 모델은 $d_\text{model}$ 차원의 벡터를 $h$개의 서로 다른 Subspace로 분할하여 병렬로 Attention 연산을 수행합니다. 논문에서는 512 차원을 8개의 Head(각 64 차원)로 분할하여 병렬로 Attention 연산을 수행했습니다.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(h_1, \dots, h_h)W^O $$

이 방식은 두 가지 이점을 가집니다.

![Multi-Head Attention](/assets/img/2025-11-19-review-attention-is-all-you-need/multi_head_attention.png){: width="400" }

1. **병렬 연산**: 쪼개진 차원들이 독립적으로 GPU에서 병렬 연산되므로 효율적입니다.
2. **표현력 향상**: 서로 다른 Head가 서로 다른 Representation Subspaces에서 정보를 수집할 수 있습니다.

### 3. Masked Self-Attention

Decoder에서는 **Masked Self-Attention**이라는 변형된 구조를 사용합니다.

Encoder는 문장 전체를 볼 수 있지만, Decoder는 번역 결과를 생성할 때 **Auto-Regressive** 구조를 유지해야 합니다. 즉, 현재 시점($t$)의 단어를 예측할 때, 아직 등장하지 않은 미래 시점($t + 1$)의 단어를 참조해서는 안 됩니다.

이를 시스템적으로 구현하기 위해, 미래의 위치에 해당하는 Attention Score에 $-\infty$ 값을 더해주는 **마스킹(Masking)** 기법을 사용합니다. 이렇게 하면 Softmax를 통과했을 때 해당 위치의 확률값이 0이 되어, 미래의 정보를 차단(Look-ahead Mask)하게 됩니다.

### 4. Position-wise Feed-Forward Networks

Attention이 단어 간의 '관계'를 파악하는 모듈이라면, **Feed-Forward Networks(FFN)**는 각 단어의 정보를 가공하고 저장하는 실질적인 연산 장치입니다.

Encoder와 Decoder의 각 레이어에는 Attention 서브 레이어 외에도, Fully Connected Layer가 존재합니다. 이 네트워크는 각 위치(Position)마다 **개별적으로, 그리고 동일하게** 적용됩니다.

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

두 번의 선형 변환(Linear Transformation)과 그 사이의 ReLU 활성화 함수로 구성됩니다.

### 5. Residual Connections & Layer Normalization

Transformer가 층을 깊게 쌓으면서도 학습이 가능한 이유는 바로 **Residual Connections**와 **Layer Normalization** 때문입니다.

각 서브 레이어(Attention, FFN)의 출력은 다음과 같은 과정을 거칩니다.

$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

1. **Residual Connections $(x + \text{SubLayer}(x))$**: 입력 x를 연산 결과에 그대로 더해줍니다. 이는 역전파 시 그래디언트가 소실되지 않고 흐를 수 있게 하여, 깊은 네트워크에서도 학습 안정성을 보장합니다.
2. **Layer Normalization**: 데이터의 분포를 정규화하여 학습 속도를 높이고, 가중치 초기화에 대한 민감도를 낮춥니다.

### 6. Positional Encoding

앞서 언급했듯, Transformer는 Recurrence(순환) 구조를 제거했습니다. 이는 모델이 입력 시퀀스의 **순서(Order)**를 인식할 수 없음을 의미합니다. 

예를 들어, **"The cat ate the mouse"**와 **"The mouse ate the cat"**은 사용된 단어가 완전히 동일합니다. 위치 정보가 없다면, 모델은 누가 포식자이고 누가 먹이인지 구별할 수 없습니다. 

이를 해결하기 위해 Transformer는 입력 임베딩에 **위치 정보(Positional Encoding)**를 더해주는 방식을 택했습니다.

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

논문에서는 사인과 코사인 함수를 이용해 각 위치($pos$)마다 고유한 주기를 가진 값을 생성하여 더해주었습니다. 이를 통해 모델은 절대적인 위치뿐만 아니라, 단어 간의 상대적인 거리 정보도 학습할 수 있게 됩니다.

## 복잡도 분석

Transformer와 기존 모델을 비교해보면, 이 모델이 왜 효율적인지 이해할 수 있습니다.

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| **Self-Attention**| $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |

_(n: 시퀀스 길이, d: 표현 차원)_

**1. 병렬 처리 극대화 (Sequential Operations: $O(1)$)**

RNN은 O($n$)의 순차 연산이 필요하지만, Self-Attention은 $O(1)$입니다. 즉, 문장 전체를 행렬로 만들어 한 번에 GPU에 밀어 넣을 수 있습니다. 이는 학습 속도를 비약적으로 상승시킵니다.

**2. 장기 의존성 해결 (Maximum Path Length: $O(1)$)**

네트워크 내에서 두 단어 사이의 거리가 멀어져도, Attention 메커니즘을 통해 단 한 번의 연산($O(1)$)으로 서로를 참조할 수 있습니다. 이는 RNN의 고질적인 문제였던 Long-Term Dependency 문제를 완벽하게 해결합니다.

**3. Trade-Off**
다만, Self-Attention의 레이어 당 복잡도는 $O(n^2 \cdot d)$로, 입력 시퀀스 길이($n$)가 길어질수록 연산량과 메모리 사용량이 제곱으로 증가합니다.

이는 현재 LLM 서비스에서 **긴 컨텍스트(Long Context)**를 처리할 때 발생하는 비용 폭증의 주원인이며, 이를 해결하기 위해 추후 **FlashAttention**과 같은 메모리 최적화 기법들이 등장하는 배경이 됩니다.

## Conclusion
Transformer는 복잡한 순차 처리 로직 없이, 단순한 **Attention 매커니즘**과 **병렬 처리**만으로 기존 번역 모델 성능을 압도했습니다. 또한, 학습 시간을 획기적으로 단축시킴으로써 대규모 데이터셋을 통한 거대 모델 학습의 길을 열었습니다.

이 논문에서 제시한 기술은 지금까지도 계속 사용되고 있습니다. GPT, Claude, Llama 같은 LLM들이 모두 이 아키텍처 위에서 설계된 것이지요. 현재 AI 시스템이 어떻게 데이터를 처리하는지 알 수 있다는 점에서, 읽어볼 가치가 높은 논문입니다.