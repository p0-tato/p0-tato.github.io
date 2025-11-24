---
title: "[Review] Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP 2023)"
date: 2025-11-24 22:00:00 +0900
categories: [Review, AI]
tags: [PagedAttention, LLM, memory, SOSP2023]
---
Efficient Memory Management for Large Language Model Serving with PagedAttention이라는 SOSP 2023에서 발표된 논문을 읽어보려고 합니다. 논문 원본은 아래에서 확인해보실 수 있습니다.
https://arxiv.org/abs/2309.06180

이 논문은 LLM 서빙 시 발생하는 메모리 병목 현상을 운영체제의 가상 메모리 기법에서 착안한 아이디어로 해결했습니다. 현재 가장 널리 쓰이는 LLM 서빙 프레임워크 중 하나인 **vLLM**의 기반이 되는 논문이기에, LLM 엔지니어링에 관심이 있다면 반드시 읽어봐야 할 논문입니다.

## Background

### LLM Serving과 KV Cache

LLM은 텍스트를 생성할 때 **Auto-Regressive** 방식을 사용합니다. 즉, 이전까지 생성된 모든 토큰들을 다시 입력으로 받아 다음 토큰 하나를 생성합니다.

이 과정에서 중복 연산을 줄이기 위해, 이전 단계의 Attention 연산 결과(Key, Value 벡터)를 메모리에 저장해두고 재사용하는데, 이를 **KV Cache**라고 부릅니다.

![KV Cache](/assets/img/2025-11-20-review-pagedattention/kv_cache.png){: width="600" }

문제는 이 KV Cache가 차지하는 메모리 용량이 어마어마하다는 것입니다. 예를 들어 LLaMA-13B 모델의 경우, 시퀀스 길이가 2048일 때 요청 하나당 약 1.7GB의 메모리를 KV Cache로 사용합니다.

### 기존 메모리 관리의 문제점: Fragmentation

PyTorch를 비롯한 기존의 딥러닝 프레임워크들은 텐서 연산을 위해 **물리적으로 연속된 메모리 공간**을 요구합니다.
하지만 LLM의 출력 길이는 생성이 끝날 때까지 알 수 없습니다. 100토큰이 생성될지, 2000토큰이 생성될지 모르는 상황에서 연속된 공간을 보장하려면 어떻게 해야 할까요?
기존 시스템들은 어쩔 수 없이 **최대 시퀀스 길이만큼의 메모리를 미리 할당**하는 방식을 택했습니다.

이러한 '최대 길이 선점' 방식은 심각한 **메모리 파편화(Fragmentation)**를 초래합니다. 논문에서는 이를 크게 세 가지 낭비로 분류합니다.

1.  **Internal Fragmentation**: 미리 잡아둔 공간보다 실제 문장이 짧아서 낭비되는 공간
2.  **External Fragmentation**: 메모리 공간이 조각나서, 합치면 충분하지만 연속된 공간이 없어 할당하지 못하는 경우

논문의 실험 결과에 따르면, 기존 시스템(Orca 등)에서는 이러한 파편화로 인해 실제 메모리의 20%~40%만이 유효한 데이터를 저장하는 데 쓰이고, 나머지는 낭비되고 있었습니다. 이로 인해 Batch Size를 키우지 못하고, 결과적으로 처리량(Throughput)이 제한되는 병목이 발생했습니다.

## PagedAttention

이 문제를 해결하기 위해 저자들은 운영체제의 **가상 메모리와 페이징** 기법에서 아이디어를 얻었습니다.

### 1. Block 단위 관리와 Block Table

PagedAttention은 KV Cache를 연속된 공간에 저장하지 않고, **블록** 단위로 쪼개서 불연속적인 메모리 공간에 저장합니다.

-   **Logical Block**: 논리적인 KV Cache 블록 (연속적)
-   **Physical Block**: 실제 GPU 메모리에 저장되는 블록 (불연속적)

그리고 이 둘을 매핑해주는 **Block Table**을 도입했습니다. OS의 Page Table과 똑같은 역할입니다. Block Table은 각 시퀀스마다 존재하며, 논리적 블록 인덱스를 물리적 블록 인덱스로 변환해줍니다.

![PagedAttention Overview](/assets/img/2025-11-20-review-pagedattention/pagedattention_overview.png){: width="700" }

### 2. PagedAttention Kernel 구현 

실제 vLLM 코드를 살펴보면(`csrc/attention/attention_kernels.cuh`), PagedAttention은 크게 두 가지 버전의 커널로 구현되어 있습니다.

-   **PagedAttention V1**: 일반적인 경우에 사용됩니다. Block Table을 참조하여 불연속적인 메모리에서 KV 데이터를 가져와 연산합니다.
-   **PagedAttention V2**: **Long Context** 상황에 최적화된 버전입니다. 시퀀스 길이가 매우 길어지면 단일 스레드 블록에서 처리하기 부담스러워지는데, V2는 시퀀스 차원을 여러 파티션으로 쪼개어 병렬 처리를 수행합니다.

```cpp
// vLLM csrc/attention/attention_kernels.cuh 예시
template <typename scalar_t, ...>
__global__ void paged_attention_kernel(
    ...
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    ...
) {
    // 1. Block Table Lookup
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    
    // 2. Calculate Physical Address
    const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride + ...;
    
    // 3. Load & Compute
    ...
}
```

이처럼 `block_tables`라는 인덱스 배열을 GPU 커널에 넘겨주어, 커널 내부에서 실시간으로 물리 주소를 계산하는 방식입니다.

### 3. Block Manager와 Scheduling (Python Level)

vLLM의 `core` 모듈(`vllm/v1/core/kv_cache_manager.py`)에서는 Python 레벨에서 블록을 관리합니다.

-   **KVCacheManager**: 전체 GPU 메모리의 블록 상태를 관리합니다. `allocate_slots` 메서드를 통해 요청이 들어올 때마다 필요한 만큼의 물리 블록을 할당합니다.
-   **Prefix Caching**: `find_longest_cache_hit` 메서드를 통해, 이미 계산된 블록(예: 시스템 프롬프트)이 있다면 재사용합니다. 이는 블록의 해시값을 기반으로 동작하며, 중복 연산을 획기적으로 줄여줍니다.

### 4. 한계점 극복: Overhead 최소화

여기서 한 가지 의문이 생길 수 있습니다. **"Block Table을 조회하고, 불연속적인 메모리에 접근하면 속도가 느려지지 않을까?"**

저자들은 이를 다음과 같이 설명하고 해결했습니다.
-   **Memory Bound Operation**: Attention 연산은 연산량보다 메모리 이동량(Memory Access)이 병목인 작업입니다. 즉, 데이터를 가져오는 시간이 연산 시간보다 훨씬 깁니다.
-   **Latency Hiding**: Block Table 자체의 크기는 KV Cache에 비해 매우 작습니다. 따라서 Block Table을 조회하는 오버헤드는 전체 메모리 로딩 시간에 비하면 미미하며, 병렬 처리를 통해 숨겨질 수 있습니다.
-   **Kernel Optimization**: 커널 수준에서 메모리 접근 패턴을 최적화하여, 불연속적인 접근으로 인한 성능 저하를 최소화했습니다.

### 5. Memory Sharing (Copy-on-Write)

PagedAttention의 가장 강력한 장점 중 하나는 **메모리 공유**가 매우 쉽다는 것입니다.

Parallel Sampling(하나의 프롬프트에서 여러 개의 답변 생성)이나 Beam Search 같은 기법을 사용할 때, 프롬프트 부분의 KV Cache는 모든 답변에서 동일합니다.

PagedAttention은 이를 물리적으로 복사하지 않고, **Block Table이 같은 물리 블록을 가리키게 함으로써** 메모리를 공유합니다. 이를 위해 각 물리 블록마다 **Reference Count**를 두어 관리합니다.

만약 특정 시퀀스가 공유된 블록의 내용을 변경해야 한다면(예: 새로운 토큰 생성으로 인해 블록 내용이 달라질 때), 그때 새로운 물리 블록을 할당하고 데이터를 복사하는 **Copy-on-Write** 방식을 사용합니다.

이 방식 덕분에 복잡한 샘플링 시나리오에서 메모리 사용량을 획기적으로(최대 55%) 줄일 수 있었습니다.

### 6. Scheduling and Preemption

앞서 살펴본 `KVCacheManager`가 `allocate_slots`를 통해 블록을 할당하다가, 만약 물리적인 GPU 메모리가 모두 소진되면 어떻게 될까요?
PagedAttention을 통해 메모리 효율을 극대화했지만, 트래픽이 폭주하거나 생성 길이가 예상보다 길어지면 결국 물리적 블록(Physical Block)이 부족해지는 상황은 피할 수 없습니다.

기존 시스템들이라면 OOM을 발생시키며 프로세스가 종료되었겠지만, vLLM은 운영체제의 스케줄링 기법을 차용하여 이 상황을 유연하게 대처합니다.

#### Preemption & Swapping

vLLM은 가용 메모리가 부족해지면, 현재 처리 중인 요청 중 일부를 잠시 중단(Preemption)시킵니다. 이때 중단된 요청의 KV Cache 데이터를 어떻게 처리하느냐에 따라 두 가지 전략을 사용합니다.

1.  **Swapping**: 운영체제의 스왑과 동일합니다. GPU 메모리에 있는 KV Cache 블록들을 **CPU 메모리**로 대피(Eviction)시킵니다. 나중에 여유가 생기면 다시 GPU로 불러와(Swap-in) 멈췄던 지점부터 생성을 재개합니다.
2.  **Recomputation**: 만약 CPU-GPU 간 데이터 이동 속도가 느리다면, 차라리 데이터를 버리고 다시 계산하는 것이 빠를 수 있습니다. 이 경우 중단된 요청의 KV Cache를 삭제하고, 재개될 때 처음부터 다시 연산합니다.

이러한 메커니즘 덕분에 vLLM은 메모리 한계 상황에서도 시스템이 죽지 않고 안정적으로 서비스를 지속할 수 있는 견고함을 갖추게 되었습니다.

## Limitations & Challenges

논문과 실제 구현을 분석해보면, PagedAttention에도 몇 가지 한계점과 고려해야 할 사항들이 존재합니다.

### 1. Kernel Overhead
Block Table을 조회하는 과정(Indirection)은 필연적으로 메모리 접근 오버헤드를 발생시킵니다. 저자들은 이를 연산 병렬성을 통해 숨겼지만(Latency Hiding), 매우 작은 배치의 경우 순수 연속 메모리 접근보다는 미세하게 느릴 수 있습니다.

### 2. Python Scheduler Overhead
현재 vLLM의 스케줄러와 블록 매니저는 Python으로 구현되어 있습니다.
GPU 연산 속도는 매우 빠르기 때문에, 트래픽이 매우 많거나 배치가 작을 경우 **CPU(Python)에서의 스케줄링 오버헤드**가 병목이 될 수 있습니다. 이를 해결하기 위해 vLLM 팀은 스케줄러의 연산 효율을 개선하고 오버헤드를 줄이기 위한 최적화 작업을 지속적으로 진행하고 있습니다.

### 3. Fixed Block Size
블록 크기는 서버 시작 시 고정해야 합니다(기본값 16).
-   블록이 너무 크면: 마지막 블록에서의 Internal Fragmentation이 커집니다.
-   블록이 너무 작으면: Block Table이 커지고, 커널에서 조회해야 할 횟수가 늘어나 오버헤드가 증가합니다.
따라서 워크로드에 맞는 적절한 블록 크기를 설정하는 것이 중요합니다.

## Performance

vLLM(PagedAttention 적용)은 기존 SOTA 시스템(FasterTransformer, Orca) 대비 압도적인 성능 향상을 보여주었습니다.

-   **Throughput**: 같은 메모리 용량에서 더 많은 요청(Batch)을 동시에 처리할 수 있어, 처리량(Throughput)이 **2~4배** 증가했습니다.
-   **Latency**: 메모리 관리에 드는 오버헤드가 거의 없어, 단일 요청의 지연 시간(Latency)에는 영향을 주지 않았습니다.

## Conclusion

PagedAttention은 시스템 수준의 최적화를 통해 LLM 서빙의 효율성을 극적으로 끌어올린 연구입니다. 단순히 모델 구조를 바꾸는 것이 아니라, **메모리 관리**라는 근본적인 시스템 문제를 해결함으로써 성능을 개선했다는 점이 인상적입니다.

이 기술은 현재 **vLLM**이라는 오픈소스 프로젝트로 공개되어 있으며, HuggingFace의 TGI 등 다양한 서빙 프레임워크에서도 표준처럼 채택되고 있습니다.