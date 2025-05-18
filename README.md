# mult-circuit

Mechanistic Analysis of Multi-Digit Multiplication in Qwen2.5-0.5B

## TL;DR

A potential explanation for how Qwen2.5-0.5B (a 0.5 billion parameter language model developed by Alibaba) performs multiplication is that it uses a combination of sequential processing and pattern matching. The model appears to:

1. Store partial products (intermediate multiplication results) in the residual stream (the main information highway of the transformer model where each layer adds its computations). This storage mechanism allows the model to maintain intermediate calculations across layers, similar to how a human might write down intermediate steps while solving multiplication problems.

2. Process subproblems (smaller multiplication steps) sequentially across layers (the building blocks of transformer models, each containing attention and MLP components). This sequential processing mirrors traditional multiplication algorithms, where each digit multiplication and addition is handled step by step.

3. Use specific attention heads (individual components within attention layers that can focus on different aspects of the input) for final addition. These specialized heads appear to be responsible for combining the partial products stored in the residual stream into the final answer.

These findings provide semi-concrete evidence for the [linearized subgraph matching hypothesis](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html), which suggests that transformers solve complex tasks by matching patterns they've seen during training. The model's approach to multiplication appears to be a learned implementation of traditional multiplication algorithms, broken down into recognizable subpatterns.

## Motivation

Recent work by [Olsson et al. (2023)](https://arxiv.org/abs/2305.18654) proposes that transformer models may perform complex reasoning through "linearized subgraph matching" - a form of pattern matching where solutions to problems are represented as directed graphs (networks showing how different steps connect), with subgraphs (smaller parts of these networks) corresponding to frequently encountered subtasks in the training data. This theory suggests that transformers learn to recognize and execute common computational patterns, much like how humans learn to recognize and apply standard algorithms.

While this theory elegantly explains model behavior at a high level, it lacks mechanistic evidence (e.g., specific logits (raw model outputs before softmax), layers, attention heads, and activations (neuron outputs)). Understanding these specific components is crucial for verifying the theory and potentially improving model performance. This project addresses this gap through a detailed mechanistic interpretability analysis of multiplication in Qwen2.5-0.5B, chosen for its demonstrated capability in performing multiplication tasks and its relatively small size, which makes it more amenable to analysis.

## Results

### Baseline Performance

Qwen2.5‑0.5B demonstrates reliable multiplication capabilities for integers up to **five digits each**, which is impressive given its relatively small parameter count. The model achieves top‑1 ranking (highest probability) for the correct answer token in 9/11 positions, showing strong performance on most digits of the output. The softmax probability distribution (normalized probabilities that sum to 1) shows a characteristic pattern: high confidence for early tokens (≈0.84 for the leading space, ≈0.74 for the first digit) with gradually decreasing confidence towards later positions (≈0.10–0.20 for the final digits). This pattern of decreasing confidence suggests that the model's uncertainty accumulates as it processes more digits, similar to how human calculators might become less certain about later digits in a long multiplication problem.

To assess the model's robustness, a comprehensive contrast‑set (a collection of systematically modified inputs) is constructed with systematically perturbed answers. These perturbations include:

- Digit-level changes: ±1, ±10, ×2/÷2
- Sequence transformations: reversal, sort, rotation
- Uniform changes: ±1 to all digits

The mean logit margin (difference in raw model outputs) between correct and perturbed answers is **1.26**, notably lower than GPT‑2‑small's margin on indirect object identification (≈3.5). This lower margin suggests that multi‑digit multiplication remains a challenging task for transformer models, requiring more precise computation than simpler tasks like identifying grammatical relationships.

### Logit Lens Analysis

By examining the residual stream after each layer and computing logit differences between correct and perturbed answers, it's observed that:

1. **Residual Stream Level**: A linear increase in logit difference across layers indicates sequential processing of subproblems. This linear progression suggests that each layer contributes incrementally to the final answer, with information building up systematically through the network.

2. **Layer Level**: MLP layers (Multi-Layer Perceptrons, the feed-forward networks in transformers) show substantial contributions while attention layers (components that determine which parts of the input to focus on) have minimal impact. This finding strongly supports the hypothesis that multiplication primarily relies on information recall and computation rather than information movement between tokens. The MLP layers appear to be doing the actual mathematical operations, while attention layers play a more minor role in organizing the computation.

3. **Head Level**: Only a few late-layer heads show significant contribution, and their attention patterns reveal nearly exclusive self-attention (where tokens attend to themselves). This suggests that the residual stream stores partial products while these heads handle the final addition, with each token primarily focusing on its own computation rather than attending to other tokens.

### Activation Patching

Through patching activations (replacing neuron outputs) from correct answers into corrupted ones, it's found that:

1. **Residual Stream Level**: All critical computation occurs on the final token, with performance nearly fully recovered by moving the residual stream to the correct position. This indicates that the model's computation is highly localized, with the final token containing all necessary information for generating the answer.

2. **Layer Level**: Results align with residual stream findings, with attention layers 7 and 22 and MLP layer 7 showing significant contributions. These specific layers appear to be crucial for the multiplication process, with layer 7 potentially handling initial digit multiplication and layer 22 managing the final combination of results.

3. **Head Level**: Analysis of weighted attention patterns (how much each token influences others) reveals that heads L0H11 and L22H7 are crucial for the final addition step. These heads appear to be specialized for combining partial products into the final answer, with L0H11 potentially handling initial digit alignment and L22H7 managing the final summation.

4. **Head Contribution Decomposition**: Late heads primarily determine information movement paths (how information flows between tokens), while both early and late heads contribute to information content (what information is stored). This separation of concerns suggests a clear division of labor in the model, with early layers handling computation and later layers managing the organization and combination of results.
