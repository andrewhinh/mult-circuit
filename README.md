# mult-circuit

Multiplication Circuits

## Description

As explored in [this post by Alexis Gallagher from Answer.AI](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html), which summarizes the main findings from [the paper "Faith and Fate: Limits of Transformers on Compositionality"](https://arxiv.org/abs/2305.18654), a potential explanation for how GPT-like models often "reason" is through a form of pattern matching called "linearized subgraph matching". In short, if the process of a solution to a problem can be represented as a directed graph, such as multiplication, there exist subgraphs that correspond to subtasks often present in the model’s training data, and that are used to approximately match those found in test-time samples. However, the paper addresses this empirically rather than pointing to concrete parts of the model such as the logits, layers, attention heads, and activations. This project aims to address this shortcoming by conducting an exploratory analysis on Qwen2.5-0.5B, which is capable of performing the task of multiplication, using mechanistic interpretability techniques to attempt to find an explainable and observable circuit involved in this task.
