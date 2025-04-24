# mult-circuit

Multiplication Circuits

## Description

As explored in [this post by Alexis Gallagher from Answer.AI](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html), which summarizes the main findings from [the paper "Faith and Fate: Limits of Transformers on Compositionality"](https://arxiv.org/abs/2305.18654), a potential explanation for how GPT-like models often "reason" is through a form of pattern matching called "linearized subgraph matching". In short, if the process of a solution to a problem can be represented as a directed graph, such as multiplication, there exist subgraphs that correspond to subtasks often present in the model’s training data, and that are used to approximately match those found in test-time samples. However, the paper addresses this empirically rather than pointing to concrete parts of the model such as the logits, layers, attention heads, and activations. This project aims to address this shortcoming by first fine-tuning GPT-2 1.3B (a.k.a. gpt2-xl) on [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), a dataset of mathematical educational content filtered from CommonCrawl [3], then conducting an exploratory analysis on the task of multiplication using mechanistic interpretability techniques to attempt to find an explainable and observable circuit involved in this task.

## Dataset

We’ve chosen the FineMath-4+ subset of the FineMath dataset from HuggingFace. It’s built from CommonCrawl web data and focuses on high-quality math-related educational content such as tutorials, problem explanations, forums. The subset contains 6.7M rows totaling 9.6B tokens of mostly English text and LaTeX-style math, but we plan to randomly sample around 100k (as per project specifications). It can easily be downloaded using datasets, a HuggingFace library, with these two lines of code:

```python
from datasets import load_dataset

data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=8)
```

## Proposed Methodology/Techniques/Resources to be used

We will fine-tune and sample from GPT-2 1.3B similarly to [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master); i.e. using minimal PyTorch and NumPy with Transformers to load GPT-2 checkpoints, Tiktoken for OpenAI's fast byte-pair encoding (BPE) code, and Weights and Biases (wandb) to log experiments.

Once we have a fine-tuned model, we will use [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to conduct exploratory analysis using a few techniques. First, we apply direct logit attribution to decompose the residual stream, the sum of the outputs of each layer and of the original token and positional embedding, into logits and work backwards to determine which layers, heads, and neurons lead to those changes. We also utilize a more advanced technique called activation patching where we run the model twice on two different inputs, producing a “clean” run and a “corrupted” run, then give the model the corrupted input but intervene on a specific activation and patch in the corresponding activation from the clean run. Then, we measure how much the output has updated towards the correct answer; if patching in an activation significantly increases the probability of the correct answer, we can localise which activations matter.
