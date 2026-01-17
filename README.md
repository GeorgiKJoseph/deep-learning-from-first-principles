# Deep Learning from First Principles

## Motivation

Modern deep learning evolves rapidly, but many techniques are often used as recipes rather than understood as principles. This repository is a personal, evolving study guide aimed at building a coherent, first-principles understanding of deep learning through papers, textbooks, blogs, lectures, and small experiments. It prioritizes depth over breadth, focusing on intuition and conceptual clarity rather than trend coverage. The goal is to understand why ideas work and how they connect into a unified mental model of modern deep learning.

## Resources

### Foundation

- [Learning representations by back propagating error](https://www.nature.com/articles/323533a0)
- [Handwritten Digit Recognition with a Back-Propagation Network (LeCun et al.)](https://papers.nips.cc/paper/1989/hash/53c3bce66e43be4f209556518c2fcb54-Abstract.html)
- [Neural Networks and Deep Learning (online book)](https://neuralnetworksanddeeplearning.com/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - Deep nets are difficult to train, paper introduced residual/skip connections for faster training convergence without compromising quality.

### Sequence Modeling & Attention (RNN → Transformer)

- [Understanding LSTM Networks (RNN + LSTM)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Introduced seq to seq, encoder-decoder model using LSTM.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Introduced attention into NMT on top of an RNN encoder–decoder model, where the hidden states of all input tokens from the encoder are stored in memory and provided to the decoder with a soft search option via attention to look at all tokens instead of relying fully on a fixed-size vector.
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Removed recurrence, relying entirely on attention (self & cross attention) for next-token prediction.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1909.10618)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### Representation Learning & Bi-Encoders

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
- [NV-EMBED: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/abs/2405.17428)

### Alignment, Instruction Tuning & RLHF

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

### Scaling and Optimization

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [GShard: Scaling Giant Models with Efficient Conditional Computation](https://arxiv.org/abs/2006.16668)
- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## Philosophy

- Understand mechanisms, not recipes.
- Prefer depth over trend coverage.
- Build intuition through iteration and connection.

This repository reflects how my understanding of deep learning evolves over time. It is not meant to be exhaustive, but coherent.