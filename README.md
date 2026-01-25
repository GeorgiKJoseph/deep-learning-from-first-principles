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
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Encoder-decoder model using LSTM for neural machine translation, the encoder processes the input tokens in language A and produces a fixed representation that the decoder uses to generate output tokens in language B.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Introduced attention into NMT on top of an RNN encoder–decoder model, where the hidden states of all input tokens from the encoder are stored in memory and provided to the decoder with a soft search option via attention to look at all tokens instead of relying fully on a fixed-size vector.
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Removed recurrence, relying entirely on attention (self & cross attention) for next-token prediction.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT-1; used the same pretrained core (a decoder-only Transformer) and fine-tuned separate models for downstream NLP tasks such as classification, entailment, similarity, etc outperforming existing task-specific approaches.
- [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1909.10618) - GPT-2; proved that a single large model can perform multiple NLP tasks without task specific fine-tuning, purely via prompting.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Introduced SwiGLU activations, empirical analysis.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Usually, Post-LN Transformers are trained with a warmup stage for the learning rate (to control gradient explosion due to multiple residual connections). Pre-LN gives more well-behaved gradients during initialization, encouraging the removal of the warmup stage and achieving comparable results. Pre-LN had been used before (e.g., GPT-2), but this work provides a theoretical explanation for why this works.
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Vision Transformer (ViT); Supervised model for image classification, less inductive bias compared to CNNs, outperforms CNNs at scale.

### Representation Learning & Bi-Encoders

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) - Similar pretrain-finetune paradigm as GPT-1 but on a encoder-only Transformer, pretrained via MLM (enabling bidirectional self-attention) and NSP (later dropped in RoBERTa). Downstream tasks are handled with task-specific heads, typically using [CLS] representation.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Sentence representation via MEAN pooling token embeddings, fine-tunes via a classification head on on the concatenated feature vector [u,v,∣u−v∣] with cross entropy loss for NLI tasks (entailment, contradiction, neutral) and further fine-tuned on STS (Semantic Textual Similarity) regression using cosine similarity with MSE loss on human-annotated similarity scores.
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Truncatable embeddings; trains a model so that a single embedding contains nested, truncated vectors of various sizes (8, 16, 32, 64, ...). It does this by applying separate losses to each prefix dimension during training.
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) - Uses a decoder-only Transformer to generate sentence embeddings by taking the [EOS] token representation. Synthetic triplets (query, positive doc, hard negative doc) are generated using GPT-4 and trained with a contrastive loss to pull query–positive embeddings closer and push query–negative embeddings apart. The approach fine-tunes Mistral-7B with minimal architectural changes and achieves state-of-the-art retrieval performance.
- [NV-EMBED: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/abs/2405.17428) - Decoder-only Transformer with a latent attention head instead of relying only on [EOS] representation, removes the causal attention mask during contrastive training to enable bidirectional context. Promptable embeddings.

### Alignment, Instruction Tuning & RLHF

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) - RL using reward models learned from human feedback rather than a reward function, enables tasks where defining rewards is impossible. Experiments on Atari (video frame input, discrete action spaces, A3C) and robot locomotion (sensor inputs, continuous action spaces, TRPO) matched or surpassed traditional methods with high sample efficiency.
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