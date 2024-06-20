---
layout: post
title:  "Speaking Probes: Self-Interpreting Models?"
date:   2023-01-16 00:00:00 +0300
categories: interpretability
---

*This post originally appeared in* [Towards Data Science](https://towardsdatascience.com/speaking-probes-self-interpreting-models-7a3dc6cb33d6).

# Introduction
In recent years, many interpretability methods have been developed in Natural Language Processing (Kadar et al., 2017; Na et al., 2019; Geva et al., 2020; Dar et al., 2022). In parallel, strong language models have taken the field by storm. One may wonder if strong language skills allow language models to communicate about their inner state. This work is a brief report on my explorations of this conjecture. In this work, we will design natural language prompts and inject model parameters as virtual tokens in the input. The prompts are designed to instruct the model to explain words — but instead of a real word they are given a virtual token representing a model parameter. The model then generates a sequence continuing the prompt. We will observe this technique's ability to explain model parameters which we have existing explanations for. We will call this new technique "speaking probes" and discuss on a high-level possible justifications for why one might expect the method to work.

Interpretability researchers are encouraged to use speaking probes as a tool to guide their analysis. I do not suggest relying on their answers indiscriminately, as they are not sufficiently grounded. However, they have the important advantage of possessing the expressive power of natural language. Our queries are out of distribution for the model in the zero-shot case as it was only trained with real tokens. However, the hypothesis is that its inherent skills at manipulating its representations will make it easy to learn the new task.

A minimal implementation is provided in: [https://github.com/guy-dar/speaking-probes]. 

# Background: Residual Stream
This has been explained in more detail in the background section of my previous post: [Analyzing Transformers in Embedding Space - Explained](https://towardsdatascience.com/analyzing-transformers-in-embedding-space-explained-ef72130a6844).

We rely on a useful view of the transformer through its residual connections originally introduced in nostalgebraist (2020). Specifically, each layer takes a hidden state as input and adds information to the hidden state through its residual connection. Under this view, the hidden state is a residual stream passed along the layers, from which information is read, and to which information is written at each layer. Elhage et al. (2021) and Geva et al. (2022b) observed that the residual stream is often barely updated in the last layers, and thus the final prediction is determined in early layers and the hidden state is mostly passed through the later layers. An exciting consequence of the residual stream view is that we can project hidden states in every layer into embedding space by multiplying the hidden state with the embedding matrix E, treating the hidden state as if it were the output of the last layer. Geva et al. (2022a) used this approach to interpret the prediction of transformer-based language models, and we will follow a similar approach.

# Showcasing Speaking Probes
## Overview
We will build on the residual stream view as our intuition. In the residual stream view, parameters of the models are added to the hidden state on a more or less equal footing with token embeddings. More generally, the residual view hints that there's a good case for considering parameter vectors, hidden states, and token embeddings to be using the same "language". "Syntactically", we can use any continuous representation - be it a parameter vector or hidden state - as a virtual token. We will use the term "neuron" interchangeably with "virtual token" throughout this paper.

We will focus on parameters in this article, as hidden states seem to be more complicated to analyze - which stands to reason since they are mixtures of parameters. We show that parameter vectors can be used alongside token embeddings in the input prompt and produce meaningful responses. My hypothesis is that every neuron, when interpreted with speaking probes, eventually collapses into a token that is related to the concepts it encodes.

Our goal is to use the strong communication skills language models possess for expressing their latent knowledge. We will explore a few prompts in which the model is requested to explain a word. Instead of a word, it is given a virtual token representing a vector in the parameters. We represent the virtual token in the prompt by the label <neuron> (when running the model, its token embedding is simply replaced with the neuron we want to interpret). We then generate the continuation of the prompt, which is the language model's response.

## Prompts
```
The term "<neuron>" means
```

```
- Synonyms of small: small, little, tiny, not big
- Synonyms of clever: clever, smart, intelligent, wise
- Synonyms of USA: USA, US, United States
- Synonyms of Paris: Paris, city of lights, city of love, capital of
France
- Synonyms of error: error, mistake, fallacy, wrong
- Synonyms of <neuron>:
```

```
The term <neuron> is a broad term related to
```

```
From Wikipedia: "<neuron> is
```

More examples are available in the repository under **prompts/**

## Method
We feed a prompt into the model and generate the continuation of the text with <neuron>'s "token embedding" being the neuron we want to interpret. To produce diverse outputs, we generate with sampling and not just greedy decoding. We will see a few examples below.

In this work, we will focus on feedforward (FF) keys (the first layer of the feed-forward sublayer), as they seem somewhat easier to interpret than FF values (the second layer). Each layer $l$ has a matrix $K_l$ (do NOT confuse with attention keys) — each of its columns can be considered interpreted *individually*.

To test our method, we will use models we already have a good idea of what they mean in embedding space. Obviously, these are the easiest cases we can consider - so these experiments are just a sanity check. For syntactic sugar, we use <param_i_j> for a neuron representing the j-th FF key in the i-th layer. All the examples below are from GPT-2 medium. The generation hyperparameters we use are:

```
temperature=0.5
repetition_penalty=2.
do_sample=True
max_new_tokens=50
min_length=1
```

## Examples 
For the examples, you are encouraged to go to the [original post](https://towardsdatascience.com/speaking-probes-self-interpreting-models-7a3dc6cb33d6) on Towards Data Science (it's to much to copy and format for me to do..).

# Discussion 
## Potential of the Method
The distinctive features that we want to capitalize on with this method are:

* **Natural language output**: both an advantage and disadvantage, it makes the output harder to evaluate, but it provides much greater flexibility than other methods.
* **Inherent ability to manipulate latent representations**: we use the model's own capabilities of manipulating its latent representations. We assume they share the same latent space with the model parameters due to the residual stream view. Other techniques need to be trained or adjusted in some other way to the model's latent space in order to "understand" it. The model is capable of decoding its own states naturally, which can be useful for interpretation.
In general, there is not much research on continuous vectors as first-class citizens in transformers. While ideas like prompt tuning (Lester et al., 2021), and exciting ideas like Hao et al. (2022) pass continuous inputs to the model, they require training to work and they are not used zero-shot. A central motif in this work is the investigation of whether some continuous vectors can be used like natural tokens without further training - under the assumption that they use the same "language" as the model.

Another useful feature of this technique is that it uses the model more or less as a black box, without much technical work involved. It is easy to implement and understand. Casting interpretation as a generation problem, we can leverage literature on generation from mainstream NLP for future work. Similarly, hallucinations are a major concern in speaking probes, but we can hope to be able to apply mainstream research approaches in the future to this method.

In total, this is perhaps the most modular interpretability method - it does not rely on a specifically tailored algorithm, and it can adopt insights from other areas in NLP to improve, without losing breath. Also, it is easy to experiment with (even for less academically inclined practitioners) and the search space landscape is very different than with other methods.

## Possible Future Directions
* **Eloquent. Too Eloquent**: language models are trained to produce eloquent explanations. Factuality is less worrisome to them. These eloquent explanations are not to be taken literally.
* **Layer Homogeneity**: in this article, we implicitly assume we can take parameters from different layers and they will react similarly to our prompts. It is possible that some layers are more amenable to use with speaking probes than others. We call this layer homogeneity. We need to be cautious in assuming that all layers can be treated the same with respect to our method.
* **Neuron Polysemy**: especially in face of word collapse, it seems that neurons that carry multiple unrelated interpretations will have to be sampled multiple times to account for all their different meanings. We would like to be able to extract the different meanings more faithfully and "in one sitting".
* **Better Prompts**: this is not the main part of our work, but many papers show the benefits of using carefully engineered prompts (e.g., Liu et al., 2021).
* **Other Types of Concepts**: we have mainly discussed neurons that represent a category or a concept in natural language. We know that language models can work with code, but we haven't considered this type of knowledge in this article. Also, it is interesting to use speaking probes to locate facts in model parameters. Facts might require a number of parameters working in unison - so it will be interesting to locate them and find prompts that will be able to extract these facts.
If you do follow-up work, please cite as:

```
@misc{speaking_probes,
      url = {https://towardsdatascience.com/speaking-probes-self-interpreting-models-7a3dc6cb33d6},
      note = {\url{https://towardsdatascience.com/speaking-probes-self-interpreting-models-7a3dc6cb33d6}},
      title = {Speaking Probes: Self-interpreting Models?},
      publisher = {Towards Data Science},
      author = {Guy, Dar},
      year = 2023
}
```