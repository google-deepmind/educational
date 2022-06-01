# DeepMind Educational Resources

This repository contains a collection of educational tutorials that we have
prepared for teaching the basics of machine learning to various audiences. The
goal is to have simple and accessible resources that can enable anyone,
including those with no machine learning background, to engage and learn from
these tutorials.

Our aim is to contribute to democratisation of machine learning by providing
accessible educational resources to inspire everyone.

The tutorials are presented as notebooks that can be launched via
[Google Colab](https://colab.sandbox.google.com/).

## Introductory Tutorials

### Fluttering Avians [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/fluttering_avians.ipynb)

Videogames and artificial intelligence have a long and happy history together. In this tutorial we will play a familiar (and hard!) game, learn what an agent is, and how we can make them learn to play the game with superhuman abilities. The agents we will build are evaluated on their performance on the game, and selected with variation in a virtual circle of life inspired by evolution.

### Fun with Language [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/fun_with_language.ipynb)

This tutorial shows how we can build important artificial intelligence models for natural language processing called language models. Language models are surprisingly effective models that learn to predict the next letter (or word) given previous letters (or words) --- they essentially learn which letters (or words) go well together. In this tutorial we teach you how the computer represents and processes language, and show how we can use a big chunk of text to learn language models and apply them on two tasks: decoding secret messages and generating text.

### Generative Models [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/generative_models.ipynb)

Creativity is central to human intelligence. In this tutorial we see where Artificial Intelligence (AI) meets creativity. We will show you how an AI can produce realistic looking images of everyday objects as well as works-of-art and imagine "spider-dogs". We refer to an AI that is able to do any one of these things as Generative Models and towards the end of this tutorial you will build your own one.

### Protein Folding  [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/protein_folding.ipynb)

This tutorial explores the use of machine learning for solving the protein
structure prediction problem. Although the tutorial does not actually solve the
problem itself, it provides students with very basic background in coding
and biology to get started, and trains their intuition on machine learning
methods, with the help of visualisation and a few examples of folding simple
protein structures.

### Basics of Reinforcement Learning [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/reinforcement_learning.ipynb)

This tutorial introduces students to a simple reinforcement learning (RL) setup
used in research. It involves running pre-existing code to set up an RL
environment and visualise it. Students can then look at how a completely random
(untrained) agent behaves in these environments. We also include simple code
that implements a reinforcement learning method that can train the agent to
solve these simple tasks. The behaviour of the trained agent can be then
visualised together with plots of how the agent evolves through training.

### Scientific Thinking [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/introductory/scientific_thinking.ipynb)

This tutorial teaches you the basic ideas that underline scientific thinking. We cover developing and testing new knowledge with the scientific method through experimentation and validation, while showing common pitfalls in the process. Through a series of games you will play as an agent trying to understand the world, you will get insight into some of the core ideas behind scientific thinking.

## Summer Schools Tutorials
The following tutorials are intended to be used by anyone who wishes to teach the basics of machine learning and artificial intelligence at a summer school, introductory university course or community meetup. The initial versions of these Colabs were written by a group of volunteers inside DeepMind, drawn on our experience in teaching at:

- Deep Learning Indaba
- Eastern European Machine Learning Summer School
- Khipu: Latin American Meeting In Artificial Intelligence
- Mediterranean Machine Learning Summer School
- Southeast Asia Machine Learning School

The material as a whole does not form a structured course. However, each Colab or group of Colabs tell their own story, so that they can also be used as self-learning materials by students who don’t have direct access to a teacher. They are released under an Apache license, which means that you could adapt them to your own unique style of teaching. You are welcome to adapt them in any way you like when you create teaching material for your own community meetup or summer school.

The Colabs are written in Python and Jax, which are some of the tools we use at DeepMind. The current Colabs in this repository are:

### Introduction to Supervised Learning 1 - Regression [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_regression.ipynb)

This tutorial presents a gentle and intuitive introduction to supervised learning. The tutorial is intended to be taught from the front of a class to a group of students, but could also be done individually. The student learns what a model for data is and what model parameters are, and experiences basic optimization of a loss function to fit data first-hand. We introduce automatic differentiation and Jax. The tutorial concludes with first steps toward non-linear models, underfitting, overfitting and basic regularization.


### Introduction to Unsupervised Learning [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_unsupervised_learning.ipynb)

This tutorial covers the basics of unsupervised learning. In the real world, this is the most common form of learning: where no labels or feedback are available. How then is the machine supposed to learn? The most important cost function guiding the machine is negative log likelihood (aka log-loss). We explain what it means in a gentle introduction to the probability theory (density estimation, maximum likelihood/log-loss and latent variable modelling). We show you how to build and train an autoencoder, and how to compress an image using the K−means clustering algorithm - all using only basic Python commands! Lots of visualisations and exercises are included to make this journey fun.

### Introduction to Graph Neural Nets with JAX/jraph [![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb)

This tutorial teaches you the fundamentals of graph neural networks (GNNs), a family of architectures designed for learning on graph-structured data consisting of nodes and edges, where the edges describe relationships between the nodes. E.g. a molecule can be represented as a graph with atoms as nodes, and the chemical bonds between them as edges. Predicting chemical properties of molecules is an example applications of GNNs, along with many others, e.g. predicting relationships between papers in a citation network.

Differently from previous setups where we have seen input features in the form of a real-valued vectors (e.g. vectorised form of images), the structure of the data itself is important here, e.g. how to atoms are connected to each other in a molecule may influence its properties. GNNs allow us to take this graph structure explicitly into account during training and inference, which is something other architectures like fully-connected networks or recurrent neural networks would not be able to do.

We will go over the basics of graph theory, code GNNs from scratch, and apply our models on three types of graph learning tasks. More specifically, we will implement two common types of GNN architectures: Graph Convolutional Networks and Graph Attention Networks.  We will also introduce you to [jraph](https://github.com/deepmind/jraph), which is a library designed for working with graph neural networks in JAX.


## Call for contributions
This repository is an ongoing effort, maintained by volunteers. As you’ve read to this point, we now count you as a co-volunteer! You are welcome to help improve the material through pull requests, bug reports, features requests and any other code contributions!



## Contact

If you have any feedback, or would like to get in touch with us,
please reach out by opening a new issue on the GitHub repo or emailing us at
`educational@deepmind.com`.

## Disclaimer

This is not an officially supported Google product.

