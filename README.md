# Understanding the Latent Spaces of OOD Detectors: A Study of Mahalanobis and IRW Approaches

Research project for the course **Machine Learning for Natural Language Processing** at ENSAE.

* Louise Demoor
* Benjamin Maurel

This project aims to address concerns about the reliability of large neural networks in Natural Language Processing (NLP).
While state-of-the-art models perform well on input data that is similar to their training datasets, they can experience dysfunction in NLP contexts due to the constantly evolving nature of languages and distributional shifts. To overcome this challenge, the project proposes measuring and detecting distributional shifts in different corpus/sentences using the latent representations of tokens, which can be analyzed using classical discrepancy measure tools adapted to the high-dimensional nature of transformers layers.

In this project we focus on understanding why using the information presents on all the layers can be usefull for Out Of Distribution detectors. Our main contribution relies on visualisations and simple heuristic to better understand the evolution of latents spaces inside a transformer.

## Introduction

Our paper is available [here]() at the root of the project.

### Exemples
Here is a link to a colab where you can reproduce the graphs and results presented in our paper:

https://colab.research.google.com/drive/1aZj7eVN7S7oqTLHCj3XkVG5oiyxEKzT3?usp=sharing

### Data

We have put directly the .npy files corresponding to the feature extractions of each layer of the ROBERTA-base model. You can find them in the drive provided for this purpose at the following link: https://drive.google.com/drive/folders/1nhwBmUW-fo12kIPIX9mrGt2X8nWvekvy?usp=sharing. You can find the embedding of sst-2 train, sst-2 test, news20, trec and wm16.

To use these files in colab's notebook, it is necessary to create a shortcut to your drive (right click on the folder then add a shortcut)

## Credits

The lib folder relies mainly on ressources from https://github.com/lancopku/Avg-Avg. 
We thank the original authors for their open-sourcing.
