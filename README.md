# gWordcomp

Code for the paper:

On the Compositionality and Semantic Interpretation of English Noun Compounds. 2016. Corina Dima. Proceedings of the 1st Workshop on Representation Learning for NLP RepL4NLP (https://sites.google.com/site/repl4nlp2016/)

Implementation for two **semantic composition models**: 
 - Matrix, described in Socher, Richard, Christopher D. Manning, and Andrew Y. Ng. "Learning continuous phrase representations and syntactic parsing with recursive neural networks." Proceedings of the NIPS-2010 Deep Learning and Unsupervised Feature Learning Workshop. 2010.
 - Full Additive, described in Zanzotto, Fabio Massimo, et al. "Estimating linear models for compositional distributional semantics." Proceedings of the 23rd International Conference on Computational Linguistics. Association for Computational Linguistics, 2010.; Dinu, Georgiana, The Pham Nghia and Baroni, Marco. "General estimation and evaluation of compositional distributional semantic models." 2013.

Such models can be used to obtain vectorial representations for linguistic units above the word level by combining the representations of the individual words. 

An example application is the composition of compounds: the vectorial representation of 'apple tree' could be obtained by combining the vectorial representations of 'apple' and 'tree'.

Implementation of a **semantic relation classifier**:
- for supervised learning and prediction of compound-internal semantic relations: e.g. for the compound 'iron fence', the semantic relation linking 'iron' to 'fence' is 'material'.

## Prerequisites

The code is written in [Lua](http://www.lua.org/about.html) and uses the [Torch scientific computing framework](http://torch.ch/). To run it, you will have to first install Torch and Torch additional packages `nn`, `nngraph`, `optim`, `paths` and `xlua`; in addition, as the code is written for GPU, the modules `cutorch`, `cunn` and `cudnn` are required.

The evaluation script for the composition is written in Python and uses [dissect](https://github.com/composes-toolkit/dissect).

## Training

Training composition models:

```
$ th script_compose.lua -model Matrix -dataset sample_dataset -embeddings embeddings_set -dim 50
```

The `-model` option specifies which one of the 2 available models should be trained (Matrix, FullAdd).

The `-dataset` option specifies which dataset should the model be trained on. The dataset should be available in the `data` folder.

The `-embeddings` option specifies which word representations should be used for training. A model can be trained on the same dataset, but using different word representations (for example embeddings or count vectors with reduced dimensionality). The different representations should be placed in an `embeddings` subfolder in the dataset folder. Each representation should be in its own subfolder.

The `-dim` option specifies the dimensionality of the word representation. (e.g. for 50 dimensions, the script will look in the dataset folder in the specified embeddings subfolder for a file with the same name as the embedding name and the suffix `.50d_cmh.emb`)

For other available options, see the help:
```
$ th script_compose.lua -help
```

Training semantic classification models:

```
$ th semantic_relation_classification.lua -model basic_600x300 -dataset semrel_dataset -embeddings embeddings_set -dim 50
```
Specify the architecture via the `-model` parameter (see paper & code comments for details).

The same provisions apply for the `-dataset` and `-embeddings` parameters, with the difference that the dataset here specifies a folder with the cross-validation splits.


# License

MIT
