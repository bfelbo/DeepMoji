### ------ Update September 2023 ------

The online demo is no longer available as it's not possible for us to renew the certificate. The code in this repo still works, but you might have to make some changes for it to work in Python 3 (see the open PRs). You can also check out the PyTorch version of this algorithm called [torchMoji](https://github.com/huggingface/torchMoji) made by HuggingFace.

# DeepMoji

[![DeepMoji Youtube](https://img.youtube.com/vi/u_JwYxtjzUs/0.jpg)](https://www.youtube.com/watch?v=u_JwYxtjzUs)  
*(click image for video demonstration)*
  
DeepMoji is a model trained on 1.2 billion tweets with emojis to understand how language is used to express emotions. Through transfer learning the model can obtain state-of-the-art performance on many emotion-related text modeling tasks.
  
See the [paper](https://arxiv.org/abs/1708.00524) or [blog post](https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0) for more details.

## Overview
* [deepmoji/](deepmoji) contains all the underlying code needed to convert a dataset to our vocabulary and use our model.
* [examples/](examples) contains short code snippets showing how to convert a dataset to our vocabulary, load up the model and run it on that dataset.
* [scripts/](scripts) contains code for processing and analysing datasets to reproduce results in the paper.
* [model/](model) contains the pretrained model and vocabulary.
* [data/](data) contains raw and processed datasets that we include in this repository for testing.
* [tests/](tests) contains unit tests for the codebase.
  
To start out with, have a look inside the [examples/](examples) directory. See [score_texts_emojis.py](examples/score_texts_emojis.py) for how to use DeepMoji to extract emoji predictions, [encode_texts.py](examples/encode_texts.py) for how to convert text into 2304-dimensional emotional feature vectors or [finetune_youtube_last.py](examples/finetune_youtube_last.py) for how to use the model for transfer learning on a new dataset.

Please consider citing our [paper](https://arxiv.org/abs/1708.00524) if you use our model or code (see below for citation).

## Frameworks

This code is based on Keras, which requires either Theano or Tensorflow as the backend. If you would rather use pyTorch there's an implementation available [here](https://github.com/huggingface/torchMoji), which has kindly been provided by Thomas Wolf.

## Installation

We assume that you're using [Python 2.7](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/installing/) installed. As a backend you need to install either [Theano (version 0.9+)](http://deeplearning.net/software/theano/install.html) or  [Tensorflow (version 1.3+)](https://www.tensorflow.org/install/). Once that's done you need to run the following inside the root directory to install the remaining dependencies:
  
```bash
pip install -e .
```
This will install the following dependencies:
* [Keras](https://github.com/fchollet/keras) (the library was tested on version 2.0.5 but anything above 2.0.0 should work)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [h5py](https://github.com/h5py/h5py)
* [text-unidecode](https://github.com/kmike/text-unidecode)
* [emoji](https://github.com/carpedm20/emoji)

Ensure that Keras uses your chosen backend. You can find the instructions [here](https://keras.io/backend/), under the *Switching from one backend to another* section.

Run the included script, which downloads the pretrained DeepMoji weights (~85MB) from [here](https://www.dropbox.com/s/xqarafsl6a8f9ny/deepmoji_weights.hdf5?dl=0) and places them in the model/ directory:

```bash
python scripts/download_weights.py
```

## Testing
To run the tests, install [nose](http://nose.readthedocs.io/en/latest/). After installing, navigate to the [tests/](tests) directory and run:

```bash
nosetests -v
```

By default, this will also run finetuning tests. These tests train the model for one epoch and then check the resulting accuracy, which may take several minutes to finish. If you'd prefer to exclude those, run the following instead: 

```bash
nosetests -v -a '!slow'
```

## Disclaimer 
This code has been tested to work with Python 2.7 on an Ubuntu 16.04 machine. It has not been optimized for efficiency, but should be fast enough for most purposes. We do not give any guarantees that there are no bugs - use the code on your own responsibility!

## Contributions
We welcome pull requests if you feel like something could be improved. You can also greatly help us by telling us how you felt when writing your most recent tweets. Just click [here](http://deepmoji.mit.edu/contribute/) to contribute.

## License
This code and the pretrained model is licensed under the MIT license. 

## Benchmark datasets
The benchmark datasets are uploaded to this repository for convenience purposes only. They were not released by us and we do not claim any rights on them. Use the datasets at your responsibility and make sure you fulfill the licenses that they were released with. If you use any of the benchmark datasets please consider citing the original authors.

## Twitter dataset
We sadly cannot release our large Twitter dataset of tweets with emojis due to licensing restrictions.

## Citation
```
@inproceedings{felbo2017,
  title={Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm},
  author={Felbo, Bjarke and Mislove, Alan and S{\o}gaard, Anders and Rahwan, Iyad and Lehmann, Sune},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2017}
}
```
