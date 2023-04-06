# Deep Learning course: lecture slides and lab notebooks

<a href="https://mybinder.org/v2/gh/rth/dl-lectures-labs/master">
  <img src="https://mybinder.org/badge.svg" />
</a>

This is a short adaptation of the
[original course](https://github.com/m2dsupsdlclass/lectures-labs) though in
[Master Year 2 Data Science IP-Paris](https://www.ip-paris.fr/education/masters/mention-mathematiques-appliquees-statistiques/master-year-2-data-science)
by Olivier Grisel and Charles Ollion

## Table of contents

The course covers the basics of Deep Learning, with a focus on applications.

### Lecture slides

  - Intro to Deep Learning
  - [Neural Networks and Backpropagation](https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html)
  - [Embeddings and Recommender Systems](https://m2dsupsdlclass.github.io/lectures-labs/slides/03_recommender_systems/index.html)
  - [Convolutional Neural Networks for Image Classification](https://m2dsupsdlclass.github.io/lectures-labs/slides/04_conv_nets/index.html)
  - [Deep Learning for Object Detection and Image Segmentation](https://m2dsupsdlclass.github.io/lectures-labs/slides/05_conv_nets_2/index.html)
  - [Recurrent Neural Networks and NLP](https://m2dsupsdlclass.github.io/lectures-labs/slides/06_deep_nlp/index.html)

Note: press "P" to display the presenter's notes that include some comments and
additional references.

### Lab and Home Assignment Notebooks

The Jupyter notebooks for the labs can be found in the `labs` folder of
the [github repository](https://github.com/rth/dl-lectures-labs/):

    git clone https://github.com/rth/dl-lectures-labs

These notebooks only work with `keras and tensorflow`
Please follow the [installation\_instructions.md](
https://github.com/rth/dl-lectures-labs/blob/master/installation_instructions.md)
to get started.

Direct links to the rendered notebooks including solutions (to be updated in rendered mode): 

#### Lab 1: Intro to Deep Learning

  - [Demo: Object Detection with pretrained RetinaNet with Keras](https://github.com/rth/dl-lectures-labs/blob/master/labs/01_keras/Demo_RetinaNet.ipynb)
  - [Intro to MLP with Keras](https://github.com/rth/dl-lectures-labs/blob/master/labs/01_keras/Intro%20Keras.ipynb)

#### Lab 2: Neural Networks and Backpropagation

  - [Backpropagation in Neural Networks using Numpy](https://github.com/rth/dl-lectures-labs/blob/master/labs/02_backprop/Backpropagation_numpy.ipynb)
  - [Bonus: Backpropagation using TensorFlow](https://github.com/rth/dl-lectures-labs/blob/master/labs/02_backprop/Backpropagation_tensorflow.ipynb)

#### Lab 3: Embeddings and Recommender Systems

  - [Short Intro to Embeddings with Keras](https://github.com/rth/dl-lectures-labs/blob/master/labs/03_neural_recsys/Short_Intro_to_Embeddings_with_Keras_rendered.ipynb)
  - [Neural Recommender Systems with Explicit Feedback](https://github.com/rth/dl-lectures-labs/blob/master/labs/03_neural_recsys/Explicit_Feedback_Neural_Recommender_System_rendered.ipynb)
  - [Neural Recommender Systems with Implicit Feedback and the Triplet Loss](https://github.com/rth/dl-lectures-labs/blob/master/labs/03_neural_recsys/Implicit_Feedback_Recsys_with_the_triplet_loss_rendered.ipynb)

#### Lab 4: Convolutional Neural Networks for Image Classification

  - [Convolutions](https://github.com/rth/dl-lectures-labs/blob/master/labs/04_conv_nets/Convolutions.ipynb)
  - [Pretrained ConvNets with Keras](https://github.com/rth/dl-lectures-labs/blob/master/labs/04_conv_nets/Pretrained_ConvNets_with_Keras_rendered.ipynb)
  - [Fine Tuning a pretrained ConvNet with Keras (GPU required)](https://github.com/rth/dl-lectures-labs/blob/master/labs/04_conv_nets/Fine_Tuning_Deep_CNNs_with_GPU_rendered.ipynb)
  - [Bonus: Convolution and ConvNets with TensorFlow](https://github.com/rth/dl-lectures-labs/blob/master/labs/04_conv_nets/ConvNets_with_TensorFlow_rendered.ipynb)

#### Lab 5: Deep Learning for Object Dection and Image Segmentation

  - [Fully Convolutional Neural Networks](https://github.com/rth/dl-lectures-labs/blob/master/labs/05_conv_nets_2/Fully_Convolutional_Neural_Networks_rendered.ipynb)
  - [ConvNets for Classification and Localization](https://github.com/rth/dl-lectures-labs/blob/master/labs/05_conv_nets_2/ConvNets_for_Classification_and_Localization_rendered.ipynb)

#### Lab 6: Text Classification, Word Embeddings and Language Models

  - [Text Classification and Word Vectors](https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/NLP_word_vectors_classification_rendered.ipynb)
  - [Character Level Language Model (GPU required)](https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/Character_Level_Language_Model_rendered.ipynb)
  - [Transformers (BERT fine-tuning): Joint Intent Classification and Slot Filling](https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/Transformers_Joint_Intent_Classification_Slot_Filling_rendered.ipynb)
  
## Acknowledgments

This lecture is built and maintained by Olivier Grisel and Charles Ollion

Charles Ollion, head of research at [Heuritech](http://www.heuritech.com) -
Olivier Grisel, software engineer at
[Inria](https://team.inria.fr/parietal/en)

<a href="http://www.heuritech.com"><img src="slides/05_conv_nets_2/images/logo heuritech v2.png"
width="300"/></a> <a href="https://team.inria.fr/parietal/en"><img
src="slides/05_conv_nets_2/images/inria-logo.png" width="250"/></a>

We thank the  Orange-Keyrus-Thalès chair for supporting this class.

## License

All the code in this repository is made available under the MIT license
unless otherwise noted.

The slides are published under the terms of the [CC-By 4.0
license](https://creativecommons.org/licenses/by/4.0/).
