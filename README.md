# COMP6248 - Reproducibility Challenge

We have looked at the paper written by Gao Huang et al. This paper was introduced on ICLR 2018 conference and was well received. The idea was to create a neural network that once trained would have a good prediction on low resource usage. MSDNet's core idea is to contain multiple interconnected branches and to use multiple classifiers that would allow for an early forward propagation exit if a level of certainty is reached.

We have replicated the architecture of their network based on their description of the initial paper. The issues we have encountered when we tried to reproduce the result was that the paper often times misses out certain details on the implementation and it comes at the judgement of the ones reimplementing it. Our implementation may differ slightly from theirs, but the core idea of their network was grasped in our architecture.

The network was trained on the FashionMNIST dataset and CIFAR10. The general results were above 90% accuracy in both cases.

Training on the CIFAR10 dataset was done over 300 epochs with an initial learning rate of 0.1. The data has been augmented as described in the paper by transforming the input images using a random centre crop with padding 4 and flipping the images horizontally with a probability of 0.5

The learning rate was slightly modified over the training period by implementing a 'scheduler' that lowered the learning rate at specific milestones (epoch 150 and epoch 225). This improved the accuracy results.

For more details about the implementation, as well as the results of the training on both datasets, read the included report in our GitHub repo.
