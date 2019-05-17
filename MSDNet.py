#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy of MSDNet.ipynb
"""

# Execute this code block to install dependencies when running on colab
try:
    import torch
    import os
except:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = os.system("!ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'")
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    os.system("!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision")

try:
    import torchbearer
    import os
except:
    os.system("!pip install torchbearer")


import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt
from torchbearer import Trial
import torchbearer
import pandas as pd

torch.manual_seed(0)

"""
  This class provides a simple iterator which returns the indexes needed to
  perform forward propagation through through 'self.grid' in 'MSDNet'. During
  forward propagation, we want MSDNet to first propagate up to depth 1, then up
  to depth 2, then to depth 3, and so on. This class provides a simple means
  of finding the indexes needed to calculate the the result of MSDNet up to
  these various depths. For example, suppose MSDNet has the following grid
  structure

      .  .  .  .
      .  .  .  .
      .  .  .  .
      .  .  .  .

  A '.' above simply represents an element in 'self.grid' in MSDNet class. In
  this example we have a 4x4 grid. If we initialise the 'MSDNetIterator' class
  and then call 'next_iter', the following indexes are returned,

      x  .  .  .
      x  .  .  .
      x  .  .  .
      x  .  .  .

  Returned indexes are marked with an 'x'. So in this case, the following indexes
  are returned by the 'next_iter' method
      [[0,0],[1,0],[2,0],[3,0]]
  These are the indexes needed to seed an input 'i' (see 4.1 MSDNET ARCHITECTURE: first layer).
  If we were to perform 'next_iter' again, we get

      .  .  .  .
      .  .  .  .
      .  .  .  .
      .  x  .  .

  returned indexes are marked with an 'x'. If we call 'next_iter', we get

      .  .  .  .
      .  .  .  .
      .  x  .  .
      .  .  x  .

  By calling 'next_iter' again, we get

      .  .  .  .
      .  x  .  .
      .  .  x  .
      .  .  .  x

  Finally if we were to call 'next_iter' again, we get

      x  .  .  .
      x  .  .  .
      x  .  .  .
      x  .  .  .


  Parameters
  ----------
      n_scales
          this integer is the number of scales in MSDNet.
      depth
          this integer is the number of layers in MSDNet

  Methods
  -------
      seed
          this method returns the indexes needed to seed an image
      next_iter()
          this method returns the indexes needed to propagate MSDNet to the next
          depth. Note that when this class is initialised 'self.indexes' is 'None'.
          So, 'next_iter' has to be called once to return the seeds.
  """

class MSDNetIterator():

  def __init__(self, n_scales, depth):
    self.ns = n_scales
    self.depth = depth - 1 # index of the maximum depth of MSDNet
    self.indexes = None
    self.end = False

  def seed(self):
    """
    get the indexes need to see MSDNet.

    returns
    -------
        list
            this list contains the indexes of the convolutions needed to seed
            MSDNet.
    """
    return [[i, 0] for i in range(self.ns)]

  def next_iter(self):
    """
    Return the indexes needed to propagate MSDNet to the next depth.
    """
    if self.indexes == None:
      self.indexes = self.seed()
      return self.indexes

    for row in range(self.ns - 1):
      if self.indexes[row][1] < self.indexes[row + 1][1]:
        self.indexes[row][1] += 1
    self.indexes[self.ns - 1][1] += 1 # update the last row

    if self.indexes[-1][1] == self.depth:
      self.end=True

    """
    if self.indexes[-1][1] > self.d - 1: # check if we have reached the max depth
      self.indexes = None # reset the iterator
      self.end = True
      return self.next_iter()
    """

    return [x for x in self.indexes if x[1] != 0] # return indexes which are not in the first column

"""
    This class implements MSDNet. In this method, the structure of MSDNet. We store the MSDNet architecture in a 2d ModuleList. We will
    refer to this 2d ModuleList as the 'grid' containing the structure of MSDNet. The dimensions of 'grid' is determined by the 'num_scales'.
    This parameter determines the number of scales that an input image is represented at. For example, if 'num_scales' was 3 the 'grid' would
    be a 3x3 2d ModuleList. Each row of the grid represents an image at a different scale with the last row being the most coarse representation
    of the image.

    Parameters:
        n_channels_in
            The number of images that are input into the network.
        n_classes
            The number of classes the images are to be classified into
        num_scales
            This is the number of scales MSDNet process the image at. So if the number of scales
            is 3 then MSDNet will process the image at 3 different scales. In other words, MSDNet
            have 3 rows with regards to figure 3.
        scale_factor
            This is the factor by which the stride changes between rows. For example if we had a
            scale factor of 2, then the strided convolutions between rows 1 and 2 will have a stride
            of 2 ** 1. Stride convolutions between row 3 and 4 will have a stride of 2 ** 3.
        k0
            This is a factor which determines how many feature maps are produced by each convolution. For
            example, the convolution h_1^1(x_0^1) (SEE FIGURE 4) will produce k0 feature maps. More generally,
            the number of feature maps produced by h^i_j will produce
                k0 * growth_rate ** i
            feature maps. Where 'growth_rate' is the parameter defined below and 'i' corresponds to the row
            where the convolution h^i_j appears. The suubscript 'j' corresponds to the depth of the network.
        growth_rate
            The growth rate dictates the number of feature maps which are produced by the convolutions at each row.
        classifier_indexes
            This is a list of the depths at which the user wishes to occur. Suppose MSDNet a depth of depths which we
            call depth_0, depth_1, depth_2 and depth_3. the depth depth_0 corresponds to the seeding layer, and the other
            depths corresponds to subsequent layers. Suppose in this case we have,
                classifier_index = [1,2,3].
            In this case, a classifier would occurs at the following depths: depth_1, depth_2, and depth_3.
            NOTE:
              At the moment, a classifier cannot occur in the initial layer
        block_sizes: list
            This is a list of the depths of each block in the architecture.
        evaluation_type: string
            'all'
                during evaluation, return the outputs of all of the classifiers in a list
            'end'
                during evaluation, return the ouput of the last classifier.
"""

class MSDNet3(nn.Module):

  NO_CLASSIFIER = 'no_classifier'

  def __init__(self, n_channels_in, n_classes, num_scales, scale_factor, k0, growth_rate, dimensions, classifier_indexes, block_depths, evaluation_type='all', threshold=0.7):

    super(MSDNet3, self).__init__()
    self.sf = scale_factor
    self.ns = num_scales
    self.k0 = k0
    self.gr = growth_rate
    self.n_classes = n_classes
    self.classifier_indexes = classifier_indexes
    self.dim = dimensions
    self.block_depths = block_depths
    self.n_channels_in = n_channels_in
    self.evaluation_type = evaluation_type
    self.threshold = threshold

    if len(block_depths) != self.ns :
      raise Exception("The number of blocks must be the same as the number of scales")

    # create the grid that stores the number of feature maps
    self.fm_grid = self.create_feature_map_grid()

    # create the grid which stores the MSDNet architecture
    self.grid = self.make_initial_layer()
    self.create_architecture()

    #create the classifiers
    self.classifiers = self.create_classifiers()

    #self.test_classifier = self.create_classifier(1)

  def create_feature_map_grid(self):
    self.block_depths = [i + 1 for i in self.block_depths]
    feature_map_grid = []
    x = []

    # seed the 'feature_map_grid'
    for i in range(self.ns):
      feature_map_grid.append([self.k0 * self.gr ** i])
      x.append(self.k0 * self.gr ** i)

    for index in range(len(self.block_depths)):
      size = self.block_depths[index]
      transition_layer_index = size - 1

      for col in range(size):
        for row in range(self.ns):
          if col == transition_layer_index:
            x[row] = int(x[row] / 2)
          elif row == 0:
            x[row] = int(x[row] + self.k0 * self.gr ** row)
          else:
            if row == index and col > 0:
              x[row] = int(x[row] + self.k0 * self.gr ** row)
            else:
              x[row] = int(x[row] + 2 * self.k0 * self.gr ** row)

        # update 'feature_map_grid'
        for row in range(self.ns):
          feature_map_grid[row].append(x[row])

    start = 0
    for row in range(len(feature_map_grid) - 1):
      depth = self.block_depths[row]
      start += depth
      for col in range(len(feature_map_grid[0])):
        if col > start:
          feature_map_grid[row][col] = None

    # remove the final transition layer
    for row in range(self.ns):
      feature_map_grid[row] = feature_map_grid[row][0 : len(feature_map_grid[row]) - 1]


    return feature_map_grid

  def make_initial_layer(self):
    """
    Maker the intial layer of the MSDNet. The first layer is the elements of 'self.grid'
    in the first column. This layer is used to seed the image at its different scales.

    Returns
    -------
        list
            this list contains all of the convolutions needed to seed the scales
    """

    layer = nn.ModuleList([nn.ModuleList([]) for i in range(self.ns)]) # stores the strided convolutions

    for scale in range(self.ns):
      if scale == 0:
        """ first scale is produced by a convolution with stride of 0 """
        conv = nn.Conv2d(
            in_channels = self.n_channels_in,
            out_channels = self.k0 * self.gr ** scale,
            kernel_size = (3,3),
            stride = 1,
            padding = 1
        )


        bn = nn.BatchNorm2d(conv.out_channels)

      else:
        """ subsequent scales are produced by convolutions with scale 'self.sf ** i' """
        prev_out_channels = layer[scale - 1][0][-1].num_features
        conv = nn.Conv2d(
            in_channels = prev_out_channels,
            out_channels = self.k0 * self.gr ** scale,
            kernel_size = (3,3),
            padding = 1,
            stride = self.sf #** scale
        )

        bn = nn.BatchNorm2d(conv.out_channels)

      operations = nn.ModuleList([conv, bn])
      layer[scale].append(operations)
    return layer

  def h(self, n_channels_in, scale_index):
    """
    create two sequential convolutions with no stride as described in appendix A.
    The convolutions comprise of a 1x1 convolution followed by a 3x3 convolution.

    Parameters
    ----------
    n_channels_in
        this is the number of feature maps which are input into the the method
    scale_index
        this is index of the row (row as shown in figure 3) which the convolutions
        will take place. The scale index allows for the number of channels which are
        output from each convolution to be calculated.

    Returns
    -------
    list
        this method returns a list of the convolutions with no stride
    """

    conv1 = nn.Conv2d(
        in_channels = n_channels_in,
        out_channels = 4 * self.k0 * self.gr ** scale_index,
        kernel_size = (1,1),
        stride = 1,
        padding = 0
    )

    bn1 = nn.BatchNorm2d(
        num_features=conv1.out_channels
    )

    conv2 = nn.Conv2d(
        in_channels = conv1.out_channels,
        out_channels = self.k0 * self.gr ** scale_index,
        kernel_size = (3,3),
        stride = 1,
        padding = 1
    )

    bn2 = nn.BatchNorm2d(
        num_features=conv2.out_channels
    )

    return torch.nn.ModuleList([conv1, bn1, conv2, bn2])

  def h_hat(self, n_channels_in, scale_index):
    """
    create 2 strided convolutions as described in appendix A. The convolutions
    comprise of a 1x1 convolution followed by a 3x3 convolution.

    Parameters
    ----------
    n_channels_in
        this is the number of feature maps which are input into the the method
    scale_index
        this is index of the row (row as shown in figure 3) which the convolutions
        will take place. The scale index allows for the number of channels which are
        output from each convolution to be calculated aswell as the stride of the
        3x3 convolution.

    Returns
    -------
    list
        this method returns a list a list of strided convolutions
    """

    conv1 = nn.Conv2d(
        in_channels = n_channels_in,
        out_channels = 4 * self.k0 * self.gr ** scale_index,
        kernel_size = (1,1),
        stride = 1,
        padding = 0
    )


    bn1 = nn.BatchNorm2d(
        num_features=conv1.out_channels
    )

    conv2 = nn.Conv2d(
        in_channels = conv1.out_channels,
        out_channels = self.k0 * self.gr ** scale_index,
        kernel_size = (3,3),
        padding = 1,
        stride = self.sf# ** scale_index
    )

    bn2 = nn.BatchNorm2d(
        num_features=conv2.out_channels
    )

    return torch.nn.ModuleList([conv1, bn1, conv2, bn2])

  def make_layer(self, layer_index):
    """
    Make a new layer in the MSDNet using the previous layer to ensure the number
    of feature maps, and scales are correct.

    The number of feature maps that each convolution take as an input is calculated
    using the following formula
        (k0 * gr ** row) + layer_index * (2 * k0 * gr ** row)
    see notes for details.
    Convolutions are paired in lists of the form
        [h_hat, h]
    So strided convolutions are stored first, and regular convolutions are stored
    second.

    Parameters
    ----------
    prev_layer
        This is a list of the convolutions from the previous layer and is used to
        determine the parameters needed from the new convolutions.
    layer_index
        This is the index of the new layer in 'self.grid'. This parameter is used
        to determine where the convolutions will go. Since the MSDNet has a diagonal
        structure, if layer_index is 2, then there does not need to be any convolutions
        in rows at index 0 or 1. See section 4 "Multi-Scale Dense Convolutional Networks
        (Network reduction and lazy evaluation)".

    Returns
    -------
    list
        returns a list of the convolutions for the layer at index 'layer_index'.
    """

    layer = nn.ModuleList([None for i in range(self.ns)]) # initialise the layer

    for row in range(layer_index, self.ns):
      convs = nn.ModuleList([]) # stores the convolutions

      # === strided convolution h_hat ===
      h_hat_num_channels_in = (self.k0 * self.gr ** (row - 1)) + (layer_index - 1) * (2 * self.k0 * self.gr ** (row - 1))
      h_hat = self.h_hat(
          n_channels_in = h_hat_num_channels_in,
          scale_index = row
      )

      # === regular convolution h ===
      h_num_channels_in = (self.k0 * self.gr ** row) + (layer_index - 1) * (2 * self.k0 * self.gr ** row)
      h = self.h(
          n_channels_in = h_num_channels_in,
          scale_index = row
      )

      convs.append(h_hat)
      convs.append(h)

      layer.insert(row, convs)

    return layer

  def create_MSDNet(self):
    """
    Create the MSDNet architecture.

    Returns
    -------
        list
            returns a grid containing the entire structure of the MSDNet
    """

    grid = self.make_initial_layer()
    self.create_architecture()

  def create_classifiers(self):
    """
    Return the layer containing the classifiers. In MSDNet, classifiers are stored,
    in a one dimensional ModuleList which we will refer to as the 'classifier_list'.
    Suppose MSDNet has a depth of 4 then 'classifier_list' is initialised and has
    the following structure,
        classifier_list = [None, None, None, None]
    So, initially, 'classifier_list' is filled with 'None' placeholders. Classifiers
    are added to 'classifier_list' at the indexes corresponding to to the indexes in
    'self.classifier_indexes'. Suppose that
        self.classifier_indexes = [1,3].
    After 'classifier_list' has been filled with classifiers it has the following
    structure:
        classifier_list = [None, classifier, None, classifier]
    The list 'classifier_list' is returned

    returns
    -------
        ModuleList
            This is a 'classifier_list' defined above. So, this is 'ModuleList' which
            has classifiers in the indexes corresponding to the indexes in 'self.classifier_indexes'.
    """
    total_depth =len(self.fm_grid[0])
    classifier_list = nn.ModuleList([None for i in range(total_depth)])

    for i in self.classifier_indexes:
      num_feature_maps = self.fm_grid[-1][i]
      classifier = self.create_classifier(num_feature_maps)
      classifier_list[i] = classifier

    return classifier_list

  def forward(self, x):
    """
    forward propagate a input x through MSDNet.

    inputs
    ------
        x
            this is an input which is to be put through the network
    """

    iterator = MSDNetIterator(self.ns, len(self.fm_grid[0]))
    indexes = iterator.next_iter()
    outputs = self.seed(x, indexes)
    pred_list = []

    if self.training == True:
      while(iterator.end == False):
        indexes = iterator.next_iter()
        outputs = self.through(outputs, indexes)
        pred = self.classify(outputs, indexes)

        if(isinstance(pred, str) and pred == self.NO_CLASSIFIER):
          continue

        pred_list.append(pred)

      #averaged_output = pred_list[0]

      #for i in range(1, len(pred_list)):
      #  averaged_output += pred_list[i]

      #averaged_output /= len(pred_list)

      return pred_list

    else:
       if self.evaluation_type == 'threshold':
        while(iterator.end == False):
          indexes = iterator.next_iter()
          outputs = self.through(outputs, indexes)
          pred = self.classify(outputs, indexes)

          if(isinstance(pred, str) and pred == self.NO_CLASSIFIER):
            continue

          pred = F.softmax(pred) # get the output of the last classifier

          max_acc = pred.max().item() # maximum output accuracy

          if max_acc > self.threshold:
            return pred
        return pred

       else:
          """
          evaluate input x. return the output of the network at each classifier
          """

          pred_list = []
          while(iterator.end == False):
            indexes = iterator.next_iter()
            outputs = self.through(outputs, indexes)
            pred = self.classify(outputs, indexes)

            if(isinstance(pred, str) and pred == self.NO_CLASSIFIER):
              continue

            pred_list.append(pred)
          if self.evaluation_type == 'all':
            return pred_list
          elif self.evaluation_type == 'end':
            return pred_list[-1]

  def through(self, x, indexes):
    for i in indexes:
      row = i[0]
      col = i[1]
      mod_list = self.grid[row][col]

      if mod_list == None:
        pass
      else:
        if len(mod_list) == 2: # 'mod_list' contains both 'h_hat' and 'h'
          up = x[row - 1]
          left = x[row]
          old_left = x[row]

          h_hat = mod_list[0]
          h = mod_list[1]

          for operation in h_hat:
            up = F.relu(operation(up))

          for operation in h:
            left = F.relu(operation(left))

          out = torch.cat((old_left, up, left), 1)
          x[row] = out
        else: # 'mod_list' contains only 'h' or a 'transition'
          h = mod_list[0]
          if len(h) == 1: # 'h' is a transition convolution
            left = x[row]

            for operation in h:
              left = operation(left)
            x[row] = left
          else: # 'h' is a regular 'h' convolution
            left = x[row]
            old_left = x[row]

            for operation in h:
              left = F.relu(operation(left))
            out = torch.cat((old_left, left), 1)
            x[row] = out

    return x



  def seed(self, x, indexes):
    """
    Seed the input x into 'self.ns' number of scales.

    parameters
    ----------
        indexes
            This is a list of the indexes of the initial layer (the layer at depth 0).
        x
            This is the input we wish to seed

    returns
    -------
        list
            A list containing the input 'x' at 'self.ns' different scales is returned
    """
    outputs = []

    for index in indexes:
      row = index[0]
      col = index[1]
      block = self.grid[row][col]
      for operation in block:
        x = operation(x)
        x = F.relu(x)
      outputs.append(x)
    return outputs

  def create_classifier(self, in_channels):
    """
    Create the classifier defined in apendix A fot the CIFAR 10 dataset. The operations in
    the classifier are stored in a 'ModuleList'. The classifier has the following structure

        Conv2d -> Conv2d -> AvgPool2d -> Linear

    parameters
    ----------
      in_channels
        this is the number of channels that are input into the classifier

    returns
    -------
      ModuleList
        a ModuleList is returned which contains the operations described above in the
        order described above. So, the returned module list looks like thus:
          [Conv2d, Conv2d, AvgPool2d, Linear]
    """
    classifier = nn.ModuleList([
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(3,3),
            stride=2,
            padding=1
        ),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=2,
            padding=1
        ),
        nn.AvgPool2d(
            kernel_size=2,
        ),
        nn.Linear(
            in_features=128,#128 * 7 * 7,
            out_features=self.n_classes
        )
    ])
    return classifier

  def classify(self, x, indexes, test=False):
    """
    Classify an input x using the classifier in 'self.classifiers' at index 'classifier_index'.
    This method is purely to make the classification process easier.

    When 'test' is true this os ONLY FOR DEBUGGING and will be romoved soon.

    parameters
    ----------
        x
            This is torch tensor and it contains the fature maps that are to be input into out classifier
        classifier_index
            This is an integer and is used to get the classifer from 'self.classifiers' at index 'classifier_index'

    returns
    -------
        tensor
            This method returns a tensor of the predictions of the input 'x'.
        string
            If the is no classifier in column 'indexes[-1][1]' then 'no classifier' is returned
    """

    if test == False:
      final_row = indexes[-1]
      classifier_index = final_row[1]
      if classifier_index not in self.classifier_indexes:
        return self.NO_CLASSIFIER
      else:
        x = x[-1] # get the coarsest feature maps
        classifier = self.classifiers[classifier_index]

        # perform the convolutions and average pooling
        for i in range(len(classifier) - 1):
          operation = classifier[i]
          x = F.relu(operation(x))

        linear_layer = classifier[-1]
        x = x.view(x.shape[0], -1) # flatten the feature maps
        x = linear_layer(x)
    else:
      classifier = self.test_classifier

      # perform the convolutions, and average pooling
      for i in range(len(classifier) - 1):
        operation = classifier[i]
        x = F.relu(operation(x))

      linear_layer = classifier[-1]
      x = x.view(x.shape[0], -1)
      x = linear_layer(x)
    return x

  def num_feature_maps(self, i, j):
    """
    Return the number of feature maps that will be produced at index '[i,j]' in 'self.grid'.

    parameters
    ----------
        i
            this is an integer which corresponds to the row
        j
            this is an integer which corresponds to the column

    returns
    -------
        integer
            the number of feature maps produced at index '[i, j]' in 'self.grid' is returned
    """
    return self.k0 * self.gr ** i + j * (2 * self.k0 * self.gr ** i)

  def create_transition(self, row, col):
    in_channels_ = self.fm_grid[row][col - 1]
    out_channels_ = self.fm_grid[row][col]
    conv = nn.Conv2d(
        in_channels = in_channels_,
        out_channels = out_channels_,
        kernel_size = (1,1),
        padding = 0,
        stride = 1
    )
    return nn.ModuleList([conv])

  def initialise_grid(self):
    total_depth = len(self.fm_grid[0])
    for row in range(self.ns):
      for col in range(total_depth - 1):
        self.grid[row].append(None)

  def create_architecture(self):
    self.initialise_grid()
    self.create_first_row()
    for row in range(1, self.ns):
      self.create_row(row)
    #self.create_block0()
    #for i in range(1, len(self.block_depths)):
    #  self.create_block_(i)
    #for block_number in range(len(self.block_depths)):
    #  self.create_block(block_number)

  def create_first_row(self):
    total_depth = len(self.fm_grid[0])
    transition_index = self.block_depths[0]
    for col in range(1, total_depth):
      if self.fm_grid[0][col] == None:
        pass
      else:
        if col == transition_index:
          left_channels_out = self.fm_grid[0][col - 1]
          transition = self.create_transition(0, col)
          self.grid[0][col] = nn.ModuleList([transition])
        else:
          left_channels_out = self.fm_grid[0][col - 1]
          h = self.h(
              n_channels_in = left_channels_out,
              scale_index = 0
          )
          self.grid[0][col] = nn.ModuleList([h])


  def create_row(self, row):
    total_depth = len(self.fm_grid[row])
    # find the indexes of the transitions
    transition_indexes = []
    transition_index = 0
    for i in range(len(self.block_depths) - 1):
      depth = self.block_depths[i]
      transition_index += depth
      transition_indexes.append(transition_index)

    for col in range(1, total_depth):
      if self.fm_grid[row][col] == None:
        return
      if col in transition_indexes:
        transition = self.create_transition(row, col)
        self.grid[row][col] = nn.ModuleList([transition])
      elif self.fm_grid[row - 1][col - 1] == None:
        left = self.fm_grid[row][col - 1]
        h = self.h(
            n_channels_in = left,
            scale_index = row
        )
        self.grid[row][col] = nn.ModuleList([h])
      else:
        left = self.fm_grid[row][col - 1]
        up = self.fm_grid[row - 1][col - 1]
        h = self.h(
            n_channels_in = left,
            scale_index = row
        )
        h_hat = self.h_hat(
            n_channels_in = up,
            scale_index = row
        )
        self.grid[row][col] = nn.ModuleList([h_hat, h])

  def print_fm_grid(self):
    str_grid = [[] for i in range(self.ns)]

    for row in range(len(self.fm_grid)):
      for col in range(len(self.fm_grid[0])):
        str_grid[row].append(str(self.fm_grid[row][col]))

    m = len(max(max(str_grid)))
    for row in range(len(str_grid)):
      for col in range(len(str_grid[0])):
        word = str_grid[row][col]
        while len(word) < m:
          word += ' '
        str_grid[row][col] = word

    for row in range(len(str_grid)):
      string = ''
      for col in range(len(str_grid[0])):
        string += ' ' + str_grid[row][col] + ' '
      print(string)

  def print_network(self):
    """
    Print the structure of the network in a readable form. This method is purely for debugging purposes
    """
    str_grid = [[] for i in range(self.ns)]
    for i in range(self.ns):
      for j in range(self.ns):
        string = str(self.grid[i][j])
        str_grid[i].append(string)

    # find string with max length in each column
    m = [0 for i in range(self.ns)]
    for col in range(self.ns):
      for row in range(self.ns):
        item = str_grid[row][col]
        if len(item) > m[col]:
          m[col] = len(item)

    # pad strings:
    for col in range(self.ns):
      for row in range(self.ns):
        item = str_grid[row][col]
        while len(item) < m[col]:
          item += ' '
        item += '   '
        str_grid[row][col] = item

    # print grid
    for row in range(self.ns):
      w = ''
      for col in range(self.ns):
        w += str_grid[row][col]
      print(w)


  def print_network2(self):
    for i in range(self.ns):
      print("################## New Row ##############")
      for j in range(self.ns):
        print(' ')
        print(self.grid[i][j])

      print(" ")
    print(type(self.grid[0]))


class AverageCrossEntropyLoss(nn.Module):
  def __init__(self):
    super(AverageCrossEntropyLoss, self).__init__()

  def forward(self, outputs, labels):

    total_loss = F.cross_entropy(outputs[0], labels)

    for i in range(len(outputs)):
      output = outputs[i]
      loss = F.cross_entropy(output, labels)
      total_loss += loss
    return total_loss / len(outputs)
    
"""
Created a function that returns a sampler for the training and validation set. Method inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb .

We also perform the data augmentation mentioned in the MSDNet paper in this function.

We normalise the images by a mean and standard deviation and we augment the data using a random centre crop with padding 4. We also flip the images horizontally with a probability of 0.5.
"""
    
############# CREATE DATA ####################
# convert each image to tensor format
import random
from torch.utils.data.sampler import SubsetRandomSampler

def train_val_sampler(trainset, valset, split):
  random.seed(10)
  size = len(trainset)
  indexes = list(range(size))
  random.shuffle(indexes)

  train_indexes = indexes[0:split]
  val_indexes = indexes[split:]

  train_sampler = SubsetRandomSampler(train_indexes)
  val_sampler = SubsetRandomSampler(val_indexes)

  return train_sampler, val_sampler

normalise = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                 std=[0.2471, 0.2435, 0.2616])

train_transform = transforms.Compose([  # convert to tensor
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalise
])

test_val_transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    normalise
])

# load data
trainset = CIFAR10(".", train=True, download=True, transform=train_transform)
valset = CIFAR10(".", train=True, download=True, transform=test_val_transform)
# testset = CIFAR10(".", train=False, download=True, transform=test_val_transform)

# get the samplers
train_sampler, val_sampler = train_val_sampler(trainset, valset, 45000)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, sampler = train_sampler)
valloader = DataLoader(valset, batch_size=128, sampler = val_sampler)
# testloader = DataLoader(testset, batch_size=128, shuffle=True)

def evaluate_model(model, test_loader, evaluation_type='all'):
  """
  Evaluate the model on the validation set.
  """
  model.evaluation_type = 'all'
  num_classifiers = len(model.classifier_indexes)
  correct = [0 for i in range(num_classifiers)]
  total = [0 for i in range(num_classifiers)]

  with torch.no_grad():
      testloader = tqdm(test_loader)
      for data in testloader:
          images, labels = data
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)

          for i in range(len(outputs)):
            classifier = outputs[i]
            _, predicted = torch.max(classifier.data, 1)

            total[i] += labels.size(0)
            correct[i] += (predicted == labels).sum().item()

  correct = [100 * correct[i] / total[i] for i in range(len(total))]
  return correct

from tqdm.auto import tqdm
import copy

model = MSDNet3(
    n_channels_in=3,
    n_classes=10,
    num_scales=3,
    scale_factor=2,
    k0=6,
    growth_rate=2,
    dimensions=(28,28),
    classifier_indexes=[4,6,8,11,13,15,17,20,22,24,26],
    block_depths=[8,8,8],
    evaluation_type = 'all'
)

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model.to(device)

epoch_range = 300
csv_data = pd.DataFrame(columns=['epoch', 'lr', 'train_acc', 'val_acc','average_train_acc', 'average_val_acc'])


crit = AverageCrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[149,224], gamma=0.1)
best_acc = 0

for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    train_loader = tqdm(trainloader)
    running_loss = 0.0

    for param_group in optimiser.param_groups:
        lr = param_group['lr']

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()


        outputs = model(inputs)
        loss = crit(outputs, labels)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        train_loader.set_postfix(epoch=epoch, lr = lr)
    scheduler.step()

    train_acc = evaluate_model(model, trainloader)
    val_acc = evaluate_model(model, valloader)
    average_train_acc = sum(train_acc) / len(train_acc)
    average_val_acc = sum(val_acc) / len(val_acc)

    # update the best model
    if average_val_acc > best_acc:
      best_model = copy.deepcopy(model)
      best_acc = average_val_acc
      best_epoch = epoch

    results = {
        'epoch' : epoch,
        'lr' : lr,
        'train_acc' : train_acc,
        'val_acc' : val_acc,
        'average_train_acc' : average_train_acc,
        'average_val_acc' : average_val_acc
    }

    csv_data = csv_data.append(results, ignore_index=True)

best_data = pd.DataFrame({
    'best_epoch' : [best_epoch],
    'best_val_acc' : [best_acc]
})

print('Finished Training')

try:
    torch.save(model.state_dict(), "MSDNET.weights")
except:
    pass
try:
    torch.save(best_model.state_dict(), "best_MSDNET.weights")
except:
    pass
try:
    csv_data.to_csv('csv_data.csv', index=False)
except:
    pass
try:
    best_data.to_csv('best_data.csv', index=False)
except:
    pass

