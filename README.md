# Geographic Aware Expert


## Introduction
This repo contains all the code necessary to train a model on the pretext task of "geographic awareness". This means the model learns geo-localisation, seasonality and identify climate zones. It is trained in a supervised approach on coordinate labels, climate zone lables and day of the year labels.

## Data
Models are trained on the minifoundation dataset. It consists of more than 100 tiles all around the world at different capture times. The labels used to train are the Koppen-Geiger climate zones, coordinates and day of year the data was captured. Models are trained on the geo-label which are of shape (31 + 3 + 2) referring to the 31 Koppen-Geiger climate classes, 3 coordinate labels (latitude, encoded sine of longitude and encoded cosine of longitude). 

More details on dataset and labels can be found here https://github.com/LuytsA/phileo-dataset or in config_geography.py

## Geographic aware loss

The geographic aware loss function (utils.training_utils.GeographicalLoss) is a weighted average of three seperate losses. In some images only sea is visible and since it is extremely hard to localise a patch of sea or predict the day of the year such a image was taken, the loss function calculates coordinate loss and time of year loss only for non-sea patches.

### Coordinate loss
1. MSELoss: 

This is a bit of a crude approach but it works ok. Minimises the squared error on encoded latitude and encoded longitude.

2. MvMFLoss: 

Based on [1].

Loss function that exploits spherical geometry of the Earth.
The loss is based on the von Mises-Fisher (vMF) distribution which is the spherical analogue of a Gaussian.

$$\text{vMF}(\textbf{y}, \textbf{center}, \text{density}) = \dfrac{\text{density}}{\text{sinh(density)}}  \exp( \text{density} \cdot \textbf{center}^\top \textbf{y})$$
with $\textbf{center},\textbf{y} \in \mathbb{S}^2, \text{density} \in \mathbb{R}^+$

The mixture distribution (MvMF) for a point y on the sphere is a convex combination of vMFs where each vMF is characterised by its own center and density. 

$$\text{MvMF}(\textbf{y}, \text{centers}, \text{densities}, \Gamma) = \sum_i \Gamma_i \cdot \text{vMF}(\textbf{y}, \text{center}_i, \text{density}_i)$$
These mixture weights $\Gamma_i(\textbf{x},\text{W})$ are the outputs of the model and depend on the input features $\textbf{x}$ and the weights of the model $\text{W}$.
They can be interpreted as logits to which cluster (class) the input image belongs. After softmaxing $\exp(\Gamma_j) / \sum_i \exp(\Gamma_i)$ represents the chance of $\textbf{x}$ belonging to cluser $j$ located at $\text{center}_j$ with spread $ \text{density}_j$.

The loss itself is the negative logarithm of the distribution: 
$$\text{loss} = - \log(\text{MvMF})$$

The use this loss it must be initialised by providing centers and densities. These centers and densities can be taken to be parameters that are optimised during training. For example when the training sets contains images from 200 locations around the world, it makes sense to sample from each location 1-5 centers resulting in a total of 200-1000 centers around the world. As described in [1] different density correspond to sensitivity at certain scales from $\text{density}=\exp 4 \sim $ 2500km to $\text{density}=\exp 14 \sim $ 25km. At the global scale this model is localising Sentinel-2 tiles it seems reasonable to initialise centers around $\exp 7$ and optimise the densities during training.

[1] Izbicki, Michael et al. “Exploiting the Earth's Spherical Geometry to Geolocate Images.” ECML/PKDD (2019).

### Koppen-Geiger climate loss

A global map of Koppen-Geiger climate zones is available at https://koeppen-geiger.vu-wien.ac.at/present.htm.
Climate zones are hierarchical meaning multiple zones might belong to the same class e.g. the zones Tropical rainforest and Tropical monsoon both belong to the class Tropical.

The loss function for climate zones consists of two times CrossentropyLoss. One tries to correctly classify the finegrained climate zone while to other only cares about the more coarse climate classes. Climate loss is a weighted average of both.

### Date loss
Date loss is the most simple of the losses. It is an MSELoss on the encoded day of the year.

## notes
- The labels contain encoded coordinates and encoded day of year instead of standard lat-long or days. Also the loss functions expect encoded values. To decode/encode values please take a look at the decode/encode functions in utils.visualisations.

- When the data scales up to $10^3-10^6$ locations or even more it is not necesarrily clear how the initalisation centers need to be chosen. In the original paper they use up to $2^{15}$ classes so maybe the amount of centers can be very high.
