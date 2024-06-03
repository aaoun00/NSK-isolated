# README

This is a prototype for an across-session unit matcher. It contains five modules; each with functions pertaining to data at increasingly large scales.

The `waveform` module concerns itself with the smallest unit of data; the single channel waveform. This module has functions for extracting features from waveforms and conveying them to the next module. The `spike` module is the next smallest unit, which is constituted by several waveform channels. The `spike` module extracts two spike level features: (1) spatial source localization and (2) a dimensionally reduced representation of the other features extracted from the waveforms. Next, the `unit` module contains functions for applying the spike-level feature extraction functions to each spike in pre-sorted unit spike-trains and for assessing the pair-wise similarity among unit feature distributions. The `session` module concerns itself with the largest unit of data---the recording sessions. This module has a function for iterating through all the across-session unit pairings and computing their respective Jensen-Shannon distances. Moreover, it has a function finding the best matches among the units. Finally, the `main` module contains a script that reads Axona formatted tetrode and cut files, applies the `session` functions to the data and writes a new cut file with the units in the second session re-ordered to match the numbering of the units to which they were paired from the first session.

Most of the features extracted at the waveform level are inspired by Caro-Martin et. al (2018).

Single unit tetrode source localization is based on Chelaru and Jog (2005).

Comparisons between units are computed using a multivariate version of the Jensen-Shannen distance that is implemented by directly estimating a multivariate version of the Kullback-Leibler divergence using the method outlined in Pérez-Cruz (2008) and informed by a discussion on a related [Gist GitHub thread](https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518)


## Modules

### `main`

The `main` module contains the main function for the prototype.
It reads the data and feeds it into `session` level functions.
It computes matches across sessions.
It then re-codes the spike trains.
It then writes the re-coded spike trains to cut files.

### `session`

The `session` module contains a three kinds of functions.
1. The first kind is for enumerating all the across-session pairs and computing distributional divergences between them. There is only one of these functions because divergence scores are aggregated into one score by the `aggregated_divergence`.
2. The second kind is for sorting possible matches by similarity in order to discover the best possible matches.
    - `extract_full_matches` extracts vectors from the bipartite divergence matrix whose minimum divergences match both ways.
    - `guess_remaining_matches` takes the smaller bipartite divergence matrix and finds the matches such that the sum of the scores of all the matches is minimized. Whenever there is ambiguity (multiple solutions), it chooses the best match for the member of the smaller side of the graph.
3. The third kind is for iterating over more than two sessions and finding matches across three or more sessions.


### `unit`

The `unit` module contains three kinds of functions.
1. The first kind is for extracting spike-level features for each spike in a unit. There is only one of these functions and it is called `extract_unit_features`. It iterates over all the spikes in a unit and yields the output from the `extract_spike_features` function detailed below.
2. The second kind is for converting feature distributions to probability distribution function estimates.
3. The third kind is for comparing the distributions of features between two units. The Jensen-Shannon distance is used for comparing distributions. (The Wasserstein / Earth Mover's Distance is sometimes used for comparing non-normalized feature vectors?)
4. The fourth kind is for packaging all the similarity measures into a single divergence score. There is only one of these functions (`aggregated_divergence`). It will be called in the `session` module to compare the units in two sessions.


### `spike`

The `spike` module contains three kinds of functions.
1. The first kind is for extracting waveform level features for each channel in a spike.
2. The second kind is for extracting features from the n-dimensional waveform level features of each spike. This kind of function takes n-dimensional features from the first kind of function and returns extracted features. There are two of these functions.
    - `estimate_source_location` is a function for extracting location estimates from tetrode peak amplitudes. It is loosely based on @ChelaruJog2005.
    - `reduce_dimensionality` is a function that takes the remaining features and reduces the dimensionality to a reasonable size (currently uses Principle Componant Analysis)
3. The third kind collects the relevant features from both of the above functions and bundles them into a single data structure. There is only one of these functions and it is called `extract_spike_features`. This is a wrapper function that will be called in the `unit` module to extract features from each spike in a unit.


### `waveform`

The `waveform` contains functions for extracting features from each waveform.
There are three domains from which features can be extracted.
1. The time domain (amplitude vs time).
2. The derivative domain (derivative of amplitude vs time).
3. The phase domain (derivative vs second derivative).
There are two functions associated with the above three domains. One to convert a waveform to the derivative domain (`convert_to_derivative`). The second is to convert a waveform to the phase domain (`convert_to_phase`).

Thera are then functions for extracting features from each domain.

- local and global extrema / peaks and valleys
- `quartiles` and `interquartile_range` are for extracting quartiles and the interquartile range from a waveform.
- `mean` and `variance` are for extracting the mean and variance from a waveform.
- `skewness` and `kurtosis` are for extracting the skewness and kurtosis from a waveform.

There is a function for organizing all the waveform features into a single data structure. This function will be called in the `spike` module to extract features from each waveform in a spike.

## Sources

Caro-Martín, Carmen Rocío, José M. Delgado-García, Agnès Gruart, and R. Sánchez-Campusano. “Spike Sorting Based on Shape, Phase, and Distribution Features, and K-TOPS Clustering with Validity and Error Indices.” Scientific Reports 8, no. 1 (December 12, 2018): 17796. https://doi.org/10.1038/s41598-018-35491-4.

Chelaru, Mircea I., and Mandar S. Jog. “Spike Source Localization with Tetrodes.” Journal of Neuroscience Methods 142, no. 2 (March 30, 2005): 305–15. https://doi.org/10.1016/j.jneumeth.2004.09.004.

Pérez-Cruz, Fernando. “Kullback-Leibler Divergence Estimation of Continuous Distributions.” In 2008 IEEE International Symposium on Information Theory, 1666–70. IEEE, 2008. https://doi.org/10.1109/ISIT.2008.4595271.
- supplementary discussion: https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
