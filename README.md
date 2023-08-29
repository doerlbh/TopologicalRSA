# Topological RSA



Code for our paper: 

**"The Topology and Geometry of Neural Representations"** 

by [Baihan Lin](https://www.neuroinference.com/) (Columbia), [Nikolaus Kriegeskorte](https://scholar.google.com/citations?user=w6M4YN4AAAAJ&hl=en&oi=sra) (Columbia).



For the latest full paper: 



All the experimental results can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.



**Abstract**

A central question for neuroscience is how to characterize brain representations of perceptual and cognitive content. An ideal characterization should distinguish different functional regions with robustness to noise and idiosyncrasies of individual brains that do not correspond to computational differences. Previous studies have characterized brain representations by their representational geometry, which is defined by the representational dissimilarity matrix (RDM), a summary statistic that abstracts from the roles of individual neurons (or responses channels) and characterizes the discriminability of stimuli. Here we explore a further step of abstraction: from the geometry to the topology of brain representations. We propose topological representational similarity analysis (tRSA), an extension of representational similarity analysis (RSA) that uses a family of geo-topological summary statistics that generalizes the RDM to characterize the topology while de-emphasizing the geometry. We evaluate this new family of statistics in terms of the sensitivity and specificity for model selection using both simulations and functional MRI (fMRI) data. In the simulations, the ground truth is a data-generating layer representation in a neural network model and the models are the same and other layers in different model instances (trained from different random seeds). In fMRI, the ground truth is a visual area and the models are the same and other areas measured in different subjects. Results show that topology-sensitive characterizations of population codes are robust to noise and interindividual variability and maintain excellent sensitivity to the unique representational signatures of different neural network layers and brain regions.


**Note:** This repository contains early benchmarking codes that generated our empirical results. We are still cleaning up this repository for easy access and run instructions. Stay tuned.

 


## Info

Language: Python 3.9


Platform: MacOS, Linux, Windows

by Baihan Lin, 2023




## Citation

If you find this work helpful, please try the models out and cite our works. Thanks!

    @article{lin2023,
      title={The Topology and Geometry of Neural Representations},
      author={Lin, Baihan and Kriegeskorte, Nikolaus},
      journal={arXiv preprint arXiv:},
      year={2023}
    }




