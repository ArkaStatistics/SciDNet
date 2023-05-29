# SciDNet# SciDNet - Screening and Cleaning Incorporated Deep Neural Network

SciDNet is a method that combines screening and cleaning techniques in a deep neural network framework to address the challenges of high-dimensional feature selection and multicollinearity in statistical modeling.

## Overview

The SciDNet method consists of the following steps:

1. Screening: This step reduces the dimensionality of the feature space by excluding the majority of null features using Hense-Zirkler tests. It focuses on identifying relevant features for further analysis.

2. Clustering: To handle excessive multicollinearity among the remaining predictors, SciDNet clusters them by estimating the precision matrix using non-paranormal transformation. This helps in identifying groups of related predictors.

3. Cleaning: After the screening and clustering steps, SciDNet further cleans out the null features by employing a sparsity-inducing deep neural network. This step helps to refine the selection of relevant features while controlling the associated False Discovery Rate (FDR) through resampling techniques.

## Repository Contents

This repository contains the basic building blocks of the SciDNet code. The code provides the necessary functionality to implement the screening, clustering, and cleaning steps of the SciDNet method.

