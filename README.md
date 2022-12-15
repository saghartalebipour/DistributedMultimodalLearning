# Distributed Training of Neural Models with Multimodal Input

This page contains a proposal for a project on using parallelization in training deep learning models that have multimodal data as inputs.
<p align="center">
 <img width="500" alt="Screen Shot 2022-12-14 at 3 33 11 PM" src="https://user-images.githubusercontent.com/46067584/207737996-bb82b418-687b-4d0c-a0e5-e4b0c52aa8e4.png">
</p>

## Problem Description
Machine learning has profoundly impacted recent advancements in many scientific domains. Furthermore, we are witnessing an explosion in the amount of raw data available to us. This, in tandem with the sheer single-core processor computing power plateauing (Moore’s law), has led researchers in the domains of high-performance computing, distributed systems, and deep learning to start focusing on proposing methods for parallelizing the training of deep neural networks (DNNs) and developing new frameworks for distributed machine learning. This entails training a model across multiple machines or devices, unlike traditional machine learning, where the model is trained on a single machine using a single processor. Distributed machine learning has several advantages over conventional machine learning, including Improved scalability, faster training times, and improved reliability.

A recent breakthrough in the area of machine learning is DNNs with multimodal inputs. This category of machine learning models can simultaneously process multiple types of input data. This is in direct contrast with traditional machine learning models, which are typically designed to process only a single type of input data. A DNN model with multimodal inputs might be trained to process both text and images as input and has proven to be effective in various tasks such as natural language processing, computer vision, and speech recognition. Multimodal neural models have several advantages over traditional models. They can provide a more complete and accurate representation of the input data, as they can incorporate multiple types of information. This can improve the model's performance on a wide range of tasks as they can handle multiple types of input data simultaneously.

Previous research work in the area of distributed machine learning has resulted in various state-of-the-art frameworks, such as Horovod, MXNet, and Parallax, that leverage different topologies and architectures for parallelization, such as all-reduce, ring-reduce, and parameter server. These frameworks have shown to be highly scalable for models that focus either on image classification tasks (Horovod, MXNet) or natural language processing tasks (Parallax). While these frameworks have been promising, to the best of my knowledge, there is a dearth of work focusing on assessing how different distributed training architectures will affect the training of neural models with multimodal inputs.

The ability to distribute the training for this type of models are essential for three reasons: (1) these type of models have more complex input data for which the parallelization in training is crucial; (2) most of the data that exists in the world around us is of the multimodal data type; and (3) the amount of training data is larger due to compiling multiple modalities into a single DNN. Therefore, in our proposed project, we aim to first asses the effect of different architectures in training neural models with multimodal inputs and later design and implement a hybrid model specifically designed for parallelization of the training process for neural models with various types of inputs.

## Project Objectives
The following shows the objectives of the proposed research project:

- Assessing the possibility of parallelizing neural models with multimodal inputs with existing distributed training solutions.
- Assessing if the state-of-the-art distributed training methods can scale up as well as their reported results for simpler networks on neural models with multimodal inputs.
- Design and implement a hybrid architecture for multimodal networks that uses the optimal architecture for each type of input variable.
- Compare the performance and scalability of the new architecture with the already existing solutions on both multimodal and simpler neural models.

## Methodology
The main idea behind the proposed research is to investigate the use of the existing distributed training techniques and architectures for neural models with multimodal inputs and to present a specific solution focusing on distributed training of this type of neural models. The detailed plan is as follows:

1. We will conduct a literature review to identify the most relevant distributed training techniques and architectures.
We should identify a set of neural models with multimodal inputs and their corresponding datasets as benchmarks. An example of these models can be found in the “Example Model and Dataset” section of the proposal.
2. We will attempt to parallelize the training of the chosen models on the previous step with the identified architectures in step 1 using various numbers of worker nodes (CPUs and GPUs), and gather the training statistics during the experiments. These architectures include but are not limited to:
    1. Ring Reduce Architecture
    2. All-Reduce Architecture
    3. Parameter Server Architecture
<img width="1100" alt="Screen Shot 2022-12-01 at 4 02 44 PM" src="https://user-images.githubusercontent.com/46067584/205184611-d87b142e-f0b2-4c82-b4ac-dafd0de5450c.png">

3. We will next analyze the gathered data from the previous step to see if the existing solutions are able to scale up as well as their reported results for simpler networks for neural models with multimodal inputs. In this step, we will also asses the effect of using different existing architectures based on performance speed up and scalability for distributed training of this type of network. The results will be reported. 

5. We hypothesize that it should be possible to propose a new hybrid architecture that handles parallelizing different parts of the model using different architectures based on the type and sparsity of the data used by that part of the model. The plan is to start by using the all-reduce architecture for the parts mostly handling the networks that handle image-based and graph-based data and parameter server architecture for the parts of the model that handles the textual data that consists of sparse variables, and again all-reduce architecture for the parts of the model that handle the data after the fusion. 5. This hybrid model should be designed and implemented at this stage.
6. We will repeat the same experiments in step 3 with the model designed and implemented in the previous step. We will then compare the results for the new hybrid model with the current state-of-the-art techniques for performance speed and scalability.


## Expected Results
We will evaluate every stage based on performance speed-up, training throughput, and scalability. We expect significantly improved training speed compared to the not parallelized version. We also expect non-optimal scalability results for models with multimodal input using the existing state-of-the-art solutions. We expect to improve the training speed, throughput, and scalability using the new hybrid architecture for distributed training of neural models with multimodal inputs that consider the input variable and sparsity. Distributed training of neural models with multimodal input is an important research area with many potential applications. The results of this study will provide valuable insights into the most effective techniques for distributed training and their trade-offs. These insights will be useful for researchers and practitioners in machine learning, HCI, and distributed systems.

## Example Model and Dataset
One of the computer science areas in which neural models with multimodal inputs are widely used is human-computer interaction (HCI), and more specifically, UI understanding. An example of these models used in this area is Screen2Vec which is responsible for inferring meaningful semantic embeddings from GUI screens and components. Being able to infer high-quality semantical representations for GUI screens and components is highly crucial to data-driven computational methods for modeling Human Computer Interactions, which have numerous applications in accessibility improvement and automated testing. The following image shows an abstract representation of the model responsible for inferring from [this](https://arxiv.org/pdf/2101.11103.pdf) paper:

<img width="1189" alt="Screen Shot 2022-09-01 at 11 30 51 AM" src="https://user-images.githubusercontent.com/46067584/187987201-f31b869a-677d-408a-bc85-0fedab72ad5d.png">

The dataset used for training this model is [RICO](https://interactionmining.org/rico) consisting of around ~70K GUI screens and user interaction sequences. Each data point contains complex tree-like hierarchies it takes a lot of computational power to properly train the models for this task.

## References
> Toby Jia-Jun Li*, Lindsay Popowski*, Tom M. Mitchell, and Brad A. Myers. [Screen2Vec: Semantic Embedding of GUI Screens and GUI Components](http://toby.li/files/li-screen2vec-chi2021.pdf). Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI 2021).

> Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols and Ranjitha Kumar. 2017. Rico: A Mobile App Dataset for Building Data-Driven Design Applications. In Proceedings of the 30th Annual Symposium on User Interface Software and Technology (UIST '17).

> Sergeev, Alexander, and Mike Del Balso. "Horovod: fast and easy distributed deep learning in TensorFlow." arXiv preprint arXiv:1802.05799 (2018).

> Chen, Tianqi, et al. "Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems." arXiv preprint arXiv:1512.01274 (2015). -->

> Jing Pan, Wendao Liu, Jing Zhou. 2020. Benchmark Tests of Convolutional Neural Network and Graph Convolu-tional Network on HorovodRunner Enabled Spark Clusters. In Proceedings of AAAI (AAAI Workshop Deep Learning on Graphs: Methodologies and Applications). AAAI, New York, New York, USA, 5 pages.
## Prerequisite - Getting familiar with one of the existing frameworks: Horovod
As a prerequisite for the proposed project I wanted to try a benchmark of one of the existing distributed machine learning frameworks to be able to see the performance speed up gained from these models. I chose Horovod, developed by Uber, which uses the ring reduce architecture (shown in the figure below), and NCCL library for communication. 

I tried the MNIST benchmark with a simple 2 layer CNN network with Horovod. For that I used the code from this [paper](https://arxiv.org/abs/2005.05510) and [Repository](https://github.com/psychologyphd/horovodRunnerBenchMark) and only very slightly changed it (the cleaned up version can be found on MNIST-Benchmark.py.) and ran it on 4 workers of type Standard_DS5_v2 on DataBricks on Microsoft Azure where each of them had 16 cores and 56GB memory for the not parallelized version, and the parallelized version with Horovod with 1, 2, and 4 CPUs and observed the results. I trained the model with batch size of 128, with base learning rate 0.1 (the base learning rate is scaled by the number of processes for the 2 and 4 CPU experiments) and for 10 epochs. Some of the observed results can be found on the table below.

| Experiment       | Total Run Time | Training Time | Average Epoch Time |  Validation Accuracy |   
| -----------------|:--------------:| --------------------------------------:|:------------------:|:--------------:|
| Not Parallalized | 3.77 min       | 196.422 sec                            | 21 sec             | 0.9918         |
| Horovod - 1 CPU  | 4.42 min       | 230.392 sec                            | 23.2 sec           | 0.9865         |
| Horovod - 2 CPUs | 2.82 min       | 136.196 sec                            | 13.35 sec          | 0.9861         |
| Horovod - 4 CPUs | 2.67 min       | 123.294 sec                            | 12.32 sec          | 0.9867         |
