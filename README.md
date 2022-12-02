# Distributed Training of Neural Models with Multimodal Input

This project is a proposal for using parallelization in training deep learning models that have multimodal data as inputs focusing on parallelizing the model used for inferring meaningful semantical embeddings from GUI screens. 

## 1. Programming: Getting familiar with programming with Horovod for more basic models

This part of the project was the prerequisite for the proposed project to get familiar with using distributed machine learning frameworks. It mostly focuses on Horovod distributed training framework that uses the ring reduce architecture for communincation between nodes (shown in the figure below), and NCCL library for communication. 

<p align="center">
<img width="660" alt="Screen Shot 2022-12-01 at 2 56 43 PM" src="https://user-images.githubusercontent.com/46067584/205176682-3e34a2fb-d578-4eb8-979f-7edef8db1337.png">
</p>
 

<p align="center">
<b>Fig.1 - Ring-Reduce Architecture</b>
</p>

I focused on parallelizing the training of basic neural models using well-known datasets such as [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). I also used Spark's DataBricks on Microsoft Azure for this part of the project. The results for training time per epoch count will be added here with the training notebooks' codes once they are finalized.
 
## 2. Problem Description
The role of machine learning and deep learning in the recent developments in many different domains and aspects of science is completely undeniable. Also, as the volume of the data that needs to be processed increases and the models themselves get more complicated, a higher computing power is required for training deep neural models. This has resulted in deep learning, high performance computing, and distributed system researches to focus on proposing methods for parallelizing the training of neural networks and developing new frameworks distributed machine learning.

Many different distributed training frameworks such as Horovod, MXNet, Parallax, etc,. have been proposed in the recent years that has shown to be scaling well for problems in the domain of image classification and natural language processing (NLP). Based on the previous studies, frameworks such as MXNet and Horovod works better with models that focus on image classification tasks, while Parallax works best with models that focus on NLP tasks, but to the best of my knowledge no research work has focused on assessing how different distributed training architectures will affect the training of neural models with multimodal outputs. Distribution training for this type of models are exteremely important for two reasons: 1) These type of models have more complex input data for which the parallelisation in training is more necessary; 2) Most of the data that exists in the word around us is type of multimodal data. 

In this project we focus on assesing how different distributed training architectures will affect the training speed of a network with multimodal input. In order to do so, we parallelize the Screen2Vec model, which we call P-Screen2Vec, which assesses the effect of using HPC and parallelism for training multimodal neural models to infer meaningful semantic embeddings from GUI screens and components. Being able to infer high-quality semantical representations for GUI screens and components is highly crucial to data-driven computational methods for modeling Human Computer Interactions which have numerous applications in accessibility improvement and automated testing. The following image shows an abstract representation of the model responsible for inferring from [this](https://arxiv.org/pdf/2101.11103.pdf) paper:

<img width="1189" alt="Screen Shot 2022-09-01 at 11 30 51 AM" src="https://user-images.githubusercontent.com/46067584/187987201-f31b869a-677d-408a-bc85-0fedab72ad5d.png">

However, since the existing dataset of the [GUI screens](https://interactionmining.org/rico) is relatively big (~70K GUI screens) and each data point contains complex tree-like hierarchies it takes a lot of computational power to properly train the models for this task.

## Specific Objectives
- Asses the possibility of parallelizing neural model with multimodal inputs with existing distributed training solutions.
- Asses if state of the art distributed training methods scale up as well as their reported results for simpler networks on neural models with multimodal inputs.  
- Design and implement a hybrid architecture for multimodal networks that uses the optimal architecture for each type of input variable. 
- Compare the performance and scalability of the new architecture with the already existing solutions on both multimodal and simpler neural models.

## Strategies and Algorithms
The first step of the solution would be to parallize the existing models with three existing architecures using a variaty of worker nodes and gather the training statistics during the experiments:
- a) Ring Reduce Architecture
- b) Basic All-Reduce Architecture
- c) Parameter Server Architecture
<img width="1100" alt="Screen Shot 2022-12-01 at 4 02 44 PM" src="https://user-images.githubusercontent.com/46067584/205184611-d87b142e-f0b2-4c82-b4ac-dafd0de5450c.png">

Analyze the gathered data to see if the existing solutions are able to scale up as well as their reported results for simpler networks for neural models with multimodal inputs. The results should be reported.

My hypothesis is that it should be possible to propose a new hybrid architecture that handles different types of input data using different architectures. I plan to use the all reduce architecture for the part handling the parts of the networks that handle image-based, and graph-based data and parameter server architecture for the parts of model that handles the textual data that consists of sparse variable. The hybrid model based on this hypothesis should be implemented and and the results should be compared with current state of the art techiniques for both performance and scalability.


## Expected Results
We will be evaluating every stage both based on training speed and scalability, and we expect to see significant improvement in training speed compared to the not paralellized version. At this point, we expect to see non-optimal scalability results for models with multimodal input using the existing state of the art solutions and we will try to improve both the training speed and performance by using our proposed hybrid architecture. 

## References

> Toby Jia-Jun Li*, Lindsay Popowski*, Tom M. Mitchell, and Brad A. Myers. [Screen2Vec: Semantic Embedding of GUI Screens and GUI Components](http://toby.li/files/li-screen2vec-chi2021.pdf). Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI 2021).





