# P-Screen2Vec: Parallelism in Training Neural Models for Inferring Semantic Embeddings from GUI Screens

P-Screen2Vec assesses the effect of using HPC and parallelism for training multimodal neural models to infer meaningful semantic embeddings from GUI screens and components. Being able to infer high-quality semantical representations for GUI screens and components is highly crucial to data-driven computational methods for modeling Human Computer Interactions which have numerous applications in accessibility improvement and automated testing. The following image shows an abstract representation of the model responsible for inferring from [this](https://arxiv.org/pdf/2101.11103.pdf) paper:

<img width="1189" alt="Screen Shot 2022-09-01 at 11 30 51 AM" src="https://user-images.githubusercontent.com/46067584/187987201-f31b869a-677d-408a-bc85-0fedab72ad5d.png">

However, since the existing dataset of the [GUI screens](https://interactionmining.org/rico) is relatively big (~70K GUI screens) and each data point contains complex tree-like hierarchies it takes a lot of computational power to properly train the models for this task. Therefore, in this project we decided to asses how parallism and HPC can help in improving the training process for this task. 

## Strategies and Algorithms
In this project we are going to use two strategies for using parallilsm in training the model:
1. Ensembles: With this strategy we will train multiple models at the same time in parallel, and during the time of inference we will use voting scheme to finalize only one decision. 
 
 <p align="center">
 <img width="550" alt="Screen Shot 2022-09-01 at 11 45 00 AM" src="https://user-images.githubusercontent.com/46067584/187989680-df5aed68-124b-43a0-b96f-b0f447bc70ab.png">
</p>

 
2. Distributed Training: With this strategy we are going to simoltaniously train one model using multiple machines where one of the machines acts as a parameter server. 
 
  <p align="center">
<img width="550" alt="Screen Shot 2022-09-01 at 12 02 27 PM" src="https://user-images.githubusercontent.com/46067584/187992456-247d30ee-cca3-4d3d-aa1e-63a11957e6ff.png">
</p>

## Expected Results
For both of the mentioned strategies above, we will be evaluating the training speed and training accuracy metrics. We are expecting to see improvement in training speed in the distributed training mode, as in this strategy we will be using several machines instead of one for training one model. We are expecting to see improvement in training accuracy in the ensemble method as for that starategy the final decision will be made based on the inference of multiple trained models.

## References

> Toby Jia-Jun Li*, Lindsay Popowski*, Tom M. Mitchell, and Brad A. Myers. [Screen2Vec: Semantic Embedding of GUI Screens and GUI Components](http://toby.li/files/li-screen2vec-chi2021.pdf). Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI 2021).





