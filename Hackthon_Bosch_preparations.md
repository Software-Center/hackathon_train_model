# Train your own model Hackathon @Bosch

## Goal
Generative AI and ChatGPT have taken software engineering with storm. They provide a lot of promises and they can help in many ways, maybe so many that we need to understand it better. We can use these models to generate test cases, we can use them to write text and we can use them to summarize text. The generally available tools like ChatGPT, Github CoPilot, GitLab CoPilot, Code Whisperer and Tabnine are already very good, but they are trained on data that is publicly available in open repositories. 

The goal of this Hackathon is to train (at least partially) a smaller model from scratch on the code available at Bosch. By training a smaller model, we build understanding of how these models work, what kind of data we need and how to build on top of larger models in the future. 

## Tasks
In this Hackathon, your task is to prepare data for training a smaller model -- RoBERTa -- available at HuggingFace, training the model to a certain extent (depending on the processing power available) and use it for completing a program. 
The task can be adjusted based on the interest in the following directions:
1) Instead of training a model from scratch -> further training an already pre-trained model (e.g., CodeBert)
2) Instead of training the model -> using an existing model and designing a tool around it for text completion. 
3) Creating a web service (and/or) a container that can provide code completion from existing models (e.g., CodeParrot) � for this, you need to read the following chapter: https://1drv.ms/b/s!Avcq_JfcNezZhpscaH8QSnAMrjI4Rw?e=iFHfUU 

Preparations
To get the most out of the Hackathon, you should prepare for the following: 
1. Before the Hackathon
* Find yourself a team of 2-3 colleagues.
* Read the following chapter: https://1drv.ms/b/s!Avcq_JfcNezZhpp5VRxu6-ulHIKvVQ?e=ccoGk0 
* Download the scripts for training the RoBERTa model: https://github.com/miroslawstaron/machine_learning_best_practices/tree/main/chapter_11; This task requires creating a new virtual environment �  see the readme.md file
* Create a large text/csv file with all the source code that you want to use for training. An example from an open source code is here: https://github.com/miroslawstaron/machine_learning_best_practices/blob/main/chapter_11/source_code_wolf_ssl.txt 
* Make sure that you can download models from HuggingFace, e.g.: https://huggingface.co/roberta-base 

2. During the Hackathon
* Training the model
    * Execute the script from point b) above for 1 epoch
    * Present the solution to everyone for fill-mask pipeline
* Training a new model
    * Instead of using roberta-base, use Microsoft CodeBert-base: microsoft/codebert-base, https://arxiv.org/abs/2002.08155 
    * Execute the script from b) above for 1 epoch
    * Present the results
*. Create a web-service
    * Follow the chapter https://1drv.ms/b/s!Avcq_JfcNezZhpscaH8QSnAMrjI4Rw?e=iFHfUU
    * Modify the code available here: https://github.com/miroslawstaron/machine_learning_best_practices/tree/main/chapter_16/predictor to use the CodeBert model
3. After the Hackathon
* Continue training the model for at least 10 epochs
* Create a set of programs that you want to use the model to complete
* Present it for your colleagues

## Plan of the day
8.30 -- 9.30: Meet and greet, welcome to the Hackathon, theory behind the models

9.00 -- 11.00: Start with the training of the model

11.00 -- 13.00: training of the model (including lunch)

13.00 -- 13.30: short check-in on the training of the model

13.30 -- 14.30: Finalizing the training and preparing for the evaluation

14.30 -- 15.30: Presentations

15.30 -- 16.00: check-out and next steps

Additional reading:
1. Staron, Miroslaw. Machine Learning Infrastructure and Best Practices for Software Engineers: Take your machine learning software from a prototype to a fully fledged software system. Packt Publishing Ltd, 2024.
2. Rothman, Denis. Transformers for Natural Language Processing: Build innovative deep neural network architectures for NLP with Python, PyTorch, TensorFlow, BERT, RoBERTa, and more. Packt Publishing Ltd, 2021.