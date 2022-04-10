![Narcissus](https://user-images.githubusercontent.com/77789132/162528344-188876a8-7809-4ca2-bc08-71338f108684.jpg)

# Narcissus Backdoor Attack

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

In this paper, we provide an affirmative answer to this question by designing an algorithm to mount clean-label backdoor attacks based only on the knowledge of representative examples from the target class.

By inserting maliciously-crafted examples totaling just 0.5\% of the target-class data size and 0.05\% of the training set size, we can manipulate a model trained on this poisoned dataset to classify test examples from arbitrary classes into the target class when the examples are patched with a backdoor trigger; 
at the same time, the trained model still maintains good accuracy on typical test examples without the trigger as if it were trained on a clean dataset. 

Our attack is highly effective across datasets and models, and even when the trigger is injected into the physical world.

# Features
- Clean label backdoor attack
- Low poison rate (0.05\%)
- All-to-one attack
- Only require target class data
- Pyhsical world attack
- Attack train from scrtch model

# Requirements
Python >= 3.6

PyTorch >= 1.10.1

TorchVisison >= 0.11.2

OpenCV >= 4.5.3

# Usage

Use the Narcissus.ipynb notebook for a quick start of our NARCISSUS backdoor attack. The default attack and defense state both use Resnet-18 as the model, CIFAR-10 as the dataset, and the default attack poisoning rate is 0.5% In-class/0.05% overall.

There are a several of optional arguments in the ```Narcissus.ipynb```:

- ```l_inf_r``` : Radius of the L-inf ball which constraint the attack stealthiness.
- ```surrogate_model```, ```generating_model``` : Define the model used to generate the trigger. In the usual case should be set to the same model.
- ```surrogate_epochs``` : The number of epochs for surrogate model training.
- ```warmup_round``` : The number of epochs for poison warmup trainging.
- ```gen_round``` : The number of epoches for poison generation.
- ```patch_mode``` : Users can change this parameter to ```change```, entering the patch trigger mode. 

