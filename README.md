![narcissus](https://user-images.githubusercontent.com/77789132/162637159-a356ba3e-a9fe-48b6-915d-502cb5c9ef67.png)

# Narcissus Backdoor Attack

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Narcissus clean-label backdoor attack provides an affirmative answer to whether backdoor attacks can present real threats: as they normally require label manipulations or strong accessibility to non-target class samples. This work demonstrates a powerful attack with access to only the target class with minimum assumptions on the attacker's knowledge and capability.

In our paper, we show inserting maliciously-crafted Narcissus poisoned examples totaling less than 0.5\% of the target-class data size or 0.05\% of the training set size, we can manipulate a model trained on the poisoned dataset to classify test examples from arbitrary classes into the target class when the examples are patched with a backdoor trigger; at the same time, the trained model still maintains good accuracy on typical test examples without the trigger as if it were trained on a clean dataset. 

Narcissus backdoor attack is highly effective across datasets and models, even when the trigger is injected into the physical world. Most surprisingly, our attack can evade the latest state-of-the-art defenses in their vanilla form, or after a simple twist, we can adapt to the downstream defenses. We study the cause of the intriguing effectiveness and find that because the trigger synthesized by our attack contains features as persistent as the original semantic features of the target class, any attempt to remove such triggers would inevitably hurt the model accuracy first.

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

