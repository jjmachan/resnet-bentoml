# ResNet-Bento ML
Bento-ml tutorial for deploying Resnet model :bento:

Bento ML is something like Django for productionizing your ML models. It’s really easy to understand and abstracts the complexities of serving your model for others to use. It’s an awesome tool for Data Scientists since you no longer have to fret about serving the model and can use that time instead to improve the model.

List of features:

- Turn your ML model into production API endpoint with just a few lines of code
- Support all major machine learning training frameworks
- High-performance API serving system with adaptive micro-batching support
- DevOps best practices baked in, simplify the transition from model development to production
- Model management for teams, providing CLI and Web UI dashboard
- Flexible model deployment orchestration with support for AWS Lambda, SageMaker, EC2, Docker, Kubernetes, KNative and more

In short, you can scale your model from your dev environment to production-ready servers, all from Bento ML. I really think this library has huge potential.

In this tutorial, we will build  a ResNet model for detecting Ants or Bees (model adapted from the official [PyTorch docs](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html))  and use Bento ML to create a production-ready service that can we directly accessed using REST APIs.

For more details checkout the [offcial docs](https://bentoml.readthedocs.io/) also I have a blog coming out, so stay tuned for that!

## Project Stucture

1. train.py - The python script to train the model and save the trained model. The saved models will be stored in saved_models folder. Typically trains for 2mins on GPU and 30mins on my CPU. It automatically saves the checkpoints if you exit from training and you can resume training from the same checkpoint by specifying the checkpoint using -c (--checkpoint) argument.

2. get_data.sh - Script to download the training data from the PyTorch servers and extract it.

3. classificationService.py - Defines the Bento service that has to be run for serving the model. This file specifies *handlers*  which are used to specify how the incoming data is to be handled, *artifacts*  or containers that store different models and *Services*  which defines the API endpoints for the various models you want to serve.

4. saveToBento.py - The script that creates, packs and saves the Bento Service. The script packs the models weights, model definition and the information on how to serve along with the dependencies and saves it into the disk. Each saved Bento Service is stand alone and containes everything needed to serve it. The saved services are also versioned for tracking the different models that you tested.

## Usage
 
First of all, download the dataset by running the script `sh get_data.sh`.

Now, install all the dependencies by running `pip install -r requirements.txt`. With that all the dependencies like *pytorch*,  *torchvision*, *imageio* and *bentoml* will be installed.

You can train the model using `python train.py` which will run by default for 25 epochs and save the model weights on the *saved_models* dir. 

With the model all trained you can now add it to Bento using `python saveToBento.py {path\to\saved_model} ` and now its saved and ready to serve. It will print out the path of the location where it is saved so note that down.

Now just run `bentoml serve {path\to\bento_file} ` and vola! Your service is running. Head over to `localhost:5000` to test it out.
