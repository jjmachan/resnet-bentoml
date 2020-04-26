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

For more details checkout my blog!
