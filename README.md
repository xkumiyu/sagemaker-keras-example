# SageMaker Example for Keras

## Requirements

- Python 3.6
- Tensorflow 1.12

## Setting Environment Variable

### Required

``` sh
SAGEMAKER_ROLE='arn:aws:iam::1234567890'
```

### Optional

``` sh
S3_BUCKET_NAME='sagemaker-example'
```

## Dataset Preparation

Download MNIST dataset and upload to S3.

``` sh
python prepare_mnist.py
```

## Training

Create SageMaker training job.

``` sh
python create_training_job.py
```

## TODO

- [ ] Deployment
- [ ] Inference
- [ ] Custom Image
