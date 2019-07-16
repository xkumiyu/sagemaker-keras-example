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

Create a SageMaker training job.
Then, the job name is `mnist-yyyy-mm-dd-HH-MM`.

``` sh
python create_training_job.py
```

Source code used for training is stored in S3.
And, after training is complete, the output and model are stored in S3.

- source code: s3://bucket-name/training/job-name/source/sourcedir.tar.gz
- output: s3://bucket-name/training/job-name/output/output.tar.gz
- model: s3://bucket-name/training/job-name/output/model.tar.gz

## Deployment

Deploy model to TensorFlow Serving-based server in SageMaker.

``` sh
python deploy_model.py <model_data>
```

`model_data` is s3 path to model.tar.gz.
For example, s3://bucket-name/training/job-name/output/model.tar.gz.

Models to be deployed are available not only for models trained in sagemaker, but also for BYO models.

## Endpoint Delete

``` sh
python delete_endpoint.py <endpoint_name>
```

## Inference

``` sh
python infer.py <endpoint_name>
```

## TODO

- [ ] Inference
  - [ ] REST (API Gateway)
  - [ ] Batch transform
- [ ] Custom Image
