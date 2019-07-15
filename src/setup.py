from setuptools import setup, find_packages

setup(name='sagemaker-keras-example',
      version='1.0',
      description='SageMaker Example for Keras.',
      author='xkumiyu',
      author_email='xkumiyu@gmail.com',
      url='https://github.com/xkumiyu/sagemaker-keras-example',
      packages=find_packages(exclude=('tests', 'docs')))
