import argparse

from sagemaker.tensorflow import TensorFlowPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_name')
    args = parser.parse_args()

    predictor = TensorFlowPredictor(args.endpoint_name)
    predictor.delete_model()
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
