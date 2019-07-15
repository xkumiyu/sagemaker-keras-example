import argparse
import os
from datetime import datetime as dt
from pathlib import Path

import tensorflow as tf

from cnn import cnn
from dataset import dataset
from utils.save_history import save_history


def main():
    print(tf.__version__)

    is_sagemaker = 'SM_CHANNEL_DATASET' in os.environ

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data-dir',
                        default=os.getenv('SM_CHANNEL_DATASET', 'data/'))
    parser.add_argument('--model-dir',
                        default=os.getenv('SM_MODEL_DIR', 'model/'))
    parser.add_argument('--output-dir',
                        default=os.getenv('SM_OUTPUT_DATA_DIR', 'outputs/'))
    args, _ = parser.parse_known_args()

    # output directory
    output_dir = Path(args.output_dir)
    if is_sagemaker:
        model_dir = args.model_dir
    else:
        output_dir /= dt.now().strftime('%Y-%m-%d-%H-%M')
        output_dir.mkdir(parents=True)
        model_dir = str(output_dir / args.model_dir)
    # export_path = os.path.join(model_dir, 'mnist/1')
    # model_file = str(output_dir / 'model.png')
    csv_file = str(output_dir / 'log.csv')

    # dataset
    x_train, y_train = dataset(args.data_dir, is_train=True)
    x_test, y_test = dataset(args.data_dir, is_train=False)

    # model
    model = cnn()
    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_file)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    # callbacks
    csv_logger = tf.keras.callbacks.CSVLogger(csv_file)

    # train
    verbose = 2 if is_sagemaker else 1
    history = model.fit(x_train,
                        y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=verbose,
                        callbacks=[csv_logger],
                        validation_data=(x_test, y_test))

    # save model
    print(f'save model to {model_dir}')
    tf.contrib.saved_model.save_keras_model(model, model_dir)
    # inputs = {t.name: t for t in model.inputs}
    # outputs = {t.name: t for t in model.outputs}
    # sess = tf.keras.backend.get_session()
    # tf.saved_model.simple_save(sess,
    #                            export_path,
    #                            inputs=inputs,
    #                            outputs=outputs)

    # save history
    save_history(history, output_dir)


if __name__ == "__main__":
    main()
