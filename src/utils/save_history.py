import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def save_history(history, output_dir):
    # save loss
    loss_file = output_dir / 'loss.png'
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig, ax = plt.subplots()
    ax.plot(loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.legend(loc='upper right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    fig.savefig(str(loss_file))

    # save accuracy
    acc_file = output_dir / 'accuracy.png'
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    fig, ax = plt.subplots()
    ax.plot(acc, label='Training Accuracy')
    ax.plot(val_acc, label='Validation Accuracy')
    ax.legend(loc='upper right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    fig.savefig(str(acc_file))
