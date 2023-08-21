import matplotlib.pyplot as plt

def plot_metrics(evaluation_metrics):
    epochs = range(1, len(evaluation_metrics['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and testing loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, evaluation_metrics['train_loss'], label='Training Loss')
    plt.plot(epochs, evaluation_metrics['test_loss'], label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()

    # Plot training and testing accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, evaluation_metrics['train_acc'], label='Training Accuracy')
    plt.plot(epochs, evaluation_metrics['test_acc'], label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()