import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.special import softmax
from mdn_sigma import load_lookup_table, sample_weight_from_lookup


def load_mnist_data(filename=None):
    """Load MNIST dataset from .mat file"""
    if filename is None:
        filename = r'Mistari\code\func\MNISTdataset01.mat'
    data = loadmat(filename)
    images_train = data['imagesTrain']
    images_test = data['imagesTest']
    y_train = data['yTrain']
    labels_train = data['labelsTrain'].flatten() + 1  # Convert to 1-10 indexing
    labels_test = data['labelsTest'].flatten() + 1
    return images_train, y_train, labels_train, images_test, labels_test


def softmax_activation(z):
    """Softmax activation function"""
    return softmax(z, axis=0)


def calc_accuracy(w1, images, labels):
    """Calculate accuracy and get predictions"""
    # Forward pass: z = w1 @ images
    z2 = w1 @ images
    # Apply softmax activation
    a2 = softmax_activation(z2)
    # Get predictions (argmax + 1 for 1-based indexing)
    predictions = np.argmax(a2, axis=0) + 1
    # Calculate accuracy
    accuracy = np.mean(predictions == labels) * 100
    
    return accuracy, a2


def shuffle_data(images_train, y_train, labels_train):
    """Shuffle training data"""
    n_samples = images_train.shape[1]
    indices = np.random.permutation(n_samples)
    
    return (images_train[:, indices], 
            y_train[:, indices], 
            labels_train[indices])


def train_mistari(images_train, y_train, labels_train, images_test, labels_test,
                  lr=1.0, epochs=10000, lookup_csv='model_run_all_voltages.csv',
                  patience=10000, seed=123):
    """
    Train using MISTARI (Coordinate Descent)
    
    Parameters:
    - lr: learning rate
    - epochs: number of epochs
    - lookup_csv: path to precomputed MDN lookup table CSV
    - patience: early stopping patience (epochs without improvement in test loss)
                Set to None to disable early stopping.
    - seed: random seed for reproducibility. Set to None to disable.
    """
    if seed is not None:
        np.random.seed(seed)

    print("Loading MDN model...")
    df_set, df_reset = load_lookup_table(lookup_csv)
    print("MDN loaded.\n")

    # Initialize dimensions
    n_in = images_train.shape[0]
    n_out = y_train.shape[0]
    n_train = images_train.shape[1]
    n_test = images_test.shape[1]
    
    # Initialize tracking arrays
    accuracy_train = np.zeros(epochs + 1)
    accuracy_test = np.zeros(epochs + 1)
    loss_train = np.zeros(epochs + 1)
    loss_test = np.zeros(epochs + 1)
    
    # Initialize weights
    init_variation = np.sqrt(2 / (n_in + n_out))
    w1 = np.random.randn(n_out, n_in) * init_variation

    # Early stopping state
    best_loss_test = np.inf
    patience_counter = 0
    best_w1 = w1.copy()
    
    # Training loop
    for k in range(epochs):
        # Calculate accuracy
        accuracy_train[k], a2 = calc_accuracy(w1, images_train, labels_train)
        accuracy_test[k], _ = calc_accuracy(w1, images_test, labels_test)
        
        # Print progress
        if (k + 1) % 100 == 0:
            print(f'Epoch {k+1}: train {accuracy_train[k]:.2f}%  '
                  f'test {accuracy_test[k]:.2f}%')
        
        # Cross-entropy loss for train set
        loss_train[k] = -np.mean(np.sum(y_train * np.log(a2 + 1e-10), axis=0))
        
        # Cross-entropy loss for test set
        _, a2_test = calc_accuracy(w1, images_test, labels_test)
        loss_test[k] = -np.mean(np.sum(np.eye(y_train.shape[0])[labels_test - 1].T * np.log(a2_test + 1e-10), axis=0))
        
        # Early stopping check
        if patience is not None:
            if loss_test[k] < best_loss_test:
                best_loss_test = loss_test[k]
                best_w1 = w1.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {k+1} (no improvement for {patience} epochs)')
                    # Fill remaining entries with last values
                    accuracy_train[k+1:] = accuracy_train[k]
                    accuracy_test[k+1:] = accuracy_test[k]
                    loss_train[k+1:] = loss_train[k]
                    loss_test[k+1:] = loss_test[k]
                    w1 = best_w1
                    return w1, accuracy_train, accuracy_test, loss_train, loss_test
        
        # Train on each sample
        for j in range(n_train):
            # Feed forward
            a1 = images_train[:, j]
            z2 = w1 @ a1
            a2_sample = softmax_activation(z2)
            
            # Get prediction
            output = np.argmax(a2_sample) + 1
            
            # Update only if wrong prediction
            if output != labels_train[j]:
                # Coordinate Descent: random weight selection
                ind1 = np.random.randint(0, n_out)
                ind2 = np.random.randint(0, n_in)
                
                # Gradient calculation
                grad2 = (a2_sample[ind1] - y_train[ind1, j]) * a1[ind2]
                delta_w = lr * grad2

                w_sampled, _, _ = sample_weight_from_lookup(
                    w_current=w1[ind1, ind2],
                    grad=delta_w,
                    df_set=df_set,
                    df_reset=df_reset,
                )
                w1[ind1, ind2] = w_sampled
        
        # Shuffle training data
        images_train, y_train, labels_train = shuffle_data(images_train, y_train, labels_train)
    
    # Final accuracy calculation
    accuracy_train[epochs], a2 = calc_accuracy(w1, images_train, labels_train)
    accuracy_test[epochs], a2_test = calc_accuracy(w1, images_test, labels_test)
    loss_train[epochs] = -np.mean(np.sum(y_train * np.log(a2 + 1e-10), axis=0))
    loss_test[epochs] = -np.mean(np.sum(np.eye(y_train.shape[0])[labels_test - 1].T * np.log(a2_test + 1e-10), axis=0))

    # Restore best weights if early stopping was active
    if patience is not None:
        w1 = best_w1
    
    return w1, accuracy_train, accuracy_test, loss_train, loss_test


if __name__ == '__main__':
    # Dataset files to train on
    dataset_files = [
        r'Mistari\code\func\MNISTdataset01.mat',
        #r'Mistari\code\func\MNISTdataset012.mat',
        #r'Mistari\code\func\MNISTdataset0123.mat',
        #r'Mistari\code\func\MNISTdataset.mat'
    ]

    #SEEDS = [43, 99, 123, 456, 789]  # Multiple seeds for robustness
    SEEDS = [123]  # Multiple seeds for robustness
    #seeds for results in paper: 42, 123
    # Store results for each dataset
    all_loss_train = {}
    all_loss_test = {}
    all_acc_train = {}
    all_acc_test = {}

    # Train on each dataset
    for dataset_file in dataset_files:
        dataset_name = dataset_file.split('\\')[-1].replace('.mat', '')
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name}...")

        images_train, y_train, labels_train, images_test, labels_test = load_mnist_data(dataset_file)

        print(f"Training set: {images_train.shape[1]} samples")
        print(f"Test set: {images_test.shape[1]} samples")

        seed_acc_train = []
        seed_acc_test = []
        seed_loss_train = []
        seed_loss_test = []

        for seed in SEEDS:
            print(f"\n  -- Seed {seed} --")
            w1, acc_train, acc_test, loss_train, loss_test = train_mistari(
                images_train, y_train, labels_train,
                images_test, labels_test,
                lr=1.0, epochs=10000, lookup_csv='model_run_all_voltages.csv',
                patience=10000, seed=seed
            )
            print(f"  Seed {seed}: Train {acc_train[acc_train != 0][-1]:.2f}%  "
                  f"Test {acc_test[acc_test != 0][-1]:.2f}%")

            seed_acc_train.append(acc_train)
            seed_acc_test.append(acc_test)
            seed_loss_train.append(loss_train)
            seed_loss_test.append(loss_test)

        # Final accuracy is the last non-zero (or simply the last) entry per seed
        final_train = np.array([a[a != 0][-1] for a in seed_acc_train])
        final_test  = np.array([a[a != 0][-1] for a in seed_acc_test])

        print(f"\n{'='*60}")
        print(f"Results over {len(SEEDS)} seeds for {dataset_name}:")
        print(f"  Train accuracy: {final_train.mean():.2f}% ± {final_train.std():.2f}%")
        print(f"  Test  accuracy: {final_test.mean():.2f}% ± {final_test.std():.2f}%")

        # Store mean curves for plotting
        all_loss_train[dataset_name] = np.mean(seed_loss_train, axis=0)
        all_loss_test[dataset_name]  = np.mean(seed_loss_test,  axis=0)
        all_acc_train[dataset_name]  = np.mean(seed_acc_train,  axis=0)
        all_acc_test[dataset_name]   = np.mean(seed_acc_test,   axis=0)

    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name, loss_train in all_loss_train.items():
        axes[0].plot(loss_train, label=dataset_name, linewidth=2)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
    axes[0].set_title('Training Loss - MISTARI (mean over seeds)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for dataset_name, loss_test in all_loss_test.items():
        axes[1].plot(loss_test, label=dataset_name, linewidth=2)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Cross-Entropy Loss', fontsize=12)
    axes[1].set_title('Test Loss - MISTARI (mean over seeds)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mistari_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print("Plot saved as 'mistari_loss_comparison.png'")

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(4, 4))

    for dataset_name, acc_train in all_acc_train.items():
        ax.plot(acc_train, label=f'{dataset_name} (Train)', linewidth=2, linestyle='-')

    for dataset_name, acc_test in all_acc_test.items():
        ax.plot(acc_test, label=f'{dataset_name} (Test)', linewidth=2, linestyle='--')

    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(95, 100)
    ax.set_xticks([0, 5000, 10000])
    #ax.set_title('Accuracy - MISTARI (mean over seeds)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mistari_accuracy_comparison.svg', dpi=300, bbox_inches='tight')
    print("Plot saved as 'mistari_accuracy_comparison.svg'")
    plt.show()
