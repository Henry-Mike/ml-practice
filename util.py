import matplotlib.pyplot as plt


def plot_history(history, names=None):
    data = history.history
    if names is None:
        names = [s for s in data.keys() if '_' not in s]

    fig = plt.figure(figsize=(8, 4))
    nrows = len(names) // 2 + len(names) % 2
    for i, name in enumerate(names):
        plt.subplot(nrows, 2, i + 1)
        plt.plot(data[name], 'o', label='Training {}'.format(name))
        val_name = 'val_' + name
        if val_name in data:
            plt.plot(data[val_name], label='Validation {}'.format(name))
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        
    plt.tight_layout()
    plt.show()


def compare_history(history1, history2):
    data1 = history1.history
    data2 = history2.history
    names = [s for s in data1.keys() if '_' not in s]

    fig = plt.figure(figsize=(8, 4))
    nrows = len(names)//2 + len(names)%2
    for i, name in enumerate(names):
        plt.subplot(nrows, 2, i+1)
        plt.plot(data1[name], 'bo', label='Train {} h1'.format(name))
        plt.plot(data2[name], 'rs', label='Train {} h2'.format(name))

        val_name = 'val_' + name
        plt.plot(data1[val_name], 'b', label='Val {} h1'.format(name))
        plt.plot(data2[val_name], 'r', label='Val {} h2'.format(name))

        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()

    plt.tight_layout()
    plt.show()