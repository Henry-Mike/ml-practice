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

    plt.show()
