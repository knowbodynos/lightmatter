import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

path = sys.argv[1]

for accuracy in ["train_accuracy", "test_accuracy"]:
    print("Plotting " + accuracy + ".")

    train_df = pd.read_csv(path + "/" + accuracy + ".csv")
    x_label, y_label = train_df.columns

    with PdfPages(path + "/" + accuracy + ".pdf") as pdf:
        fig, ax = plt.subplots()
        fig.set_size_inches(1.5 * 11, 1.5 * 8.5)
        ax.grid()
        ax.set_title(x_label + " vs. " + y_label)
        ax.set_xlabel(x_label)
        ax.set_xlim((train_df[x_label].min(), train_df[x_label].max()))
        ax.set_ylabel(y_label)
        ax.set_ylim((train_df[y_label].min(), train_df[y_label].max()))
        ax.plot(train_df[x_label], train_df[y_label])
        pdf.savefig(fig)

print("Finished plotting accuracies.")