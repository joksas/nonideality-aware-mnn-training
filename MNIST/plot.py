import numpy as np
import matplotlib.pyplot as plt


def get_accuracy(log_dir_path):
    filename = "{}/accuracy.csv".format(log_dir_path)
    csv = np.genfromtxt(filename, delimiter=",")
    accuracy = 100*csv
    return accuracy

def get_error(log_dir_path):
    accuracy = get_accuracy(log_dir_path)
    error = 100 - accuracy
    return error

def get_power(log_dir_path):
    filename = "{}/power.csv".format(log_dir_path)
    csv = np.genfromtxt(filename, delimiter=",")
    power = csv[0::2] + csv[1::2]
    return power


dataset = "MNIST"
network_types = ["regular", "non-regularized-aware", "regularized-aware"]
group_idxs = [0, 2]
colors = ["#e69f00", "#0072b2", "#009e73"]
labels = ["Standard", "Nonideality-aware (non-regularised)", "Nonideality-aware (regularised)"]

axis_label_font_size = 14
legend_font_size = 12
ticks_font_size = 10
markersize = 7

fig, ax = plt.subplots()
boxplots = []

for network_idx, network_type in enumerate(network_types):
    for group_idx in group_idxs:
        log_dir_path = "models/{}/{}/group-{}".format(
                dataset, network_type, group_idx)
        power = get_power(log_dir_path)
        error = get_error(log_dir_path)
        print("network_type:", network_type, "group_idx:", group_idx, "median error:", np.median(error), "median power:", np.median(power))
        # plt.scatter(power, accuracy, marker="x", s=markersize, color=colors[network_idx], label=labels[network_idx])
        w = 0.1
        x_pos = np.median(power)
        boxplots.append(plt.boxplot(error, positions=[x_pos], widths=[10**(np.log10(x_pos)+w/2.)-10**(np.log10(x_pos)-w/2.)], sym=colors[network_idx]))
        bplot = boxplots[-1]
        plt.setp(bplot['fliers'], marker='x', markersize=4)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bplot[element], color=colors[network_idx])


ax.legend([boxplot["boxes"][0] for boxplot in boxplots[::len(group_idxs)]],
        [label for label in labels], fontsize=legend_font_size)
plt.xticks(fontsize=ticks_font_size)
plt.yticks(fontsize=ticks_font_size)
plt.xlabel("Ohmic power consumption (W)", fontsize=axis_label_font_size)
plt.ylabel("Inference test error (%)", fontsize=axis_label_font_size)
plt.semilogx()
plt.semilogy()
# x.ticklabel_format(style='plain')
# ax.get_yaxis().get_major_formatter().set_scientific(False)
# plt.show()
plt.savefig("error-box-plot-two-groups.pdf", bbox_inches='tight')

