import copy

import crossbar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simulations
from training import architecture
from training.iterator import (D2DLognormal, Inference, Iterator,
                               IVNonlinearity, StuckAtGMin, Training)

from . import utils

AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 8
TICKS_FONT_SIZE = 8
SUBPLOT_LABEL_SIZE = 12
LINEWIDTH = 0.75
# Advanced Science
ONE_COLUMN_WIDTH = 8.5 / 2.54
TWO_COLUMNS_WIDTH = 17.8 / 2.54


def iv_nonlinearity_error_curves():
    num_rows = 2
    num_cols = 3
    training_idx = 0
    colors = utils.color_dict()
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 2),
    )

    temp_iterators = simulations.iv_nonlinearity.get_iterators()
    for i in range(len(temp_iterators)):
        temp_iterators[i].training.repeat_idx = training_idx
    iterators = np.array(
        [
            [temp_iterators[idx] for idx in row]
            for row in [
                [0, 1, 2],
                [0, 3, 4],
            ]
        ]
    )

    test_histories = np.array(
        [
            [iterators[i, j].train_test_histories()[idx] for j, idx in enumerate(row)]
            for i, row in enumerate(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                ]
            )
        ]
    )

    for i in range(num_rows):
        for j in range(num_cols):
            iterator = iterators[i, j]
            test_history = test_histories[i, j]
            axis = axes[i, j]

            # Training curve.
            train_epochs, train_accuracy = iterator.train_epochs_and_accuracy()
            train_error = 100 * (1 - train_accuracy)
            axis.plot(
                train_epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH
            )

            # Validation curve.
            (
                validation_epochs,
                validation_accuracy,
            ) = iterator.validation_epochs_and_accuracy()
            validation_error = 100 * (1 - validation_accuracy)
            if len(validation_error.shape) > 1:
                validation_error_median = np.median(validation_error, axis=1)
                validation_error_min = np.min(validation_error, axis=1)
                validation_error_max = np.max(validation_error, axis=1)
                axis.fill_between(
                    validation_epochs,
                    validation_error_min,
                    validation_error_max,
                    color=colors["sky-blue"],
                    alpha=0.25,
                    linewidth=0,
                )
                axis.plot(
                    validation_epochs,
                    validation_error_median,
                    color=colors["sky-blue"],
                    linewidth=LINEWIDTH / 2,
                )
            else:
                axis.plot(
                    validation_epochs,
                    validation_error,
                    color=colors["sky-blue"],
                    linewidth=LINEWIDTH,
                )

            # Testing (during training) curve.
            test_epochs = test_history["epoch_no"]
            test_accuracy = np.array(test_history["accuracy"])
            test_error = 100 * (1 - test_accuracy)
            test_error_median = np.median(test_error, axis=1)
            test_error_min = np.min(test_error, axis=1)
            test_error_max = np.max(test_error, axis=1)
            axis.fill_between(
                test_epochs,
                test_error_min,
                test_error_max,
                color=colors["reddish-purple"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                test_epochs,
                test_error_median,
                color=colors["reddish-purple"],
                linewidth=LINEWIDTH / 2,
            )

            utils.add_subfigure_label(fig, axis, i * num_cols + j, SUBPLOT_LABEL_SIZE)
            axis.set_yscale("log")
            plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

            if i + 1 == num_rows:
                axes[i, j].set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

            if j == 0:
                axes[i, j].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, len(train_epochs)])

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.8, 1.05),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig(
        "plotting/iv-nonlinearity-error-curves.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def iv_nonlinearity_losses():
    num_rows = 2
    num_cols = 3
    training_idx = 0
    colors = utils.color_dict()
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 2),
    )

    temp_iterators = simulations.iv_nonlinearity.get_iterators()
    for i in range(len(temp_iterators)):
        temp_iterators[i].training.repeat_idx = training_idx
    iterators = np.array(
        [
            [temp_iterators[idx] for idx in row]
            for row in [
                [0, 1, 2],
                [0, 3, 4],
            ]
        ]
    )

    test_histories = np.array(
        [
            [iterators[i, j].train_test_histories()[idx] for j, idx in enumerate(row)]
            for i, row in enumerate(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                ]
            )
        ]
    )

    for i in range(num_rows):
        for j in range(num_cols):
            iterator = iterators[i, j]
            test_history = test_histories[i, j]
            axis = axes[i, j]

            # Training curve.
            train_epochs, train_loss = iterator.train_epochs_and_loss()
            axis.plot(
                train_epochs, train_loss, color=colors["orange"], linewidth=LINEWIDTH
            )

            # Validation curve.
            validation_epochs, validation_loss = iterator.validation_epochs_and_loss()
            if len(validation_loss.shape) > 1:
                validation_loss_median = np.median(validation_loss, axis=1)
                validation_loss_min = np.min(validation_loss, axis=1)
                validation_loss_max = np.max(validation_loss, axis=1)
                axis.fill_between(
                    validation_epochs,
                    validation_loss_min,
                    validation_loss_max,
                    color=colors["sky-blue"],
                    alpha=0.25,
                    linewidth=0,
                )
                axis.plot(
                    validation_epochs,
                    validation_loss_median,
                    color=colors["sky-blue"],
                    linewidth=LINEWIDTH / 2,
                )
            else:
                axis.plot(
                    validation_epochs,
                    validation_loss,
                    color=colors["sky-blue"],
                    linewidth=LINEWIDTH,
                )

            # Testing (during training) curve.
            test_epochs = test_history["epoch_no"]
            test_loss = np.array(test_history["loss"])
            test_loss_median = np.median(test_loss, axis=1)
            test_loss_min = np.min(test_loss, axis=1)
            test_loss_max = np.max(test_loss, axis=1)
            axis.fill_between(
                test_epochs,
                test_loss_min,
                test_loss_max,
                color=colors["reddish-purple"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                test_epochs,
                test_loss_median,
                color=colors["reddish-purple"],
                linewidth=LINEWIDTH / 2,
            )

            utils.add_subfigure_label(fig, axis, i * num_cols + j, SUBPLOT_LABEL_SIZE)
            axis.set_yscale("log")
            plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

            if i + 1 == num_rows:
                axes[i, j].set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

            if j == 0:
                axes[i, j].set_ylabel("Loss", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, len(train_epochs)])

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.8, 1.05),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig(
        "plotting/iv-nonlinearity-loss-curves.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def iv_nonlinearity_boxplots():
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.8 * ONE_COLUMN_WIDTH))
    fig.tight_layout()
    iterators = simulations.iv_nonlinearity.get_iterators()
    iterators.insert(3, iterators[0])
    indices = [0, 0, 0, 1, 0, 0]
    errors = [
        100 * iterator.test_error()[idx].flatten()
        for idx, iterator in zip(indices, iterators)
    ]
    powers = [
        iterator.avg_power()[idx].flatten() for idx, iterator in zip(indices, iterators)
    ]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue", "bluish-green"]]
    labels = ["Standard", "Nonideality-aware", "Nonideality-aware (regularised)"]

    boxplots = []

    for idx, (error, power) in enumerate(zip(errors, powers)):
        x_pos = np.mean(power)
        w = 0.2
        color = colors[idx % 3]
        boxplot = plt.boxplot(
            error,
            positions=[x_pos],
            widths=[
                10 ** (np.log10(x_pos) + w / 2.0) - 10 ** (np.log10(x_pos) - w / 2.0)
            ],
            sym=color,
        )
        plt.setp(boxplot["fliers"], marker="x", markersize=1, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(boxplot[element], color=color, linewidth=0.5)

        boxplots.append(boxplot)

    leg = axes.legend(
        [boxplot["boxes"][0] for boxplot in boxplots[:3]],
        labels,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
    plt.xlabel("Ohmic power consumption (W)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Test error (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    axes.set_xscale("log")
    axes.set_yscale("log")

    plt.savefig(
        "plotting/iv-nonlinearity-boxplots.pdf", bbox_inches="tight", transparent=True
    )


def cnn_results():
    fig, axes = plt.subplots(
        1, 3, sharey=True, figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 3)
    )
    fig.tight_layout()
    colors = utils.color_dict()
    iterators = simulations.iv_nonlinearity_cnn.get_iterators()

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        train_epochs, train_accuracy = iterator.train_epochs_and_accuracy()
        train_error = 100 * (1 - train_accuracy)
        axis.plot(
            train_epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH
        )

        (
            validation_epochs,
            validation_accuracy,
        ) = iterator.validation_epochs_and_accuracy()
        validation_error = 100 * (1 - validation_accuracy)
        if len(validation_error.shape) > 1:
            validation_error_median = np.median(validation_error, axis=1)
            validation_error_min = np.min(validation_error, axis=1)
            validation_error_max = np.max(validation_error, axis=1)
            axis.fill_between(
                validation_epochs,
                validation_error_min,
                validation_error_max,
                color=colors["sky-blue"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                validation_epochs,
                validation_error_median,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH / 2,
            )
        else:
            axis.plot(
                validation_epochs,
                validation_error,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH,
            )

        test_history = iterator.train_test_histories()[0]
        test_epochs = test_history["epoch_no"]
        test_accuracy = np.array(test_history["accuracy"])
        test_error = 100 * (1 - test_accuracy)
        test_error_median = np.median(test_error, axis=1)
        test_error_min = np.min(test_error, axis=1)
        test_error_max = np.max(test_error, axis=1)
        axis.fill_between(
            test_epochs,
            test_error_min,
            test_error_max,
            color=colors["reddish-purple"],
            alpha=0.25,
            linewidth=0,
        )
        axis.plot(
            test_epochs,
            test_error_median,
            color=colors["reddish-purple"],
            linewidth=LINEWIDTH / 2,
        )

        axis.set_xlim([0, len(train_epochs)])
        axis.set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    axes[0].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    # Box plots.
    errors = [100 * iterator.test_error()[0].flatten() for iterator in iterators]
    for idx, (error, color) in enumerate(
        zip(errors, [colors["vermilion"], colors["blue"]])
    ):
        bplot = axes[2].boxplot(error, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=1, markeredgewidth=0.2)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.2)
        plt.xticks([0, 1], ["Standard", "Nonideality-aware"])
        axes[2].set_xlabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)

    # Common properties.
    for idx, axis in enumerate(axes):
        axis.set_yscale("log")
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.65, 1.15),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig("plotting/cnn-error-results.pdf", bbox_inches="tight", transparent=True)


def cnn_results_loss():
    fig, axes = plt.subplots(
        1, 3, sharey=True, figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 3)
    )
    fig.tight_layout()
    colors = utils.color_dict()
    iterators = simulations.iv_nonlinearity_cnn.get_iterators()

    # loss curves.
    for axis, iterator in zip(axes, iterators):
        train_epochs, train_loss = iterator.train_epochs_and_loss()
        axis.plot(train_epochs, train_loss, color=colors["orange"], linewidth=LINEWIDTH)

        validation_epochs, validation_loss = iterator.validation_epochs_and_loss()
        if len(validation_loss.shape) > 1:
            validation_loss_median = np.median(validation_loss, axis=1)
            validation_loss_min = np.min(validation_loss, axis=1)
            validation_loss_max = np.max(validation_loss, axis=1)
            axis.fill_between(
                validation_epochs,
                validation_loss_min,
                validation_loss_max,
                color=colors["sky-blue"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                validation_epochs,
                validation_loss_median,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH / 2,
            )
        else:
            axis.plot(
                validation_epochs,
                validation_loss,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH,
            )

        test_history = iterator.train_test_histories()[0]
        test_epochs = test_history["epoch_no"]
        test_loss = np.array(test_history["loss"])
        test_loss_median = np.median(test_loss, axis=1)
        test_loss_min = np.min(test_loss, axis=1)
        test_loss_max = np.max(test_loss, axis=1)
        axis.fill_between(
            test_epochs,
            test_loss_min,
            test_loss_max,
            color=colors["reddish-purple"],
            alpha=0.25,
            linewidth=0,
        )
        axis.plot(
            test_epochs,
            test_loss_median,
            color=colors["reddish-purple"],
            linewidth=LINEWIDTH / 2,
        )

        axis.set_xlim([0, len(train_epochs)])
        axis.set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    axes[0].set_ylabel("Loss", fontsize=AXIS_LABEL_FONT_SIZE)

    # Box plots.
    losses = [iterator.test_loss()[0].flatten() for iterator in iterators]
    for idx, (loss, color) in enumerate(
        zip(losses, [colors["vermilion"], colors["blue"]])
    ):
        bplot = axes[2].boxplot(loss, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=1, markeredgewidth=0.2)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.2)
        plt.xticks([0, 1], ["Standard", "Nonideality-aware"])
        axes[2].set_xlabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)

    # Common properties.
    for idx, axis in enumerate(axes):
        axis.set_yscale("log")
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.65, 1.15),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig("plotting/cnn-loss-results.pdf", bbox_inches="tight", transparent=True)


def d2d_conductance_histograms():
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(ONE_COLUMN_WIDTH, 1.5 * ONE_COLUMN_WIDTH),
    )

    iterators = simulations.d2d_asymmetry.get_iterators()
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (axis, iterator, color) in enumerate(zip(axes, iterators, colors)):
        iterator.is_training = True
        model = architecture.get_model(
            iterator, custom_weights_path=iterator.weights_path()
        )
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(
            weights, iterator.training.G_min, iterator.training.G_max
        )
        G = 1000 * G
        axis.hist(G.numpy().flatten(), bins=100, color=color)
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
        axis.set_ylabel("Count (#)", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[1].set_xlabel("Conductance (mS)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.savefig("plotting/d2d-G-histograms.pdf", bbox_inches="tight", transparent=True)


def d2d_boxplots():
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.8 * ONE_COLUMN_WIDTH))
    fig.tight_layout()
    iterators = simulations.d2d_asymmetry.get_iterators()
    errors = [100 * iterator.test_error()[0].flatten() for iterator in iterators]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    boxplots = []

    for idx, (error, color) in enumerate(zip(errors, colors)):
        bplot = plt.boxplot(error, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.5)
        plt.xticks([0, 1], ["Uniform", "Asymmetric"])

    axes.set_yscale("log")
    plt.xlabel("D2D variability", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    plt.savefig("plotting/d2d-boxplots.pdf", bbox_inches="tight", transparent=True)


def iv_nonlinearity_and_stuck_error_curves():
    num_rows = 1
    num_cols = 2
    training_idx = 0
    colors = utils.color_dict()
    fig, axes = plt.subplots(
        num_rows, num_cols, sharex=True, sharey=True, figsize=(12 / 2.54, 4.5 / 2.54)
    )

    iterators = simulations.iv_nonlinearity_and_stuck.get_iterators()
    test_histories = [
        iterator.train_test_histories()[idx]
        for (idx, iterator) in zip([5, 0], iterators)
    ]

    for idx, (iterator, test_history, axis) in enumerate(
        zip(iterators, test_histories, axes)
    ):
        # Training curve.
        train_epochs, train_accuracy = iterator.train_epochs_and_accuracy()
        train_error = 100 * (1 - train_accuracy)
        axis.plot(
            train_epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH
        )

        # Validation curve.
        (
            validation_epochs,
            validation_accuracy,
        ) = iterator.validation_epochs_and_accuracy()
        validation_error = 100 * (1 - validation_accuracy)
        if len(validation_error.shape) > 1:
            validation_error_median = np.median(validation_error, axis=1)
            validation_error_min = np.min(validation_error, axis=1)
            validation_error_max = np.max(validation_error, axis=1)
            axis.fill_between(
                validation_epochs,
                validation_error_min,
                validation_error_max,
                color=colors["sky-blue"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                validation_epochs,
                validation_error_median,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH / 2,
            )
        else:
            axis.plot(
                validation_epochs,
                validation_error,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH,
            )

        # Testing (during training) curve.
        test_epochs = test_history["epoch_no"]
        test_accuracy = np.array(test_history["accuracy"])
        test_error = 100 * (1 - test_accuracy)
        test_error_median = np.median(test_error, axis=1)
        test_error_min = np.min(test_error, axis=1)
        test_error_max = np.max(test_error, axis=1)
        axis.fill_between(
            test_epochs,
            test_error_min,
            test_error_max,
            color=colors["reddish-purple"],
            alpha=0.25,
            linewidth=0,
        )
        axis.plot(
            test_epochs,
            test_error_median,
            color=colors["reddish-purple"],
            linewidth=LINEWIDTH / 2,
        )

        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
        axis.set_yscale("log")

        axis.set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

        if idx == 0:
            axis.set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, len(train_epochs)])

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.9, 1.15),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig(
        "plotting/iv-nonlinearity-and-stuck-error-curves.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def iv_nonlinearity_and_stuck_boxplots():
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.8 * ONE_COLUMN_WIDTH))
    fig.tight_layout()
    iterators = simulations.iv_nonlinearity_and_stuck.get_iterators()
    errors = [100 * iterator.test_error()[0].flatten() for iterator in iterators]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    boxplots = []

    for idx, (error, color) in enumerate(zip(errors, colors)):
        bplot = plt.boxplot(error, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.5)
    plt.xticks([0, 1], ["Standard", "Nonideality-aware"])

    axes.set_yscale("log")
    plt.xlabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    plt.savefig(
        "plotting/iv-nonlinearity-and-stuck-boxplots.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def d2d_error_curves():
    num_rows = 1
    num_cols = 2
    training_idx = 0
    colors = utils.color_dict()
    fig, axes = plt.subplots(
        num_rows, num_cols, sharex=True, sharey=True, figsize=(12 / 2.54, 4.5 / 2.54)
    )

    iterators = simulations.d2d_asymmetry.get_iterators()
    test_histories = [iterator.train_test_histories()[0] for iterator in iterators]

    for idx, (iterator, test_history, axis) in enumerate(
        zip(iterators, test_histories, axes)
    ):
        # Training curve.
        train_epochs, train_accuracy = iterator.train_epochs_and_accuracy()
        train_error = 100 * (1 - train_accuracy)
        axis.plot(
            train_epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH
        )

        # Validation curve.
        (
            validation_epochs,
            validation_accuracy,
        ) = iterator.validation_epochs_and_accuracy()
        validation_error = 100 * (1 - validation_accuracy)
        if len(validation_error.shape) > 1:
            validation_error_median = np.median(validation_error, axis=1)
            validation_error_min = np.min(validation_error, axis=1)
            validation_error_max = np.max(validation_error, axis=1)
            axis.fill_between(
                validation_epochs,
                validation_error_min,
                validation_error_max,
                color=colors["sky-blue"],
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(
                validation_epochs,
                validation_error_median,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH / 2,
            )
        else:
            axis.plot(
                validation_epochs,
                validation_error,
                color=colors["sky-blue"],
                linewidth=LINEWIDTH,
            )

        # Testing (during training) curve.
        test_epochs = test_history["epoch_no"]
        test_accuracy = np.array(test_history["accuracy"])
        test_error = 100 * (1 - test_accuracy)
        test_error_median = np.median(test_error, axis=1)
        test_error_min = np.min(test_error, axis=1)
        test_error_max = np.max(test_error, axis=1)
        axis.fill_between(
            test_epochs,
            test_error_min,
            test_error_max,
            color=colors["reddish-purple"],
            alpha=0.25,
            linewidth=0,
        )
        axis.plot(
            test_epochs,
            test_error_median,
            color=colors["reddish-purple"],
            linewidth=LINEWIDTH / 2,
        )

        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
        axis.set_yscale("log")

        axis.set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

        if idx == 0:
            axis.set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, len(train_epochs)])

    leg = plt.figlegend(
        ["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0, 0, 0.9, 1.15),
        frameon=False,
    )
    for line in leg.get_lines():
        line.set_linewidth(1)

    plt.savefig("plotting/d2d-error-curves.pdf", bbox_inches="tight", transparent=True)


def checkpoint_comparison_boxplots():
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.8 * ONE_COLUMN_WIDTH))
    fig.tight_layout()

    iterators = simulations.checkpoint_comparison.get_iterators()
    errors = [100 * iterator.test_error()[0].flatten() for iterator in iterators]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    boxplots = []

    for idx, (error, color) in enumerate(zip(errors, colors)):
        bplot = plt.boxplot(error, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.5)
        plt.xticks([0, 1], ["Standard", "Memristive"])

    axes.set_yscale("log")
    plt.xlabel("Checkpoint", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    plt.savefig(
        "plotting/checkpoint-comparison-boxplots.pdf",
        bbox_inches="tight",
        transparent=True,
    )
