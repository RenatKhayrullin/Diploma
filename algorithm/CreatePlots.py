# coding=utf-8
import os
import io
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


reload(sys)
sys.setdefaultencoding('utf-8')


initial_markup_size = [1, 7, 15, 20, 30, 40, 50]
K = 3, 10


rootdir = "/Users/Reist/PycharmProjects/Diploma/graphics/"


def create_loss_graphics(df, filename):
    plt.figure(figsize=(10, 10))
    # Initialize the figure
    plt.style.use('ggplot')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # multiple line plot

    num = 0
    for column in df.drop("iterations", axis=1):
        num += 1
        # Find the right spot on the plot
        plt.subplot(3, 2, num)
        # Plot the lineplot
        # plt.plot(df['iterations'], df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
        plt.plot(df['iterations'], df[column], marker='', color=palette(num), label=column)
        # Same limits for everybody!
        plt.xlim(0, 150)
        # plt.ylim(0, 50000)

        # Not ticks everywhere
        if num in range(5):
            plt.tick_params(labelbottom='off')

        # if num not in [1, 4, 7]:
        #      plt.tick_params(labelleft='off')

        # Add title
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))

    # general title
    # plt.suptitle("Decrease in error through iterations", fontsize=13, fontweight=0, color='black', style='italic', y=1)

    # Axis title
    plt.text(0.5, 0.04, 'iterations', ha='center', va='center')
    plt.text(0.04, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    plt.savefig(filename + ".png")


def draw_loss_plot(loss_graphics_data, filename):
    errors_dict = {}
    # HACK: fixed counts of iterations
    iter_counts = range(1, 151)
    errors_dict["iterations"] = iter_counts
    for seed_size, cluster_size_values in loss_graphics_data.items():
        for clusters_count, loss_valies in cluster_size_values.items():
            if clusters_count not in errors_dict:
                errors_dict[clusters_count] = {}
            y = "seed size:" + seed_size + " " + "seed clusters:" + clusters_count
            errors_dict[clusters_count][y] = loss_valies["itr_values"]

    for clusters_count in errors_dict.keys():
        graphic_file_name = filename + "_clusters-"+clusters_count
        df = pd.DataFrame(errors_dict[clusters_count])
        create_loss_graphics(df,graphic_file_name)


def create_score_plots(df, filename):
    plt.plot('seed_size', 'precision', data=df, marker='', color='skyblue', linewidth=2)
    plt.plot('seed_size', 'recall', data=df, marker='', color='green', linewidth=2)
    plt.plot('seed_size', 'f1', data=df, marker='', color='red', linewidth=2)
    plt.legend()
    plt.savefig(filename + ".png")


def draw_score_plots(score_graphics_data, filename):
    score_dict = {}
    for seed_size, cluster_size_scores in score_graphics_data.items():
        for clusters_count, score_valies in cluster_size_scores.items():
            if clusters_count not in score_dict:
                score_dict[clusters_count] = {}

            for class_id, class_scores in score_valies.items():
                if class_id not in score_dict[clusters_count]:
                    score_dict[clusters_count][class_id] = {}
                    score_dict[clusters_count][class_id]["seed_size"] = []
                    score_dict[clusters_count][class_id]["precision"] = []
                    score_dict[clusters_count][class_id]["recall"] = []
                    score_dict[clusters_count][class_id]["f1"] = []

                score_dict[clusters_count][class_id]["seed_size"].append(int(seed_size))
                score_dict[clusters_count][class_id]["precision"].append(float(score_valies["precision"]))
                score_dict[clusters_count][class_id]["recall"].append(float(score_valies["recall"]))
                score_dict[clusters_count][class_id]["f1"].append(float(score_valies["f1"]))

    for clusters_count in score_dict.keys():
        graphic_file_name = filename + "_clusters-" + clusters_count
        for class_id in score_dict[clusters_count].keys():
            seed_size = score_dict[clusters_count][class_id]["seed_size"]
            precision = score_dict[clusters_count][class_id]["precision"]
            recall = score_dict[clusters_count][class_id]["recall"]
            f1 = score_dict[clusters_count][class_id]["f1"]
            graph_data = {"seed_size": seed_size, "precision": precision, "recall": recall, "f1": f1}
            df = pd.DataFrame(graph_data)
            create_score_plots(df, graphic_file_name)


def create_plots():
    entity_types = {0: "PERS", 1: "LOC", 2: "ORG"}
    loss_graphics = {}
    score_graphics = {}

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            folder_name = str(os.path.basename(subdir))
            file_base_name = str(os.path.splitext(file)[0])
            print subdir, file

            algorithm_type = file_base_name.split("_")[0]
            file_type = file_base_name.split("_")[1]
            seed_size = folder_name.split("_")[1]
            clusters_count = folder_name.split("_")[2]

            data_file = str(os.path.join(subdir, file))
            with io.open(data_file, 'r', encoding='utf8') as r_file:
                if file_type == "err":
                    itrs = []
                    itrs_values = []
                    for row in r_file:
                        values = row.replace("\n","").split(";")
                        itrs.append(int(values[0]))
                        itrs_values.append(float(values[1]))

                    if algorithm_type not in loss_graphics:
                        loss_graphics[algorithm_type] = {}
                    if seed_size not in loss_graphics[algorithm_type]:
                        loss_graphics[algorithm_type][seed_size] = {}

                    loss_graphics[algorithm_type][seed_size][clusters_count] = {"iterations": itrs, "itr_values": itrs_values}

                if file_type == "score":
                    score_dict = {}
                    for row in r_file:
                        values = row.replace("\n","").split(";")
                        class_id = int(values[0])
                        precsision_value = float(values[1])
                        recall_value = float(values[2])
                        f1_value = float(values[3])
                        score_dict[entity_types[class_id]] = {"precision": precsision_value, "recall": recall_value, "f1": f1_value}

                    score_graphics[algorithm_type] = {}
                    score_graphics[algorithm_type][seed_size] = {}
                    score_graphics[algorithm_type][seed_size][clusters_count] = score_dict

                r_file.close()

    draw_loss_plot(loss_graphics["NOCLUS"], "NOCLUS_Loss")
    draw_loss_plot(loss_graphics["CLUS"], "CLUS_Loss")

    create_score_plots(score_graphics["NOCLUS"], "NOCLUS_score")
    create_score_plots(score_graphics["CLUS"], "CLUS_score")


create_plots()