import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import jsonlines
import re
import string
from tqdm import tqdm


def plot_generated_count_histogram():
    data = np.loadtxt("adversarial_count.txt")
    x = np.arange(0, len(data), 1)

    plt.xticks([])
    plt.bar(x, data)
    

def plot_performance_metrics():
    fig, (ax, ax1) = plt.subplots(2,1, figsize=(8,7))

    base_model_on_squad = {"eval_exact_match": 76.34815515610218, "eval_f1": 84.63653172984633}
    base_model_on_adv = {"eval_exact_match": 48.87640449438202, "eval_f1": 56.000914991676616}
    base_model_on_custom_adv = {'eval_exact_match': 72.07190160832545, 'eval_f1': 78.96822803713549}

    trained_model_on_squad = {'eval_exact_match': 77.23746452223273, 'eval_f1': 85.3504380057704}
    trained_model_on_adv = {"eval_exact_match": 59.353932584269664, "eval_f1": 66.37959943513269}
    trained_model_on_custom_adv = {"eval_exact_match": 76.85903500473037, "eval_f1": 84.59640934275986}

    # categories = ("SQuAD", "SQuAD", "Adv", "Adv", "Custom", "Custom")
    # colors = ["blue", "orange", "blue", "orange", "blue", "orange"]
    categories = ("SQuAD", "Adv")
    colors = ["blue", "blue"]
    eval_exact_match = [
        76.34815515610218, 
        # 77.23746452223273, 
        48.87640449438202, 
        # 59.353932584269664,
        # 72.07190160832545,
        # 76.85903500473037
    ]
    eval_f1 = [
        84.63653172984633,
        # 85.3504380057704,
        56.000914991676616,
        # 66.37959943513269,
        # 78.96822803713549,
        # 84.59640934275986
    ]
    x = np.arange(len(categories))
    ax.bar(x, eval_exact_match, color=colors)
    ax.set_xticks([])
    for i, v in enumerate(eval_exact_match):
        ax.text(x[i], v - 8.5, str(round(v, 2)), ha="center", va="bottom", color="white")
    ax.set_title("Exact Match")

    ax1.bar(x, eval_f1, color=colors)
    ax1.set_xticks(x, labels=categories)
    for i, v in enumerate(eval_f1):
        ax1.text(x[i], v - 8.5, str(round(v, 2)), ha="center", va="bottom", color="white")
    ax1.set_title("F1")

    base_model_patch = mpatches.Patch(color='blue', label='Base Model')
    # custom_model_patch = mpatches.Patch(color='orange', label='StructAdv Model')

    # Add the legend with both the automatic and manual entries
    fig.legend(handles=[base_model_patch], loc='upper right')


def get_squad_adv_n_gram_metrics():
    data = []
    with jsonlines.open("eval/base_model_squad_adv/eval_predictions.jsonl", mode='r') as reader:
        for qa_pair in tqdm(reader):

            advers_sent = re.split(r"[\.\!\?]+", qa_pair["context"])[-2].lower()
            advers_sent = advers_sent.translate(str.maketrans('', '', string.punctuation)).split(" ")
            advers_sent = set(advers_sent)
            if '' in advers_sent:
                advers_sent.remove('')
            
            
            question = qa_pair["question"].lower()
            question = question.translate(str.maketrans('', '', string.punctuation)).split(" ")
            question = set(question)
            if '' in question:
                question.remove('')
            
            overlap = question.intersection(advers_sent)
            data.append(len(overlap) / len(question))

    data = np.array(data)

    x = np.arange(len(data))
    plt.xticks([])
    plt.bar(x, data)

if __name__ == "__main__":
    # plot_generated_count_histogram()
    # plot_performance_metrics()
    get_squad_adv_n_gram_metrics()
    plt.show()