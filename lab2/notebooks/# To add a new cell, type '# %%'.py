# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import nltk
import numpy as np

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

from importlib.util import find_spec

if find_spec("text_recognizer") is None:
    import sys

    sys.path.append("/workspace/lab2")

from text_recognizer.data.emnist_lines import (
    EMNISTLines,
    construct_image_from_string,
    get_samples_by_char,
)
from text_recognizer.data.sentence_generator import SentenceGenerator

# %% [markdown]
# ## Synthetic Dataset
#
# We can put together EMNIST characters into sequences.
# The sequences will be drawn from a natural language corpus.
#
# Let's start by just drawing 8 characters at a time, with no overlap between the letters.

# %%
sentence_generator = SentenceGenerator()
for _ in range(4):
    print(sentence_generator.generate(max_length=16))


# %%
import argparse

args = argparse.Namespace(max_length=16, max_overlap=0)
dataset = EMNISTLines(args)
dataset.prepare_data()
dataset.setup()
print(dataset)
print("Mapping:", dataset.mapping)


# %%
def convert_y_label_to_string(y, dataset=dataset):
    return "".join([dataset.mapping[i] for i in y])


y_example = dataset.data_train[0][1]
print(y_example, y_example.shape)
convert_y_label_to_string(y_example)


# %%
num_samples_to_plot = 9

for i in range(num_samples_to_plot):
    plt.figure(figsize=(20, 20))
    x, y = dataset.data_train[i]
    sentence = convert_y_label_to_string(y)
    print(sentence)
    plt.title(sentence)
    plt.imshow(x.squeeze(), cmap="gray")

# %% [markdown]
# ## Making it more difficult
#
# Let's now expand the maximum number of characters in a line, and add a random amount of overlap between the letters.

# %%
args = argparse.Namespace(max_length=34, max_overlap=0.33)
dataset = EMNISTLines(args)
dataset.prepare_data()
dataset.setup()
print(dataset)


# %%
num_samples_to_plot = 9

for i in range(num_samples_to_plot):
    plt.figure(figsize=(20, 20))
    x, y = dataset.data_train[i]
    sentence = convert_y_label_to_string(y)
    print(sentence)
    plt.title(sentence)
    plt.imshow(x.squeeze(), cmap="gray")
