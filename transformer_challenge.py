"""VS code cell block notebook for Stephen Casper's Transformer Challenge."""
# %%
import itertools
import os
import pickle

import tqdm

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from helpers import cross_entropy_high_precision, to_numpy
from transformers import (
    Config,
    gen_train_test,
    HookPoint,
    Transformer,
    Trainer,
    train_model,
)

# %%
p = 113
config = Config(
    lr=5e-4,
    weight_decay=0.1,
    p=p,
    d_model=128,
    frac_train=0.5,
    num_epochs=50000,
    stopping_thresh=-1,
    seed=0,
    num_layers=1,
    batch_style="full",
    d_vocab=p + 1,
    n_ctx=3,
    d_mlp=8 * 128,
    num_heads=8,
    act_type="ReLU",
)

# %%
train, test = gen_train_test(config)
all_pairs = [(i, j, p) for i, j in itertools.product(range(p), repeat=2)]

# Creates an array of Boolean indices according to whether each data point is in
# train or test
# Used to index into the big batch of all possible data
is_train = []
is_test = []
for i, j, p in all_pairs:
    if (i, j, p) in train:
        is_train.append(True)
        is_test.append(False)
    else:
        is_train.append(False)
        is_test.append(True)
is_train = np.array(is_train)
is_test = np.array(is_test)

# %%
model = Transformer(config, use_ln=False)
model.to("cuda")
# %%
save_dict = torch.load("transformer_model.pt")
model.load_state_dict(save_dict["model"])
# %%
with open("transformer_label_info.pkl", "rb") as f:
    label_info = pickle.load(f)
ground_truth_labels = label_info["labels"]

# %%
save_dict["epoch"]
# %%
logits = model(all_pairs)[:, -1]
model_labels = np.zeros((p, p))
for x in range(p):
    for y in range(p):
        if logits[x * p + y][0] < logits[x * p + y][1]:
            model_labels[x][y] = 1

train_acc = 1 - np.mean(
    np.abs(ground_truth_labels.flatten()[is_train] - model_labels.flatten()[is_train])
)
test_acc = 1 - np.mean(
    np.abs(ground_truth_labels.flatten()[is_test] - model_labels.flatten()[is_test])
)

print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
# %%
# train the model by hand
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.98),
)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
labels = torch.tensor(ground_truth_labels, dtype=torch.long).to("cuda").reshape(-1)
for n in range(config.num_epochs):
    # train the model
    logits = model(all_pairs)[:, -1]
    train_loss = cross_entropy_high_precision(logits, labels)
    test_loss = cross_entropy_high_precision(logits[is_test], labels[is_test])
    train_loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    print(f"Epoch {n}, train loss: {train_loss.item():.4f}")

    # get the logits
    logits = model(all_pairs)[:, -1]
    model_labels = np.zeros((p, p))
    for x in range(p):
        for y in range(p):
            if logits[x * p + y][0] < logits[x * p + y][1]:
                model_labels[x][y] = 1

    train_acc = 1 - np.mean(
        np.abs(
            ground_truth_labels.flatten()[is_train] - model_labels.flatten()[is_train]
        )
    )
    test_acc = 1 - np.mean(
        np.abs(ground_truth_labels.flatten()[is_test] - model_labels.flatten()[is_test])
    )
    print(f"\tTrain accuracy: {train_acc*100:.2f}%")
    print(f"\tTest accuracy: {test_acc*100:.2f}%")
    print()

# %%
# get ground truth labels in train and test (note that we need to reshape is_train  and is_test to 113*113)
ground_truth_labels_train = ground_truth_labels[is_train.reshape(113, 113)]
ground_truth_labels_test = ground_truth_labels[is_test.reshape(113, 113)]
model_labels_train = model_labels[is_train.reshape(113, 113)]
model_labels_test = model_labels[is_test.reshape(113, 113)]
# %%
# plot using grey scale, with white = 1 and black = 0
# increase plotly figure size
pio.templates.default = "plotly_white"
fig = px.imshow(
    ground_truth_labels,
    color_continuous_scale=["black", "white"],
    width=800,
    height=800,
)
# set x and y axis to be 0-113
# plot y = 133 - x on the heatmap
df = pd.DataFrame({"x": np.arange(20, 113), "y": 133 - np.arange(20, 113)})

fig = go.Figure(fig).add_trace(
    go.Scatter(x=df.x, y=df.y, mode="lines", line=dict(color="red", width=2))
)

fig.update_layout(
    title="Ground Truth Labels",
    xaxis_title="y",
    yaxis_title="x",
    xaxis=dict(tickmode="array", tickvals=np.arange(0, 113, 10)),
    yaxis=dict(tickmode="array", tickvals=np.arange(0, 113, 10), autorange=True),
)
fig.update_xaxes(range=[0, 113])
fig.update_yaxes(range=[0, 113])
fig.show()
# %%
from scipy.optimize import curve_fit

edge_points = [
    (29, 2),
    (36, 5),
    (39, 7),
    (42, 9),
    (48, 15),
    (51, 19),
    (60, 37),
    (63, 45),
    (65, 50),
    (68, 60),
    (74, 86),
    (75, 90),
    (76, 95),
    (77, 102),
]

# fit an exponential curve to the edge points
x = np.array([x for x, y in edge_points])
y = np.array([y - 1 for x, y in edge_points])

popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(1, 1e-6, 1))
y_fitted = np.round((popt[0] * np.exp(popt[1] * x) + popt[2]), 0)
print(f"Residual sum of squares: {np.sum((y_fitted - y) ** 2)}")
y_fitted = y_fitted[y_fitted < 113]

# plot the fitted curve
# fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
fig.add_trace(go.Scatter(x=x, y=y_fitted, mode="lines", name="fitted"))

# %%
# fit a degree 3 polynomial to the edge points
x = np.array([x for x, y in edge_points])
y = np.array([y for x, y in edge_points])

p, r, _, _, _ = np.polyfit(x, y, deg=3, full=True)
y_fitted = np.round(np.polyval(p, np.arange(0, 113)), 0)
# ensure y_fitted is less than 113
x_trimmed = np.arange(0, 113)[y_fitted < 113]
y_fitted = y_fitted[y_fitted < 113]
print(f"Residual sum of squares: {np.sum((np.round(np.polyval(p, x)) - y)**2)}")

# plot the fitted curve
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
fig.add_trace(go.Scatter(x=np.arange(0, 113), y=y_fitted, mode="lines", name="fitted"))
fig.update_yaxes(range=[0, 113])
fig.update_xaxes(range=[0, 113])
# %%
deg = 10
p, r, _, _, _ = np.polyfit(x, y, deg=deg, full=True)
y_fitted = np.round(np.polyval(p, x), 0)
print(f"Residual sum of squares normal rounding: {np.sum((y_fitted - y)**2)}")

# now round down
y_fitted = np.floor(np.polyval(p, x))
print(f"Residual sum of squares floor rounding: {np.sum((y_fitted - y)**2)}")

# round up
y_fitted = np.ceil(np.polyval(p, x))
print(f"Residual sum of squares ceil rounding: {np.sum((y_fitted - y)**2)}")

# %%
popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(1, 1e-6, 1))
y_fitted = np.round((popt[0] * np.exp(popt[1] * x) + popt[2]), 0)
y_fitted = y_fitted[y_fitted < 113]

# round normally
print(f"Residual sum of squares normal rounding: {np.sum((y_fitted - y)**2)}")

# round down
y_fitted = np.floor((popt[0] * np.exp(popt[1] * x) + popt[2]))
print(f"Residual sum of squares floor rounding: {np.sum((y_fitted - y)**2)}")

# round up
y_fitted = np.ceil((popt[0] * np.exp(popt[1] * x) + popt[2]))
print(f"Residual sum of squares ceil rounding: {np.sum((y_fitted - y)**2)}")

# %%
popt
# %%
from transformers import make_fourier_basis

# %%
px.imshow(logits.reshape(113, 113))
# %%
W_E = model.embed.W_E[:, :-1]
W_U = model.unembed.W_U[:, :-1].T
W_out = model.blocks[0].mlp.W_out
# %%
px.imshow(to_numpy(W_E))
# %%
px.imshow(to_numpy(W_U))
# %%
basis = make_fourier_basis(config)

W_E @ basis

# %%
W_Lff = basis @ W_U @ W_out
# %%
px.imshow(to_numpy(W_Lff))
# %%

logits = model(all_pairs)[:, -1, :2]
logits.retain_grad()
train_loss = cross_entropy_high_precision(logits, labels)
test_loss = cross_entropy_high_precision(logits[is_test], labels[is_test])
train_loss.backward()
# %%
grad = logits.grad[:]
# %%
grad.shape
# %%
logits.shape
# %%
model(all_pairs).shape
# %%
logits.shape
# %%
logits = logits.reshape(113, 113, 2)
# %%
px.imshow(to_numpy(logits[:, :, 0]))
# %%
fig = px.imshow(to_numpy(logits[:, :, 1]))
fig.update_layout(yaxis=dict(autorange=True))

# %%
pio.templates.default = "plotly_white"
fig = px.imshow(
    ground_truth_labels,
    color_continuous_scale=["black", "white"],
    width=800,
    height=800,
)
# set x and y axis to be 0-113
# plot y = 133 - x on the heatmap
df = pd.DataFrame({"x": np.arange(20, 113), "y": 133 - np.arange(20, 113)})

fig = go.Figure(fig).add_trace(
    go.Scatter(x=df.x, y=df.y, mode="lines", line=dict(color="red", width=2))
)

fig.update_layout(
    title="Ground Truth Labels",
    xaxis_title="y",
    yaxis_title="x",
    xaxis=dict(tickmode="array", tickvals=np.arange(0, 113, 10)),
    yaxis=dict(tickmode="array", tickvals=np.arange(0, 113, 10), autorange=True),
)
fig.update_xaxes(range=[0, 113])
fig.update_yaxes(range=[0, 113])
fig.show()

# %%
logits = logits.reshape(113, 113, -1)
grad = grad.reshape(113, 113, -1)
# %%
from plotly.subplots import make_subplots

# show both logits and ground truth labels next to each other
fig = make_subplots(
    rows=1,
    cols=3,
    column_widths=[0.33, 0.33, 0.33],
    subplot_titles=("Logits Gradients", "Model Labels", "Ground Truth Labels"),
)

fig.add_trace(
    go.Heatmap(
        z=to_numpy(grad[:, :, 0]),
        showscale=False,
        x=np.arange(0, 113),
        y=np.arange(0, 113),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=model_labels,
        showscale=False,
        colorscale=["black", "white"],
        x=np.arange(0, 113),
        y=np.arange(0, 113),
    ),
    row=1,
    col=2,
)


fig.add_trace(
    go.Heatmap(
        z=ground_truth_labels,
        colorscale=["black", "white"],
        showscale=False,
        x=np.arange(0, 113),
        y=np.arange(0, 113),
    ),
    row=1,
    col=3,
)
fig.add_trace(
    go.Scatter(x=df.x, y=df.y, mode="lines", line=dict(color="red", width=2)),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(x=df.x, y=df.y, mode="lines", line=dict(color="red", width=2)),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(x=df.x, y=df.y, mode="lines", line=dict(color="red", width=2)),
    row=1,
    col=3,
)

# %%
model(all_pairs).shape
# %%
all_pairs
# %%
einops.rearrange(W_E[:, all_pairs], "n d -> n () d")
# %%
W_E[:, all_pairs].shape
# %%
# bar plot of ground_truth_labels summed x axis
fig = px.bar(
    x=np.arange(0, 113),
    y=np.sum(ground_truth_labels, axis=0),
    labels={"x": "y", "y": "Number of 1s"},
)
fig.show()
# %%
fig = px.bar(
    x=np.arange(0, 113),
    y=np.sum(ground_truth_labels, axis=1),
    labels={"x": "y", "y": "Number of 1s"},
)
fig.show()
# %%
ground_truth_labels_shifted = np.roll(ground_truth_labels, -1, axis=1)
fig = px.imshow(ground_truth_labels_shifted)
fig.show()

# %%

shifts = -np.arange(0, 113)
for i, (shift, row) in enumerate(zip(shifts, ground_truth_labels)):
    ground_truth_labels_shifted[i] = np.roll(row, shift, axis=0)

fig = px.imshow(ground_truth_labels_shifted)
fig.show()
# %%
