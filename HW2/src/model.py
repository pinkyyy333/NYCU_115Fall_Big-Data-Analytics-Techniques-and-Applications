"""A small heterogeneous GNN using PyTorch only (no PyG installation required)."""
import math

import numpy as np
import torch
from torch import nn


def relation_name(key):
    return "__".join(key)


class RelationalLayer(nn.Module):
    def __init__(self, node_types, relations, hidden):
        super().__init__()
        self.self_linear = nn.ModuleDict({name: nn.Linear(hidden, hidden) for name in node_types})
        self.relation_linear = nn.ModuleDict({relation_name(key): nn.Linear(hidden, hidden, bias=False)
                                             for key in relations})
        self.norm = nn.ModuleDict({name: nn.LayerNorm(hidden) for name in node_types})

    def forward(self, values, edges):
        out = {name: self.self_linear[name](value) for name, value in values.items()}
        incoming = {name: 0 for name in values}
        for key, edge in edges.items():
            source, _, destination = key
            if edge.shape[1] == 0:
                continue
            count = values[destination].shape[0]
            aggregated = torch.zeros_like(values[destination])
            aggregated.index_add_(0, edge[1], values[source][edge[0]])
            degree = torch.bincount(edge[1], minlength=count).clamp_min(1).unsqueeze(1)
            aggregated = aggregated / degree
            out[destination] = out[destination] + self.relation_linear[relation_name(key)](aggregated)
            incoming[destination] += 1
        return {name: torch.relu(self.norm[name](out[name] / math.sqrt(1 + incoming[name])))
                for name in values}


class ActivityModel(nn.Module):
    def __init__(self, dimensions, relations, hidden=32, layers=2, relational=True):
        super().__init__()
        self.relational = relational
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
                                       for name, dim in dimensions.items()})
        self.layers = nn.ModuleList([RelationalLayer(dimensions, relations, hidden)
                                    for _ in range(layers if relational else 0)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, features, edges, cohort_nodes):
        if self.relational:
            values = {name: self.encoders[name](x) for name, x in features.items()}
            for layer in self.layers:
                values = layer(values, edges)
            user = values["Users"][cohort_nodes]
        else:
            user = self.encoders["Users"](features["Users"][cohort_nodes])
        return self.head(user)


def tensors(snapshot, device="cpu"):
    return ({name: torch.as_tensor(x, dtype=torch.float32, device=device)
             for name, x in snapshot.features.items()},
            {key: torch.as_tensor(x, dtype=torch.long, device=device)
             for key, x in snapshot.edges.items()},
            torch.as_tensor(snapshot.cohort_nodes, dtype=torch.long, device=device))


def fit(train_snapshots, targets, config, relational=True):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    first = train_snapshots[0]
    dimensions = {name: value.shape[1] for name, value in first.features.items()}
    model = ActivityModel(dimensions, first.edges, config["hidden_dim"],
                          config["graph_layers"], relational)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    all_y = np.concatenate(targets)
    positives = all_y.sum(0)
    weight = np.minimum((len(all_y) - positives) / np.maximum(positives, 1), 15)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    prepared = [tensors(s) for s in train_snapshots]
    target_tensors = [torch.tensor(y, dtype=torch.float32) for y in targets]
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        for inputs, target in zip(prepared, target_tensors):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{config['epochs']}, loss={history[-1]:.5f}", flush=True)
    return model, history


@torch.no_grad()
def predict(model, snapshot):
    model.eval()
    return torch.sigmoid(model(*tensors(snapshot))).numpy()


def tune_thresholds(probabilities, target):
    """Tune each label's F1 on the separate historical tuning week only."""
    thresholds = np.full(6, 0.5)
    grid = np.r_[np.linspace(0.05, 0.95, 37), 0.99, 1.01]
    for k in range(6):
        best = (-1, -1)
        for threshold in grid:
            p, y = probabilities[:, k] >= threshold, target[:, k].astype(bool)
            tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
            score = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
            # Prefer the larger threshold when F1 ties, to avoid needless positives.
            if (score, threshold) > best:
                best = (score, threshold)
        thresholds[k] = best[1]
    return thresholds
