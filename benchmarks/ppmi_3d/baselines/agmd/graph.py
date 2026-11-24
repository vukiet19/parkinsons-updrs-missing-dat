"""Anatomical-guided graph learning (Sec. 2.3, Eq. 3-6) for 2D/3D feature maps.

Nodes. Official code (orig_repo/utils.py:100-132, model.py:168-198): AAL3 ROIs of an atlas resampled to the
feature-map size; each ROI is a node of the DEEPEST graph layer where it still exists ("private nodes":
ROIs of layer l that vanish at layer l+1; the last layer keeps all its ROIs); node feature = mean of the
feature map inside the ROI. Two node layouts are offered here:

  * ``atlas`` -- the official scheme, for MNI-registered 3D volumes (PPMI). Needs an integer label volume.
  * ``grid``  -- [Proposal] stand-in for images that are not atlas-registered (NTUA 2D slices): every graph
    layer is split into ``grid`` non-overlapping blocks per axis (torch.tensor_split, as DFTD-dagger).
  * ``atlas_area`` -- [Proposal] STEWARD D-032 (PPMI 3D): every atlas label is a node at every graph layer
    (no private nodes); node feature = feature map averaged with area (volume-fraction) weights of the label in
    each cell, so key ROIs never vanish at coarse stages (prompt P1b, notes/AGMD_atlas_analysis.md).

Edges. Semantic graph (Eq. 3-4): per subject, k nearest neighbours by cosine similarity of the (detached)
node features, symmetrised, no self loops [Code utils.py:190-229]; k = ceil(sqrt(N_V)) per layer [Paper Sec.
2.3]. Anatomical graph (Eq. 5): edges between the M key (disease-sensitive) nodes and their spatially
adjacent nodes, symmetric [Paper]; fused adjacency A = A_sem + A_anat [Paper].

GCN (Eq. 6): two GCNConv layers, ReLU after the first only, hidden 64 [Code model.py:201-221], implemented
densely with PyG's GCNConv normalisation D^-1/2 (A + I) D^-1/2 (torch_geometric is not a dependency here).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------------------
def resnet_stage_sizes(input_size: Sequence[int]) -> list[list[int]]:
    """Spatial size of the 4 ResNet-18 stages (stride 4/8/16/32 with ceil at every stride-2 op)."""
    size = [int(s) for s in input_size]
    size = [math.ceil(s / 2) for s in size]          # stem conv 7x7 s2 p3
    size = [math.ceil(s / 2) for s in size]          # max-pool 3 s2 p1
    out = [list(size)]
    for _ in range(3):                               # stages 2-4: first conv 3x3 s2 p1
        size = [math.ceil(s / 2) for s in size]
        out.append(list(size))
    return out


def grid_masks(size: Sequence[int], grid: int) -> torch.Tensor:
    """(N, *size) one-hot masks of grid^dims non-overlapping blocks, row-major over the axes."""
    dims = len(size)
    for s in size:
        if s < grid:
            raise ValueError(f"Feature map {tuple(size)} is smaller than the node grid {grid}.")
    bounds = [[len(part) for part in torch.tensor_split(torch.arange(s), grid)] for s in size]
    masks = []
    for index in torch.cartesian_prod(*[torch.arange(grid)] * dims).tolist() if dims > 1 else [[i] for i in range(grid)]:
        m = torch.zeros(size)
        sl = []
        for axis, i in enumerate(index):
            start = sum(bounds[axis][:i])
            sl.append(slice(start, start + bounds[axis][i]))
        m[tuple(sl)] = 1.0
        masks.append(m)
    return torch.stack(masks)


def grid_key_nodes(grid: int, dims: int, key_size: int) -> list[int]:
    """[Assumption] key nodes of the grid layout = the central key_size^dims block of the grid."""
    start = (grid - key_size) // 2
    centre = range(start, start + key_size)
    keys = []
    for flat in range(grid ** dims):
        index = [(flat // grid ** (dims - 1 - a)) % grid for a in range(dims)]
        if all(i in centre for i in index):
            keys.append(flat)
    return keys


def grid_adjacent_pairs(grid: int, dims: int) -> set[tuple[int, int]]:
    """Face-adjacent block pairs of the grid (4-neighbourhood in 2D, 6 in 3D), both directions."""
    pairs = set()
    for flat in range(grid ** dims):
        index = [(flat // grid ** (dims - 1 - a)) % grid for a in range(dims)]
        for axis in range(dims):
            if index[axis] + 1 < grid:
                other = flat + grid ** (dims - 1 - axis)
                pairs.add((flat, other))
                pairs.add((other, flat))
    return pairs


def load_atlas(path: str | Path) -> torch.Tensor:
    """Integer label volume (NIfTI or a tensor saved with torch.save)."""
    path = Path(path)
    if path.suffix == ".pt":
        atlas = torch.load(path, map_location="cpu", weights_only=True)
    else:
        import nibabel as nib
        import numpy as np
        atlas = torch.from_numpy(np.asarray(nib.load(str(path)).get_fdata(dtype="float32")))
    return atlas.round().long()


def resample_labels(atlas: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Nearest-neighbour resampling of a label volume (code: pre-computed raal3_{l}.pt per layer)."""
    x = atlas[None, None].float()
    return F.interpolate(x, size=tuple(int(s) for s in size), mode="nearest")[0, 0].round().long()


def atlas_adjacency(atlas: torch.Tensor) -> set[tuple[int, int]]:
    """Pairs of different labels sharing a voxel face in the full-resolution atlas (both directions).
    [Proposal] automatic version of the hand-written AAL3_ADJACENT_ROIS table (orig_repo/utils.py:135-145)."""
    pairs = set()
    for axis in range(atlas.ndim):
        a = atlas.narrow(axis, 0, atlas.shape[axis] - 1).flatten()
        b = atlas.narrow(axis, 1, atlas.shape[axis] - 1).flatten()
        diff = a != b
        for u, v in torch.unique(torch.stack([a[diff], b[diff]], 1), dim=0).tolist():
            pairs.add((u, v))
            pairs.add((v, u))
    return pairs


def area_coverage(atlas: torch.Tensor, size: Sequence[int], labels: Sequence[int]) -> torch.Tensor:
    """Area (volume-fraction) coverage of every label in every feature-map cell (STEWARD D-032, prompt P1b).

    The atlas must be on the INPUT grid. Input voxel i belongs to cell floor(i * out / in) on every axis (the index
    map of F.interpolate(mode="nearest"), as in the P1a analysis), so cells do not overlap and every voxel counts
    once. Returns float64 (N, *size) with coverage[n, cell] = #voxels of labels[n] in cell / #voxels of cell.
    [Assumption] Padding and receptive fields are ignored.
    """
    size = [int(s) for s in size]
    if atlas.ndim != len(size):
        raise ValueError(f"Atlas has {atlas.ndim} dims, feature map {len(size)}")
    axes = [(torch.arange(n) * o) // n for n, o in zip(atlas.shape, size)]
    cell = torch.zeros(atlas.shape, dtype=torch.long)
    for a, idx in enumerate(axes):
        shape = [1] * atlas.ndim
        shape[a] = -1
        cell = cell * size[a] + idx.view(shape)
    lut = torch.full((int(atlas.max()) + 1,), -1, dtype=torch.long)
    lut[torch.tensor([int(v) for v in labels])] = torch.arange(len(labels))
    lab = lut[atlas.long()]
    n_cells = math.prod(size)
    keep = lab >= 0
    counts = torch.bincount(lab[keep] * n_cells + cell[keep], minlength=len(labels) * n_cells)
    per_cell = torch.bincount(cell.flatten(), minlength=n_cells).double()
    return (counts.view(len(labels), n_cells).double() / per_cell).view(len(labels), *size)


def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_label_sets(label_maps: list[torch.Tensor], include_background: bool) -> list[list[int]]:
    """orig_repo/utils.py:109-118: layer l keeps ROIs absent from layer l+1; the last layer keeps all."""
    sets = [set(torch.unique(m).tolist()) for m in label_maps]
    if not include_background:
        sets = [s - {0} for s in sets]
    out = []
    for i, s in enumerate(sets):
        out.append(sorted(s - sets[i + 1]) if i + 1 < len(sets) else sorted(s))
    return out


# ---------------------------------------------------------------------------------------------------------
# graph layout (static per model) and semantic edges (per subject)
# ---------------------------------------------------------------------------------------------------------
class GraphLayout(nn.Module):
    """Node masks and anatomical adjacency of every graph layer (registered as buffers)."""

    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        g = cfg["graph"]
        self.layers = [int(l) for l in g["layers"]]
        dims = int(protocol["data"]["spatial_dims"])
        sizes = resnet_stage_sizes(protocol["preprocessing"]["input_size"])
        self.n_nodes, self.k = [], []
        self.mode = g["nodes"]
        self.stage_sizes = [list(sizes[l]) for l in self.layers]
        self.atlas_file, self.atlas_sha256, self.key_ids = None, None, []
        self.key_sum_coverage = [dict() for _ in self.layers]
        weights = None                                  # set only by atlas_area (already normalised)
        if g["nodes"] in ("atlas", "atlas_area"):
            if not g.get("atlas_path"):
                raise ValueError(f"graph.nodes={g['nodes']} needs graph.atlas_path (MNI-registered 3D data only)")
            self.atlas_file = str(_resolve(g["atlas_path"]))
            self.atlas_sha256 = file_sha256(self.atlas_file)
            if g.get("atlas_sha256") and g["atlas_sha256"] != self.atlas_sha256:
                raise ValueError(f"Atlas {self.atlas_file} has SHA-256 {self.atlas_sha256}, config expects "
                                 f"{g['atlas_sha256']}")
            self.key_ids = [int(v) for v in g["atlas_key_rois"]]
        if g["nodes"] == "grid":
            grids = [int(v) for v in g["grid"]]
            if len(grids) != len(self.layers):
                raise ValueError("graph.grid needs one entry per graph layer")
            masks = [grid_masks(sizes[l], gr) for l, gr in zip(self.layers, grids)]
            anat = []
            for gr in grids:
                keys = set(grid_key_nodes(gr, dims, int(g["grid_key_size"])))
                anat.append({(u, v) for (u, v) in grid_adjacent_pairs(gr, dims) if u in keys or v in keys})
            self.node_ids = [list(range(m.shape[0])) for m in masks]
        elif g["nodes"] == "atlas":
            if not g.get("atlas_path"):
                raise ValueError("graph.nodes=atlas needs graph.atlas_path (MNI-registered 3D data only)")
            atlas = load_atlas(_resolve(g["atlas_path"]))
            label_maps = [resample_labels(atlas, sizes[l]) for l in self.layers]
            labels = private_label_sets(label_maps, bool(g["atlas_include_background"]))
            masks = [torch.stack([(m == v).float() for v in labs]) for m, labs in zip(label_maps, labels)]
            adjacent, keys = atlas_adjacency(atlas), {int(v) for v in g["atlas_key_rois"]}
            anat = []
            for labs in labels:
                pos = {v: i for i, v in enumerate(labs)}
                anat.append({(pos[u], pos[v]) for (u, v) in adjacent
                             if u in pos and v in pos and (u in keys or v in keys)})
            self.node_ids = labels
            for l, labs in zip(self.layers, labels):    # known behaviour of the official scheme: warn, do not fail
                lost = [v for v in sorted(keys) if v not in labs]
                if lost:
                    print(f"[agmd graph] WARNING nodes=atlas: key ROIs {lost} are not nodes at stage {l} "
                          f"(nearest + private nodes)", flush=True)
        elif g["nodes"] == "atlas_area":
            # STEWARD D-032 (a)+(c): every atlas label is a node at EVERY graph layer (no private nodes); node feature
            # = coverage-weighted mean of the feature map. [Proposal] resolution-driven change, neither paper nor code.
            atlas = load_atlas(self.atlas_file)
            in_size = [int(s) for s in protocol["preprocessing"]["input_size"]]
            if list(atlas.shape) != in_size:
                raise ValueError(f"nodes=atlas_area needs the atlas on the input grid {in_size}, got {list(atlas.shape)}")
            labs = sorted(int(v) for v in torch.unique(atlas).tolist())
            if not g["atlas_include_background"]:
                labs = [v for v in labs if v != 0]
            missing = [v for v in self.key_ids if v not in labs]
            if missing:
                raise ValueError(f"Key ROIs {missing} are absent from the atlas {self.atlas_file}")
            adjacent, keys = atlas_adjacency(atlas), set(self.key_ids)
            pos = {v: i for i, v in enumerate(labs)}
            edges = {(pos[u], pos[v]) for (u, v) in adjacent
                     if u in pos and v in pos and u != v and (u in keys or v in keys)}
            masks, weights, anat = [], [], []
            for li, l in enumerate(self.layers):
                cov = area_coverage(atlas, sizes[l], labs)
                total = cov.flatten(1).sum(1)
                if (total <= 0).any():
                    empty = [labs[i] for i in torch.nonzero(total <= 0).flatten().tolist()]
                    raise ValueError(f"Labels {empty} have zero coverage at stage {l}")
                masks.append(cov.float())
                weights.append((cov / total.view(-1, *([1] * dims))).float())
                anat.append(edges)
                self.key_sum_coverage[li] = {v: float(total[pos[v]]) for v in self.key_ids}
            self.node_ids = [list(labs) for _ in self.layers]
        else:
            raise ValueError(f"Unknown graph.nodes: {g['nodes']}")
        for i, (m, edges) in enumerate(zip(masks, anat)):
            n = m.shape[0]
            if n < 2:
                raise ValueError(f"Graph layer {self.layers[i]} has {n} node(s)")
            if weights is not None:                     # atlas_area: sum_cells coverage / sum(coverage)
                weight = weights[i]
            else:
                weight = m / m.flatten(1).sum(1).clamp_min(1.0).view(-1, *([1] * dims))   # mean inside the node
            a = torch.zeros(n, n)
            if g["anatomical"]:
                for u, v in edges:
                    a[u, v] = a[v, u] = 1.0                      # Eq. 5 is symmetric [Paper]
            self.register_buffer(f"pool_{i}", weight, persistent=False)
            self.register_buffer(f"anat_{i}", a, persistent=False)
            self.n_nodes.append(n)
            self.k.append(_knn_k(g["knn"], n))
        if self.mode in ("atlas", "atlas_area"):
            print("[agmd graph] " + json.dumps(self.node_info()), flush=True)

    def node_info(self) -> dict:
        """Per graph layer: stage, size, #nodes, k, #undirected anatomical edges, key ROIs present and (atlas_area)
        sum of coverage of every key ROI. Used for logs and checks (prompt P1b section 4.2.7)."""
        layers = []
        for i, l in enumerate(self.layers):
            present = [v for v in self.key_ids if v in self.node_ids[i]]
            layers.append(dict(stage=l, size=self.stage_sizes[i], n_nodes=self.n_nodes[i], k=self.k[i],
                               anat_edges=int(torch.triu(self.anatomical(i), 1).sum()),
                               key_rois_present=present, n_key_present=len(present),
                               key_sum_coverage={str(k): round(v, 6) for k, v in self.key_sum_coverage[i].items()}))
        return dict(mode=self.mode, atlas=self.atlas_file, atlas_sha256=self.atlas_sha256, layers=layers)

    def pool(self, i: int) -> torch.Tensor:
        return getattr(self, f"pool_{i}")

    def deployment_payload(self) -> dict:
        """Freeze the exact pooling geometry; deployment never needs the atlas file."""
        names = ("layers", "stage_sizes", "n_nodes", "k", "mode", "node_ids", "key_ids", "key_sum_coverage",
                 "atlas_sha256")
        return dict(metadata={key: getattr(self, key) for key in names},
                    pools=[self.pool(i).detach().cpu().clone() for i in range(len(self.layers))],
                    anatomical=[self.anatomical(i).detach().cpu().clone() for i in range(len(self.layers))])

    @classmethod
    def from_deployment_payload(cls, payload: dict):
        """Restore tensor-only geometry without opening an atlas or a training manifest."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        for key, value in payload["metadata"].items():
            if key not in ("layers", "stage_sizes", "n_nodes", "k", "mode", "node_ids", "key_ids",
                           "key_sum_coverage", "atlas_sha256"):
                raise ValueError(f"Unexpected graph-layout field: {key}")
            setattr(obj, key, value)
        obj.atlas_file = None
        count = len(obj.layers)
        if not count or any(len(payload[key]) != count for key in ("pools", "anatomical")):
            raise ValueError("Incomplete deployment graph layout")
        for i, (pool, anat) in enumerate(zip(payload["pools"], payload["anatomical"])):
            n = obj.n_nodes[i]
            if list(pool.shape) != [n, *obj.stage_sizes[i]] or list(anat.shape) != [n, n]:
                raise ValueError("Deployment graph-layout shape mismatch")
            if not torch.isfinite(pool).all() or not torch.isfinite(anat).all() or (pool < 0).any() or (anat < 0).any():
                raise ValueError("Invalid deployment graph-layout values")
            if not torch.allclose(pool.flatten(1).sum(1), torch.ones(n), atol=1e-5):
                raise ValueError("Deployment graph pooling must sum to one per node")
            if not torch.equal(anat, anat.T) or not 0 < obj.k[i] < n:
                raise ValueError("Invalid deployment graph adjacency or k")
            obj.register_buffer(f"pool_{i}", pool.clone(), persistent=False)
            obj.register_buffer(f"anat_{i}", anat.clone(), persistent=False)
        return obj

    def anatomical(self, i: int) -> torch.Tensor:
        return getattr(self, f"anat_{i}")

    def node_features(self, i: int, fmap: torch.Tensor) -> torch.Tensor:
        """(B, C, *S) -> (B, N, C): mean of the feature map inside each node [Code utils.py:123-132]."""
        return torch.einsum("bcs,ns->bnc", fmap.flatten(2), self.pool(i).flatten(1))


def _knn_k(spec, n: int) -> int:
    if spec == "sqrt_n":                    # Paper Sec. 2.3: k = sqrt(N_V)
        return max(1, min(n - 1, math.ceil(math.sqrt(n))))
    return max(1, min(n - 1, int(spec)))


def _resolve(path: str) -> Path:
    from utils.config import resolve_path
    return resolve_path(path)


@torch.no_grad()
def semantic_adjacency(h: torch.Tensor, k: int) -> torch.Tensor:
    """Eq. 3-4 per subject: cosine kNN on detached node features, symmetrised, no self loops.
    (B, N, C) -> (B, N, N) in {0, 1}. [Code utils.py:190-229 computes it on CPU with sklearn]"""
    x = F.normalize(h.detach().float(), dim=-1)
    sim = x @ x.transpose(1, 2)
    eye = torch.eye(sim.shape[1], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, float("-inf"))
    idx = sim.topk(k, dim=-1).indices
    a = torch.zeros_like(sim).scatter_(-1, idx, 1.0)
    return torch.maximum(a, a.transpose(1, 2))


def gcn_norm(a: torch.Tensor) -> torch.Tensor:
    """PyG GCNConv normalisation: D^-1/2 (A + I) D^-1/2 with edge multiplicities as weights. (B, N, N)."""
    a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    d = a.sum(-1).pow(-0.5)
    return d.unsqueeze(-1) * a * d.unsqueeze(-2)


class DenseGCNConv(nn.Module):
    """out = norm(A) @ (X W) + b, the dense equivalent of torch_geometric.nn.GCNConv (glorot W, zero b)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=False)
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.lin(x) + self.bias


class GCNModule(nn.Module):
    """orig_repo/model.py:201-221: GCNConv -> ReLU -> GCNConv (drop_edge is defined there but never used)."""

    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.conv1 = DenseGCNConv(cin, hidden)
        self.conv2 = DenseGCNConv(hidden, hidden)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x, a_norm)), a_norm)


class BrainGraph(nn.Module):
    """Feature maps of the graph layers -> fused graphs -> GCN node embeddings H^l (Eq. 3-7).

    Returns the node embeddings of all layers concatenated as in orig_repo/model.py:297-305 (deepest layer
    first), shape (B, sum_l N_l, hidden), plus the adjacency matrices (for checks)."""

    def __init__(self, protocol: dict, cfg: dict, channels: Sequence[int], layout: GraphLayout | None = None):
        super().__init__()
        self.layout = GraphLayout(protocol, cfg) if layout is None else layout
        hidden = int(cfg["graph"]["gcn_hidden"])
        self.gcn = nn.ModuleList([GCNModule(int(channels[l]), hidden) for l in self.layout.layers])
        self.n_nodes = sum(self.layout.n_nodes)
        self.hidden = hidden

    def forward(self, feats: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outs, adjs = [], []
        for i in reversed(range(len(self.layout.layers))):          # code order: layer -1, then -2
            h = self.layout.node_features(i, feats[self.layout.layers[i]])
            a = semantic_adjacency(h, self.layout.k[i]) + self.layout.anatomical(i)   # A_fused, Sec. 2.3
            adjs.append(a)
            outs.append(self.gcn[i](h, gcn_norm(a)))
        return torch.cat(outs, dim=1), adjs
