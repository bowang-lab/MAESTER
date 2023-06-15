import os
from typing import Dict
import torch
import yaml
import numpy as np


class PluginManager:
    """
    Plugin manager for MAESTER, which is used to register and get plugins.

    Code is adopted from https://gist.github.com/mepcotterell/6004997
    """

    def __init__(self):
        self.plugin_container: Dict[str : Dict[str:object]] = {}  # type: ignore

    def register_plugin(
        self, plugin_type: str, plugin_name: str, plugin_object: object
    ):
        if plugin_type not in self.plugin_container:
            self.plugin_container[plugin_type] = {}

        self.plugin_container[plugin_type][plugin_name] = plugin_object

    def get(self, plugin_type: str, plugin_name: str):
        return self.plugin_container[plugin_type][plugin_name]


def register_plugin(plugin_type: str, plugin_name: str):
    def decorator(cls):
        plugin_manager.register_plugin(plugin_type, plugin_name, cls)
        return cls

    return decorator


def get_plugin(plugin_type: str, plugin_name: str):
    return plugin_manager.get(plugin_type, plugin_name)


def read_yaml(path) -> Dict:
    """
    Helper function to read yaml file
    """
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    data_dict = yaml.safe_load(string)

    return data_dict


def save_checkpoint(path, state_dict, name):
    filename = os.path.join(path, name)
    torch.save(state_dict, filename)
    print("Saving checkpoint:", filename)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    code is adopted from https://github.com/facebookresearch/mae
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


plugin_manager = PluginManager()
