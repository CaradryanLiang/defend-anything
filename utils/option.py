import logging
import os
import os.path as osp
import sys
from collections import OrderedDict

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.
    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, root_path=".", is_train=True):

    opt_path = osp.abspath(opt_path)
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    print(opt)
    return opt
