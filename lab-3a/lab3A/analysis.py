
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2Model
import os
import sys
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict
from networkx.drawing.nx_agraph import to_agraph
from IPython.display import Image
import networkx as nx

from transformers.utils.fx import (
    symbolic_trace as symbolic_trace_transformers,
)

import pandas as pd
import time

def trace(model):
    graphmodule: torch.fx.GraphModule = transformers.utils.fx.symbolic_trace(model)
    return graphmodule

def dump_graph(graphmodule):
    entries = []

    for node in graphmodule.graph.nodes:
        shape = None if not hasattr(node, "shape") else node.shape
        latency = None if not hasattr(node, "latency") else node.latency

        entry = {"op":node.op, "target":node.target, "args":node.args, "kwargs":node.kwargs, "shape":shape, "latency":latency,}
        entries.append(entry)

    entries = pd.DataFrame(entries)
    display(entries)
    entries.to_csv("nodes.csv")

#---------------------------------------------------#
#---------- Do not modify above this line ----------#
#---------------------------------------------------#

# This function prints out some of the attributes of the nodes you may find helpful. This function is provided only for debugging, so you may modify this function at your convenience.
def print_graph(graphmodule):
    for node in graphmodule.graph.nodes:
        entry = {"op":node.op, "target":node.target, "args":node.args, "kwargs":node.kwargs}
        print(entry)

class NodeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        # This loop iterates through all the nodes in the given graph.
        # Set shape and latency attributes of each node to proper value.
        # You may find it helpful to refer to the result of print_graph to start

        node_to_val = {}
        placeholder_idx = 0

        # define helper function.
        def eval_arg(arg):
          if isinstance(arg, Node):
              return node_to_val[arg]
          elif isinstance(arg, torch.Tensor):
            # If the tensor has a single element and is of integer type, convert it to an int.
            if arg.numel() == 1 and arg.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
              return int(arg.item())
            else:
              return arg
          elif isinstance(arg, slice):  # <-- NEW BRANCH for slice objects
              return slice(
                  eval_arg(arg.start) if arg.start is not None else None,
                  eval_arg(arg.stop) if arg.stop is not None else None,
                  eval_arg(arg.step) if arg.step is not None else None
              )
          elif isinstance(arg, (list, tuple)):
              # print(f"list or tuple instance: {arg}")
              return type(arg)(eval_arg(x) for x in arg)
          elif isinstance(arg, dict):
              # print(f"dict instance: {arg}")
              return {k: eval_arg(v) for k, v in arg.items()}
          else:
              return arg

        def get_latency(fn, *args, **kwargs):
          times = []
          iterations = 10
          for _ in range(iterations):
            start = time.time()
            _ = fn(*args, **kwargs)
            times.append((time.time() - start)*1000)
          return sum(times) / iterations

        for node in self.graph.nodes:
          node.latency = 0

          ## set shapes.
          if node.op == 'placeholder':
            val = args[placeholder_idx]
            placeholder_idx += 1
            node_to_val[node] = val
                      
          elif node.op == 'get_attr':
            try:
              val = getattr(self.mod, node.target)
            except AttributeError:
              val = self.mod.state_dict()[node.target]
            node_to_val[node] = val

          elif node.op == 'call_module':
              evaled_args = eval_arg(node.args)
              evaled_kwargs = eval_arg(node.kwargs)
              submod = self.modules[node.target]
              val = submod(*evaled_args, **evaled_kwargs)
              node_to_val[node] = val
              node.latency = get_latency(submod, *evaled_args, **evaled_kwargs)

          elif node.op == 'call_method':
            evaled_args = eval_arg(node.args)
            evaled_kwargs = eval_arg(node.kwargs)
            obj = evaled_args[0]
            method = getattr(obj, node.target)
            # print(f"method = {method}, evaled_args = {evaled_args}")
            val = method(*evaled_args[1:], **evaled_kwargs)
            node_to_val[node] = val

            if node.target == 'size':
              node.shape = None
            
            node.latency = get_latency(method, *evaled_args[1:], **evaled_kwargs)

          elif node.op == 'call_function':
            evaled_args = eval_arg(node.args)
            evaled_kwargs = eval_arg(node.kwargs)

            val = node.target(*evaled_args, **evaled_kwargs)
            node_to_val[node] = val

            node.latency = get_latency(node.target, *evaled_args, **evaled_kwargs)

          elif node.op == 'output':
            val = eval_arg(node.args)
            node_to_val[node] = val
            node.shape = None
            
          else:
            # set dummy.
            node.shape = [10,10] # 10x10 tensor
            node.latency = 1 # in ms
        
          # for all, set node.shape and also assign to dictionary.
          node.shape = list(val.shape) if isinstance(val, torch.Tensor) else None
        return self.mod

## A.3 Return the top 3 nodes with the highest latency as a list. 
def findHeavyOps(graphmodule):
  # ignore the latency out "output" node
  pairs = []
  for n in graphmodule.graph.nodes:
    # print(f"node.name = {n.name}, node.latency = {n.latency}")
    if n.op == "output":
      n.latency = None
      break
    
    pairs.append((n.name, n.latency))

  # Iterate through all the nodes and get top three nodes with highest latency.
  # Each value should be (node.name, node.latency)
  pairs.sort(key=lambda x: x[1], reverse=True)
  return pairs[:3]

#---------------------------------------------------#
#---------- Do not modify below this line ----------#
#---------------------------------------------------#

def visualize(graphmodule, dump=False):
    # create graph
    nx_graph = nx.DiGraph()

    for node in graphmodule.graph.nodes:
        idx_node = list(graphmodule.graph.nodes).index(node)
        # add nodes to the graph
        node_tuple = (idx_node, {"node": node, "idx" : idx_node})
        nx_graph.add_nodes_from([node_tuple])
        # add edges to the graph
        for user in node.users:
            nx_graph.add_edge(idx_node, list(graphmodule.graph.nodes).index(user),)

    for n in nx_graph.nodes:
        node = nx_graph.nodes[n]["node"]
        nx_graph.nodes[n]["shape"] = "box"
        nx_graph.nodes[n]["style"] = "filled"
        nx_graph.nodes[n]["label"] = node.name
        if hasattr(node, "latency") and node.latency is not None:
            nx_graph.nodes[n]["label"] += f" ({node.latency:.2f} ms)"

    for n in nx_graph.edges:
        edge = nx_graph.edges[n]
        prev_node = nx_graph.nodes[n[0]]["node"]
        if hasattr(prev_node, "shape") and prev_node.shape is not None:
            edge["label"] = str(list(prev_node.shape))

    agraph = to_agraph(nx_graph)
    agraph.layout("dot")
    out_filename = "./graph.png"
    agraph.draw(out_filename)

    display(Image(filename=out_filename))

