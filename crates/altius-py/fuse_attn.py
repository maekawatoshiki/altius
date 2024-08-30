import collections
import copy
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import onnx
from onnx import ModelProto, NodeProto, helper
import onnxruntime as ort

# from onnxruntime.transformers.onnx_model import OnnxModel
# from onnxruntime.transformers.fusion_attention import FusionAttention
# from onnxruntime.transformers.fusion_attention import AttentionMask

# class DeiT(OnnxModel):
#     def __init__(self, model: ModelProto):
#         super().__init__(model)
#         self.attn_mask = AttentionMask(self)
#         self.attn_fusion = FusionAttention(self, 384, 6, self.attn_mask)
#
#     def fuse(self):
#         self.attn_fusion.apply()


def create_value_to_users(
    model: onnx.ModelProto,
) -> DefaultDict[str, List[onnx.NodeProto]]:
    value_to_users = collections.defaultdict(lambda: [])
    for node in model.graph.node:
        for input in node.input:
            value_to_users[input].append(node)
    return value_to_users


def fuse_mha(
    root: NodeProto,
    visited: Set[str],
    value_to_users: DefaultDict[str, List[NodeProto]],
) -> Optional[NodeProto]:
    if root.op_type != "LayerNormalization":
        return None

    ln = root
    if len(value_to_users[ln.output[0]]) != 3:
        return None

    mm1, mm2, mm3 = value_to_users[ln.output[0]]
    if mm1.op_type != "MatMul" or mm2.op_type != "MatMul" or mm3.op_type != "MatMul":
        return None
    if mm1.op_type != "MatMul" or mm2.op_type != "MatMul" or mm3.op_type != "MatMul":
        return None

    add1, add2, add3 = (
        value_to_users[mm1.output[0]],
        value_to_users[mm2.output[0]],
        value_to_users[mm3.output[0]],
    )
    if len(add1) != 1 or len(add2) != 1 or len(add3) != 1:
        return None
    add1, add2, add3 = add1[0], add2[0], add3[0]
    if add1.op_type != "Add" or add2.op_type != "Add" or add3.op_type != "Add":
        return None

    key = None
    query = None
    value = None
    for out in [add1.output[0], add2.output[0], add3.output[0]]:
        if "attention/key" in out:
            key = out
        elif "attention/query" in out:
            query = out
        elif "attention/value" in out:
            value = out
    if key is None or query is None or value is None:
        return None

    que = []
    que.extend(value_to_users[key])
    que.extend(value_to_users[query])
    que.extend(value_to_users[value])
    exit_reshape_node = None
    while que:
        node = que.pop(0)
        visited.add(node.name)
        if node.op_type == "Reshape" and "attention/attention/Reshape_3" in node.name:
            print(node.name)
            exit_reshape_node = node
            break
        users = value_to_users[node.output[0]]
        que.extend(users)
    assert exit_reshape_node is not None

    num_heads = 6
    mha_node = helper.make_node(
        "MultiHeadAttention",
        inputs=[query, key, value],
        outputs=[exit_reshape_node.output[0]],
        name=f"MultiHeadAttention@{ln.name}",
    )
    mha_node.domain = "com.microsoft"
    mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

    print("Fusing MHA")

    return mha_node


def topo_sort(
    model: onnx.ModelProto, nodes: List[onnx.NodeProto]
) -> List[onnx.NodeProto]:
    node_to_order = {}
    for i, n in enumerate(model.graph.node):
        node_to_order[n.output[0]] = i

    order_and_nodes = []
    for n in nodes:
        order_and_nodes.append((node_to_order[n.output[0]], n))

    order_and_nodes.sort(key=lambda x: x[0])
    return [n for _, n in order_and_nodes]


def fuse(model: ModelProto) -> ModelProto:
    users = create_value_to_users(model)
    new_model = copy.deepcopy(model)
    del new_model.graph.node[:]

    visited: Set[str] = set()
    nodes = []
    for node in model.graph.node:
        if node.name in visited:
            continue

        nodes.append(node)

        mha = fuse_mha(node, visited, users)
        if mha is not None:
            nodes.append(mha)

    sorted_nodes = topo_sort(model, nodes)
    for node in sorted_nodes:
        new_model.graph.node.add().CopyFrom(node)

    new_model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    onnx.checker.check_model(new_model)

    return new_model


def main():
    model = onnx.load("../../models/deit.onnx")
    # deit = DeiT(copy.deepcopy(model))
    # deit.fuse()

    new_model = fuse(model)

    onnx.save(new_model, "./fused_deit.onnx")


if __name__ == "__main__":
    main()
