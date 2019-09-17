// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseSplitConvConcatIntoGroupConv final : public PredicateBasedPass {
  explicit FuseSplitConvConcatIntoGroupConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_split_conv_concat_into_group_conv";
  }

  bool check_conv_attributes(Node* n,
                             const std::vector<int>& dilations,
                             const std::vector<int>& pads,
                             const std::vector<int>& strides,
                             const int& group,
                             const size_t& dim_size
  ) {
    bool res = true;
    const std::vector<int> default_dilations(dim_size, 1);
    const std::vector<int> default_pads(dim_size * 2, 0);
    const std::vector<int> default_strides(dim_size, 1);
    const int default_group = 1;

    for (size_t i = 0; i < default_dilations.size(); i++) {
      int d = (n->hasAttribute(kdilations)) ? (int)n->is(kdilations)[i] : default_dilations[i];
      if (d != dilations[i]) { res = false; }
    }
    for (size_t i = 0; i < default_pads.size(); i++) {
      int p = (n->hasAttribute(kpads)) ? (int)n->is(kpads)[i] : default_pads[i];
      if (p != pads[i]) { res = false; }
    }
    for (size_t i = 0; i < default_strides.size(); i++) {
      int s = (n->hasAttribute(kstrides)) ? (int)n->is(kstrides)[i] : default_strides[i];
      if (s != strides[i]) { res = false; }
    }
    int g = (n->hasAttribute(kgroup)) ? (int)n->i(kgroup) : default_group;
    if (g != group) { res = false; }

    return res;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConcat) { return false; }

    for (auto input : node->inputs()) {
      if (input->node()->kind() != kConv) { return false; }
      if (input->node()->inputs()[0]->node()->kind() != kSplit) { return false; }
    }
    return true;
  }

  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    std::vector<Node*> destory_nodes;
    Node* split_node = n->inputs()[0]->node()->inputs()[0]->node();
    destory_nodes.push_back(split_node);
    ONNX_ASSERT(split_node->hasAttribute(ksplit));
    if (std::adjacent_find(split_node->is(ksplit).begin(),
                           split_node->is(ksplit).end(),
                           std::not_equal_to<bool>()) != split_node->is(ksplit).end()) {
      return false;
    };

    if (split_node->outputs().size() != n->inputs().size()) { return false; }

    int64_t split_axis = 0;
    if (split_node->hasAttribute(kaxis)) {
      split_axis = split_node->i(kaxis);
    }
    ONNX_ASSERT(n->hasAttribute(kaxis));
    if (split_axis != n->i(kaxis)) { return false; }

    Node* first_conv_n = n->inputs()[0]->node();

    std::vector<Dimension> w_sizes(first_conv_n->inputs()[1]->sizes());
    int32_t w_elem_type = (*graph.getInitializer(first_conv_n->inputs()[1]->uniqueName())).elem_type();

    // Create new weight tensor
    Tensor new_w_t;
    new_w_t.elem_type() = w_elem_type;
    for (size_t i = 0; i < w_sizes.size(); i++) {
      int64_t dim = w_sizes[i].dim;
      if (i == 0) {
        dim *= n->inputs().size();
      }
      new_w_t.sizes().push_back(dim);
    }

    // Create new bias tensor
    Tensor new_b_t;
    new_b_t.elem_type() = w_elem_type;
    new_b_t.sizes().emplace_back(first_conv_n->inputs()[1]->sizes()[0].dim * n->inputs().size());
    if (first_conv_n->inputs().size() > 2) {
      if (w_elem_type == TensorProto_DataType_FLOAT || w_elem_type == TensorProto_DataType_FLOAT16) {
        std::vector<float> zero((size_t)new_b_t.sizes()[0], 0.0);
        new_b_t.floats().insert(new_w_t.floats().end(),
                                zero.begin(),
                                zero.end());
      }
      else if (w_elem_type == TensorProto_DataType_DOUBLE) {
        std::vector<double> zero((size_t)new_b_t.sizes()[0], 0.0);
        new_b_t.doubles().insert(new_w_t.doubles().end(),
                                 zero.begin(),
                                 zero.end());
      }
    }

    // Make reference attributes for comparison
    std::vector<int> dilations(w_sizes.size() - 2, 1);
    if (first_conv_n->hasAttribute(kdilations)) {
      for (size_t i = 0; i < first_conv_n->is(kdilations).size(); i++) {
        dilations[i] = (int)first_conv_n->is(kdilations)[i];
      }
    }
    std::vector<int> pads((w_sizes.size() - 2) * 2, 0);
    if (first_conv_n->hasAttribute(kpads)) {
      for (size_t i = 0; i < first_conv_n->is(kpads).size(); i++) {
        pads[i] = (int)first_conv_n->is(kpads)[i];
      }
    }
    std::vector<int> strides(w_sizes.size() - 2, 1);
    if (first_conv_n->hasAttribute(kstrides)) {
      for (size_t i = 0; i < first_conv_n->is(kstrides).size(); i++) {
        strides[i] = (int)first_conv_n->is(kstrides)[i];
      }
    }
    const int group = 1;

    // For loop conv node to group
    for (size_t i = 0; i < n->inputs().size(); i++) {
      auto input = n->inputs()[i];
      auto conv_n = input->node();
      if (conv_n->output()->uses().size() > 1) { return false;}

      Value* w_v = input->node()->inputs()[1];
      Tensor w_t = *graph.getInitializer(w_v->uniqueName());

      // Check if kernel shape and type is same
      ONNX_ASSERT(w_v->has_sizes());
      if (!check_conv_attributes(conv_n, dilations, pads, strides, group, w_sizes.size() - 2)) {
        return false;
      }
      if (!std::equal(w_sizes.begin(), w_sizes.end(), w_v->sizes().begin(),
          [](Dimension l, Dimension r) { return l.dim == r.dim; })) {
        return false;
      }
      if (w_elem_type != w_t.elem_type()) { return false; }

      // Group weights into new weights tensor
      if (w_elem_type == TensorProto_DataType_FLOAT
          || w_elem_type == TensorProto_DataType_FLOAT16) {
        new_w_t.floats().insert(new_w_t.floats().end(),
                                w_t.floats().begin(),
                                w_t.floats().end());
      }
      else if (w_elem_type == TensorProto_DataType_DOUBLE) {
        new_w_t.doubles().insert(new_w_t.doubles().end(),
                                 w_t.doubles().begin(),
                                 w_t.doubles().end());
      }

      // Group bias into new bias tensor if current conv node has bias
      bool has_bias = input->node()->inputs().size() > 2;
      Value* b_v;
      if (has_bias) {
        b_v = input->node()->inputs()[2];
        Tensor b_t = *graph.getInitializer(b_v->uniqueName());
        ONNX_ASSERT(b_v->has_sizes());
        if (w_sizes[0].dim != b_v->sizes()[0].dim) { return false; }
        if (w_elem_type != b_t.elem_type()) { return false; }
        if (w_elem_type == TensorProto_DataType_FLOAT
            || w_elem_type == TensorProto_DataType_FLOAT16) {
          for (int j = 0; j < b_v->sizes()[0].dim; j++) {
            new_b_t.floats()[i * b_v->sizes()[0].dim + j] = b_t.floats()[j];
          }
        }
        else if (w_elem_type == TensorProto_DataType_DOUBLE) {
          for (int j = 0; j < b_v->sizes()[0].dim; j++) {
            new_b_t.doubles()[i * b_v->sizes()[0].dim + j] = b_t.doubles()[j];
          }
        }
      }

      destory_nodes.push_back(conv_n);
      conv_n->removeAllInputs();
      graph.eraseInitializerAndInput(w_v);
      if (has_bias) {
        graph.eraseInitializerAndInput(b_v);
      }
    }

    // Add new weights and new bias to graph
    Value* new_w_v = graph.addInitializerAndInput(new_w_t);
    Value* new_b_v = graph.addInitializerAndInput(new_b_t);

    // Create new conv node
    Node* new_conv_n = graph.create(kConv, 1);
    new_conv_n->addInput(split_node->input());
    split_node->removeAllInputs();
    new_conv_n->addInput(new_w_v);
    new_conv_n->addInput(new_b_v);
    new_conv_n->insertBefore(n);
    new_conv_n->copyAttributes(*first_conv_n);
    new_conv_n->i_(kgroup, (int64_t)n->inputs().size());
    if (n->output()->has_sizes()) {
      new_conv_n->output()->setSizes(n->output()->sizes());
    }

    // Destory split and conv nodes
    n->replaceAllUsesWith(new_conv_n);
    n->removeAllInputs();
    for (size_t i = destory_nodes.size() - 1; i != (size_t)-1; i--) {
      Node* node = destory_nodes[i];
      for (auto o : node->outputs()) {
        if (!o->uses().empty()) { return false; }
      }
      node->destroy();
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
