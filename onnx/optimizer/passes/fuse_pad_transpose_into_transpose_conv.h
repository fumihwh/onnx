// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"
#include "onnx/optimizer/passes/fuse_pad_into_conv.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FusePadTransposeIntoTransposeConv final : public PredicateBasedPass {
  explicit FusePadTransposeIntoTransposeConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_pad_transpose_into_transpose_conv";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConv
        && node->inputs()[0]->node()->kind() == kTranspose
        && node->inputs()[0]->node()->input()->node()->kind() == kPad;
  }

  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    auto trans_n = n->inputs()[0]->node();
    auto pad_n = trans_n->input()->node();
    if (trans_n->output()->uses().size() > 1
        || pad_n->output()->uses().size() > 1) {
      return false;
    }
    ONNX_ASSERT(trans_n->hasAttribute(kperm));
    auto perm = trans_n->is(kperm);
    std::vector<long long> pad_perm;
    std::copy(perm.begin(), perm.end(), std::back_inserter(pad_perm));
    auto rank = perm.size();
    std::for_each(pad_perm.begin(), pad_perm.end(),
                  [&](long long& l) { l += rank;});
    pad_perm.insert(pad_perm.begin(), perm.begin(), perm.end());

    std::vector<int64_t> pads;
    if (pad_n->hasAttribute(kpads)) {
      // opset 10 and below
      pads = pad_n->is(kpads);
      std::vector<int64_t> new_pads(pads.size());
      for (int i = 0; i < pad_perm.size(); i++) {
        new_pads[i] = pads[pad_perm[i]];
      }
      pad_n->is_(kpads, std::move(new_pads));
    } else {
      // opset 11 and above - first check if 'pad' node has 'pads' input
      // initialized
      const auto& pads_name = pad_n->inputs()[1]->uniqueName();
      const auto pads_initializer = graph.getInitializer(pads_name);
      // 'pad' node has the 'pads' input which has not been initialized -
      // can't proceed with fusing
      if (pads_initializer == graph.initializers().end()) {
        return false;
      }

      // make sure the type of 'pads' is INT64
      if (pads_initializer->elem_type() != TensorProto::INT64) {
        return false;
      }

      // parse 'pads' data from the initialized input
      pads = ParseData<int64_t>(&*pads_initializer);

      std::vector<int64_t> new_pads(pads.size());
      for (int i = 0; i < pad_perm.size(); i++) {
        new_pads[i] = pads[pad_perm[i]];
      }

      Tensor new_pads_t;
      new_pads_t.elem_type() = TensorProto::INT64;
      new_pads_t.sizes().emplace_back(new_pads.size());
      for (int i = 0; i < new_pads.size(); i++) {
        new_pads_t.int64s().emplace_back(new_pads[i]);
      }

      auto pads_v = pad_n->inputs()[1];
      Value* new_v = graph.addInitializerAndInput(new_pads_t);
      pad_n->replaceInput(1, new_v);
      graph.eraseInitializerAndInput(pads_v);
    }

    trans_n->moveBefore(pad_n);
    trans_n->replaceInputWith(trans_n->input(), pad_n->inputs()[0]);
    trans_n->replaceAllUsesWith(pad_n);
    trans_n->output()->setSizes(trans_n->input()->sizes());

    std::vector<Dimension> new_i_sizes(trans_n->input()->sizes().size(), Dimension(0));
    for (int i = 0; i < new_i_sizes.size(); i++) {
      new_i_sizes[i].dim = trans_n->input()->sizes()[perm[i]].dim;
    }
    trans_n->output()->setSizes(new_i_sizes);
    pad_n->replaceInput(0, trans_n->output());

    return FusePadIntoConv().runTransform(n, graph, destroy_current);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
