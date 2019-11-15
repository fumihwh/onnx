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

    auto pads = pad_n->is(kpads);
    std::vector<int64_t> new_pads(pads.size());
    for (int i = 0; i < pad_perm.size(); i++) {
      new_pads[i] = pads[pad_perm[i]];
    }
    pad_n->is_(kpads, std::move(new_pads));

    trans_n->moveBefore(pad_n);
    trans_n->replaceInputWith(trans_n->input(), pad_n->input());
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
