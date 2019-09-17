// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSqueezeUnsqueeze final : public PredicateBasedPass {
  explicit FuseConsecutiveSqueezeUnsqueeze()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_squeeze_unsqueeze";
  }
  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kSqueeze && node->kind() != kUnsqueeze) {
      return false;
    }
    if ((node->kind() == kSqueeze && node->input()->node()->kind() != kUnsqueeze)
        || (node->kind() == kUnsqueeze && node->input()->node()->kind() != kSqueeze)) {
      return false;
    }
    return true;
  }

  bool runTransform(Node* n, Graph&, NodeDestroyType& destroy_current)
      override {
    ONNX_ASSERT(n->hasAttribute(kaxes));
    auto curr_axes = n->is(kaxes);
    auto orig_input = n->input();
    ONNX_ASSERT(orig_input->node()->hasAttribute(kaxes));
    auto prev_axes = orig_input->node()->is(kaxes);
    if (curr_axes.size() != prev_axes.size()) {
      return false;
    }
    if (!std::equal(curr_axes.begin(), curr_axes.end(), prev_axes.begin())) {
      return false;
    }
    n->output()->replaceAllUsesWith(orig_input->node()->input());
    n->removeAllInputs();
    if (!orig_input->uses().empty()) {
      return false;
    }
    orig_input->node()->destroy();
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
