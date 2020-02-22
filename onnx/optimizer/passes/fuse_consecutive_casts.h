// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveCasts final : public PredicateBasedPass {
  explicit FuseConsecutiveCasts()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Partial,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_casts";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kCast && node->input()->node()->kind() == kCast;
  }
  bool runTransform(Node* n, Graph&, NodeDestroyType& destroy_current)
      override {
    auto pre_cast_node = n->input()->node();
    n->replaceInput(0, pre_cast_node->input());
    pre_cast_node->removeAllInputs();
    if (!pre_cast_node->hasUses()) {
      pre_cast_node->destroy();
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
