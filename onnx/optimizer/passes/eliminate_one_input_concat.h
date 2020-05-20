// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateOneInputConcat final : public PredicateBasedPass {
  explicit EliminateOneInputConcat()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_one_input_concat";
  }

  bool patternMatchPredicate(Node* node) override {
    return (node->kind() == kConcat && node->inputs().size() == 1);
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    node->output()->replaceAllUsesWith(node->input());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
