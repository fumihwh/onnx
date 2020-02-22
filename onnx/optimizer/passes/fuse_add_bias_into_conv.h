// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = Conv(X, Y)
//   B = Z + A
// After:
//   B = Conv(X, Y, A)
//
// the pass can handle the following cases:
//   case 1: A is 1D tensor and A.dim[0] == Z.dim[1]
//   case 2: A is 1-element 1D tensor

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseAddBiasIntoConv final : public PredicateBasedPass {
  explicit FuseAddBiasIntoConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_add_bias_into_conv";
  }

  void replace_inputs(Tensor& t, int idx, Node* conv, Graph& graph) {
    Value* new_t_value = graph.addInitializerAndInput(t);
    if (idx == 1) {
      Value* old_t_value = conv->inputs()[idx];
      conv->replaceInput(idx, new_t_value);
      if (old_t_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_t_value);
      }
    }
    if (idx == 2) {
      if (conv->inputs().size() == 3) {
        Value* old_t_value = conv->inputs()[idx];
        conv->replaceInput(idx, new_t_value);
        if (old_t_value->uses().size() == 0) {
          graph.eraseInitializerAndInput(old_t_value);
        }
      } else {
        Value* new_b_value = graph.addInitializerAndInput(t);
        conv->addInput(new_b_value);
      }
    }
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kAdd &&
        (node->inputs()[0]->node()->kind() == kConv ||
         node->inputs()[0]->node()->kind() == kConvTranspose);
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    // due to current broadcasting's constraint, Conv has to be the first
    // operand
    destroy_current = NodeDestroyType::DestroyZero;
    auto orig_conv = n->inputs()[0];
    auto orig_bias = n->inputs()[1];
    // check if bias is Const or in graph's initializers
    if (orig_bias->node()->kind() != kConstant &&
        orig_bias->node()->kind() != kParam) {
      return false;
    }
    // check if conv is only used by Add
    if (orig_conv->uses().size() > 1) {
      return false;
    }
    auto conv_shape = orig_conv->sizes();
    auto bias_shape = orig_bias->sizes();
    auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
    int64_t M = -1;
    int64_t rank = -1;
    // try to get feature M and rank from conv_shape
    if (conv_shape.size() > 1 && conv_shape[1].is_int) {
      M = conv_shape[1].dim;
      rank = conv_shape.size();
    }
    // try to get feature M and rank from weight_shape
    else if (weight_shape.size() > 0 && weight_shape[0].is_int) {
      ONNX_ASSERT(M == -1 || M == weight_shape[0].dim);
      M = weight_shape[0].dim;
      ONNX_ASSERT(
          rank == -1 || rank == static_cast<int64_t>(weight_shape.size()));
      rank = weight_shape.size();
    }
    int64_t num_el = 1;
    for (int i = 0; i < static_cast<int64_t>(bias_shape.size()); ++i) {
      if (bias_shape[i].is_int) {
        num_el *= bias_shape[i].dim;
      } else {
        num_el = -1;
        return false;
      }
    }
    if (M == -1 || num_el == -1) {
      // No enough information, bail out
      return false;
    }
    if (rank < static_cast<int64_t>(bias_shape.size())) {
      return false;
    }

    auto end_iter = graph.initializers().end();
    auto orig_bias_iter = graph.getInitializer(orig_bias->uniqueName());
    if (orig_bias_iter == end_iter) {
      return false;
    }
    Tensor orig_b = *orig_bias_iter;
    Tensor new_b;
    new_b.elem_type() = orig_b.elem_type();
    new_b.sizes().push_back(M);

#define DO_COMPUTATION(vec)                          \
  new_b.vec().clear();                               \
  if (num_el == 1) {                                 \
    for (int64_t i = 0; i < new_b.sizes()[0]; ++i) { \
      new_b.vec().push_back(orig_b.vec()[0]);        \
    }                                                \
  } else {                                           \
    for (int64_t i = 0; i < new_b.sizes()[0]; ++i) { \
      new_b.vec().push_back(orig_b.vec()[i]);        \
    }                                                \
  }

    switch (orig_b.elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        DO_COMPUTATION(floats)
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        DO_COMPUTATION(doubles)
        break;
      }
      default:
        return false;
    }
    if (orig_conv->node()->inputs().size() == 2) {
      Value* new_b_value = graph.addInitializerAndInput(new_b);
      orig_conv->node()->addInput(new_b_value);
    } else if (orig_conv->node()->inputs().size() == 3) {
      auto conv_b =
          *graph.getInitializer(orig_conv->node()->inputs()[2]->uniqueName());
      conv_b.add(new_b);
      replace_inputs(conv_b, 2, orig_conv->node(), graph);
    }

#undef DO_COMPUTATION

    if (orig_conv->sizes().size() == 0 && n->output()->sizes().size() > 0) {
      orig_conv->setSizes(n->output()->sizes());
    }
    if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
      orig_conv->setElemType(n->output()->elemType());
    }
    n->replaceAllUsesWith(orig_conv->node());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
