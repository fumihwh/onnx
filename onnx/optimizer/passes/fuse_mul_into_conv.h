// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = Conv(X, Y)
//   B = Z * A
// After:
//   B = Conv(X, Y * A)
//
// the pass can handle the following cases:
//   case 1: A is 4D tensor and A.dim[1] == Y.dim[0]

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseMulIntoConv final : public PredicateBasedPass {
  explicit FuseMulIntoConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_mul_into_conv";
  }

  void replace_inputs(Tensor& t, int idx, Node* conv, Graph& graph) {
    Value* new_t_value = graph.addInitializerAndInput(t);
    Value* old_t_value = conv->inputs()[idx];
    conv->replaceInput(idx, new_t_value);
    if (idx == 1) {
      if (old_t_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_t_value);
      }
    }
    if (idx == 2) {
      if (conv->inputs().size() == 3) {
        conv->replaceInput(2, new_t_value);
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
    return node->kind() == kMul && node->inputs()[0]->node()->kind() == kConv;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    auto orig_conv = n->inputs()[0];
    auto orig_scale = n->inputs()[1];
    // check if bias is Const or in graph's initializers
    if (orig_scale->node()->kind() != kConstant &&
        orig_scale->node()->kind() != kParam) {
      return false;
    }
    // check if conv is only used by Add
    if (orig_conv->uses().size() > 1) {
      return false;
    }
    auto conv_shape = orig_conv->sizes();
    auto scale_shape = orig_scale->sizes();
    auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
    int64_t M = -1;
    int64_t rank = -1;
    // try to get feature M and rank from conv_shape
    if (conv_shape.size() > 1 && conv_shape[1].is_int) {
      M = conv_shape[1].dim;
      rank = conv_shape.size();
    }
    // try to get feature M and rank from weight_shape
    if (weight_shape.size() > 0 && weight_shape[0].is_int) {
      ONNX_ASSERT(M == -1 || M == weight_shape[0].dim);
      M = weight_shape[0].dim;
      ONNX_ASSERT(
          rank == -1 || rank == static_cast<int64_t>(weight_shape.size()));
      rank = weight_shape.size();
    }
    for (int i = 0; i < rank; i++) {
      if (scale_shape[i].dim != (i == 1 ? M : 1)) {
        return false;
      }
    }
    auto end_iter = graph.initializers().end();
    auto w_iter =
        graph.getInitializer(orig_conv->node()->inputs()[1]->uniqueName());
    auto s_iter = graph.getInitializer(orig_scale->uniqueName());
    if (w_iter == end_iter || s_iter == end_iter) {
      return false;
    }
    Tensor w = *w_iter;
    Tensor orig_s = *s_iter;
    Tensor s;
    s.elem_type() = orig_s.elem_type();
    s.sizes().push_back(M);

#define DO_COMPUTATION(t, vec)                   \
  if (s.vec().size() != M) {                     \
    for (int64_t i = 0; i < s.sizes()[0]; ++i) { \
      s.vec().push_back(orig_s.vec()[i]);        \
    }                                            \
  }                                              \
  (t).scale_by_first_dim(s);

    switch (s.elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        DO_COMPUTATION(w, floats)
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        DO_COMPUTATION(w, doubles)
        break;
      }
      default:
        return false;
    }
    replace_inputs(w, 1, orig_conv->node(), graph);

    if (orig_conv->node()->inputs().size() == 3) {
      auto b_iter =
          graph.getInitializer(orig_conv->node()->inputs()[2]->uniqueName());
      if (b_iter == end_iter || s_iter == end_iter) {
        return false;
      }
      Tensor b = *b_iter;
      switch (s.elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          DO_COMPUTATION(b, floats)
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          DO_COMPUTATION(b, doubles)
          break;
        }
        default:
          return false;
      }
      replace_inputs(b, 2, orig_conv->node(), graph);
    }

#undef DO_COMPUTATION

    n->replaceAllUsesWith(orig_conv->node());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
