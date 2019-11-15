// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseSplitGemmConcatIntoReshapeGemmReshape final : public PredicateBasedPass {
  explicit FuseSplitGemmConcatIntoReshapeGemmReshape()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_split_gemm_concat_into_reshape_gemm_reshape";
  }

  bool check_gemm_attributes(Node* n,
                             const float& alpha,
                             const float& beta,
                             const int& trans_a,
                             const int& trans_b
  ) {
    bool res = true;
    const float default_alpha = 1.0;
    const float default_beta = 1.0;
    const int default_trans_a = 0;
    const int default_trans_b = 0;
    const int default_group = 1;

    float a = n->hasAttribute(kalpha) ? (float)n->f(kalpha) : default_alpha;
    if (a != alpha) { res = false; }
    float b = n->hasAttribute(kbeta) ? (float)n->f(kbeta) : default_beta;
    if (b != beta) { res = false; }
    float t_a = n->hasAttribute(ktransA) ? (int)n->i(ktransA) : default_trans_a;
    if (t_a != trans_a) { res = false; }
    float t_b = n->hasAttribute(ktransB) ? (int)n->i(ktransB) : default_trans_b;
    if (t_b != trans_b) { res = false; }

    return res;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConcat) { return false; }

    for (auto input : node->inputs()) {
      auto unsqueeze_n = input->node();
      if (unsqueeze_n->kind() != kUnsqueeze) { return false; }

      auto gemm_n = input->node()->input()->node();
      if (gemm_n->kind() != kGemm) { return false; }

      auto squeeze_n = gemm_n->inputs()[0]->node();
      if (squeeze_n->kind() != kSqueeze) { return false; }

      auto split_n = squeeze_n->input()->node();
      if (split_n->kind() != kSplit) { return false; }
      if (split_n->outputs().size() != node->inputs().size()) { return false; }
    }
    return true;
  }

  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    std::vector<Node*> destory_nodes;
    Node* first_gemm_n = n->inputs()[0]->node()->input()->node();
    Node* split_n = first_gemm_n->inputs()[0]->node()->input()->node();
    destory_nodes.push_back(split_n);
    ONNX_ASSERT(split_n->hasAttribute(ksplit));
    if (std::adjacent_find(split_n->is(ksplit).begin(),
                           split_n->is(ksplit).end(),
                           std::not_equal_to<bool>()) != split_n->is(ksplit).end()) {
      return false;
    };
    if (split_n->is(ksplit)[0] != 1) { return false;}

    if (split_n->outputs().size() != n->inputs().size()) { return false; }

    int64_t split_axis = 0;
    if (split_n->hasAttribute(kaxis)) {
      split_axis = split_n->i(kaxis);
    }
    ONNX_ASSERT(n->hasAttribute(kaxis));
    if (split_axis != n->i(kaxis)) { return false; }

    Value* weights_v = first_gemm_n->inputs()[1];
    Value* bias_v = first_gemm_n->inputs()[2];
    ONNX_ASSERT(first_gemm_n->inputs()[0]->has_sizes()
                    && first_gemm_n->inputs()[1]->has_sizes()
                    && first_gemm_n->inputs()[2]->has_sizes());
    std::vector<Dimension> w_sizes(weights_v->sizes());

    // Make reference attributes for comparison
    const float alpha = first_gemm_n->hasAttribute(kalpha) ? (float)first_gemm_n->f(kalpha) : 1;
    const float beta = first_gemm_n->hasAttribute(kbeta) ? (float)first_gemm_n->f(kbeta) : 1;
    const int trans_a = first_gemm_n->hasAttribute(ktransA) ? (int)first_gemm_n->i(ktransA) : 0;
    const int trans_b = first_gemm_n->hasAttribute(ktransB) ? (int)first_gemm_n->i(ktransB) : 0;

    int64_t batch = first_gemm_n->inputs()[0]->sizes()[0].dim;

    // For loop gemm node to check
    for (size_t i = 0; i < n->inputs().size(); i++) {
      auto unsqueeze_n = n->inputs()[i]->node();
      auto gemm_n = unsqueeze_n->input()->node();
      auto squeeze_n = gemm_n->inputs()[0]->node();
      ONNX_ASSERT(squeeze_n->hasAttribute(kaxes) && squeeze_n->is(kaxes).size() == 1);
      ONNX_ASSERT(unsqueeze_n->hasAttribute(kaxes) && unsqueeze_n->is(kaxes).size() == 1);
      if (squeeze_n->is(kaxes)[0] != 0 || unsqueeze_n->is(kaxes)[0] != 0) {
        return false;
      }

      if (unsqueeze_n->output()->uses().size() > 1
          || gemm_n->output()->uses().size() > 1
          || squeeze_n->output()->uses().size() > 1) {
        return false;
      }

      ONNX_ASSERT(gemm_n->inputs().size() == 3)
      Value* w_v = gemm_n->inputs()[1];
      Value* b_v = gemm_n->inputs()[2];
      ONNX_ASSERT(gemm_n->inputs()[0]->has_sizes()
                      && w_v->has_sizes()
                      && b_v->has_sizes());
      if (!check_gemm_attributes(gemm_n, alpha, beta, trans_a, trans_b)) {
        return false;
      }
      if (!std::equal(w_sizes.begin(), w_sizes.end(), w_v->sizes().begin(),
                      [](Dimension l, Dimension r) { return l.dim == r.dim; })) {
        return false;
      }
      if (w_sizes[!trans_b].dim != b_v->sizes()[0].dim) {
        return false;
      }
      if (gemm_n->inputs()[1] != w_v || gemm_n->inputs()[2] != b_v) {
        return false;
      }

      unsqueeze_n->removeAllInputs();
      gemm_n->removeAllInputs();
      squeeze_n->removeAllInputs();

      destory_nodes.push_back(squeeze_n);
      destory_nodes.push_back(gemm_n);
      destory_nodes.push_back(unsqueeze_n);
    }

    Tensor prev_shape_t;
    prev_shape_t.elem_type() = TensorProto_DataType_INT64;
    prev_shape_t.sizes().push_back(2);
    prev_shape_t.int64s().push_back((int)split_n->outputs().size() * batch);
    prev_shape_t.int64s().push_back(w_sizes[trans_b].dim);
    Value* prev_shape_v = graph.addInitializerAndInput(prev_shape_t);

    Node* prev_reshape_n = graph.create(kReshape, 1);
    prev_reshape_n->addInput(split_n->input());
    prev_reshape_n->addInput(prev_shape_v);
    prev_reshape_n->insertBefore(n);
    prev_reshape_n->output()->setSizes(std::vector<Dimension>{Dimension(prev_shape_t.int64s()[0]),
                                                              Dimension(prev_shape_t.int64s()[1])});
    prev_reshape_n->output()->setElemType(split_n->input()->elemType());

    Node* new_gemm_n = graph.create(kGemm, 1);
    new_gemm_n->addInput(prev_reshape_n->output());
    new_gemm_n->addInput(weights_v);
    new_gemm_n->addInput(bias_v);
    new_gemm_n->copyAttributes(*first_gemm_n);
    new_gemm_n->insertBefore(n);
    new_gemm_n->output()->setSizes(std::vector<Dimension>{Dimension(prev_shape_t.int64s()[0]),
                                                          Dimension(w_sizes[!trans_b].dim)});
    new_gemm_n->output()->setElemType(first_gemm_n->output()->elemType());

    Tensor post_shape_t;
    post_shape_t.elem_type() = TensorProto_DataType_INT64;
    post_shape_t.sizes().push_back(3);
    post_shape_t.int64s().push_back((int)split_n->outputs().size());
    post_shape_t.int64s().push_back(batch);
    post_shape_t.int64s().push_back(w_sizes[!trans_b].dim);
    Value* post_shape_v = graph.addInitializerAndInput(post_shape_t);

    Node* post_reshape_n = graph.create(kReshape, 1);
    post_reshape_n->addInput(new_gemm_n->output());
    post_reshape_n->addInput(post_shape_v);
    post_reshape_n->insertBefore(n);
    post_reshape_n->output()->setSizes(std::vector<Dimension>{Dimension(post_shape_t.int64s()[0]),
                                                              Dimension(post_shape_t.int64s()[1]),
                                                              Dimension(w_sizes[!trans_b].dim)});
    post_reshape_n->output()->setElemType(first_gemm_n->output()->elemType());

    // Destory nodes
    n->replaceAllUsesWith(post_reshape_n);
    n->removeAllInputs();
    for (size_t i = destory_nodes.size() - 1; i != (size_t)-1; i--) {
      Node* node = destory_nodes[i];
      for (auto o : node->outputs()) {
        if (!o->uses().empty()) {
          return false;
        }
      }
      node->destroy();
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
