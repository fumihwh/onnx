// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> target_operators{kRelu,
                                                    kPad,
                                                    kSplit,
                                                    kConcat,
                                                    kResize,
                                                    kLeakyRelu,
                                                    kAdd,
                                                    kMul,
                                                    kSum,
                                                    kClip};

struct SwapTransposeDown final : public PredicateBasedPass {
  explicit SwapTransposeDown()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "swap_transpose_down";
  }

  void simple_swap(
      Node* n,
      Node* trans_node,
      Graph& graph,
      NodeDestroyType& destroy_current) {
    bool is_output_node = false;
    for (auto it = graph.outputs().begin(); it != graph.outputs().end(); it++) {
      auto output = *it;
      if (n->output()->uniqueName() == output->uniqueName()) {
        is_output_node = true;
        break;
      }
    }
    n->replaceInput(0, trans_node->input());
    const auto uses = n->output()->uses();
    auto output_sizes = n->output()->sizes();
    for (auto use : uses) {
      auto user_n = use.user;
      Node* t_n = graph.create(kTranspose, 1);
      t_n->copyAttributes(*trans_node);
      t_n->insertBefore(user_n);
      t_n->addInput(n->output());
      user_n->replaceInputWith(n->output(), t_n->output());
      reset_sizes(t_n, output_sizes);
      t_n->output()->setElemType(t_n->input()->elemType());
      if (is_output_node) {
        t_n->output()->setUniqueName(n->output()->uniqueName());
        n->output()->setUniqueName(
            n->output()->uniqueName() + "_" + graph.generate_hex());
      }
    }
    if (!trans_node->hasUses()) {
      trans_node->removeAllInputs();
      trans_node->destroy();
    }
  }

  void reset_sizes(Node* trans_node, std::vector<Dimension>& output_sizes) {
    auto perm = trans_node->is(kperm);
    std::vector<Dimension> new_i_sizes(perm.size(), Dimension(0));
    for (int i = 0; i < output_sizes.size(); i++) {
      new_i_sizes[perm[i]].dim = output_sizes[i].dim;
    }
    trans_node->output()->setSizes(output_sizes);
    trans_node->input()->setSizes(new_i_sizes);
  }

  bool patternMatchPredicate(Node* node) override {
    if (target_operators.find(node->kind()) == target_operators.end()) {
      return false;
    }
    if (node->kind() == kConcat) {
      return std::all_of(
          node->inputs().begin(), node->inputs().end(), [](Value* input) {
            return input->node()->kind() == kTranspose;
          });
    } else {
      return node->inputs()[0]->node()->kind() == kTranspose;
    }
  }

  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    if (n->kind() == kRelu or n->kind() == kLeakyRelu or n->kind() == kClip) {
      simple_swap(n, n->input()->node(), graph, destroy_current);
    }

    else if (n->kind() == kAdd or n->kind() == kMul or n->kind() == kSum) {
      auto input_0 = n->inputs()[0];
      auto input_1 = n->inputs()[1];
      if (input_1->node()->kind() == kParam) {
        auto it = graph.getInitializer(input_1->uniqueName());
        if (it == graph.initializers().end()) {
          return false;
        }
        auto perm = input_0->node()->is(kperm);
        auto t = *it;
        Tensor new_t;
        new_t.elem_type() = t.elem_type();
        if (t.sizes().size() == 1) {
          for (int i = 0; i < perm.size(); i++) {
            auto s = i == (int)perm[perm.size() - 1] ? t.sizes()[0] : 1;
            new_t.sizes().emplace_back(s);
          }
        } else if (t.sizes().size() == 0) {
          for (int i = 0; i < perm.size(); i++) {
            new_t.sizes().emplace_back(1);
          }
        } else if (t.sizes().size() == perm.size()) {
          new_t.sizes().resize(perm.size());
          for (int i = 0; i < perm.size(); i++) {
            new_t.sizes()[perm[i]] = t.sizes()[i];
          }
        } else {
          return false;
        }

        if (new_t.elem_type() == TensorProto_DataType_FLOAT) {
          if (t.is_raw_data()) {
            const auto d = ParseData<float>(&t);
            std::copy(d.begin(), d.end(), std::back_inserter(new_t.floats()));
          } else if (new_t.elem_type() == TensorProto_DataType_FLOAT) {
            std::copy(
                t.floats().begin(),
                t.floats().end(),
                std::back_inserter(new_t.floats()));
          }
        }
        Value* new_v = graph.addInitializerAndInput(new_t);
        n->replaceInput(1, new_v);
        if (input_1->uses().size() == 0) {
          graph.eraseInitializerAndInput(input_1);
        }
        simple_swap(n, input_0->node(), graph, destroy_current);
      } else if (
          input_0->node()->kind() == kTranspose &&
          input_1->node()->kind() == kTranspose) {
        auto perm_0 = input_0->node()->is(kperm);
        auto perm_1 = input_1->node()->is(kperm);
        if (!std::equal(perm_0.begin(), perm_0.end(), perm_1.begin())) {
          return false;
        }
        if (input_0->uses().size() != 1 || input_1->uses().size() != 1) {
          return false;
        }
        for (int i = 0; i < n->inputs().size(); i++) {
          auto trans_node = n->inputs()[i]->node();
          trans_node->moveAfter(n);
          n->replaceInput(i, trans_node->input());
          trans_node->removeInput(0);
          if (!trans_node->hasUses()) {
            trans_node->destroy();
          }
        }
        auto uses = n->output()->uses();
        auto output_sizes = n->output()->sizes();
        for (int i = 0; i < n->output()->uses().size(); i++) {
          Node* new_trans_node = graph.create(kTranspose, 1);
          new_trans_node->is_(kperm, std::move(perm_0));
          new_trans_node->insertAfter(n);
          auto user = uses[i].user;
          user->replaceInputWith(n->output(), new_trans_node->output());
          new_trans_node->addInput(n->output());
          reset_sizes(new_trans_node, output_sizes);
        }
      } else if (
          input_0->node()->kind() == kTranspose ||
          input_1->node()->kind() == kTranspose) {
        Node* trans_node;
        if (input_0->node()->kind() == kTranspose) {
          trans_node = input_0->node();
        } else {
          trans_node = input_1->node();
        }
        auto perm = trans_node->is(kperm);
        if (input_0->uses().size() != 1 || input_1->uses().size() != 1) {
          return false;
        }

        if (n->inputs().size() != perm.size()) {
          return false;
        }

        for (int i = 0; i < n->inputs().size(); i++) {
          if (n->inputs()[i]->node()->kind() == kTranspose) {
            trans_node->moveAfter(n);
            n->replaceInput(i, trans_node->input());
            trans_node->removeInput(0);
            if (!trans_node->hasUses()) {
              trans_node->destroy();
            }
          } else {
            std::vector<int64_t> reverse_perm(perm.size());
            for (int j = 0; j < perm.size(); j++) {
              reverse_perm[perm[j]] = j;
            }
            Node* reverse_trans_node = graph.create(kTranspose, 1);
            reverse_trans_node->is_(kperm, std::move(reverse_perm));
            reverse_trans_node->insertBefore(n);
            reverse_trans_node->addInput(n->inputs()[i]);
            n->replaceInput(i, reverse_trans_node->output());
            auto old_i_sizes = reverse_trans_node->input()->sizes();
            std::vector<Dimension> new_i_sizes(perm.size(), Dimension(0));
            for (int i = 0; i < perm.size(); i++) {
              new_i_sizes[perm[i]].dim = old_i_sizes[i].dim;
            }
            reverse_trans_node->output()->setSizes(new_i_sizes);
            reverse_trans_node->output()->setElemType(
                reverse_trans_node->input()->elemType());
          }
        }
        auto uses = n->output()->uses();
        auto output_sizes = n->output()->sizes();
        for (int i = 0; i < n->output()->uses().size(); i++) {
          Node* new_trans_node = graph.create(kTranspose, 1);
          new_trans_node->is_(kperm, std::move(perm));
          new_trans_node->insertAfter(n);
          auto user = uses[i].user;
          user->replaceInputWith(n->output(), new_trans_node->output());
          new_trans_node->addInput(n->output());
          reset_sizes(new_trans_node, output_sizes);
        }
      } else {
        return false;
      }
    }

    else if (n->kind() == kPad) {
      auto perm = n->inputs()[0]->node()->is(kperm);
      std::vector<long long> pad_perm;
      std::copy(perm.begin(), perm.end(), std::back_inserter(pad_perm));
      auto rank = perm.size();
      std::for_each(
          pad_perm.begin(), pad_perm.end(), [&](long long& l) { l += rank; });
      pad_perm.insert(pad_perm.begin(), perm.begin(), perm.end());

      std::vector<int64_t> pads;
      if (n->hasAttribute(kpads)) {
        // opset 10 and below
        pads = n->is(kpads);
        std::vector<int64_t> new_pads(pads.size());
        for (int i = 0; i < pad_perm.size(); i++) {
          new_pads[pad_perm[i]] = pads[i];
        }
        n->is_(kpads, std::move(new_pads));
      } else {
        // opset 11 and above - first check if 'pad' node has 'pads' input
        // initialized
        const auto& pads_name = n->inputs()[1]->uniqueName();
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
          new_pads[pad_perm[i]] = pads[i];
        }

        Tensor new_pads_t;
        new_pads_t.elem_type() = TensorProto::INT64;
        new_pads_t.sizes().emplace_back(new_pads.size());
        for (int i = 0; i < new_pads.size(); i++) {
          new_pads_t.int64s().emplace_back(new_pads[i]);
        }

        auto pads_v = n->inputs()[1];
        Value* new_v = graph.addInitializerAndInput(new_pads_t);
        n->replaceInput(1, new_v);
        graph.eraseInitializerAndInput(pads_v);
      }

      simple_swap(n, n->inputs()[0]->node(), graph, destroy_current);
    }

    else if (n->kind() == kSplit) {
      auto orig_input = n->input();
      auto trans_node = orig_input->node();
      auto perm = trans_node->is(kperm);
      trans_node->moveAfter(n);
      n->replaceInput(0, trans_node->input());
      int64_t axis = 0;
      if (n->hasAttribute(kaxis)) {
        axis = n->i(kaxis);
      }
      if (axis < 0) {
        axis += perm.size();
      }
      auto new_axis = perm[axis];
      n->i_(kaxis, new_axis);
      for (int i = 0; i < n->outputs().size(); i++) {
        Node* new_trans_node = graph.create(kTranspose, 1);
        new_trans_node->is_(kperm, std::move(perm));
        new_trans_node->insertAfter(n);
        n->outputs()[i]->replaceAllUsesWith(new_trans_node->output());
        new_trans_node->addInput(n->outputs()[i]);
        auto output_sizes = n->outputs()[i]->sizes();
        reset_sizes(new_trans_node, output_sizes);
      }
      if (!trans_node->hasUses()) {
        trans_node->destroy();
      }
    }

    else if (n->kind() == kConcat) {
      auto inputs = n->inputs();
      std::vector<int64_t> input_perm;
      for (auto input : inputs) {
        if (input->node()->kind() != kTranspose) {
          return false;
        }
        if (!input_perm.empty() &&
            !std::equal(
                input_perm.begin(),
                input_perm.end(),
                input->node()->is(kperm).begin())) {
          return false;
        }
        if (input_perm.empty()) {
          for (int i = 0; i < input->node()->is(kperm).size(); i++) {
            input_perm.emplace_back(input->node()->is(kperm)[i]);
          }
        }
      }
      int32_t elem_type;
      for (int i = 0; i < inputs.size(); i++) {
        auto trans_node = inputs[i]->node();
        elem_type = trans_node->output()->elemType();
        trans_node->moveAfter(n);
        n->replaceInput(i, trans_node->input());
        trans_node->removeInput(0);
        if (!trans_node->hasUses()) {
          trans_node->destroy();
        }
      }

      auto axis = n->i(kaxis);
      if (axis < 0) {
        axis += input_perm.size();
      }
      auto new_axis = input_perm[axis];
      n->i_(kaxis, new_axis);
      Node* new_trans_node = graph.create(kTranspose, 1);
      new_trans_node->is_(kperm, std::move(input_perm));
      new_trans_node->insertAfter(n);
      new_trans_node->output()->setElemType(elem_type);
      n->output()->replaceAllUsesWith(new_trans_node->output());
      new_trans_node->addInput(n->output());
      auto output_sizes = n->output()->sizes();
      reset_sizes(new_trans_node, output_sizes);
    }

    else if (n->kind() == kResize && !n->has_domain()) {
      auto perm = n->inputs()[0]->node()->is(kperm);
      auto scales_it = graph.getInitializer(n->inputs()[1]->uniqueName());
      if (scales_it == graph.initializers().end()) {
        return false;
      }

      Tensor t;
      t.elem_type() = TensorProto_DataType_FLOAT;
      auto& data = t.floats();
      std::vector<float> new_scales(perm.size());
      for (int64_t i : perm) {
        new_scales[perm[i]] = (*scales_it).floats()[i];
      }
      for (float s : new_scales) {
        data.emplace_back(s);
      }
      auto& sizes = t.sizes();
      sizes.emplace_back(perm.size());
      Value* new_scales_v = graph.addInitializerAndInput(t);
      Value* old_scales_v = n->inputs()[1];
      n->replaceInput(1, new_scales_v);
      graph.eraseInitializerAndInput(old_scales_v);
      simple_swap(n, n->inputs()[0]->node(), graph, destroy_current);
    }

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
