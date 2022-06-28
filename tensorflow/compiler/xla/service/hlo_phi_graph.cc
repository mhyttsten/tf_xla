#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_phi_graph.h"

#include <queue>

namespace xla {
HloValue::Id PhiGraph::GetOptimizedId(const HloValue& value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_0(mht_0_v, 190, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::GetOptimizedId");

  Node* node = value_id_to_node_[value.id()];
  CHECK(!node->mark_as_dead);
  return node->value_id;
}

// Returns true if the inputs to a hlo value are the same as `inputs`.
bool PhiGraph::InputsEqualTo(const HloValue& value,
                             absl::Span<const HloValue* const> inputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_1(mht_1_v, 201, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::InputsEqualTo");

  auto iter = value_id_to_node_.find(value.id());
  CHECK(iter != value_id_to_node_.end());
  absl::flat_hash_set<HloValue::Id> existing_set;
  for (Node* operand : iter->second->operands) {
    existing_set.insert(operand->value_id);
  }
  absl::flat_hash_set<HloValue::Id> new_set;
  for (const HloValue* input : inputs) {
    new_set.insert(input->id());
  }
  return existing_set == new_set;
}

HloValue::Id PhiGraph::FindOptimizedValue(const HloValue::Id id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::FindOptimizedValue");

  auto iter = value_id_to_node_.find(id);
  CHECK(iter != value_id_to_node_.end());
  CHECK(!iter->second->mark_as_dead);
  return iter->second->value_id;
}

PhiGraph::Node* PhiGraph::CreateOrReuseNode(const HloValue& value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_3(mht_3_v, 228, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::CreateOrReuseNode");

  auto iter = value_id_to_node_.find(value.id());
  if (iter == value_id_to_node_.end()) {
    node_storage_.emplace_back(absl::make_unique<Node>());
    Node* node = node_storage_.back().get();
    node->value_id = value.id();
    value_id_to_node_[value.id()] = node;
    node_to_value_id_[node].push_back(value.id());
    return node;
  } else {
    // A node is already registered with this value, check the value_id
    // is the same as previously registrated.
    CHECK_NE(iter->second, nullptr);
    CHECK_EQ(iter->second->value_id, value.id());
    return iter->second;
  }
}

void PhiGraph::ReplaceNodeWith(PhiGraph::Node* node, PhiGraph::Node* replace) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_4(mht_4_v, 249, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::ReplaceNodeWith");

  // Update users.
  CHECK(node->is_phi);
  if (node->mark_as_dead) {
    // The node has already been replaced with another.
    return;
  }
  if (replace->mark_as_dead) {
    // The node we are placing with has already been replaced with another node.
    auto iter = value_id_to_node_.find(replace->value_id);
    CHECK(iter != value_id_to_node_.end());
    return ReplaceNodeWith(node, iter->second);
  }
  CHECK(!replace->mark_as_dead);
  for (Node* user : node->users) {
    absl::c_replace(user->operands, node, replace);
  }

  // Update operand's users
  for (Node* operand : node->operands) {
    absl::c_replace(operand->users, node, replace);
  }

  for (HloValue::Id value_id : node_to_value_id_[node]) {
    CHECK(value_id_to_node_.contains(value_id));
    value_id_to_node_[value_id] = replace;
  }
  // Update mappings to HloValue::Id.
  absl::c_copy(node_to_value_id_[node],
               std::back_inserter(node_to_value_id_[replace]));
  node_to_value_id_[node].clear();
  node->mark_as_dead = true;
}

void PhiGraph::RegisterPhi(const HloValue& value,
                           absl::Span<const HloValue* const> inputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_5(mht_5_v, 287, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::RegisterPhi");

  Node* node = CreateOrReuseNode(value);
  CHECK(value.is_phi());
  node->is_phi = true;
  node->operands.clear();
  for (auto input : inputs) {
    CHECK(input != nullptr);
    Node* input_node = CreateOrReuseNode(*input);
    node->operands.push_back(input_node);
  }
}

std::string PhiGraph::ToString() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_6(mht_6_v, 302, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::ToString");

  std::string out = "PhiGraph: \n";
  for (auto& node : node_storage_) {
    std::string is_phi = node->is_phi ? ", phi" : "";
    std::string is_optimized = node->mark_as_dead ? ", dead" : "";
    absl::StrAppend(&out, node->value_id);
    absl::StrAppend(&out, is_phi);
    absl::StrAppend(&out, is_optimized, ":\n");
    for (Node* input : node->operands) {
      absl::StrAppend(&out, "  ", input->value_id);
      absl::StrAppend(&out, "\n");
    }
  }
  return out;
}

void PhiGraph::Optimize() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_phi_graphDTcc mht_7(mht_7_v, 321, "", "./tensorflow/compiler/xla/service/hlo_phi_graph.cc", "PhiGraph::Optimize");

  VLOG(2) << "Optimizing phi graph:";
  XLA_VLOG_LINES(2, ToString());
  // Set up users for each node.
  for (auto& node : node_storage_) {
    for (Node* input : node->operands) {
      input->users.push_back(node.get());
    }
  }

  // input_node->users.push_back(node);
  bool changed = true;

  // Run the optimization to a fixed point.
  while (changed) {
    changed = false;
    absl::flat_hash_set<Node*> checked_for_closure;
    for (auto& node : node_storage_) {
      // Only optimize phi node.
      if (!node->is_phi) {
        continue;
      }
      // Skip dead nodes
      if (node->mark_as_dead) {
        continue;
      }

      Node* node_ptr = node.get();

      VLOG(2) << "Optimizing: " << node_ptr->value_id;

      CHECK_GE(node_ptr->operands.size(), 1);

      // Remove self-referencing ids from users and operands.
      auto it = absl::c_find(node_ptr->operands, node_ptr);
      while (it != node_ptr->operands.end()) {
        node_ptr->operands.erase(it);
        it = absl::c_find(node_ptr->operands, node_ptr);
      }

      it = absl::c_find(node_ptr->users, node_ptr);
      while (it != node_ptr->users.end()) {
        node_ptr->users.erase(it);
        it = absl::c_find(node_ptr->users, node_ptr);
      }

      // If all inputs to phi (after self referencing ids are removed) are the
      // same value, replace the phi with that value.
      //
      // phi(A, A, ... A) => A
      // phi(A, self) = phi(A) => A
      CHECK_GE(node_ptr->operands.size(), 1);
      bool all_inputs_are_same = absl::c_all_of(
          node_ptr->operands,
          [&](Node* elem) { return elem == node_ptr->operands[0]; });

      if (all_inputs_are_same) {
        VLOG(1) << "All inputs to node " << node_ptr->value_id
                << " are the same, replacing it with "
                << node_ptr->operands[0]->value_id;
        ReplaceNodeWith(node_ptr, node_ptr->operands[0]);
        changed = true;
        continue;
      }

      // Find a closure of inter-connected phis and one non-phi node. Replace
      // all phis with that non-phi node.
      //
      // def A = phi(B, C)
      // def B = phi(C, D)
      // def C = phi(A, B)
      // def D = non-phi
      // Replace A, B, and C with D:
      // A = phi(B, C) => D
      // B = phi(C, D) => D
      // C = phi(A, B) => D
      if (checked_for_closure.contains(node_ptr)) {
        continue;
      }
      // Keeps track of nodes in the current closure being tested.
      absl::flat_hash_set<Node*> workset;
      std::queue<Node*> worklist;
      Node* non_phi = nullptr;
      worklist.push(node_ptr);
      while (!worklist.empty()) {
        Node* todo = worklist.front();
        worklist.pop();
        if (workset.contains(todo)) {
          continue;
        }
        checked_for_closure.insert(todo);
        workset.insert(todo);
        for (Node* operand : todo->operands) {
          worklist.push(operand);
        }
        if (!todo->is_phi) {
          if (non_phi != nullptr && non_phi != todo) {
            // We see distinct non-phi nodes in the closure, can't apply the
            // optimization.
            non_phi = nullptr;
            // Break the while loop non_phi setting to nullptr, signaling that
            // the optimization can't be applied.
            break;
          } else {
            // This is the non_phi node we are seeing so far.
            non_phi = todo;
          }
        }
      }
      if (non_phi != nullptr) {
        // Replace all phi nodes in the closure/workset with the non_phi node.
        for (Node* node : workset) {
          if (!node->is_phi) {
            CHECK_EQ(node, non_phi);
            continue;
          }
          VLOG(1) << "Replace node " << node->value_id
                  << " in the closure with node " << non_phi->value_id;
          ReplaceNodeWith(node, non_phi);
          changed = true;
        }
      }
    }
  }
}
}  // namespace xla
