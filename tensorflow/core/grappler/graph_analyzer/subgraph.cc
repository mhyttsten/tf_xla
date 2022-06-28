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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"

#include <functional>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

//=== Subgraph::Identity

Subgraph::Identity::Identity(InitializerList init) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::Identity::Identity");

  for (auto element : init) {
    insert(element);
  }
}

bool Subgraph::Identity::operator<(const Identity& other) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::Identity::operator<");

  // Shorter sets go first.
  if (this->size() < other.size()) {
    return true;
  }
  if (this->size() > other.size()) {
    return false;
  }
  for (auto lit = this->begin(), rit = other.begin(); lit != this->end();
       ++lit, ++rit) {
    if (*lit < *rit) {
      return true;
    }
    if (*lit > *rit) {
      return false;
    }
  }
  return false;  // Equal.
}

bool Subgraph::Identity::operator==(const Identity& other) const {
  if (this->size() != other.size()) {
    return false;
  }
  for (auto lit = this->begin(), rit = other.begin(); lit != this->end();
       ++lit, ++rit) {
    if (*lit != *rit) {
      return false;
    }
  }
  return true;  // Equal.
}

size_t Subgraph::Identity::Hash() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::Identity::Hash");

  std::hash<const GenNode*> hasher;
  size_t result = 0;
  for (auto ptr : *this) {
    CombineHash(hasher(ptr), &result);
  }
  return result;
}

string Subgraph::Dump() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::Dump");

  // TODO(babkin): this is simplified for now.
  std::vector<string> nodes;
  for (const auto& n : id_) {
    if (specific_) {
      nodes.emplace_back(absl::StrFormat("%s(%s)", n->opcode(), n->name()));
    } else {
      nodes.emplace_back(n->opcode());
    }
  }
  std::sort(nodes.begin(), nodes.end());

  return absl::StrFormat("%d: ", collation_count_) + absl::StrJoin(nodes, ", ");
}

void Subgraph::ExtractForSignature(SigNodeMap* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::ExtractForSignature");

  // Mapping of nodes from the original graph to the new one.
  SigNode::TranslationMap full_to_new;

  for (auto node : id_) {
    auto newnode_ref = absl::make_unique<SigNode>(node->node_def());
    auto newnode = newnode_ref.get();
    (*result)[node->name()] = std::move(newnode_ref);
    full_to_new[node] = newnode;
  }

  for (const auto& mapping : full_to_new) {
    mapping.second->CopyLinks(*mapping.first, full_to_new);
  }
}

//=== Subgraph

Subgraph::Subgraph(const Identity& parent_id, GenNode* add_node)
    : id_(parent_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_5(mht_5_v, 297, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "Subgraph::Subgraph");

  id_.insert(add_node);
  hash_ = id_.Hash();
}

//=== SubgraphIterator

SubgraphIterator::SubgraphIterator(const Subgraph::Identity* id)
    : id_(id), id_it_(id_->begin()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_6(mht_6_v, 308, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::SubgraphIterator");

  if (!id_->empty()) {
    link_map_it_ = (*id_it_)->links().begin();
    // In case if the node has no links.
    while (link_map_it_ == (*id_it_)->links().end()) {
      if (++id_it_ == id_->end()) {
        return;
      }
      link_map_it_ = (*id_it_)->links().begin();
    }
    link_idx_ = 0;
    // The LinkTargetVector should never be empty but just in case safeguard
    // against that too.
    PropagateNext();
  }
}

bool SubgraphIterator::Next() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_7(mht_7_v, 328, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::Next");

  if (AtEnd()) {
    return false;
  }
  ++link_idx_;
  return PropagateNext();
}

bool SubgraphIterator::NextIfSamePort() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_8(mht_8_v, 339, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::NextIfSamePort");

  if (AtEnd()) {
    return false;
  }
  const int64_t link_map_it_second_size = link_map_it_->second.size();
  if (link_idx_ + 1 < link_map_it_second_size) {
    ++link_idx_;
    return true;
  } else {
    return false;
  }
}

void SubgraphIterator::SkipPort() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_9(mht_9_v, 355, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::SkipPort");

  if (AtEnd()) {
    return;
  }
  link_idx_ = link_map_it_->second.size() - 1;
}

void SubgraphIterator::SkipNode() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_10(mht_10_v, 365, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::SkipNode");

  if (AtEnd()) {
    return;
  }
  for (auto next = link_map_it_; next != (*id_it_)->links().end(); ++next) {
    link_map_it_ = next;
  }
  link_idx_ = link_map_it_->second.size() - 1;
}

bool SubgraphIterator::PropagateNext() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_11(mht_11_v, 378, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphIterator::PropagateNext");

  // Loops are used to skip over the empty entries.
  const int64_t link_map_it_second_size = link_map_it_->second.size();
  while (link_idx_ >= link_map_it_second_size) {
    ++link_map_it_;
    while (link_map_it_ == (*id_it_)->links().end()) {
      if (++id_it_ == id_->end()) {
        return false;
      }
      link_map_it_ = (*id_it_)->links().begin();
    }
    link_idx_ = 0;
  }
  return true;
}

bool SubgraphIterator::operator==(const SubgraphIterator& other) const {
  if (id_ != other.id_) {
    return false;
  }
  if (id_it_ != other.id_it_) {
    return false;
  }
  // When AtEnd(), the rest of the fields are not valid.
  if (AtEnd()) {
    return true;
  }
  if (link_map_it_ != other.link_map_it_) {
    return false;
  }
  if (link_idx_ != other.link_idx_) {
    return false;
  }
  return true;
}

//=== SubgraphPtrSet

Subgraph* SubgraphPtrSet::ExtendParent(const Subgraph::Identity& parent_id,
                                       GenNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTcc mht_12(mht_12_v, 420, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.cc", "SubgraphPtrSet::ExtendParent");

  if (parent_id.find(node) != parent_id.end()) {
    // This was another link to the node that is already in the parent.
    return nullptr;
  }

  // Constructing an object just to check that an equivalent one is already
  // present is kind of ugly but storing the references rather than the objects
  // in the set avoids the need to make the object copyable.
  auto sg = absl::make_unique<Subgraph>(parent_id, node);
  if (find(sg) != end()) {
    // This subgraph was already found by extending from a different path.
    return nullptr;
  }

  Subgraph* ptr = sg.get();
  insert(std::move(sg));
  return ptr;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
