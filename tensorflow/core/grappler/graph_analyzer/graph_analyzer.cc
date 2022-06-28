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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc() {
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

#include <deque>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/graph_analyzer.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

GraphAnalyzer::GraphAnalyzer(const GraphDef& graph, int subgraph_size)
    : graph_(graph), subgraph_size_(subgraph_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::GraphAnalyzer");
}

GraphAnalyzer::~GraphAnalyzer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::~GraphAnalyzer");
}

Status GraphAnalyzer::Run() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::Run");

  // The signature computation code would detect this too, but better
  // to report it up front than spend time computing all the graphs first.
  if (subgraph_size_ > Signature::kMaxGraphSize) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrFormat("Subgraphs of %d nodes are not supported, "
                                  "the maximal supported node count is %d.",
                                  subgraph_size_, Signature::kMaxGraphSize));
  }

  Status st = BuildMap();
  if (!st.ok()) {
    return st;
  }

  FindSubgraphs();
  DropInvalidSubgraphs();
  st = CollateResult();
  if (!st.ok()) {
    return st;
  }

  return Status::OK();
}

Status GraphAnalyzer::BuildMap() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::BuildMap");

  nodes_.clear();
  return GenNode::BuildGraphInMap(graph_, &nodes_);
}

void GraphAnalyzer::FindSubgraphs() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::FindSubgraphs");

  result_.clear();

  if (subgraph_size_ < 1) {
    return;
  }

  partial_.clear();
  todo_.clear();  // Just in case.

  // Start with all subgraphs of size 1.
  const Subgraph::Identity empty_parent;
  for (const auto& node : nodes_) {
    if (subgraph_size_ == 1) {
      result_.ExtendParent(empty_parent, node.second.get());
    } else {
      // At this point ExtendParent() is guaranteed to not return nullptr.
      todo_.push_back(partial_.ExtendParent(empty_parent, node.second.get()));
    }
  }

  // Then extend the subgraphs until no more extensions are possible.
  while (!todo_.empty()) {
    ExtendSubgraph(todo_.front());
    todo_.pop_front();
  }

  partial_.clear();
}

void GraphAnalyzer::ExtendSubgraph(Subgraph* parent) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::ExtendSubgraph");

  const int next_parent_id = parent->id().size() + 1;
  bool will_complete = (next_parent_id == subgraph_size_);
  SubgraphPtrSet& sg_set = will_complete ? result_ : partial_;

  const GenNode* last_all_or_none_node = nullptr;
  for (SubgraphIterator sit(parent); !sit.AtEnd(); sit.Next()) {
    const GenNode* node = sit.GetNode();
    GenNode::Port port = sit.GetPort();
    const GenNode::LinkTarget& neighbor = sit.GetNeighbor();

    if (node->AllInputsOrNone() && port.IsInbound() && !port.IsControl()) {
      if (node != last_all_or_none_node) {
        ExtendSubgraphAllOrNone(parent, node);
        last_all_or_none_node = node;
      }
      sit.SkipPort();
    } else if (neighbor.node->AllInputsOrNone() && !port.IsInbound() &&
               !port.IsControl()) {
      if (parent->id().find(neighbor.node) == parent->id().end()) {
        // Not added yet.
        ExtendSubgraphAllOrNone(parent, neighbor.node);
      }
    } else if (node->IsMultiInput(port)) {
      ExtendSubgraphPortAllOrNone(parent, node, port);
      sit.SkipPort();
    } else if (neighbor.node->IsMultiInput(neighbor.port)) {
      // Would need to add all inputs of the neighbor node at this port at
      // once.
      if (parent->id().find(neighbor.node) != parent->id().end()) {
        continue;  // Already added.
      }
      ExtendSubgraphPortAllOrNone(parent, neighbor.node, neighbor.port);
    } else {
      Subgraph* sg = sg_set.ExtendParent(parent->id(), neighbor.node);
      if (!will_complete && sg != nullptr) {
        todo_.push_back(sg);
      }
    }
  }
}

void GraphAnalyzer::ExtendSubgraphAllOrNone(Subgraph* parent,
                                            const GenNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_6(mht_6_v, 324, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::ExtendSubgraphAllOrNone");

  Subgraph::Identity id = parent->id();
  id.insert(node);

  auto range_end = node->links().end();

  for (auto nbit = node->links().begin(); nbit != range_end; ++nbit) {
    auto port = nbit->first;
    if (!port.IsInbound() || port.IsControl()) {
      continue;
    }

    // Since there might be multiple links to the same nodes,
    // have to add all links one-by-one to check whether the subgraph
    // would grow too large. But if it does grow too large, there is no
    // point in growing it more, can just skip over the rest of the links.
    for (const auto& link : nbit->second) {
      id.insert(link.node);
      const int id_size = id.size();
      if (id_size > subgraph_size_) {
        return;  // Too big.
      }
    }
  }

  AddExtendedSubgraph(parent, id);
}

void GraphAnalyzer::ExtendSubgraphPortAllOrNone(Subgraph* parent,
                                                const GenNode* node,
                                                GenNode::Port port) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_7(mht_7_v, 357, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::ExtendSubgraphPortAllOrNone");

  auto nbit = node->links().find(port);
  if (nbit == node->links().end()) {
    return;  // Should never happen.
  }

  Subgraph::Identity id = parent->id();
  id.insert(node);

  // Since there might be multiple links to the same nodes,
  // have to add all links one-by-one to check whether the subgraph
  // would grow too large. But if it does grow too large, there is no
  // point in growing it more, can just skip over the rest of the links.
  for (const auto& link : nbit->second) {
    id.insert(link.node);
    const int id_size = id.size();
    if (id_size > subgraph_size_) {
      return;  // Too big.
    }
  }

  AddExtendedSubgraph(parent, id);
}

void GraphAnalyzer::AddExtendedSubgraph(Subgraph* parent,
                                        const Subgraph::Identity& id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::AddExtendedSubgraph");

  if (id.size() == parent->id().size()) {
    return;  // Nothing new was added.
  }

  auto sg = absl::make_unique<Subgraph>(id);
  SubgraphPtrSet& spec_sg_set =
      (id.size() == subgraph_size_) ? result_ : partial_;
  if (spec_sg_set.find(sg) != spec_sg_set.end()) {
    // This subgraph was already found by extending from a different path.
    return;
  }
  const int id_size = id.size();
  if (id_size != subgraph_size_) {
    todo_.push_back(sg.get());
  }
  spec_sg_set.insert(std::move(sg));
}

void GraphAnalyzer::DropInvalidSubgraphs() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_9(mht_9_v, 407, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::DropInvalidSubgraphs");

  auto resit = result_.begin();
  while (resit != result_.end()) {
    if (HasInvalidMultiInputs(resit->get())) {
      auto delit = resit;
      ++resit;
      result_.erase(delit);
    } else {
      ++resit;
    }
  }
}

bool GraphAnalyzer::HasInvalidMultiInputs(Subgraph* sg) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_10(mht_10_v, 423, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::HasInvalidMultiInputs");

  // Do the all-or-none-input nodes.
  for (auto const& node : sg->id()) {
    if (!node->AllInputsOrNone()) {
      continue;
    }

    bool anyIn = false;
    bool anyOut = false;

    auto range_end = node->links().end();
    for (auto nbit = node->links().begin(); nbit != range_end; ++nbit) {
      auto port = nbit->first;
      if (!port.IsInbound() || port.IsControl()) {
        continue;
      }

      // Since there might be multiple links to the same nodes,
      // have to add all links one-by-one to check whether the subgraph
      // would grow too large. But if it does grow too large, there is no
      // point in growing it more, can just skip over the rest of the links.
      for (const auto& link : nbit->second) {
        if (sg->id().find(link.node) == sg->id().end()) {
          anyOut = true;
        } else {
          anyIn = true;
        }
      }
    }

    if (anyIn && anyOut) {
      return true;
    }
  }

  // Do the multi-input ports.
  for (SubgraphIterator sit(sg); !sit.AtEnd(); sit.Next()) {
    if (sit.GetNode()->IsMultiInput(sit.GetPort())) {
      bool anyIn = false;
      bool anyOut = false;
      do {
        GenNode* peer = sit.GetNeighbor().node;
        if (sg->id().find(peer) == sg->id().end()) {
          anyOut = true;
        } else {
          anyIn = true;
        }
      } while (sit.NextIfSamePort());

      if (anyIn && anyOut) {
        return true;
      }
    }
  }
  return false;
}

Status GraphAnalyzer::CollateResult() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_11(mht_11_v, 483, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::CollateResult");

  ordered_collation_.clear();
  collation_map_.clear();

  // Collate by the signatures of the graphs.
  for (const auto& it : result_) {
    auto sig = absl::make_unique<Signature>();
    it->ExtractForSignature(&sig->map);
    Status status = sig->Compute();
    if (!status.ok()) {
      return status;
    }

    auto& coll_entry = collation_map_[sig.get()];
    if (coll_entry.sig == nullptr) {
      coll_entry.sig = std::move(sig);
    }
    ++coll_entry.count;
  }

  // Then order them by the count.
  for (auto& entry : collation_map_) {
    ordered_collation_.insert(&entry.second);
  }

  result_.clear();  // Not needed after collation.

  return Status::OK();
}

std::vector<string> GraphAnalyzer::DumpRawSubgraphs() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_12(mht_12_v, 516, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::DumpRawSubgraphs");

  std::vector<string> result;
  for (const auto& it : result_) {
    result.emplace_back(it->Dump());
  }
  return result;
}

std::vector<string> GraphAnalyzer::DumpSubgraphs() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_13(mht_13_v, 527, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::DumpSubgraphs");

  std::vector<string> result;
  for (auto ptr : ordered_collation_) {
    result.emplace_back(
        absl::StrFormat("%d %s", ptr->count, ptr->sig->ToString()));
  }
  return result;
}

Status GraphAnalyzer::OutputSubgraphs() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzerDTcc mht_14(mht_14_v, 539, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc", "GraphAnalyzer::OutputSubgraphs");

  size_t total = 0;
  for (auto ptr : ordered_collation_) {
    std::cout << ptr->count << ' ' << ptr->sig->ToString() << '\n';
    total += ptr->count;
  }
  std::cout << "Total: " << total << '\n';
  if (std::cout.fail()) {
    return Status(error::DATA_LOSS, "Failed to write to stdout");
  } else {
    return Status::OK();
  }
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
