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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh() {
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


#include <initializer_list>
#include <set>
#include <unordered_set>

#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/map_tools.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

// The description of a single subgraph for processing.
class Subgraph {
 public:
  // Identity of a single subgraph as a set of nodes.
  class Identity : public gtl::FlatSet<const GenNode*> {
   public:
    using InitializerList = std::initializer_list<GenNode*>;

    Identity() = default;
    Identity(InitializerList init);
    bool operator<(const Identity& other) const;
    bool operator==(const Identity& other) const;

    // Compute the hash.
    size_t Hash() const;
  };

  explicit Subgraph(Identity id) : id_(std::move(id)), hash_(id_.Hash()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "Subgraph");
}

  // Construct by extending the parent identity with an extra node.
  Subgraph(const Identity& parent_id, GenNode* add_node);

  Subgraph() = delete;
  Subgraph(const Subgraph& other) = delete;
  void operator=(const Subgraph& other) = delete;

  // Order for building sets of subgraphs.
  bool operator<(const Subgraph& other) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "operator<");
 return this->id_ < other.id_; }
  // Support for hashed sets.
  bool operator==(const Subgraph& other) const {
    return this->id_ == other.id_;
  }
  size_t Hash() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "Hash");
 return hash_; }

  // Dump the subgraph information to a string.
  string Dump();

  // Extract this subgraph into a separate graph representation for signature
  // building, that includes only the links between the nodes in the subgraph
  // and drops all the external links. The result map should be clear before the
  // call.
  void ExtractForSignature(SigNodeMap* result);

  const Identity& id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_3(mht_3_v, 253, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "id");
 return id_; }
  bool specific() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_4(mht_4_v, 257, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "specific");
 return specific_; }
  void SetSpecific(bool value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_5(mht_5_v, 261, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "SetSpecific");
 specific_ = value; }
  int32_t collation_count() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_6(mht_6_v, 265, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "collation_count");
 return collation_count_; }
  void AddCollation(int32_t n = 1) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_7(mht_7_v, 269, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "AddCollation");
 collation_count_ += n; }
  void ResetCollation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_8(mht_8_v, 273, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "ResetCollation");
 collation_count_ = 1; }
  void MergeCollation(const Subgraph& other) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_9(mht_9_v, 277, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "MergeCollation");

    collation_count_ += other.collation_count_;
  }

 private:
  // Identity also serves as the list of nodes. It never changes throughout the
  // life of subgraph.
  Identity id_;
  size_t hash_;  // Cached from the identity.
  // Whether the dump should include the specific names of the nodes. The
  // non-specific (i.e. generic) subgraphs represent a collation of multiple
  // subgraphs.
  bool specific_ = true;
  // How many collated subgraphs are represented by this subgraph.
  int32_t collation_count_ = 1;
};

// Iteration of all links in a subgraph. This is more like Java iterators than
// the normal C++ iterators. It's simpler this way and there seems to be no
// major reason to make it a proper C++ iterator.
class SubgraphIterator {
 public:
  // Obviously an iterator is valid only until the original object
  // gets destroyed.
  explicit SubgraphIterator(const Subgraph::Identity* id);
  explicit SubgraphIterator(const Subgraph* sg) : SubgraphIterator(&sg->id()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_10(mht_10_v, 305, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "SubgraphIterator");
}

  // Check whether the built-in iterator is at the end.
  bool AtEnd() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_11(mht_11_v, 311, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "AtEnd");
 return id_it_ == id_->end(); }

  // Get the neighbor at the current iterator.
  // MUST NOT be called when AtEnd();
  const GenNode::LinkTarget& GetNeighbor() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_12(mht_12_v, 318, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "GetNeighbor");

    return link_map_it_->second[link_idx_];
  }

  // Get the node at the current iterator.
  // MUST NOT be called when AtEnd();
  const GenNode* GetNode() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_13(mht_13_v, 327, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "GetNode");
 return *id_it_; }

  // Get the port leading to the neighbor at the current iterator.
  // MUST NOT be called when AtEnd();
  GenNode::Port GetPort() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsubgraphDTh mht_14(mht_14_v, 334, "", "./tensorflow/core/grappler/graph_analyzer/subgraph.h", "GetPort");
 return link_map_it_->first; }

  // Increases the iterator.
  // Returns true if NOT AtEnd() after increasing the iterator.
  // Safe to call if already AtEnd().
  bool Next();

  // If there are more links at the same port, increases the iterator and
  // returns true. Otherwise leaves the iterator unchanged and returns false.
  bool NextIfSamePort();

  // Increases the iterator directly to the last position on the current port
  // (or if already there then doesn't increase). Equivalent to calling
  // NextIfSamePort() while it returns true, but faster.
  // Safe to call if already AtEnd().
  void SkipPort();

  // Increases the iterator directly to the last position on the current node.
  // Safe to call if already AtEnd().
  void SkipNode();

  // Returns true if the iterators are exactly the same.
  bool operator==(const SubgraphIterator& other) const;
  bool operator!=(const SubgraphIterator& other) const {
    return !(*this == other);
  }

 private:
  // After link_idx_ has been increased, make sure that it points to the
  // next valid element (or end) by increasing the higher levels of iteration if
  // needed.
  // Returns true if NOT AtEnd() after increasing the iterator.
  // NOT safe to call if already AtEnd().
  bool PropagateNext();

  // Identity of the subgraph being iterated over.
  const Subgraph::Identity* id_;

  // The current position, allowing to iterate through the links (see the
  // reasoning for it in the public section).
  //
  // (1) Iterator of the nodes in the subgraph.
  Subgraph::Identity::const_iterator id_it_;
  // (2) Iterator in the link map of the node.
  GenNode::LinkMap::const_iterator link_map_it_;
  // (3) Index in the vector of the links.
  int32_t link_idx_;
};

// A convenient way to store subgraphs: in a set of unique_ptrs. This way the
// addresses of subgraph objects will stay stable, and the objects themselves
// won't be copied.
class SubgraphPtrSet
    : public std::unordered_set<std::unique_ptr<Subgraph>,
                                HashAtPtr<std::unique_ptr<Subgraph>>,
                                EqAtPtr<std::unique_ptr<Subgraph>>> {
 public:
  // Attempts to extend the set by adding a new subgraph that gets created by
  // adding one node to the parent subgraph. If such a subgraph already exists,
  // returns nullptr, otherwise returns the pointer to the new subgraph.
  Subgraph* ExtendParent(const Subgraph::Identity& parent_id, GenNode* node);
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_
