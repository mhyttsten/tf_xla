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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh() {
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


#include <map>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

namespace test {
class SigBaseTest;
}  // end namespace test

class SigNode;

// To find nodes by name. Having the map ordered makes the tests easier,
// and it isn't used in production code often enough to get any win from
// using an unordered map.
using SigNodeMap = std::map<string, std::unique_ptr<SigNode>>;

// One node in the graph, in the form convenient for generation of the signature
// of the graph, and comparison of two (sub)graphs for equivalence. It refers to
// the original NodeDef protobuf for most information and adds the extra
// enrichment.
//
// The graph building is 2-stage: first match a SigNode with each NodeDef and
// collect them into a map that finds them by name, then process the map,
// deep-parse the underlying NodeDefs and connect the SigNodes together.
class SigNode {
 public:
  friend struct Signature;

  // Will keep the pointer to the underlying NodeDef, so that
  // underlying object must not be deleted while SigNode is alive.
  explicit SigNode(const NodeDef* node);

  // Access wrappers.
  const string& name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "name");
 return node_->name(); }
  const string& opcode() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_1(mht_1_v, 235, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "opcode");
 return node_->op(); }
  const NodeDef* node_def() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "node_def");
 return node_; }

  // For extraction of subgraphs into a separate SigNodeMap, copies the links
  // that point inside the subgraph from a full-graph SigNode to a subgraph
  // SigNode. The translation map defines the subgraph and gives the mapping
  // from the nodes in the full graph to the matching nodes in subgraph.
  using TranslationMap =
      std::unordered_map<const GenNode* /*full_graph*/, SigNode* /*subgraph*/>;
  void CopyLinks(const GenNode& from, const TranslationMap& map);

  // A link is an edge of the graph that connects 2 nodes. Each of the connected
  // nodes has its own perspective on the link, seeing its local port, remote
  // port and the remote node. The direction of the link is encoded in the
  // ports, one port is always incoming and another one outgoing.
  //
  // The link tag here contains both ports of the link viewed from the
  // perspective of this node; consisting of both the local port (i.e. at this
  // node) and remote port (i.e. on the other node), the local one going first.
  struct LinkTag {
    struct Hasher {
      size_t operator()(const LinkTag& tag) const noexcept {
        size_t hval = port_hasher(tag.local);
        CombineHash(port_hasher(tag.remote), &hval);
        return hval;
      }
      GenNode::Port::Hasher port_hasher;
    };

    LinkTag(GenNode::Port a_local, GenNode::Port a_remote)
        : local(a_local), remote(a_remote) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_3(mht_3_v, 271, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "LinkTag");
}

    // The default constructor is used for the default values in maps.
    // (false, 99) is an arbitrary value that makes the uninitialized
    // links easy to tell when debugging (they should never happen).
    LinkTag() : local(false, 99), remote(false, 99) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_4(mht_4_v, 279, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "LinkTag");
}

    // Port of the link on the local node.
    GenNode::Port local;
    // Port of the link on the remote node.
    GenNode::Port remote;

    bool operator==(const LinkTag& other) const {
      return local == other.local && remote == other.remote;
    }
    bool operator<(const LinkTag& other) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_5(mht_5_v, 292, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "operator<");

      return local < other.local ||
             (local == other.local && remote < other.remote);
    }
  };

  // Since the signature logic doesn't differentiate between the links
  // with the same tag (other than by the "peer" nodes on their other ends),
  // all the links with the same tag are grouped into a single structure.
  struct Link {
    LinkTag tag;
    size_t unique_hash;  // Hash of the tag after conflict resolution.
    // The remote node(s) on the other side on the link(s).
    using PeerVector = std::vector<SigNode*>;
    PeerVector peers;
  };

  // A way to look up the link description by its hash.
  using LinkHashMap = std::map<size_t, Link>;
  const LinkHashMap& hash_to_link() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_6(mht_6_v, 314, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "hash_to_link");
 return hash_to_link_; }

  // The enumeration of all the peer nodes in a predictable order.
  // Before the signature generation, only the link values determine the
  // order, after the signature generation the entries at the same
  // links get further sorted by their peer node ranks.
  struct HashedPeer {
    HashedPeer(size_t l, SigNode* p) : link_hash(l), peer(p) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_7(mht_7_v, 324, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "HashedPeer");
}

    struct LessByRank {
      bool operator()(const SigNode::HashedPeer& left,
                      const SigNode::HashedPeer& right) {
        return left.peer->unique_rank_ < right.peer->unique_rank_;
      }
    };

    size_t link_hash;
    SigNode* peer;
  };
  using HashedPeerVector = std::vector<HashedPeer>;
  const HashedPeerVector& hashed_peers() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_8(mht_8_v, 340, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "hashed_peers");
 return hashed_peers_; }

  // Compares two nodes in two different graphs for equivalence (two nodes in
  // the same graph would never be equivalent). Expects that the signatures of
  // the graphs have already been computed, so unique_rank_ is filled in and
  // the hashed_peers_ properly ordered.
  bool operator==(const SigNode& other) const;

  bool operator!=(const SigNode& other) const { return !(*this == other); }

 private:
  friend class test::SigBaseTest;

  // The CopyLinks code is split into 2 parts for testability.
  // The first pass builds a map ordered by LinkTag for predictability.
  void CopyLinksPass1(const GenNode& from, const TranslationMap& map,
                      std::map<LinkTag, Link>* link_map);
  // The second pass converts to the map by hash value,
  // resolves any hash conflicts, and builds the hashed peer vector.
  void CopyLinksPass2(std::map<LinkTag, Link>* link_map);

  // Computes the topological hash at distance 0. Resets the topo_hash_ vector
  // and hashed_nodes_;
  void ComputeTopoHash0();

  // Compute the topological has at the given distance. The hashes for all the
  // lower distances must be already computed for all the nodes in the graph.
  // Also computes next_hashed_nodes_ from last_hashed_nodes_.
  void ComputeTopoHash(int distance);

  // Get the hash value for a particular distance. It must be previously
  // computed.
  size_t GetTopoHash(int distance) const;

  // The hash value for the highest computed distance. It must be previously
  // computed.
  size_t GetHighTopoHash() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_9(mht_9_v, 379, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "GetHighTopoHash");

    CHECK(!topo_hash_.empty());
    return topo_hash_.back();
  }

  // Rehash the topmost hash, to avoid conflicts.
  void ReHighTopoHash() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_10(mht_10_v, 388, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "ReHighTopoHash");

    CHECK(!topo_hash_.empty());
    CombineHash(1, &topo_hash_.back());
  }

  // Ordering by node order and highest available hash (it must be
  // previously computed).
  struct NodeOrderLess {
    bool operator()(const SigNode* left, const SigNode* right) {
      return left->topo_hash_.back() < right->topo_hash_.back();
    }
  };

 private:
  const NodeDef* node_;

  // The bitmap mask with 1 bit set that represents this node in the set
  // during the computation of the signature.
  uint64_t node_mask_ = 0;

  // The code that populates this map makes sure that there are no hash
  // conflicts, rehashing if necessary.
  LinkHashMap hash_to_link_;

  // The enumeration of all the direct peers in the predictable order (which
  // happens to be the order ot their link tags, but the order of the hashes
  // would do too). It is used for the quick enumeration during the signature
  // computation. After the signature building is completed, the entries that
  // have the same link tag get further sorted in the order of the ranks of
  // their nodes.
  HashedPeerVector hashed_peers_;

  // The unique rank represents the order in which the node will be included
  // into the signature. It gets assigned in order either when the topo_hash_ of
  // this node becomes unique in the graph, or when the nodes are completely
  // equivalent, one of them is picked at random to assign the next rank, and
  // then the rest of the nodes attempt to disambiguate based on that
  // information.
  size_t unique_rank_ = ~0;
  // When hash_is_final_ is set, the topo_has_ vector stops growing, and the
  // last value from it is used for all the further hashes.
  bool hash_is_final_ = false;
  // The hashes that include the topology of the nodes up to the distance N. The
  // hash for distance 0 is produced from the attributes of this node itself and
  // its general connectivity properties but no information about the
  // neighboring nodes. The hash for distance D+1 is build from hashes at level
  // D of this node and of all its immediate neighbors. The neighbors that are
  // connected by equivalent links are included in a commutative way.
  std::vector<size_t> topo_hash_;
  // The set of nodes that got included into the computation of the
  // last topo_hash_ entry.
  uint64_t last_hashed_nodes_ = 0;
  // The next set of nodes that gets used for the current topo_hash entry.
  uint64_t next_hashed_nodes_ = 0;
};

// Signature of a graph. The computation is intertwined with the private methods
// of SigNode, so keeping both in the same file looks more convenient.
struct Signature {
  friend class test::SigBaseTest;

  // Maximal size of the graphs for which the signature can be computed.
  // Changing this constant won't magically add the support for a larger size,
  // the rest of implementation would have to be extended. The value of 64 is
  // driven by the size of a bitset in an uint64_t, and should be enough for our
  // purposes, while having a high efficiency of implementation.
  static constexpr int kMaxGraphSize = 64;

  // Using the map, computes the rest of the fields of a signature.
  // Returns an error is the graph is too big.
  Status Compute();

  // Convert the computed signature to a string representation.
  string ToString() const;

  SigNodeMap map;        // The nodes in the graph, accessible by name.
  size_t sig_short = 0;  // Hash of the signature, for the quick equality check.
  // The full signature: hashes of the nodes in a predictable order.
  std::vector<size_t> sig_full;
  // The nodes in the same order as they go in the signature.
  std::vector<SigNode*> nodes;

  // For building the unordered maps.
  size_t Hash() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_nodeDTh mht_11(mht_11_v, 474, "", "./tensorflow/core/grappler/graph_analyzer/sig_node.h", "Hash");
 return sig_short; }

  // Returns true if the graphs are equivalent. The signature must be already
  // computed.
  bool operator==(const Signature& other) const;

 private:
  // Populates the nodes vector from the map and initializes the state of the
  // nodes for the signature computation.
  void PrepareNodes();

  // Finds the nodes with the hashes that are unique and assigns the unique ids
  // to them. If there are nodes with non-unique hashes, exactly one node from
  // the first such sequence (in the order of hash values) will be picked and
  // assigned a unique id. Assumes that the nodes[0...(next_node_id-1)] have
  // been already assigned the unique ids. Advances next_node_id by at least 1.
  void FindUniqueHashes(size_t* next_node_id_p);

  // One round of the signature computation. Assumes that the
  // nodes[0...(next_node_id-1)] have been already assigned the fixed
  // positions, and thus computes the hashes only for the remaining nodes.
  void ComputeOneRound(size_t next_node_id);

  // Additional ordering of the hashed_peers_ links in the nodes, so that they
  // can be compared and printed in a predictable order.
  void OrderLinks();
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_
