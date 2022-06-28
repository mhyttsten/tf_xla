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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc() {
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

#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"
#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::Ne;
using ::testing::SizeIs;

//===

TEST(SigNodeLinkTag, Compare) {
  SigNode::LinkTag a(GenNode::Port(false, 1), GenNode::Port(false, 2));
  SigNode::LinkTag b(GenNode::Port(false, 1), GenNode::Port(false, 2));
  SigNode::LinkTag c(GenNode::Port(false, 2), GenNode::Port(false, 1));
  SigNode::LinkTag d(GenNode::Port(false, 1), GenNode::Port(false, 3));
  SigNode::LinkTag e(GenNode::Port(false, 2), GenNode::Port(false, 2));

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a == e);

  EXPECT_FALSE(a < b);
  EXPECT_FALSE(b < a);

  EXPECT_TRUE(a < c);
  EXPECT_FALSE(c < a);

  EXPECT_TRUE(a < d);
  EXPECT_FALSE(d < a);
}

//===

class SigBaseTest : public ::testing::Test, protected TestGraphs {
 protected:
  void BuildSigMap(const GraphDef& graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "BuildSigMap");

    gen_map_.clear();
    sig_.map.clear();
    CHECK(GenNode::BuildGraphInMap(graph, &gen_map_).ok());
    Subgraph::Identity id;
    for (const auto& entry : gen_map_) {
      id.insert(entry.second.get());
    }
    Subgraph sg(id);
    sg.ExtractForSignature(&sig_.map);
  }

  static void CopyLinksPass2(
      std::map<SigNode::LinkTag, SigNode::Link>* link_map, SigNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "CopyLinksPass2");

    node->CopyLinksPass2(link_map);
  }

  static void ComputeTopoHash0(SigNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "ComputeTopoHash0");
 node->ComputeTopoHash0(); }

  static void ComputeTopoHash(int distance, SigNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "ComputeTopoHash");

    node->ComputeTopoHash(distance);
  }

  static size_t GetTopoHash(int distance, SigNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "GetTopoHash");

    return node->GetTopoHash(distance);
  }

  static size_t GetHighTopoHash(SigNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "GetHighTopoHash");

    return node->GetHighTopoHash();
  }

  static void ReHighTopoHash(SigNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_6(mht_6_v, 282, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "ReHighTopoHash");
 node->ReHighTopoHash(); }

  static SigNode::HashedPeerVector& RefHashedPeers(SigNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_7(mht_7_v, 287, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefHashedPeers");

    return node->hashed_peers_;
  }
  static size_t& RefUniqueRank(SigNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_8(mht_8_v, 293, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefUniqueRank");
 return node->unique_rank_; }
  static bool& RefHashIsFinal(SigNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_9(mht_9_v, 297, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefHashIsFinal");
 return node->hash_is_final_; }
  static std::vector<size_t>& RefTopoHash(SigNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_10(mht_10_v, 301, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefTopoHash");

    return node->topo_hash_;
  }
  static uint64_t& RefNodeMask(SigNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_11(mht_11_v, 307, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefNodeMask");
 return node->node_mask_; }
  static uint64_t& RefLastHashedNodes(SigNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_12(mht_12_v, 311, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefLastHashedNodes");

    return node->last_hashed_nodes_;
  }
  static uint64_t& RefNextHashedNodes(SigNode* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_13(mht_13_v, 317, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "RefNextHashedNodes");

    return node->next_hashed_nodes_;
  }

  static void PrepareNodes(Signature* signature) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_14(mht_14_v, 324, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "PrepareNodes");
 signature->PrepareNodes(); }

  static void FindUniqueHashes(size_t* next_node_id_p, Signature* signature) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_15(mht_15_v, 329, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "FindUniqueHashes");

    signature->FindUniqueHashes(next_node_id_p);
  }

  static void ComputeOneRound(size_t next_node_id, Signature* signature) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_16(mht_16_v, 336, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "ComputeOneRound");

    signature->ComputeOneRound(next_node_id);
  }

  static void OrderLinks(Signature* signature) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_17(mht_17_v, 343, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "OrderLinks");
 signature->OrderLinks(); }

  // These get initialized in BuildSigMap().
  GenNodeMap gen_map_;
  Signature sig_;
};

//===

class SigNodeTest : public SigBaseTest {};

// Tests that the duplicate hashes get resolved by rehashing.
TEST_F(SigNodeTest, DuplicateHash) {
  NodeDef node1 = MakeNodeConst("node1");
  NodeDef node2 = MakeNodeConst("node2");
  NodeDef node3 = MakeNodeShapeN("node3", "node1", "node2");

  SigNode sn1(&node1);
  SigNode sn2(&node2);
  SigNode sn3(&node3);

  constexpr size_t kSameHash = 999;

  SigNode::Link link1;
  link1.tag = SigNode::LinkTag(GenNode::Port(true, 0), GenNode::Port(false, 0));
  link1.unique_hash = kSameHash;
  link1.peers.emplace_back(&sn1);

  SigNode::Link link2;
  link2.tag = SigNode::LinkTag(GenNode::Port(true, 1), GenNode::Port(false, 0));
  link2.unique_hash = kSameHash;
  link2.peers.emplace_back(&sn2);

  SigNode::Link link3;
  link3.tag = SigNode::LinkTag(GenNode::Port(true, 2), GenNode::Port(false, 0));
  link3.unique_hash = kSameHash;
  link3.peers.emplace_back(&sn3);

  std::map<SigNode::LinkTag, SigNode::Link> link_map;
  link_map[link1.tag] = link1;
  link_map[link2.tag] = link2;
  link_map[link3.tag] = link3;

  CopyLinksPass2(&link_map, &sn3);
  auto& hl = sn3.hash_to_link();
  EXPECT_THAT(hl, SizeIs(3));

  // Check that the hashes are self_consistent, and put the entries into
  // another map with a known order.
  std::map<SigNode::LinkTag, SigNode::Link> rehashed;
  auto hlit = hl.begin();
  ASSERT_THAT(hlit, Ne(hl.end()));
  EXPECT_THAT(hlit->second.unique_hash, Eq(hlit->first));
  rehashed[hlit->second.tag] = hlit->second;
  ++hlit;
  ASSERT_THAT(hlit, Ne(hl.end()));
  EXPECT_THAT(hlit->second.unique_hash, Eq(hlit->first));
  rehashed[hlit->second.tag] = hlit->second;
  ++hlit;
  ASSERT_THAT(hlit, Ne(hl.end()));
  EXPECT_THAT(hlit->second.unique_hash, Eq(hlit->first));
  rehashed[hlit->second.tag] = hlit->second;

  // Just in case.
  ASSERT_THAT(rehashed, SizeIs(3));

  auto rhit = rehashed.begin();
  ASSERT_THAT(rhit, Ne(rehashed.end()));
  EXPECT_TRUE(rhit->second.tag == link1.tag);
  EXPECT_THAT(rhit->second.unique_hash, Eq(kSameHash));
  EXPECT_THAT(rhit->second.peers, ElementsAre(&sn1));

  ++rhit;
  ASSERT_THAT(rhit, Ne(rehashed.end()));
  EXPECT_TRUE(rhit->second.tag == link2.tag);
  // This hash must be rehashed.
  EXPECT_THAT(rhit->second.unique_hash, Ne(kSameHash));
  size_t hash2 = rhit->second.unique_hash;
  EXPECT_THAT(rhit->second.peers, ElementsAre(&sn2));

  ++rhit;
  ASSERT_THAT(rhit, Ne(rehashed.end()));
  EXPECT_TRUE(rhit->second.tag == link3.tag);
  // This hash must be rehashed.
  EXPECT_THAT(rhit->second.unique_hash, Ne(kSameHash));
  EXPECT_THAT(rhit->second.unique_hash, Ne(hash2));
  size_t hash3 = rhit->second.unique_hash;
  EXPECT_THAT(rhit->second.peers, ElementsAre(&sn3));

  auto& peers = sn3.hashed_peers();
  EXPECT_THAT(peers, SizeIs(3));

  auto peerit = peers.begin();
  ASSERT_THAT(peerit, Ne(peers.end()));
  EXPECT_THAT(peerit->link_hash, Eq(kSameHash));
  EXPECT_THAT(peerit->peer, Eq(&sn1));

  ++peerit;
  ASSERT_THAT(peerit, Ne(peers.end()));
  EXPECT_THAT(peerit->link_hash, Eq(hash2));
  EXPECT_THAT(peerit->peer, Eq(&sn2));

  ++peerit;
  ASSERT_THAT(peerit, Ne(peers.end()));
  EXPECT_THAT(peerit->link_hash, Eq(hash3));
  EXPECT_THAT(peerit->peer, Eq(&sn3));
}

// The full CopyLinks() is tested in (SubgraphTest, ExtractForSignature).

TEST_F(SigNodeTest, GetTopoHash) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);

  // Fake some hash values.
  RefTopoHash(&sn1).emplace_back(123);
  RefTopoHash(&sn1).emplace_back(456);

  EXPECT_THAT(GetTopoHash(0, &sn1), Eq(123));
  EXPECT_THAT(GetTopoHash(1, &sn1), Eq(456));

  RefHashIsFinal(&sn1) = true;

  EXPECT_THAT(GetTopoHash(0, &sn1), Eq(123));
  EXPECT_THAT(GetTopoHash(1, &sn1), Eq(456));
  EXPECT_THAT(GetTopoHash(2, &sn1), Eq(456));

  EXPECT_THAT(GetHighTopoHash(&sn1), Eq(456));
}

TEST_F(SigNodeTest, ReTopoHash) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);

  // Fake some hash values.
  RefTopoHash(&sn1).emplace_back(123);
  RefTopoHash(&sn1).emplace_back(456);

  EXPECT_THAT(GetTopoHash(0, &sn1), Eq(123));
  EXPECT_THAT(GetTopoHash(1, &sn1), Eq(456));

  ReHighTopoHash(&sn1);

  size_t expected_hash = 456;
  CombineHash(1, &expected_hash);

  EXPECT_THAT(GetTopoHash(0, &sn1), Eq(123));
  EXPECT_THAT(GetTopoHash(1, &sn1), Eq(expected_hash));
}

TEST_F(SigNodeTest, ComputeTopoHash0) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);

  // Fake a topology.
  RefUniqueRank(&sn1) = 10;
  RefNodeMask(&sn1) = 0x02;

  RefTopoHash(&sn1).emplace_back(123);
  RefTopoHash(&sn1).emplace_back(456);

  // Fake a state.
  RefLastHashedNodes(&sn1) = 0xFF;
  RefNextHashedNodes(&sn1) = 0xFF;

  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(1, nullptr));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(1, nullptr));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(2, nullptr));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(3, nullptr));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(3, nullptr));

  // Run the test.
  ComputeTopoHash0(&sn1);

  EXPECT_THAT(RefLastHashedNodes(&sn1), Eq(0x02));
  EXPECT_THAT(RefNextHashedNodes(&sn1), Eq(0x02));
  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(1));

  size_t exp_hval = std::hash<string>()(sn1.opcode());
  CombineHash(1, &exp_hval);
  CombineHash(1, &exp_hval);
  CombineHash(2, &exp_hval);
  CombineHash(3, &exp_hval);
  CombineHash(3, &exp_hval);

  EXPECT_THAT(GetTopoHash(0, &sn1), Eq(exp_hval));
}

TEST_F(SigNodeTest, ComputeTopoHashNotFinal) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  SigNode sn3(&node3);

  // Fake a topology.
  RefUniqueRank(&sn1) = 0;
  RefNodeMask(&sn1) = 0x01;
  RefUniqueRank(&sn2) = 0;
  RefNodeMask(&sn2) = 0x02;
  RefUniqueRank(&sn3) = 0;
  RefNodeMask(&sn3) = 0x04;

  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(10, &sn2));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(10, &sn3));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(20, &sn2));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(30, &sn3));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(30, &sn2));

  // Fake a state.
  RefTopoHash(&sn1).emplace_back(123);
  RefTopoHash(&sn1).emplace_back(321);

  RefTopoHash(&sn2).emplace_back(456);
  RefTopoHash(&sn2).emplace_back(654);

  RefTopoHash(&sn3).emplace_back(789);
  RefTopoHash(&sn3).emplace_back(987);

  // These values are not realistic in the way that they don't include the bits
  // from the mask of nodes themselves, but that's the point of this test: only
  // the previous nodes' node sets are used in the computation, not their own
  // masks directly.
  RefLastHashedNodes(&sn1) = 0x8;
  RefLastHashedNodes(&sn2) = 0x10;
  RefLastHashedNodes(&sn3) = 0x20;

  // A scratch value to get overwritten.
  RefNextHashedNodes(&sn1) = 0x100;

  ComputeTopoHash(2, &sn1);

  EXPECT_THAT(RefLastHashedNodes(&sn1), Eq(0x8));  // Unchanged.
  EXPECT_THAT(RefNextHashedNodes(&sn1), Eq(0x38));

  // This computes the hash form the explicit numbers above.
  size_t exp_hash = 123;  // The 0th hash is the starting point.
  size_t comm_hash;

  comm_hash = 0;
  CombineHashCommutative(654, &comm_hash);
  CombineHashCommutative(987, &comm_hash);

  CombineHash(10, &exp_hash);
  CombineHash(comm_hash, &exp_hash);

  comm_hash = 0;
  CombineHashCommutative(654, &comm_hash);

  CombineHash(20, &exp_hash);
  CombineHash(comm_hash, &exp_hash);

  comm_hash = 0;
  CombineHashCommutative(654, &comm_hash);
  CombineHashCommutative(987, &comm_hash);

  CombineHash(30, &exp_hash);
  CombineHash(comm_hash, &exp_hash);

  EXPECT_THAT(GetTopoHash(2, &sn1), Eq(exp_hash));
  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(3));
}

TEST_F(SigNodeTest, ComputeTopoHashFinal) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  SigNode sn3(&node3);

  // Fake a topology - same as for ComputeTopoHashNotFinal.
  RefUniqueRank(&sn1) = 0;
  RefNodeMask(&sn1) = 0x01;
  RefUniqueRank(&sn2) = 0;
  RefNodeMask(&sn2) = 0x02;
  RefUniqueRank(&sn3) = 0;
  RefNodeMask(&sn3) = 0x04;

  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(10, &sn2));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(10, &sn3));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(20, &sn2));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(30, &sn3));
  RefHashedPeers(&sn1).emplace_back(SigNode::HashedPeer(30, &sn2));

  // Fake a state - mostly same as for ComputeTopoHashNotFinal.
  RefTopoHash(&sn1).emplace_back(123);
  RefTopoHash(&sn1).emplace_back(321);

  RefTopoHash(&sn2).emplace_back(456);
  RefTopoHash(&sn2).emplace_back(654);

  RefTopoHash(&sn3).emplace_back(789);
  RefTopoHash(&sn3).emplace_back(987);

  // These values are not realistic in the way that they don't include the bits
  // from the mask of nodes themselves, but that's the point of this test: only
  // the previous nodes' node sets are used in the computation, not their own
  // masks directly.
  RefLastHashedNodes(&sn1) = 0x8;
  RefLastHashedNodes(&sn2) = 0x10;
  RefLastHashedNodes(&sn3) = 0x20;

  // A scratch value to get overwritten.
  RefNextHashedNodes(&sn1) = 0x100;

  // This is the difference in configuration.
  RefHashIsFinal(&sn1) = true;

  ComputeTopoHash(2, &sn1);

  EXPECT_THAT(RefLastHashedNodes(&sn1), Eq(0x8));  // Unchanged.
  EXPECT_THAT(RefNextHashedNodes(&sn1), Eq(0x8));
  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(2));
  EXPECT_THAT(GetTopoHash(2, &sn1), Eq(321));
}

TEST_F(SigNodeTest, EqualsOpcode) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);

  EXPECT_TRUE(sn1 == sn2);
  EXPECT_FALSE(sn1 != sn2);

  node2.set_op("Mul");

  EXPECT_TRUE(sn1 != sn2);
  EXPECT_FALSE(sn1 == sn2);
}

TEST_F(SigNodeTest, EqualsRank) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);

  EXPECT_TRUE(sn1 == sn2);
  EXPECT_FALSE(sn1 != sn2);

  RefUniqueRank(&sn1) = 1;
  RefUniqueRank(&sn2) = 2;

  EXPECT_TRUE(sn1 != sn2);
  EXPECT_FALSE(sn1 == sn2);
}

// Checks that if the nodes have a different number of links,
// they will be considered unequal.
TEST_F(SigNodeTest, EqualsLinkSize) {
  GraphDef graph1;
  (*graph1.add_node()) = MakeNodeConst("node1");
  (*graph1.add_node()) = MakeNodeMul("node2", "node1", "node1");

  GenNodeMap gen_map1;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph1, &gen_map1), Eq(Status::OK()));

  Subgraph::Identity id1;
  id1.insert(gen_map1["node1"].get());
  id1.insert(gen_map1["node2"].get());
  Subgraph sg1(id1);

  SigNodeMap sig_map1;
  sg1.ExtractForSignature(&sig_map1);

  GraphDef graph2;
  (*graph2.add_node()) = MakeNodeConst("node1");
  // The difference between graph1 and graph2: one more input.
  auto node22 = graph2.add_node();
  *node22 = MakeNodeMul("node2", "node1", "node1");
  node22->add_input("node2");

  GenNodeMap gen_map2;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph2, &gen_map2), Eq(Status::OK()));

  Subgraph::Identity id2;
  id2.insert(gen_map2["node1"].get());
  id2.insert(gen_map2["node2"].get());
  Subgraph sg2(id2);

  SigNodeMap sig_map2;
  sg2.ExtractForSignature(&sig_map2);

  EXPECT_TRUE(*sig_map1["node1"] == *sig_map2["node1"]);
  EXPECT_FALSE(*sig_map1["node2"] == *sig_map2["node2"]);
  EXPECT_FALSE(*sig_map2["node2"] == *sig_map1["node2"]);
}

TEST_F(SigNodeTest, EqualsLinks) {
  // Start with 2 copies of the same graph.
  GraphDef graph1;
  (*graph1.add_node()) = MakeNodeConst("node1");
  (*graph1.add_node()) = MakeNodeMul("node2", "node1", "node1");

  GenNodeMap gen_map1;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph1, &gen_map1), Eq(Status::OK()));

  Subgraph::Identity id1;
  id1.insert(gen_map1["node1"].get());
  id1.insert(gen_map1["node2"].get());
  Subgraph sg1(id1);

  SigNodeMap sig_map1;
  sg1.ExtractForSignature(&sig_map1);

  GenNodeMap gen_map2;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph1, &gen_map2), Eq(Status::OK()));

  Subgraph::Identity id2;
  id2.insert(gen_map2["node1"].get());
  id2.insert(gen_map2["node2"].get());
  Subgraph sg2(id2);

  SigNodeMap sig_map2;
  sg2.ExtractForSignature(&sig_map2);

  EXPECT_TRUE(*sig_map1["node1"] == *sig_map2["node1"]);
  EXPECT_TRUE(*sig_map1["node2"] == *sig_map2["node2"]);

  // Alter the link hash of one of the nodes.
  SigNode* sn2 = sig_map2["node2"].get();
  ++RefHashedPeers(sn2)[0].link_hash;

  EXPECT_FALSE(*sig_map1["node2"] == *sig_map2["node2"]);

  // Restore back.
  --RefHashedPeers(sn2)[0].link_hash;
  EXPECT_TRUE(*sig_map1["node2"] == *sig_map2["node2"]);

  // Alter the unique rank of a referenced node.
  ++RefUniqueRank(sig_map2["node1"].get());

  EXPECT_FALSE(*sig_map1["node2"] == *sig_map2["node2"]);
}

//===

class SignatureTest : public SigBaseTest {
 protected:
  // Initializeds the state used to generate the permutations of a given size.
  static void InitPermutation(size_t size,
                              std::vector<size_t>* plain_permutation,
                              std::vector<size_t>* countdown) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_18(mht_18_v, 792, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "InitPermutation");

    plain_permutation->clear();
    countdown->clear();
    for (size_t i = 0; i < size; ++i) {
      plain_permutation->emplace_back(i);
      countdown->emplace_back(size - 1 - i);
    }
  }

  // Builds a permutation guided by the count-down value.
  static void BuildPermutation(const std::vector<size_t>& plain_permutation,
                               const std::vector<size_t>& countdown,
                               std::vector<size_t>* result) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_19(mht_19_v, 807, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "BuildPermutation");

    *result = plain_permutation;
    for (int i = 0; i < result->size(); ++i) {
      std::swap((*result)[i], (*result)[i + countdown[i]]);
    }
  }

  // Returns false when the count-down is finished.
  static bool CountDown(std::vector<size_t>* countdown) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_20(mht_20_v, 818, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "CountDown");

    // The last position always contains 0, so skip it.
    int pos;
    for (pos = countdown->size() - 2; pos >= 0; --pos) {
      if ((*countdown)[pos] > 0) {
        --(*countdown)[pos];
        break;
      }
      (*countdown)[pos] = (countdown->size() - 1 - pos);
    }

    return pos >= 0;
  }

  // Permutes the nodes every which way and checks that all the signatures
  // produced are the same. This is reasonable for the graphs up to the
  // size 5, maybe 6 at the stretch. After that the number of permutation grows
  // huge and the test becomes very slow.
  void TestGraphEveryWay(const GraphDef& graph) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSsig_node_testDTcc mht_21(mht_21_v, 839, "", "./tensorflow/core/grappler/graph_analyzer/sig_node_test.cc", "TestGraphEveryWay");

    size_t graph_size = graph.node_size();

    gen_map_.clear();
    sig_.map.clear();
    Status result = GenNode::BuildGraphInMap(graph, &gen_map_);
    ASSERT_THAT(result, Eq(Status::OK()));
    Subgraph::Identity id;
    for (const auto& entry : gen_map_) {
      id.insert(entry.second.get());
    }
    Subgraph sg(id);
    sg.ExtractForSignature(&sig_.map);

    std::vector<size_t> plain_permutation;
    std::vector<size_t> countdown;
    InitPermutation(graph_size, &plain_permutation, &countdown);

    std::set<string> signatures;
    std::vector<size_t> permutation;
    do {
      BuildPermutation(plain_permutation, countdown, &permutation);

      constexpr bool kDebugPermutation = false;
      if (kDebugPermutation) {
        string p;
        for (int i = 0; i < permutation.size(); ++i) {
          p.push_back('0' + permutation[i]);
        }
        LOG(INFO) << "Permutation: " << p;
      }

      std::vector<std::unique_ptr<SigNode>> hold(graph_size);
      int idx;

      // Permute the nodes.
      sig_.nodes.clear();
      idx = 0;
      if (kDebugPermutation) {
        LOG(INFO) << "    nodes before permutation:";
      }
      for (auto& entry : sig_.map) {
        if (kDebugPermutation) {
          LOG(INFO) << "        " << entry.second.get();
        }
        hold[idx++] = std::move(entry.second);
      }
      idx = 0;
      if (kDebugPermutation) {
        LOG(INFO) << "    nodes after permutation:";
      }
      for (auto& entry : sig_.map) {
        entry.second = std::move(hold[permutation[idx++]]);
        if (kDebugPermutation) {
          LOG(INFO) << "        " << entry.second.get();
        }
        // This is used to order the links per permutation.
        sig_.nodes.emplace_back(entry.second.get());
        RefUniqueRank(entry.second.get()) = idx;
      }
      // Order the links with the same tags per permutation.
      OrderLinks(&sig_);

      // The test as such.
      ASSERT_THAT(sig_.Compute(), Eq(Status::OK()));

      signatures.insert(sig_.ToString());

      EXPECT_THAT(sig_.sig_full, SizeIs(graph_size));
      size_t hval = 0;
      for (size_t ih : sig_.sig_full) {
        // The space 1..graph_size is reserved.
        EXPECT_THAT(ih, Gt(graph_size));
        CombineHash(ih, &hval);
      }
      EXPECT_THAT(sig_.sig_short, Eq(hval));

      // Un-permute the nodes for the next iteration.
      idx = 0;
      for (auto& entry : sig_.map) {
        hold[permutation[idx++]] = std::move(entry.second);
      }
      idx = 0;
      if (kDebugPermutation) {
        LOG(INFO) << "    nodes after un-permutation:";
      }
      for (auto& entry : sig_.map) {
        entry.second = std::move(hold[idx++]);
        if (kDebugPermutation) {
          LOG(INFO) << "        " << entry.second.get();
        }
      }
    } while (CountDown(&countdown));

    for (const auto& s : signatures) {
      LOG(INFO) << "Signature: " << s;
    }

    // All the permutations should produce the same signature.
    EXPECT_THAT(signatures, SizeIs(1));
  }
};

TEST_F(SignatureTest, PrepareNodes) {
  NodeDef node1 = MakeNodeConst("node1");
  sig_.map["node1"] = absl::make_unique<SigNode>(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  sig_.map["node2"] = absl::make_unique<SigNode>(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  sig_.map["node3"] = absl::make_unique<SigNode>(&node3);

  PrepareNodes(&sig_);

  ASSERT_THAT(sig_.nodes, SizeIs(3));

  int idx = 0;
  for (const auto& entry : sig_.map) {
    EXPECT_THAT(RefNodeMask(entry.second.get()), Eq(1 << idx))
        << " at index " << idx;
    EXPECT_THAT(RefUniqueRank(entry.second.get()), Eq(static_cast<size_t>(~0)))
        << " at index " << idx;
    EXPECT_THAT(RefHashIsFinal(entry.second.get()), false)
        << " at index " << idx;
    EXPECT_THAT(RefTopoHash(entry.second.get()), SizeIs(1))
        << " at index " << idx;
    ++idx;
  }
}

TEST_F(SignatureTest, FindUniqueHashesAllDifferent) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  SigNode sn3(&node3);
  NodeDef node4 = MakeNodeConst("node4");
  SigNode sn4(&node4);

  // The last values in the arrays values go in the backwards order.
  RefTopoHash(&sn1).emplace_back(100);
  RefTopoHash(&sn1).emplace_back(900);

  RefTopoHash(&sn2).emplace_back(200);
  RefTopoHash(&sn2).emplace_back(800);

  RefTopoHash(&sn3).emplace_back(300);
  RefTopoHash(&sn3).emplace_back(700);

  RefTopoHash(&sn4).emplace_back(400);
  RefTopoHash(&sn4).emplace_back(600);

  sig_.nodes.emplace_back(&sn1);
  sig_.nodes.emplace_back(&sn2);
  sig_.nodes.emplace_back(&sn3);
  sig_.nodes.emplace_back(&sn4);

  size_t next = 1;  // Skips over sn1.

  FindUniqueHashes(&next, &sig_);
  EXPECT_THAT(next, Eq(4));

  EXPECT_THAT(sig_.nodes[0], Eq(&sn1));
  // The nodes after first one get sorted by the high hash.
  EXPECT_THAT(sig_.nodes[1], Eq(&sn4));
  EXPECT_THAT(sig_.nodes[2], Eq(&sn3));
  EXPECT_THAT(sig_.nodes[3], Eq(&sn2));

  EXPECT_THAT(RefHashIsFinal(&sn1), Eq(false));
  // Nodes that get finalized are marked as such.
  EXPECT_THAT(RefHashIsFinal(&sn2), Eq(true));
  EXPECT_THAT(RefHashIsFinal(&sn3), Eq(true));
  EXPECT_THAT(RefHashIsFinal(&sn4), Eq(true));

  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(2));
  ASSERT_THAT(RefTopoHash(&sn2), SizeIs(1));
  ASSERT_THAT(RefTopoHash(&sn3), SizeIs(1));
  ASSERT_THAT(RefTopoHash(&sn4), SizeIs(1));

  EXPECT_THAT(RefTopoHash(&sn2)[0], Eq(4));
  EXPECT_THAT(RefTopoHash(&sn3)[0], Eq(3));
  EXPECT_THAT(RefTopoHash(&sn4)[0], Eq(2));

  EXPECT_THAT(sig_.sig_full, ElementsAre(600, 700, 800));

  size_t exp_short_hash = 0;
  CombineHash(600, &exp_short_hash);
  CombineHash(700, &exp_short_hash);
  CombineHash(800, &exp_short_hash);
  EXPECT_THAT(sig_.sig_short, Eq(exp_short_hash));
}

TEST_F(SignatureTest, FindUniqueHashesDuplicatesExceptOne) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  SigNode sn3(&node3);
  NodeDef node4 = MakeNodeConst("node4");
  SigNode sn4(&node4);
  NodeDef node5 = MakeNodeConst("node5");
  SigNode sn5(&node5);

  RefTopoHash(&sn1).emplace_back(100);
  RefTopoHash(&sn1).emplace_back(600);

  RefTopoHash(&sn2).emplace_back(200);
  RefTopoHash(&sn2).emplace_back(600);

  RefTopoHash(&sn3).emplace_back(300);
  RefTopoHash(&sn3).emplace_back(700);

  RefTopoHash(&sn4).emplace_back(400);
  RefTopoHash(&sn4).emplace_back(800);

  RefTopoHash(&sn5).emplace_back(500);
  RefTopoHash(&sn5).emplace_back(800);

  sig_.nodes.emplace_back(&sn1);
  sig_.nodes.emplace_back(&sn2);
  sig_.nodes.emplace_back(&sn3);
  sig_.nodes.emplace_back(&sn4);
  sig_.nodes.emplace_back(&sn5);

  size_t next = 0;

  FindUniqueHashes(&next, &sig_);
  EXPECT_THAT(next, Eq(1));

  // The unique node goes first.
  EXPECT_THAT(sig_.nodes[0], Eq(&sn3));

  // The rest of the nodes are assumed to be sorted in a stable order.
  EXPECT_THAT(sig_.nodes[1], Eq(&sn2));
  // Node 1 gets swapped with node 3.
  EXPECT_THAT(sig_.nodes[2], Eq(&sn1));
  EXPECT_THAT(sig_.nodes[3], Eq(&sn4));
  EXPECT_THAT(sig_.nodes[4], Eq(&sn5));

  EXPECT_THAT(RefHashIsFinal(&sn1), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn2), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn3), Eq(true));
  EXPECT_THAT(RefHashIsFinal(&sn4), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn5), Eq(false));

  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn2), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn3), SizeIs(1));
  EXPECT_THAT(RefTopoHash(&sn4), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn5), SizeIs(2));

  EXPECT_THAT(RefTopoHash(&sn3)[0], Eq(1));
}

TEST_F(SignatureTest, FindUniqueHashesDuplicates) {
  NodeDef node1 = MakeNodeConst("node1");
  SigNode sn1(&node1);
  NodeDef node2 = MakeNodeConst("node2");
  SigNode sn2(&node2);
  NodeDef node3 = MakeNodeConst("node3");
  SigNode sn3(&node3);
  NodeDef node4 = MakeNodeConst("node4");
  SigNode sn4(&node4);
  NodeDef node5 = MakeNodeConst("node5");
  SigNode sn5(&node5);

  RefTopoHash(&sn1).emplace_back(100);
  RefTopoHash(&sn1).emplace_back(600);

  RefTopoHash(&sn2).emplace_back(200);
  RefTopoHash(&sn2).emplace_back(600);

  RefTopoHash(&sn3).emplace_back(300);
  RefTopoHash(&sn3).emplace_back(700);

  RefTopoHash(&sn4).emplace_back(400);
  RefTopoHash(&sn4).emplace_back(700);

  RefTopoHash(&sn5).emplace_back(500);
  RefTopoHash(&sn5).emplace_back(700);

  sig_.nodes.emplace_back(&sn1);
  sig_.nodes.emplace_back(&sn2);
  sig_.nodes.emplace_back(&sn3);
  sig_.nodes.emplace_back(&sn4);
  sig_.nodes.emplace_back(&sn5);

  size_t next = 0;

  FindUniqueHashes(&next, &sig_);
  EXPECT_THAT(next, Eq(1));

  // The last copy of the last duplicate wins.
  EXPECT_THAT(sig_.nodes[0], Eq(&sn5));

  // The rest of the nodes are assumed to be sorted in a stable order.
  // Node 1 gets swapped.
  EXPECT_THAT(sig_.nodes[1], Eq(&sn2));
  EXPECT_THAT(sig_.nodes[2], Eq(&sn3));
  EXPECT_THAT(sig_.nodes[3], Eq(&sn4));
  EXPECT_THAT(sig_.nodes[4], Eq(&sn1));

  EXPECT_THAT(RefHashIsFinal(&sn1), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn2), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn3), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn4), Eq(false));
  EXPECT_THAT(RefHashIsFinal(&sn5), Eq(true));

  EXPECT_THAT(RefTopoHash(&sn1), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn2), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn3), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn4), SizeIs(2));
  EXPECT_THAT(RefTopoHash(&sn5), SizeIs(1));

  EXPECT_THAT(RefTopoHash(&sn5)[0], Eq(1));
}

// On a circular topology.
TEST_F(SignatureTest, ComputeOneRoundCircular) {
  BuildSigMap(graph_circular_onedir_);
  PrepareNodes(&sig_);

  ASSERT_THAT(sig_.nodes, SizeIs(5));

  // This skips FindUniqueHashes() which would pick one node, so that
  // all the nodes are equivalent for ComputeOneRound().

  ComputeOneRound(0, &sig_);

  // All the nodes are the same, so the computed hashes will also be the same.
  size_t hval = GetHighTopoHash(sig_.nodes[0]);
  for (int i = 0; i < 5; ++i) {
    EXPECT_THAT(GetHighTopoHash(sig_.nodes[i]), Eq(hval)) << " at index " << i;
    EXPECT_THAT(RefHashIsFinal(sig_.nodes[i]), Eq(true)) << " at index " << i;
    EXPECT_THAT(RefLastHashedNodes(sig_.nodes[i]), Eq(0x1F))
        << " at index " << i;
    EXPECT_THAT(RefNextHashedNodes(sig_.nodes[i]), Eq(0x1F))
        << " at index " << i;
    // The sets of hashed nodes go like this:
    // Step 0: self.
    // Step 1: self, previous (-1) and next (+1) node.
    // Step 2: self, (-1), (-2), (+1), (+2): all 5 nodes in the graph
    // Step 3: still all 5 nodes in the graph
    EXPECT_THAT(RefTopoHash(sig_.nodes[i]), SizeIs(4)) << " at index " << i;
  }
}

// On a linear topology.
TEST_F(SignatureTest, ComputeOneRoundLinear) {
  BuildSigMap(graph_linear_);
  PrepareNodes(&sig_);

  ASSERT_THAT(sig_.nodes, SizeIs(5));

  // This skips FindUniqueHashes() which would pick one node, so that
  // all the nodes are equivalent for ComputeOneRound().

  ComputeOneRound(0, &sig_);

  std::vector<size_t> hash_size;
  for (int i = 0; i < 5; ++i) {
    EXPECT_THAT(RefHashIsFinal(sig_.nodes[i]), Eq(true)) << " at index " << i;
    EXPECT_THAT(RefLastHashedNodes(sig_.nodes[i]), Eq(0x1F))
        << " at index " << i;
    EXPECT_THAT(RefNextHashedNodes(sig_.nodes[i]), Eq(0x1F))
        << " at index " << i;
    hash_size.emplace_back(RefTopoHash(sig_.nodes[i]).size());
  }

  // The sets of hashed nodes for the central node go like this:
  // Step 0: self.
  // Step 1: self, previous (-1) and next (+1) node.
  // Step 2: self, (-1), (-2), (+1), (+2): all 5 nodes in the graph
  // Step 3: still all 5 nodes in the graph
  //
  // The nodes one step closer to the ends require one more step. The end nodes
  // require one more step yet.
  std::sort(hash_size.begin(), hash_size.end());
  EXPECT_THAT(hash_size, ElementsAre(4, 5, 5, 6, 6));
}

// On a linear topology where the central node has been already marked as unique
// (yeah, not a very realistic case but tests the situations when the
// disconnected subgraphs get created).
TEST_F(SignatureTest, ComputeOneRoundSplitLinear) {
  BuildSigMap(graph_linear_);
  PrepareNodes(&sig_);

  ASSERT_THAT(sig_.nodes, SizeIs(5));

  // This test relies on the order of SigNodeMap imposed on sig_.nodes.

  // The middle node gets separated by moving it to the front.
  std::swap(sig_.nodes[0], sig_.nodes[2]);
  ASSERT_THAT(RefNodeMask(sig_.nodes[0]), Eq(0x04));
  ASSERT_THAT(RefLastHashedNodes(sig_.nodes[0]), Eq(0x04));
  ASSERT_THAT(RefNextHashedNodes(sig_.nodes[0]), Eq(0x04));
  RefHashIsFinal(sig_.nodes[0]) = true;

  ComputeOneRound(1, &sig_);

  // These should stay unchanged.
  EXPECT_THAT(RefLastHashedNodes(sig_.nodes[0]), Eq(0x04));
  EXPECT_THAT(RefNextHashedNodes(sig_.nodes[0]), Eq(0x04));

  std::vector<size_t> hash_size;
  for (int i = 1; i < 5; ++i) {
    EXPECT_THAT(RefHashIsFinal(sig_.nodes[i]), Eq(true)) << " at index " << i;
    hash_size.emplace_back(RefTopoHash(sig_.nodes[i]).size());
  }

  std::sort(hash_size.begin(), hash_size.end());
  // The end nodes take 4 steps, closer to the center 3 steps.
  EXPECT_THAT(hash_size, ElementsAre(3, 3, 4, 4));

  EXPECT_THAT(RefLastHashedNodes(sig_.nodes[1]), Eq(0x07));
  EXPECT_THAT(RefNextHashedNodes(sig_.nodes[1]), Eq(0x07));
  EXPECT_THAT(RefLastHashedNodes(sig_.nodes[2]), Eq(0x07));
  EXPECT_THAT(RefNextHashedNodes(sig_.nodes[2]), Eq(0x07));

  EXPECT_THAT(RefLastHashedNodes(sig_.nodes[3]), Eq(0x1C));
  EXPECT_THAT(RefNextHashedNodes(sig_.nodes[3]), Eq(0x1C));
  EXPECT_THAT(RefLastHashedNodes(sig_.nodes[4]), Eq(0x1C));
  EXPECT_THAT(RefNextHashedNodes(sig_.nodes[4]), Eq(0x1C));
}

TEST_F(SignatureTest, OrderLinks) {
  gen_map_.clear();
  sig_.map.clear();
  Status result = GenNode::BuildGraphInMap(graph_for_link_order_, &gen_map_);
  ASSERT_THAT(result, Eq(Status::OK()));
  Subgraph::Identity id;
  for (const auto& entry : gen_map_) {
    id.insert(entry.second.get());
  }
  Subgraph sg(id);
  sg.ExtractForSignature(&sig_.map);

  // Populate the fake signature and assign the ranks in the backwards order.
  for (auto it = sig_.map.rbegin(); it != sig_.map.rend(); ++it) {
    auto& entry = *it;
    RefUniqueRank(entry.second.get()) = sig_.nodes.size();
    sig_.nodes.emplace_back(entry.second.get());
  }

  // How it was ordered in the original graph.
  string before = sig_.ToString();
  // clang-format off
  EXPECT_THAT(before, Eq(
    "0:Mul[i0:o0:5][i0:o0:4][i0:o1:4][i0:o2:3][i0:o2:2][i0:o3:2],"
    "1:Mul[i0:o0:5][i0:o0:4][i0:o0:3][i0:o0:2],"
    "2:Const,"
    "3:Const,"
    "4:Const,"
    "5:Const,"
    ));
  // clang-format on

  OrderLinks(&sig_);

  string after = sig_.ToString();
  // clang-format off
  EXPECT_THAT(after, Eq(
      "0:Mul[i0:o0:4][i0:o0:5][i0:o1:4][i0:o2:2][i0:o2:3][i0:o3:2],"
      "1:Mul[i0:o0:2][i0:o0:3][i0:o0:4][i0:o0:5],"
      "2:Const,"
      "3:Const,"
      "4:Const,"
      "5:Const,"
      ));
  // clang-format on
}

TEST_F(SignatureTest, GraphTooBig) {
  GraphDef graph;
  for (int i = 0; i <= Signature::kMaxGraphSize; ++i) {
    (*graph.add_node()) = MakeNodeConst(absl::StrFormat("node%d", i));
  }

  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &gen_map_), Eq(Status::OK()));

  Subgraph::Identity id;
  for (const auto& entry : gen_map_) {
    id.insert(entry.second.get());
  }
  Subgraph sg(id);
  sg.ExtractForSignature(&sig_.map);

  ASSERT_THAT(sig_.Compute(),
              Eq(Status(error::INVALID_ARGUMENT,
                        "A graph of 65 nodes is too big for signature "
                        "computation, the maximal supported node count is "
                        "64.")));
}

TEST_F(SignatureTest, ToString) {
  BuildSigMap(graph_circular_onedir_);
  PrepareNodes(&sig_);

  ASSERT_THAT(sig_.nodes, SizeIs(5));

  // Fake the works by assigning unique ranks as they go in the initial order.
  for (int i = 0; i < 5; ++i) {
    RefUniqueRank(sig_.nodes[i]) = i;
    RefHashIsFinal(sig_.nodes[i]) = true;
  }

  string result = sig_.ToString();

  // clang-format off
  ASSERT_THAT(result, Eq(
      "0:Mul[i0:o0:4][i0:o0:4],"
      "1:Mul[i0:o0:0][i0:o0:0],"
      "2:Mul[i0:o0:1][i0:o0:1],"
      "3:Mul[i0:o0:2][i0:o0:2],"
      "4:Mul[i0:o0:3][i0:o0:3],"
      ));
  // clang-format on
}

// This is a test of the permutation logic itself.
TEST_F(SignatureTest, Permutation) {
  std::vector<size_t> plain_permutation;
  std::vector<size_t> countdown;
  InitPermutation(5, &plain_permutation, &countdown);

  std::set<string> results;

  std::vector<size_t> permutation;
  do {
    BuildPermutation(plain_permutation, countdown, &permutation);
    EXPECT_THAT(permutation, SizeIs(5));

    string p;
    for (int i = 0; i < permutation.size(); ++i) {
      p.push_back('0' + permutation[i]);
    }
    LOG(INFO) << "Permutation: " << p;
    results.insert(p);
  } while (CountDown(&countdown));

  EXPECT_THAT(results, SizeIs(5 * 4 * 3 * 2 * 1));
}

TEST_F(SignatureTest, ComputeCircularOneDir) {
  TestGraphEveryWay(graph_circular_onedir_);
}

TEST_F(SignatureTest, ComputeCircularBiDir) {
  TestGraphEveryWay(graph_circular_bidir_);
}

TEST_F(SignatureTest, ComputeLinear) { TestGraphEveryWay(graph_linear_); }

TEST_F(SignatureTest, ComputeMultiInput) {
  TestGraphEveryWay(graph_multi_input_);
}

TEST_F(SignatureTest, ComputeAllOrNone) {
  TestGraphEveryWay(graph_all_or_none_);
}

TEST_F(SignatureTest, ComputeCross) { TestGraphEveryWay(graph_small_cross_); }

TEST_F(SignatureTest, Equals) {
  // Start with 2 copies of the same graph.
  GenNodeMap gen_map1;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph_circular_bidir_, &gen_map1),
              Eq(Status::OK()));

  Subgraph::Identity id1;
  id1.insert(gen_map1["node1"].get());
  id1.insert(gen_map1["node2"].get());
  Subgraph sg1(id1);

  Signature sig1;
  sg1.ExtractForSignature(&sig1.map);
  ASSERT_THAT(sig1.Compute(), Eq(Status::OK()));

  GenNodeMap gen_map2;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph_circular_bidir_, &gen_map2),
              Eq(Status::OK()));

  Subgraph::Identity id2;
  id2.insert(gen_map2["node1"].get());
  id2.insert(gen_map2["node2"].get());
  Subgraph sg2(id2);

  Signature sig2;
  sg2.ExtractForSignature(&sig2.map);
  ASSERT_THAT(sig2.Compute(), Eq(Status::OK()));

  EXPECT_TRUE(sig1 == sig2);

  // Change the short hash.
  ++sig2.sig_short;
  EXPECT_FALSE(sig1 == sig2);

  // Restore back.
  --sig2.sig_short;
  EXPECT_TRUE(sig1 == sig2);

  // Change the full hash.
  ++sig2.sig_full[0];
  EXPECT_FALSE(sig1 == sig2);

  // Restore back.
  --sig2.sig_full[0];
  EXPECT_TRUE(sig1 == sig2);

  // Make the nodes different.
  std::swap(sig2.nodes[0], sig2.nodes[1]);
  EXPECT_FALSE(sig1 == sig2);

  // Restore back.
  std::swap(sig2.nodes[0], sig2.nodes[1]);
  EXPECT_TRUE(sig1 == sig2);

  // Different number of nodes.
  sig2.nodes.emplace_back(sig2.nodes[0]);
  EXPECT_FALSE(sig1 == sig2);
  EXPECT_FALSE(sig2 == sig1);
}

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
