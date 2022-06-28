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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc() {
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

#include "mlir-hlo/utils/cycle_detector.h"

#include <algorithm>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

namespace {

using NodeSet = llvm::DenseSet<int32_t>;
using OrderedNodeSet = OrderedSet<int32_t>;

template <typename T>
struct VecStruct {
  using type = llvm::SmallVector<T, 4>;
};
template <typename T>
using Vec = typename VecStruct<T>::type;

struct Node {
  // rank number assigned by Pearce-Kelly algorithm
  int32_t rank;
  // Temporary marker used by depth-first-search
  bool visited;
  // User-supplied data
  void* data;
  // List of immediate predecessor nodes in graph
  OrderedNodeSet in;
  // List of immediate successor nodes in graph
  OrderedNodeSet out;
};

}  // namespace

struct GraphCycles::Rep {
  Vec<Node*> nodes;
  // Indices for unused entries in nodes
  Vec<int32_t> free_nodes;

  // Temporary state.
  // Results of forward DFS
  Vec<int32_t> deltaf;
  // Results of backward DFS
  Vec<int32_t> deltab;
  // All nodes to reprocess
  Vec<int32_t> list;
  // Rank values to assign to list entries
  Vec<int32_t> merged;
  // Emulates recursion stack when doing depth first search
  Vec<int32_t> stack;
};

GraphCycles::GraphCycles(int32_t num_nodes) : rep_(new Rep) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_0(mht_0_v, 239, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::GraphCycles");

  rep_->nodes.reserve(num_nodes);
  for (int32_t i = 0; i < num_nodes; ++i) {
    Node* n = new Node;
    n->visited = false;
    n->data = nullptr;
    n->rank = rep_->nodes.size();
    rep_->nodes.push_back(n);
  }
}

GraphCycles::~GraphCycles() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_1(mht_1_v, 253, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::~GraphCycles");

  for (Vec<Node*>::size_type i = 0, e = rep_->nodes.size(); i < e; ++i) {
    delete rep_->nodes[i];
  }
  delete rep_;
}

bool GraphCycles::HasEdge(int32_t x, int32_t y) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::HasEdge");

  return rep_->nodes[x]->out.Contains(y);
}

void GraphCycles::RemoveEdge(int32_t x, int32_t y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::RemoveEdge");

  rep_->nodes[x]->out.Erase(y);
  rep_->nodes[y]->in.Erase(x);
  // No need to update the rank assignment since a previous valid
  // rank assignment remains valid after an edge deletion.
}

static bool ForwardDFS(GraphCycles::Rep* r, int32_t n, int32_t upper_bound);
static void BackwardDFS(GraphCycles::Rep* r, int32_t n, int32_t lower_bound);
static void Reorder(GraphCycles::Rep* r);
static void Sort(const Vec<Node*>&, Vec<int32_t>* delta);
static void MoveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst);
static void ClearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes);

bool GraphCycles::InsertEdge(int32_t x, int32_t y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_4(mht_4_v, 288, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::InsertEdge");

  if (x == y) return false;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  if (!nx->out.Insert(y)) {
    // Edge already exists.
    return true;
  }

  Node* ny = r->nodes[y];
  ny->in.Insert(x);

  if (nx->rank <= ny->rank) {
    // New edge is consistent with existing rank assignment.
    return true;
  }

  // Current rank assignments are incompatible with the new edge.  Recompute.
  // We only need to consider nodes that fall in the range [ny->rank,nx->rank].
  if (ForwardDFS(r, y, nx->rank)) {
    // Found a cycle.  Undo the insertion and tell caller.
    nx->out.Erase(y);
    ny->in.Erase(x);
    // Since we do not call Reorder() on this path, clear any visited
    // markers left by ForwardDFS.
    ClearVisitedBits(r, r->deltaf);
    return false;
  }
  BackwardDFS(r, x, ny->rank);
  Reorder(r);
  return true;
}

// Follows the edges from producer to consumer and searchs if the node having
// rank `n` can reach the node having rank `upper_bound` using a DFS search.
// When doing DFS search, We only consider the pathes that satisfy the ranks
// of the nodes of the path are all smaller than `upper_bound`.
//
// Returns true if such path exists.
static bool ForwardDFS(GraphCycles::Rep* r, int32_t n, int32_t upper_bound) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "ForwardDFS");

  // Avoid recursion since stack space might be limited.
  // We instead keep a stack of nodes to visit.
  r->deltaf.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltaf.push_back(n);

    for (auto w : nn->out.GetSequence()) {
      Node* nw = r->nodes[w];
      if (nw->rank == upper_bound) {
        return true;
      }
      if (!nw->visited && nw->rank < upper_bound) {
        r->stack.push_back(w);
      }
    }
  }
  return false;
}

// Follows the edges from consumer to producer and visit all the nodes that
// is reachable from node `n` and have rank larger than `lower_bound`.
static void BackwardDFS(GraphCycles::Rep* r, int32_t n, int32_t lower_bound) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_6(mht_6_v, 363, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "BackwardDFS");

  r->deltab.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltab.push_back(n);

    for (auto w : nn->in.GetSequence()) {
      Node* nw = r->nodes[w];
      if (!nw->visited && lower_bound < nw->rank) {
        r->stack.push_back(w);
      }
    }
  }
}

// Recomputes rank assignments to make them compatible with the edges (producer
// has smaller rank than its consumer)
static void Reorder(GraphCycles::Rep* r) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_7(mht_7_v, 390, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "Reorder");

  Sort(r->nodes, &r->deltab);
  Sort(r->nodes, &r->deltaf);

  // Adds contents of delta lists to list (backwards deltas first).
  r->list.clear();
  MoveToList(r, &r->deltab, &r->list);
  MoveToList(r, &r->deltaf, &r->list);

  // Produce sorted list of all ranks that will be reassigned.
  r->merged.resize(r->deltab.size() + r->deltaf.size());
  std::merge(r->deltab.begin(), r->deltab.end(), r->deltaf.begin(),
             r->deltaf.end(), r->merged.begin());

  // Assign the ranks in order to the collected list.
  for (Vec<int32_t>::size_type i = 0, e = r->list.size(); i < e; ++i) {
    r->nodes[r->list[i]]->rank = r->merged[i];
  }
}

// Sorts nodes in the vector according to their ranks. Small rank first.
static void Sort(const Vec<Node*>& nodes, Vec<int32_t>* delta) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_8(mht_8_v, 414, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "Sort");

  struct ByRank {
    const Vec<Node*>* nodes;
    bool operator()(int32_t a, int32_t b) const {
      return (*nodes)[a]->rank < (*nodes)[b]->rank;
    }
  };
  ByRank cmp;
  cmp.nodes = &nodes;
  std::sort(delta->begin(), delta->end(), cmp);
}

// Collects ranks of nodes in vector `src` to vector `dst`
static void MoveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_9(mht_9_v, 431, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "MoveToList");

  for (Vec<int32_t>::size_type i = 0, e = src->size(); i < e; i++) {
    int32_t w = (*src)[i];
    // Replace src entry with its rank
    (*src)[i] = r->nodes[w]->rank;
    // Prepare for future DFS calls
    r->nodes[w]->visited = false;
    dst->push_back(w);
  }
}

// Clears bookkeeping fileds used during the last DFS process.
static void ClearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_10(mht_10_v, 446, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "ClearVisitedBits");

  for (Vec<int32_t>::size_type i = 0, e = nodes.size(); i < e; i++) {
    r->nodes[nodes[i]]->visited = false;
  }
}

bool GraphCycles::IsReachable(int32_t x, int32_t y) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_11(mht_11_v, 455, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::IsReachable");

  if (x == y) return true;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  Node* ny = r->nodes[y];

  if (nx->rank >= ny->rank) {
    // x cannot reach y since it is after it in the topological ordering
    return false;
  }

  // See if x can reach y using a DFS search that is limited to y's rank
  bool reachable = ForwardDFS(r, x, ny->rank);

  // Clear any visited markers left by ForwardDFS.
  ClearVisitedBits(r, r->deltaf);
  return reachable;
}

llvm::Optional<int32_t> GraphCycles::ContractEdge(int32_t a, int32_t b) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_12(mht_12_v, 477, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::ContractEdge");

  assert(HasEdge(a, b));
  RemoveEdge(a, b);

  if (IsReachable(a, b)) {
    // Restore the graph to its original state.
    InsertEdge(a, b);
    return {};
  }

  if (rep_->nodes[b]->in.Size() + rep_->nodes[b]->out.Size() >
      rep_->nodes[a]->in.Size() + rep_->nodes[a]->out.Size()) {
    // Swap "a" and "b" to minimize copying.
    std::swap(a, b);
  }

  Node* nb = rep_->nodes[b];
  OrderedNodeSet out = std::move(nb->out);
  OrderedNodeSet in = std::move(nb->in);
  for (int32_t y : out.GetSequence()) {
    rep_->nodes[y]->in.Erase(b);
  }
  for (int32_t y : in.GetSequence()) {
    rep_->nodes[y]->out.Erase(b);
  }
  rep_->free_nodes.push_back(b);

  rep_->nodes[a]->out.Reserve(rep_->nodes[a]->out.Size() + out.Size());
  for (int32_t y : out.GetSequence()) {
    InsertEdge(a, y);
  }

  rep_->nodes[a]->in.Reserve(rep_->nodes[a]->in.Size() + in.Size());
  for (int32_t y : in.GetSequence()) {
    InsertEdge(y, a);
  }

  // Note, if the swap happened it might be what originally was called "b".
  return a;
}

std::vector<int32_t> GraphCycles::SuccessorsCopy(int32_t node) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_13(mht_13_v, 521, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::SuccessorsCopy");

  return rep_->nodes[node]->out.GetSequence();
}

namespace {
void SortInPostOrder(const Vec<Node*>& nodes, std::vector<int32_t>* to_sort) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_14(mht_14_v, 529, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "SortInPostOrder");

  std::sort(to_sort->begin(), to_sort->end(), [&](int32_t a, int32_t b) {
    return nodes[a]->rank > nodes[b]->rank;
  });
}
}  // namespace

std::vector<int32_t> GraphCycles::AllNodesInPostOrder() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScycle_detectorDTcc mht_15(mht_15_v, 539, "", "./tensorflow/compiler/mlir/hlo/lib/utils/cycle_detector.cc", "GraphCycles::AllNodesInPostOrder");

  llvm::DenseSet<int32_t> free_nodes_set;
  for (int32_t n : rep_->free_nodes) free_nodes_set.insert(n);

  std::vector<int32_t> all_nodes;
  all_nodes.reserve(rep_->nodes.size() - free_nodes_set.size());
  for (size_t i = 0, e = rep_->nodes.size(); i < e; i++) {
    if (!free_nodes_set.count(i)) {
      all_nodes.push_back(i);
    }
  }

  SortInPostOrder(rep_->nodes, &all_nodes);
  return all_nodes;
}

}  // namespace mlir
