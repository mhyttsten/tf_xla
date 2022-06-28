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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/memory_types.h"

#include <utility>

#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

struct Endpoint {
  int node_id;
  int output_index;
};

struct EndpointHash {
  uint32 operator()(const Endpoint& x) const {
    return Hash32(reinterpret_cast<const char*>(&x.node_id), sizeof(int),
                  x.output_index);
  }
};

struct EndpointEq {
  uint32 operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node_id == y.node_id) && (x.output_index == y.output_index);
  }
};

static Status ProcessMemoryTypes(
    const DeviceType& device_type, const Graph* g,
    const std::function<Status(const Edge*, MemoryType, MemoryType)>& fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/common_runtime/memory_types.cc", "ProcessMemoryTypes");

  if (device_type != DEVICE_GPU &&
      !DeviceFactory::IsPluggableDevice(device_type.type_string())) {
    // On non-GPU devices, HOST_MEMORY and DEVICE_MEMORY are always compatible.
    return Status::OK();
  }
  // For GPU, HOST_MEMORY and DEVICE_MEMORY is not compatible. I.e., a
  // conversion/transfer must be done.
  //
  // {node id, slot id} -> memory type.
  typedef std::unordered_map<Endpoint, MemoryType, EndpointHash, EndpointEq>
      MemTypeMap;
  MemTypeMap inp;
  MemTypeMap out;
  MemoryTypeVector inp_mvec;
  MemoryTypeVector out_mvec;
  for (const Node* n : g->nodes()) {
    TF_RETURN_IF_ERROR(MemoryTypesForNode(g->op_registry(), device_type,
                                          n->def(), &inp_mvec, &out_mvec));
    for (size_t i = 0; i < inp_mvec.size(); ++i) {
      VLOG(2) << "inp mvec " << n->id() << " " << i << " " << inp_mvec[i];
      inp[{n->id(), static_cast<int>(i)}] = inp_mvec[i];
    }
    for (size_t i = 0; i < out_mvec.size(); ++i) {
      VLOG(2) << "out mvec " << n->id() << " " << i << " " << out_mvec[i];
      out[{n->id(), static_cast<int>(i)}] = out_mvec[i];
    }
  }
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    MemoryType sm = gtl::FindWithDefault(out, {e->src()->id(), e->src_output()},
                                         DEVICE_MEMORY);
    MemoryType dm = gtl::FindWithDefault(inp, {e->dst()->id(), e->dst_input()},
                                         DEVICE_MEMORY);
    VLOG(1) << e->src()->id() << ":" << e->src_output() << " -> "
            << e->dst()->id() << ":" << e->dst_input() << ": " << sm << " -> "
            << dm;
    TF_RETURN_IF_ERROR(fn(e, sm, dm));
  }
  return Status::OK();
}

Status ValidateMemoryTypes(const DeviceType& device_type, const Graph* g) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_1(mht_1_v, 267, "", "./tensorflow/core/common_runtime/memory_types.cc", "ValidateMemoryTypes");

  return ProcessMemoryTypes(
      device_type, g, [](const Edge* e, MemoryType sm, MemoryType dm) {
        if (sm == dm) {
          return Status::OK();
        }
        return errors::Internal("Memory type mismatch (", sm, " ", dm,
                                ") between :", e->src()->id(), ":",
                                e->src_output(), " and ", e->dst()->id(), ":",
                                e->dst_input(), " : from ",
                                FormatNodeForError(*e->src()), " to ",
                                FormatNodeForError(*e->dst()));
      });
}

// Given an Edge whose two endpoints have different memory types and
// are gonna to insert a pair of HostSend/Recv or Send/HostRecv nodes,
// GetTensorName() returns a unique string that we can use as part of
// the rendezvous key. The return string is guaranteed to be unique
// within this process. That is sufficient because EnsureMemoryTypes
// is only used on a TensorFlow graph that is gonna to be executed in
// a single tf device (hence within a single process).
static string GetTensorName(const Edge* edge) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_2(mht_2_v, 292, "", "./tensorflow/core/common_runtime/memory_types.cc", "GetTensorName");

  static std::atomic<int64_t> counter(0);
  return strings::StrCat("memtype_", counter.fetch_add(1), "_",
                         edge->src()->name());
}

static Node* Send(Graph* g, const string& tensor_name,
                  const string& device_name, bool host, const Edge* edge) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tensor_name: \"" + tensor_name + "\"");
   mht_3_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_3(mht_3_v, 304, "", "./tensorflow/core/common_runtime/memory_types.cc", "Send");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), host ? "_HostSend" : "_Send")
                  .Input(edge->src(), edge->src_output())
                  .Attr("tensor_name", tensor_name)
                  .Attr("send_device", device_name)
                  .Attr("send_device_incarnation", 0)  // Do not care.
                  .Attr("recv_device", device_name)
                  .Attr("_hostmem_sendrecv", true)
                  .Attr("_src", edge->src()->name())
                  .Attr("_dst", edge->dst()->name())
                  .Finalize(g, &ret));
  return ret;
}

static Node* Recv(Graph* g, const string& tensor_name,
                  const string& device_name, bool host, const Edge* edge) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tensor_name: \"" + tensor_name + "\"");
   mht_4_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_4(mht_4_v, 325, "", "./tensorflow/core/common_runtime/memory_types.cc", "Recv");

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), host ? "_HostRecv" : "_Recv")
          .Attr("tensor_type", edge->src()->output_type(edge->src_output()))
          .Attr("tensor_name", tensor_name)
          .Attr("send_device", device_name)
          .Attr("send_device_incarnation", 0)
          .Attr("recv_device", device_name)
          .Attr("_hostmem_sendrecv", true)
          .Attr("_src", edge->src()->name())
          .Attr("_dst", edge->dst()->name())
          .Finalize(g, &ret));
  return ret;
}

Status EnsureMemoryTypes(const DeviceType& device_type,
                         const string& device_name, Graph* g) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_5(mht_5_v, 346, "", "./tensorflow/core/common_runtime/memory_types.cc", "EnsureMemoryTypes");

  struct Item {
    const Edge* edge;
    MemoryType sm;
    MemoryType dm;
  };
  std::vector<Item> edges;
  TF_RETURN_IF_ERROR(ProcessMemoryTypes(
      device_type, g, [&edges](const Edge* e, MemoryType sm, MemoryType dm) {
        if (sm == dm) {
          return Status::OK();
        }
        if (((sm == HOST_MEMORY) && (dm == DEVICE_MEMORY)) ||
            ((sm == DEVICE_MEMORY) && (dm == HOST_MEMORY))) {
          edges.push_back({e, sm, dm});
          return Status::OK();
        }
        return errors::Internal("Unexpected memory type pair on an edge: ", sm,
                                " vs. ", dm);
      }));

  // edges contains edges in 'g' that memtype is not
  // compatible. Therefore, if we found any, we need to insert
  // HostSend/Recv and Send/HostRecv pairs.  recv_nodes records all
  // nodes we added so that we don't copy the same tensor more than
  // once.
  if (!edges.empty()) {
    std::unordered_map<Endpoint, Node*, EndpointHash, EndpointEq> recv_nodes;
    for (const auto& item : edges) {
      const Edge* e = item.edge;
      const bool has_ref = IsRefType(e->src()->output_type(e->src_output()));
      Node* recv = nullptr;
      Endpoint key{e->src()->id(), e->src_output()};
      auto iter = recv_nodes.find(key);
      if (iter == recv_nodes.end()) {
        const string tensor_name = GetTensorName(e);
        Node* send =
            Send(g, tensor_name, device_name, (item.sm == HOST_MEMORY), e);
        recv = Recv(g, tensor_name, device_name, (item.dm == HOST_MEMORY), e);
        if (!has_ref) {
          // We only cache if there is no ref is involved.
          recv_nodes[key] = recv;
        }
        g->AddControlEdge(send, recv);
      } else {
        recv = iter->second;
      }
      g->AddEdge(recv, 0, e->dst(), e->dst_input());
      g->RemoveEdge(e);
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Dumped graph after EnsureMemoryTypes to "
            << DumpGraphToFile("EnsureMemoryTypes", *g);
  }

  return ValidateMemoryTypes(device_type, g);
}

Status MemoryTypeForOutput(const DeviceType& device_type, const Graph* g,
                           const Node* n, int index, MemoryType* memory_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmemory_typesDTcc mht_6(mht_6_v, 410, "", "./tensorflow/core/common_runtime/memory_types.cc", "MemoryTypeForOutput");

  MemoryTypeVector inp_mvec;
  MemoryTypeVector out_mvec;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(g->op_registry(), device_type, n->def(),
                                        &inp_mvec, &out_mvec));
  if (out_mvec.size() <= index) {
    return errors::Internal("Trying to get the memory type for ", index,
                            "'th output of node ", FormatNodeForError(*n),
                            " that has only ", out_mvec.size(), " outputs");
  }
  *memory_type = out_mvec[index];
  return Status::OK();
}

}  // end namespace tensorflow
