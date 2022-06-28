/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_C_API_INTERNAL_H_
#define TENSORFLOW_C_C_API_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPSc_api_internalDTh {
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
   MHTracer_DTPStensorflowPScPSc_api_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSc_api_internalDTh() {
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


#include "tensorflow/c/c_api.h"

#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/core/framework/op_gen_lib.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
class Device;
class DeviceMgr;
class ServerInterface;
}  // namespace tensorflow

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

struct TF_SessionOptions {
  tensorflow::SessionOptions options;
};

struct TF_DeprecatedSession {
  tensorflow::Session* session;
};

struct TF_Library {
  void* lib_handle;
  TF_Buffer op_list;
};

struct TF_Graph {
  TF_Graph();

  mutable tensorflow::mutex mu;
  tensorflow::Graph graph TF_GUARDED_BY(mu);

  // Runs shape inference.
  tensorflow::ShapeRefiner refiner TF_GUARDED_BY(mu);

  // Maps from name of an operation to the Node* in 'graph'.
  std::unordered_map<tensorflow::string, tensorflow::Node*> name_map
      TF_GUARDED_BY(mu);

  // The keys of this map are all the active sessions using this graph. Each
  // value records whether the graph has been mutated since the corresponding
  // session has been run (this is detected in RecordMutation function). If the
  // string is empty, no mutation has occurred. Otherwise the string is a
  // description of the mutation suitable for returning to the user.
  //
  // Sessions are added to this map in TF_NewSession, and removed in
  // TF_DeleteSession.
  // TF_Graph may only / must be deleted when
  //   sessions.size() == 0 && delete_requested == true
  //
  // TODO(b/74949947): mutations currently trigger a warning instead of a bad
  // status, this should be reverted when possible.
  tensorflow::gtl::FlatMap<TF_Session*, tensorflow::string> sessions
      TF_GUARDED_BY(mu);
  bool delete_requested TF_GUARDED_BY(mu);  // set true by TF_DeleteGraph

  // Used to link graphs contained in TF_WhileParams to the parent graph that
  // will eventually contain the full while loop.
  TF_Graph* parent;
  TF_Output* parent_inputs;
};

struct TF_OperationDescription {
  TF_OperationDescription(TF_Graph* g, const char* op_type,
                          const char* node_name)
      : node_builder(node_name, op_type, g->graph.op_registry()), graph(g) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_type: \"" + (op_type == nullptr ? std::string("nullptr") : std::string((char*)op_type)) + "\"");
   mht_0_v.push_back("node_name: \"" + (node_name == nullptr ? std::string("nullptr") : std::string((char*)node_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_internalDTh mht_0(mht_0_v, 280, "", "./tensorflow/c/c_api_internal.h", "TF_OperationDescription");
}

  tensorflow::NodeBuilder node_builder;
  TF_Graph* graph;
  std::set<tensorflow::string> colocation_constraints;
};

struct TF_Operation {
  tensorflow::Node node;
};

struct TF_Session {
  TF_Session(tensorflow::Session* s, TF_Graph* g);

  tensorflow::Session* session;
  TF_Graph* const graph;

  tensorflow::mutex mu TF_ACQUIRED_AFTER(TF_Graph::mu);
  int last_num_graph_nodes;

  // If true, TF_SessionRun and similar methods will call
  // ExtendSessionGraphHelper before running the graph (this is the default
  // public behavior). Can be set to false if the caller needs to call
  // ExtendSessionGraphHelper manually.
  std::atomic<bool> extend_before_run;
};

struct TF_ImportGraphDefOptions {
  tensorflow::ImportGraphDefOptions opts;

  // Backing memory for TensorId fields in opts.
  // TODO(skyewm): it'd be better if ImportGraphDefOptions owned this.
  std::list<tensorflow::string> tensor_id_data;
};

struct TF_ImportGraphDefResults {
  std::vector<TF_Output> return_tensors;
  std::vector<TF_Operation*> return_nodes;
  std::vector<const char*> missing_unused_key_names;
  std::vector<int> missing_unused_key_indexes;

  // Backing memory for missing_unused_key_names values.
  std::list<tensorflow::string> missing_unused_key_names_data;
};

struct TF_DeviceList {
  std::vector<tensorflow::DeviceAttributes> response;
};

struct TF_Function {
  tensorflow::FunctionDef fdef;
  tensorflow::StackTracesMap stack_traces;
};

struct TF_ApiDefMap {
  explicit TF_ApiDefMap(const tensorflow::OpList& op_list)
      :
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
        api_def_map(op_list),
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
        update_docs_called(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSc_api_internalDTh mht_1(mht_1_v, 343, "", "./tensorflow/c/c_api_internal.h", "TF_ApiDefMap");

  }

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
  tensorflow::ApiDefMap api_def_map TF_GUARDED_BY(lock);
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
  bool update_docs_called TF_GUARDED_BY(lock);
  tensorflow::mutex lock;
};

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
struct TF_Server {
  TF_Server(std::unique_ptr<tensorflow::ServerInterface> server);

  const tensorflow::string target;
  std::unique_ptr<tensorflow::ServerInterface> server;
};
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

namespace tensorflow {

Status MessageToBuffer(const tensorflow::protobuf::MessageLite& in,
                       TF_Buffer* out);

Status BufferToMessage(const TF_Buffer* in,
                       tensorflow::protobuf::MessageLite* out);

// Set the shapes and types of the output's handle.
//
// The lengths of the arrays pointed to by `shapes`, `ranks`, and `types` must
// all be equal to `num_shapes_and_types`. If `ranks[i] != -1`, (i.e., if the
// rank is known), then it must be equal to the length of `shapes[i]`; if
// `ranks[i] == 1`, then `shapes[i]` may be nullptr.
//
// TODO(akshayka): Implement a corresponding getter method.
void TF_GraphSetOutputHandleShapesAndTypes(TF_Graph* graph, TF_Output output,
                                           int num_shapes_and_types,
                                           const int64_t** shapes,
                                           const int* ranks,
                                           const TF_DataType* types,
                                           TF_Status* status);

void RecordMutation(TF_Graph* graph, const TF_Operation& op,
                    const char* mutation_type)
    TF_EXCLUSIVE_LOCKS_REQUIRED(graph->mu);

bool ExtendSessionGraphHelper(TF_Session* session, TF_Status* status)
    TF_LOCKS_EXCLUDED(session->graph->mu, session->mu);

std::string getTF_OutputDebugString(TF_Output node);

}  // end namespace tensorflow

#endif  // TENSORFLOW_C_C_API_INTERNAL_H_
