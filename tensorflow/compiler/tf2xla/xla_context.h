/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the contexts used during XLA compilation.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh() {
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


#include <vector>

#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class XlaOpKernelContext;
class XlaCompiler;

// The XlaContext is the data structure that holds the state of an XLA
// compilation, that is accessible from OpKernelContexts when compiling a
// subgraph of Ops using XLA.
class XlaContext : public ResourceBase {
 public:
  // Retrieves the XlaContext of the current compilation.
  static XlaContext& Get(const OpKernelContext* ctx);

  // Creates a new XlaContext. See the documentation on the class data fields
  // for descriptions of the arguments.
  XlaContext(XlaCompiler* compiler, xla::XlaBuilder* builder,
             const Graph* graph);

  // Virtual method defined by ResourceBase.
  string DebugString() const override;

  XlaCompiler* compiler() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/tf2xla/xla_context.h", "compiler");
 return compiler_; }

  const AbstractStackTrace* StackTraceForNodeName(const std::string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_1(mht_1_v, 230, "", "./tensorflow/compiler/tf2xla/xla_context.h", "StackTraceForNodeName");

    const auto& it = stack_traces_.find(name);
    if (it != stack_traces_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  // Returns the XlaBuilder that Ops use for compiling new expressions.
  xla::XlaBuilder* builder() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_2(mht_2_v, 242, "", "./tensorflow/compiler/tf2xla/xla_context.h", "builder");
 return builder_; }

  const std::vector<XlaExpression>& args() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_3(mht_3_v, 247, "", "./tensorflow/compiler/tf2xla/xla_context.h", "args");
 return args_; }
  void set_args(std::vector<XlaExpression> args);

  const std::vector<XlaExpression>& retvals() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_4(mht_4_v, 253, "", "./tensorflow/compiler/tf2xla/xla_context.h", "retvals");
 return retvals_; }

  // Sets a return value.
  // Since we do not always know in advance how many return values there are,
  // grows the return values vector to size index+1 if it is smaller.
  void SetRetval(int index, const XlaExpression& expression);

  // Adds 'resource' to the set of resources owned by the context.
  XlaResource* AddResource(std::unique_ptr<XlaResource> resource);

  const std::vector<std::unique_ptr<XlaResource>>& resources() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_5(mht_5_v, 266, "", "./tensorflow/compiler/tf2xla/xla_context.h", "resources");

    return resources_;
  }

  // Get an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMax(const DataType type);

  // Get an XLA lambda to compute Min. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMin(const DataType type);

  // Get an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateAdd(const DataType type);

  // Get an XLA lambda to compute Mul. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMul(const DataType type);

  // The name of the XlaContext resource during symbolic graph execution.
  static const char kXlaContextResourceName[];

  // Records the collective information from the nested compilation `result`.
  Status RecordCollectiveInfoFromNestedCompilationResult(
      const XlaCompilationResult& result);

  // Records the collective configurations for all the collectives in the XLA
  // cluster and returns the channel_id to be used for the next collective.
  StatusOr<int64_t> RecordCollectiveInfo(int group_key, int group_size);

  const absl::optional<XlaCompilationResult::CollectiveInfo>&
  GetCollectiveInfo() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTh mht_6(mht_6_v, 305, "", "./tensorflow/compiler/tf2xla/xla_context.h", "GetCollectiveInfo");

    return collective_info_;
  }

 private:
  XlaCompiler* const compiler_;

  // The XlaBuilder used to construct the subgraph's compiled representation.
  xla::XlaBuilder* builder_;

  // Stack traces for the graph used for compilation.
  StackTracesMap stack_traces_;

  // Arguments to the Tensorflow graph, indexed by _Arg index.
  // Includes both compile-time constant arguments and runtime parameters.
  std::vector<XlaExpression> args_;

  // Return values of the Tensorflow graph, indexed by _Retval index.
  std::vector<XlaExpression> retvals_;

  // Holds ownership of resources. The resources are not ordered.
  std::vector<std::unique_ptr<XlaResource>> resources_;

  // Information about encountered collective ops. We allow only a
  // single configuration per cluster.
  absl::optional<XlaCompilationResult::CollectiveInfo> collective_info_;

  // Cache of prebuilt computations indexed by their type.
  using ComputationMap = std::map<DataType, xla::XlaComputation>;

  // Finds the value for the given type in out map if it already
  // exists or makes a new value with create function and keeps it the
  // map. The returned value != nullptr and is owned by the map.
  const xla::XlaComputation* LookupOrCreate(
      DataType type, ComputationMap* out,
      const std::function<xla::XlaComputation()>& create);

  // Cached computation to compute Max of two elements, specialized by type.
  ComputationMap max_func_;

  // Cached computation to compute Min of two elements, specialized by type.
  ComputationMap min_func_;

  // Cached computation to compute Sum of two elements, specialized by type.
  ComputationMap add_func_;

  // Cached computation to compute Mul of two elements, specialized by type.
  ComputationMap mul_func_;

  // Cached computation to compute Sigmoid of an element, specialized by type.
  ComputationMap sigmoid_func_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
