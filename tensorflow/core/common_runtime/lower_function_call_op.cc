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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_opDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/lower_function_call_op.h"

#include "absl/algorithm/container.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;

Status RewriteFunctionCallNode(Node* n, Graph* g,
                               const FunctionLibraryDefinition& flib_def,
                               bool keep_caller_fetchable) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/common_runtime/lower_function_call_op.cc", "RewriteFunctionCallNode");

  VLOG(2) << "Lower function call node: " << SummarizeNode(*n);

  // We support lowering of two types of functions that could be invoked by the
  // node `n`: 1) native functions and 2) multi-device functions.
  // NOTE(ezhulenev): We explicitly choose not to deal with SymbolicGradient,
  // because it has been deprecated for a long time.
  InlineFunctionBodyOptions inline_options;
  inline_options.keep_caller_node = keep_caller_fetchable
                                        ? KeepCallerNode::kFetchable
                                        : KeepCallerNode::kTargetable;

  FunctionCallInlinePolicy policy = GetFunctionCallInlinePolicy(n);
  if (policy == FunctionCallInlinePolicy::kMultiDevicePlacer) {
    // Multi-device function calls (PartitionedCall or StatefulPartitionedCall
    // ops) can execute on multiple devices and accept DT_RESOURCE inputs that
    // belong to different devices. This type of functions was added in
    // Tensorflow 2.0 Eager mode, and it has control outputs to represent
    // side-effects that must always execute (see `control_ret` in FunctionDef).
    inline_options.output_control_src = OutputControlSrc::kControlOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::MultiDevice();
  } else if (policy == FunctionCallInlinePolicy::kSingleDevicePlacer) {
    // Native function call (node.type_string() is the function name). These
    // functions are always executed on a single-device, which is the device of
    // the function call node.
    inline_options.output_control_src = OutputControlSrc::kDataOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::SingleDevice();
  } else {
    return errors::InvalidArgument("Unsupported function inlining policy");
  }

  const FunctionDef* fdef;
  if (n->IsPartitionedCall()) {
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "f", &func));
    fdef = flib_def.Find(func.name());
  } else if (n->type_string() == FunctionLibraryDefinition::kGradientOp) {
    VLOG(2) << "Skip SymbolicGradient lowering";
    return Status::OK();
  } else {
    fdef = flib_def.Find(n->type_string());
  }

  if (fdef == nullptr) {
    return errors::Internal("Can't find a function: node=", SummarizeNode(*n));
  }

  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*fdef, n->attrs(), &flib_def, &fbody));

  Status can_inline_function_call =
      ValidateInlining(n, fbody.get(), inline_options);
  if (can_inline_function_call.ok()) {
    TF_RETURN_IF_ERROR(
        InlineFunctionBody(flib_def, g, n, fbody.get(), inline_options));
  } else {
    VLOG(2) << "Failed to inline function call node: "
            << can_inline_function_call.error_message();
  }

  return Status::OK();
}

}  // namespace tensorflow
