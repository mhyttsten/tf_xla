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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc() {
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

#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

FlatTensorFunction::FlatTensorFunction(
    const std::string& name, std::vector<ImmediateTensorHandlePtr> captures,
    ImmediateExecutionContext* ctx)
    : name_(name), captures_(std::move(captures)), ctx_(ctx) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc mht_0(mht_0_v, 208, "", "./tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.cc", "FlatTensorFunction::FlatTensorFunction");
}

FlatTensorFunction::~FlatTensorFunction() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc mht_1(mht_1_v, 213, "", "./tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.cc", "FlatTensorFunction::~FlatTensorFunction");

  Status status = ctx_->RemoveFunction(name_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to remove functiondef " << name_ << ". "
               << status.error_message();
  }
}

Status FlatTensorFunction::Create(
    const FunctionDef* function_def,
    std::vector<ImmediateExecutionTensorHandle*> captures,
    ImmediateExecutionContext* ctx, std::unique_ptr<FlatTensorFunction>* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc mht_2(mht_2_v, 227, "", "./tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.cc", "FlatTensorFunction::Create");

  TF_RETURN_IF_ERROR(ctx->AddFunctionDef(*function_def));
  std::vector<ImmediateTensorHandlePtr> owned_captures;
  owned_captures.reserve(captures.size());
  for (ImmediateExecutionTensorHandle* capture : captures) {
    capture->Ref();
    owned_captures.push_back(ImmediateTensorHandlePtr(capture));
  }

  out->reset(new FlatTensorFunction(function_def->signature().name(),
                                    std::move(owned_captures), ctx));
  return Status();
}

Status FlatTensorFunction::MakeCallOp(
    absl::Span<AbstractTensorHandle* const> inputs, ImmediateOpPtr* out) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSflat_tensor_functionDTcc mht_3(mht_3_v, 245, "", "./tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.cc", "FlatTensorFunction::MakeCallOp");

  out->reset(ctx_->CreateOperation());
  // In eager mode, TF2 python executes functions by constructing an op with
  // the name of the functiondef:
  // https://github.com/tensorflow/tensorflow/blob/66668ec0ca432e2f38a575b814f45b6d299d01ed/tensorflow/python/eager/function.py#L545
  // In graph mode, we create a PartitionedCallOp instead:
  // https://github.com/tensorflow/tensorflow/blob/66668ec0ca432e2f38a575b814f45b6d299d01ed/tensorflow/python/eager/function.py#L573

  // TODO(bmzhao): After discussing with Allen, we should execute this via a
  // PartitionedCallOp for compatibility with "tooling that assumes functions in
  // graphs are PartitionedCallOps".
  TF_RETURN_IF_ERROR((*out)->Reset(name_.c_str(), nullptr));

  // Adding the user-provided inputs to the function.
  TF_RETURN_IF_ERROR((*out)->AddInputList(inputs));

  absl::Span<AbstractTensorHandle* const> captures(
      reinterpret_cast<AbstractTensorHandle* const*>(captures_.data()),
      captures_.size());

  // Adding the captures of the function.
  TF_RETURN_IF_ERROR((*out)->AddInputList(captures));
  return Status();
}

}  // namespace tensorflow
