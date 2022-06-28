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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/resource_variable_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: VarHandleOp()
// Summary: Creates a handle to a Variable resource.
//
// Description:
Status VarHandleOp(AbstractContext* ctx, AbstractTensorHandle** resource,
                   DataType dtype, const PartialTensorShape shape,
                   const char* container, const char* shared_name,
                   absl::Span<string const> allowed_devices, const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("container: \"" + (container == nullptr ? std::string("nullptr") : std::string((char*)container)) + "\"");
   mht_0_v.push_back("shared_name: \"" + (shared_name == nullptr ? std::string("nullptr") : std::string((char*)shared_name)) + "\"");
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/c/experimental/ops/resource_variable_ops.cc", "VarHandleOp");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("VarHandleOp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("container", container, strlen(container)));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("shared_name", shared_name, strlen(shared_name)));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrShape("shape", shape));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrStringList("allowed_devices", allowed_devices));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(resource, 1), &num_retvals);
}

// Op: ReadVariableOp()
// Summary: Reads the value of a variable.
//
// Description:
//   The tensor returned by this operation is immutable.
//
//   The value returned by this operation is guaranteed to be influenced by all
//   the writes on which this operation depends directly or indirectly, and to
//   not be influenced by any of the writes which depend directly or indirectly
//   on this operation.
Status ReadVariableOp(AbstractContext* ctx,
                      AbstractTensorHandle* const resource,
                      AbstractTensorHandle** value, DataType dtype,
                      const char* name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc mht_1(mht_1_v, 243, "", "./tensorflow/c/experimental/ops/resource_variable_ops.cc", "ReadVariableOp");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      op_ptr->Reset("ReadVariableOp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("dtype", dtype));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(value, 1), &num_retvals);
}

// Op: AssignVariableOp()
// Summary: Assigns a new value to a variable.
//
// Description:
//   Any ReadVariableOp with a control dependency on this op is guaranteed to
//   return this value or a subsequent newer value of the variable.
Status AssignVariableOp(AbstractContext* ctx,
                        AbstractTensorHandle* const resource,
                        AbstractTensorHandle* const value, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc mht_2(mht_2_v, 266, "", "./tensorflow/c/experimental/ops/resource_variable_ops.cc", "AssignVariableOp");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      op_ptr->Reset("AssignVariableOp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(value));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}

// Op: DestroyResourceOp()
// Summary: Deletes the resource specified by the handle.
//
// Description:
//   All subsequent operations using the resource will result in a NotFound
//   error status.
Status DestroyResourceOp(AbstractContext* ctx,
                         AbstractTensorHandle* const resource,
                         bool ignore_lookup_error, const char* name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSresource_variable_opsDTcc mht_3(mht_3_v, 290, "", "./tensorflow/c/experimental/ops/resource_variable_ops.cc", "DestroyResourceOp");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      op_ptr->Reset("DestroyResourceOp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrBool("ignore_lookup_error", ignore_lookup_error));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
