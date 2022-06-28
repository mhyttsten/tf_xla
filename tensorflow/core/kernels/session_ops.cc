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
class MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc() {
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

// See docs in ../ops/data_flow_ops.cc.

#include <limits.h>

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class GetSessionHandleOp : public OpKernel {
 public:
  explicit GetSessionHandleOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/session_ops.cc", "GetSessionHandleOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/session_ops.cc", "Compute");

    const Tensor& val = ctx->input(0);
    auto session_state = ctx->session_state();
    OP_REQUIRES(ctx, session_state != nullptr,
                errors::FailedPrecondition(
                    "GetSessionHandle called on null session state"));
    int64_t id = session_state->GetNewId();
    TensorStore::TensorAndKey tk{val, id, requested_device()};
    OP_REQUIRES_OK(ctx, ctx->tensor_store()->AddTensor(name(), tk));

    Tensor* handle = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      ResourceHandle resource_handle = MakeResourceHandle<Tensor>(
          ctx, SessionState::kTensorHandleResourceTypeName,
          tk.GetHandle(name()));
      resource_handle.set_maybe_type_name(
          SessionState::kTensorHandleResourceTypeName);
      handle->scalar<ResourceHandle>()() = resource_handle;
    } else {
      // Legacy behavior in V1.
      handle->flat<tstring>().setConstant(tk.GetHandle(name()));
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GetSessionHandleOp);
};

REGISTER_KERNEL_BUILDER(Name("GetSessionHandle").Device(DEVICE_CPU),
                        GetSessionHandleOp);
REGISTER_KERNEL_BUILDER(Name("GetSessionHandleV2").Device(DEVICE_CPU),
                        GetSessionHandleOp);

#define REGISTER_DEFAULT_KERNEL(type)                     \
  REGISTER_KERNEL_BUILDER(Name("GetSessionHandle")        \
                              .Device(DEVICE_DEFAULT)     \
                              .HostMemory("handle")       \
                              .TypeConstraint<type>("T"), \
                          GetSessionHandleOp)             \
  REGISTER_KERNEL_BUILDER(Name("GetSessionHandleV2")      \
                              .Device(DEVICE_DEFAULT)     \
                              .HostMemory("handle")       \
                              .TypeConstraint<type>("T"), \
                          GetSessionHandleOp)

TF_CALL_NUMBER_TYPES(REGISTER_DEFAULT_KERNEL);
REGISTER_DEFAULT_KERNEL(bool);
#undef REGISTER_DEFAULT_KERNEL

class GetSessionTensorOp : public OpKernel {
 public:
  explicit GetSessionTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/kernels/session_ops.cc", "GetSessionTensorOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/session_ops.cc", "Compute");

    const Tensor& handle = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(handle.shape()),
                errors::InvalidArgument("handle must be scalar"));
    const string& name = handle.scalar<tstring>()();
    Tensor val;
    auto session_state = ctx->session_state();
    OP_REQUIRES(ctx, session_state != nullptr,
                errors::FailedPrecondition(
                    "GetSessionTensor called on null session state"));
    OP_REQUIRES_OK(ctx, session_state->GetTensor(name, &val));
    ctx->set_output(0, val);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GetSessionTensorOp);
};

REGISTER_KERNEL_BUILDER(Name("GetSessionTensor").Device(DEVICE_CPU),
                        GetSessionTensorOp);

#define REGISTER_DEFAULT_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("GetSessionTensor")            \
                              .Device(DEVICE_DEFAULT)         \
                              .HostMemory("handle")           \
                              .TypeConstraint<type>("dtype"), \
                          GetSessionTensorOp)

TF_CALL_NUMBER_TYPES(REGISTER_DEFAULT_KERNEL);
REGISTER_DEFAULT_KERNEL(bool);
#undef REGISTER_DEFAULT_KERNEL

class DeleteSessionTensorOp : public OpKernel {
 public:
  explicit DeleteSessionTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_4(mht_4_v, 314, "", "./tensorflow/core/kernels/session_ops.cc", "DeleteSessionTensorOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsession_opsDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/kernels/session_ops.cc", "Compute");

    const Tensor& handle = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(handle.shape()),
                errors::InvalidArgument("`handle` must be scalar"));
    const string& name = handle.scalar<tstring>()();
    auto session_state = ctx->session_state();
    OP_REQUIRES(ctx, session_state != nullptr,
                errors::FailedPrecondition(
                    "DeleteSessionTensor called on null session state"));
    OP_REQUIRES_OK(ctx, session_state->DeleteTensor(name));
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DeleteSessionTensorOp);
};

REGISTER_KERNEL_BUILDER(Name("DeleteSessionTensor").Device(DEVICE_CPU),
                        DeleteSessionTensorOp);
REGISTER_KERNEL_BUILDER(
    Name("DeleteSessionTensor").Device(DEVICE_DEFAULT).HostMemory("handle"),
    DeleteSessionTensorOp);

}  // namespace tensorflow
