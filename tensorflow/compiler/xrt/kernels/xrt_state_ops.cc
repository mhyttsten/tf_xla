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
class MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_state_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_state_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_state_opsDTcc() {
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

// Classes for allocating XLA literals in device memory and managing handles
// that refer to them.

#include "tensorflow/compiler/xrt/kernels/xrt_state_ops.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"

namespace tensorflow {
namespace {

class XRTMetricsCollectOp : public OpKernel {
 public:
  explicit XRTMetricsCollectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_state_opsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xrt/kernels/xrt_state_ops.cc", "XRTMetricsCollectOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_state_opsDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xrt/kernels/xrt_state_ops.cc", "Compute");

    VLOG(1) << "XRTMetricsCollectOp::Compute";

    const Tensor& metrics_proto = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(metrics_proto.shape()),
                errors::Internal("request input should be a string scalar"));
    xrt::XRTMetricsCollect metrics;
    OP_REQUIRES(ctx,
                ParseFromTString(metrics_proto.scalar<tstring>()(), &metrics),
                errors::InvalidArgument(
                    "Unable to parse request input to XRTMetricsCollect"));

    xla::StatusOr<xrt::MetricsReport> collected_metrics_or =
        CollectMetrics(metrics);
    OP_REQUIRES_OK(ctx, collected_metrics_or.status());
    xrt::MetricsReport collected_metrics =
        collected_metrics_or.ConsumeValueOrDie();
    Tensor output(DT_STRING, TensorShape({}));
    output.scalar<tstring>()() = collected_metrics.SerializeAsString();
    ctx->set_output(0, output);
  }
};

}  // namespace

REGISTER_KERNEL_BUILDER(Name("XRTAllocate")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("allocation")
                            .HostMemory("handle"),
                        XRTAllocateOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocate")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("allocation")
                            .HostMemory("handle"),
                        XRTAllocateOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTAllocateUninitialized")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTAllocateUninitializedOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocateUninitialized")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTAllocateUninitializedOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTAllocateFromTensor")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("inputs")
                            .HostMemory("handle"),
                        XRTAllocateFromTensorOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocateFromTensor")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("inputs")
                            .HostMemory("handle"),
                        XRTAllocateFromTensorOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTuple")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<false, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTSubTuple")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<false, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTupleAndRelease")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<true, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTSubTupleAndRelease")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<true, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMakeTuple")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("tuple_description")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTMakeTupleOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTMakeTuple")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("tuple_description")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTMakeTupleOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteral")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<false, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadLiteral")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<false, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTWriteLiteral")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal")
                            .HostMemory("output_handle"),
                        XRTWriteLiteralOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTWriteLiteral")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal")
                            .HostMemory("output_handle"),
                        XRTWriteLiteralOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteralAndRelease")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<true, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadLiteralAndRelease")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<true, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadToTensor")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handles")
                            .HostMemory("tensors"),
                        XRTReadToTensorOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadToTensor")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handles")
                            .HostMemory("tensors"),
                        XRTReadToTensorOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllocationHandle")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTReleaseAllocationOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllocationHandle")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTReleaseAllocationOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllAllocations").Device(DEVICE_XLA_GPU),
                        XRTReleaseAllAllocationsOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllAllocations").Device(DEVICE_XLA_CPU),
                        XRTReleaseAllAllocationsOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTCompactAllocations").Device(DEVICE_XLA_GPU),
                        XRTCompactAllocationsOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTCompactAllocations").Device(DEVICE_XLA_CPU),
                        XRTCompactAllocationsOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMetricsCollect").Device(DEVICE_CPU),
                        XRTMetricsCollectOp);

REGISTER_KERNEL_BUILDER(Name("XRTMemoryInfo").Device(DEVICE_XLA_GPU),
                        XRTMemoryInfoOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTMemoryInfo").Device(DEVICE_XLA_CPU),
                        XRTMemoryInfoOp<XRTGenericDeviceAccessor>);

}  // namespace tensorflow
