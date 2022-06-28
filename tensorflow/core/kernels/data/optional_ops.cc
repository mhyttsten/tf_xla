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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data/optional_ops.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {
namespace {

static Status OptionalDeviceCopy(
    const OptionalVariant& from, OptionalVariant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/data/optional_ops.cc", "OptionalDeviceCopy");

  if (from.has_value()) {
    const std::vector<Tensor>& from_values = from.get_values();
    std::vector<Tensor> to_values;
    to_values.reserve(from_values.size());
    for (const Tensor& t : from_values) {
      if (DMAHelper::CanUseDMA(&t) || t.dtype() == DT_VARIANT) {
        // NOTE(skyewm): we're careful to make sure the lifetime of the 'to'
        // Tensor passed to `copy` (i.e. to_values.back()) is the same as the
        // returned 'to' OptionalVariant. This is because `copy` may spawn async
        // callbacks that don't run until after this function returns and access
        // the 'to' Tensor (e.g. BaseGPUDevice::MaybeCopyTensorToGPU).
        to_values.emplace_back(t.dtype());
        TF_RETURN_IF_ERROR(copy(t, &to_values.back()));
      } else {
        to_values.push_back(t);
      }
    }
    *to = OptionalVariant(std::move(to_values));
  } else {
    *to = from;
  }
  return Status::OK();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      OptionalVariant, DIRECTION, OptionalDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(OptionalVariant,
                                       kOptionalVariantTypeName);

}  // namespace

void OptionalNoneOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/data/optional_ops.cc", "OptionalNoneOp::Compute");

  OP_REQUIRES_OK(ctx, WriteOptionalNoneToOutput(ctx, 0));
}

void OptionalFromValueOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/data/optional_ops.cc", "OptionalFromValueOp::Compute");

  OpInputList components_input;
  OP_REQUIRES_OK(ctx, ctx->input_list("components", &components_input));
  std::vector<Tensor> components(components_input.begin(),
                                 components_input.end());
  OP_REQUIRES_OK(ctx,
                 WriteOptionalWithValueToOutput(ctx, 0, std::move(components)));
}

void OptionalHasValueOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/data/optional_ops.cc", "OptionalHasValueOp::Compute");

  const Tensor* optional_input;
  OP_REQUIRES_OK(ctx, ctx->input("optional", &optional_input));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(optional_input->shape()),
              errors::InvalidArgument(
                  "Input to OptionalHasValue must be a scalar tensor "
                  "containing an OptionalVariant object."));
  const OptionalVariant* optional =
      optional_input->scalar<Variant>()().get<OptionalVariant>();
  OP_REQUIRES(
      ctx, optional != nullptr,
      errors::InvalidArgument(
          "Input to OptionalHasValue must be an OptionalVariant object."));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &result));
  result->scalar<bool>()() = optional->has_value();
}

void OptionalGetValueOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/kernels/data/optional_ops.cc", "OptionalGetValueOp::Compute");

  const Tensor* optional_input;
  OP_REQUIRES_OK(ctx, ctx->input("optional", &optional_input));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(optional_input->shape()),
              errors::InvalidArgument(
                  "Input to OptionalHasValue must be a scalar tensor "
                  "containing an OptionalVariant object."));
  const OptionalVariant* optional =
      optional_input->scalar<Variant>()().get<OptionalVariant>();
  OP_REQUIRES(
      ctx, optional != nullptr,
      errors::InvalidArgument(
          "Input to OptionalHasValue must be an OptionalVariant object."));
  OP_REQUIRES(
      ctx, optional->has_value(),
      errors::InvalidArgument("The given optional does not have a value."));
  const auto& components = optional->get_values();
  OP_REQUIRES(
      ctx, components.size() == output_types_.size(),
      errors::InvalidArgument("The given optional has ", components.size(),
                              " components, expected ", output_types_.size()));
  for (int i = 0; i < components.size(); ++i) {
    OP_REQUIRES(ctx, components[i].dtype() == output_types_[i],
                errors::InvalidArgument(
                    "The given optional does not match the expected type for "
                    "component ",
                    i, ". Expected: ", DataTypeString(output_types_[i]),
                    ". Actual: ", DataTypeString(components[i].dtype()), "."));
    OP_REQUIRES(ctx, output_shapes_[i].IsCompatibleWith(components[i].shape()),
                errors::InvalidArgument(
                    "The given optional does not match the expected shape "
                    "for component ",
                    i, ". Expected: ", output_shapes_[i].DebugString(),
                    ". Actual: ", components[i].shape().DebugString(), "."));
    ctx->set_output(i, components[i]);
  }
}

Status WriteOptionalWithValueToOutput(OpKernelContext* ctx, int output_index,
                                      std::vector<Tensor> value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_5(mht_5_v, 323, "", "./tensorflow/core/kernels/data/optional_ops.cc", "WriteOptionalWithValueToOutput");

  OptionalVariant v(std::move(value));
  Tensor* variant_t;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  TF_RETURN_IF_ERROR(ctx->allocate_output(output_index, TensorShape({}),
                                          &variant_t, cpu_alloc));
  variant_t->scalar<Variant>()() = v;
  return Status::OK();
}

Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTcc mht_6(mht_6_v, 337, "", "./tensorflow/core/kernels/data/optional_ops.cc", "WriteOptionalNoneToOutput");

  OptionalVariant v;
  Tensor* variant_t;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  TF_RETURN_IF_ERROR(ctx->allocate_output(output_index, TensorShape({}),
                                          &variant_t, cpu_alloc));
  variant_t->scalar<Variant>()() = v;
  return Status::OK();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_CPU).Priority(2),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_GPU).Priority(1),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(
    Name("OptionalFromValue").Device(DEVICE_CPU).Priority(2),
    OptionalFromValueOp);
REGISTER_KERNEL_BUILDER(
    Name("OptionalFromValue").Device(DEVICE_GPU).Priority(1),
    OptionalFromValueOp);

REGISTER_KERNEL_BUILDER(Name("OptionalHasValue").Device(DEVICE_CPU).Priority(2),
                        OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalHasValue")
                            .Device(DEVICE_GPU)
                            .HostMemory("has_value")
                            .Priority(1),
                        OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_CPU).Priority(2),
                        OptionalGetValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_GPU).Priority(1),
                        OptionalGetValueOp);

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, OptionalVariant,
                                         OptionalZerosLike<CPUDevice>);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          OptionalVariant,
                                          OptionalBinaryAdd<CPUDevice>);

}  // namespace data
}  // namespace tensorflow
