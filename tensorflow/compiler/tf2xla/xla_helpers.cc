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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc() {
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

// This file defines helper routines for XLA compilation.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"

#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

xla::XlaOp XlaHelpers::Zero(xla::XlaBuilder* b, DataType data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::Zero");

  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::Zero(type));
}

xla::XlaOp XlaHelpers::One(xla::XlaBuilder* b, DataType data_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::One");

  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::One(type));
}

xla::XlaOp XlaHelpers::IntegerLiteral(xla::XlaBuilder* b, DataType data_type,
                                      int64_t value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::IntegerLiteral");

  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::IntegerLiteral(b, type, value);
}

xla::XlaOp XlaHelpers::FloatLiteral(xla::XlaBuilder* b, DataType data_type,
                                    double value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::FloatLiteral");

  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::FloatLiteral(b, type, value);
}

/* static */ Status XlaHelpers::ReshapeLiteral(
    const xla::Literal& input, absl::Span<const int64_t> dimensions,
    xla::Literal* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::ReshapeLiteral");

  if (input.shape().IsTuple()) {
    return errors::InvalidArgument("ReshapeLiteral does not support tuples.");
  }
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(input.shape().element_type(), dimensions);
  int64_t elements_before = xla::ShapeUtil::ElementsIn(input.shape());
  int64_t elements_after = xla::ShapeUtil::ElementsIn(shape);
  if (elements_before != elements_after) {
    return errors::InvalidArgument(
        "Shapes before and after ReshapeLiteral have different numbers of "
        "elements.");
  }

  *output = input.Clone();
  output->mutable_shape_do_not_use()->Swap(&shape);
  return Status::OK();
}

Status XlaHelpers::OneHot(xla::XlaBuilder* builder, int64_t depth, int axis,
                          DataType index_type, const TensorShape& indices_shape,
                          const xla::XlaOp& indices, const xla::XlaOp& on_value,
                          const xla::XlaOp& off_value, xla::XlaOp* one_hot) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_5(mht_5_v, 275, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::OneHot");

  // Broadcast the linspace constant across the indices along the new axis,
  // and test equality at each position.
  std::vector<int64_t> broadcast_dims(indices_shape.dims());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

  TensorShape output_shape = indices_shape;
  output_shape.InsertDim(axis, depth);
  xla::Shape iota_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(index_type, output_shape, &iota_shape));

  // Selects the user-provided off_value and on_value values.
  *one_hot = xla::Select(
      xla::Eq(indices, xla::Iota(builder, iota_shape, axis), broadcast_dims),
      xla::Broadcast(on_value, output_shape.dim_sizes()),
      xla::Broadcast(off_value, output_shape.dim_sizes()));
  return Status::OK();
}

DataType XlaHelpers::SumAccumulationType(const DataType& dtype) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::SumAccumulationType");

  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  if (dtype == DT_BFLOAT16 || dtype == DT_HALF) {
    return DT_FLOAT;
  }
  // Upcast small integer types to 32 bit to avoid overflow.
  if (dtype == DT_INT8 || dtype == DT_INT16) {
    return DT_INT32;
  }
  if (dtype == DT_UINT8 || dtype == DT_UINT16) {
    return DT_UINT32;
  }
  return dtype;
}

xla::XlaOp XlaHelpers::ConvertElementType(const xla::XlaOp& operand,
                                          const DataType new_element_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_7(mht_7_v, 319, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "XlaHelpers::ConvertElementType");

  xla::PrimitiveType convert_to;
  TF_CHECK_OK(DataTypeToPrimitiveType(new_element_type, &convert_to));
  return xla::ConvertElementType(operand, convert_to);
}

XlaHelpers::ShapeRepresentationFn IdentityShapeRepresentationFn() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_8(mht_8_v, 328, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "IdentityShapeRepresentationFn");

  return [](const TensorShape& shape, DataType dtype, bool use_fast_memory,
            XlaLayoutPreference layout_preference) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, shape, &xla_shape));
    return xla_shape;
  };
}

Status ResolveDeviceAssignment(
    OpKernelContext* ctx,
    const XlaCompilationResult::CollectiveInfo& collective_info,
    xla::ExecutableRunOptions& run_options,
    xla::DeviceAssignment& device_assignment,
    xla::gpu::GpuExecutableRunOptions& gpu_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_helpersDTcc mht_9(mht_9_v, 345, "", "./tensorflow/compiler/tf2xla/xla_helpers.cc", "ResolveDeviceAssignment");

  // TODO(nnigania): workaround for b/199436990
  static const int kTimeoutSeconds = 1000;
  if (ctx->collective_executor() == nullptr) {
    return errors::InvalidArgument(
        "CollectiveExecutor is required but not available");
  }

  auto params = core::RefCountPtr<CollectiveParams>(new CollectiveParams());
  params->name = "xla-reduction-compilation";
  params->group.device_type =
      DeviceType{static_cast<Device*>(ctx->device())->device_type()};
  params->group.group_size = collective_info.group_size;
  params->group.group_key = collective_info.group_key;
  params->instance.type = REDUCTION_COLLECTIVE;
  params->instance.impl_details.communication_hint = "nccl";
  params->instance.impl_details.timeout_seconds = kTimeoutSeconds;
  params->instance.impl_details.collective_name = "NcclReduce";
  // TODO(cheshire): Avoid passing a dummy shape, TF runtime does not resolve
  // devices otherwise.
  params->instance.shape = TensorShape({1});

  Status st;
  absl::Notification n;
  ctx->collective_executor()->CompleteParamsAsync(
      ctx->device()->attributes(), params.get(), ctx->cancellation_manager(),
      [&](const Status& s) {
        st = s;
        n.Notify();
      });
  if (!n.WaitForNotificationWithTimeout(absl::Seconds(kTimeoutSeconds))) {
    return errors::InvalidArgument("Timeout reached");
  }
  TF_RETURN_IF_ERROR(st);
  VLOG(5) << "Using collective params to resolve device assignment: "
          << params->ToString();

  // Identify the physical device associated with each replica.
  device_assignment = xla::DeviceAssignment(params->group.group_size, 1);
  for (int device_idx = 0; device_idx < params->group.group_size;
       device_idx++) {
    const DeviceAttributes& device = params->group.members[device_idx].device;
    if (device.xla_global_id() == -1) {
      if (params->group.device_type == DEVICE_TPU) {
        return errors::InvalidArgument(
            absl::StrCat("No global ID was set for TPU device ", device.name(),
                         ". Try initializing the TPU system, e.g. "
                         "`tf.tpu.experimental.initialize_tpu_system()`."));
      } else if (params->group.device_type == DEVICE_GPU) {
        return errors::Internal(
            absl::StrCat("No global ID was set for ", device.name(),
                         ". This is unexpected, please file a bug."));
      } else {
        // TODO(b/194942685): Implement CPU collectives.
        return errors::Unimplemented(
            absl::StrCat("Collectives are not yet implemented for ",
                         params->group.device_type.type_string(),
                         " devices when compiling with XLA. Attempted to "
                         "compile a collective running on",
                         device.name(),
                         ". Please comment on b/194942685 or "
                         "file a new bug if you don't have access."));
      }
    }
    VLOG(2) << "Assigning physical id " << device.xla_global_id()
            << " for replica " << device_idx << " (" << device.name() << ")";
    device_assignment(device_idx, 0) = device.xla_global_id();
  }
  VLOG(5) << "Generated device assignment: " << device_assignment.ToString();
  if (params->group.device_type == DEVICE_GPU) {
    // For GPU collectives, `xla_global_id`s are arbitrary integers, and XLA
    // requires a mapping from local device IDs to global device IDs.
    const DeviceMgr* device_mgr = ctx->function_library()->device_mgr();
    std::vector<xla::GlobalDeviceId> global_device_ids(
        device_mgr->NumDeviceType(params->group.device_type.type_string()));

    for (int device_idx = 0; device_idx < params->group.group_size;
         device_idx++) {
      const DeviceAttributes& device_attributes =
          params->group.members[device_idx].device;
      Device* resolved_device = nullptr;
      Status lookup_status =
          device_mgr->LookupDevice(device_attributes.name(), &resolved_device);
      if (lookup_status.ok()) {
        // This is a local device, so include it in the mapping.
        const DeviceBase::AcceleratorDeviceInfo* gpu_device_info =
            resolved_device->tensorflow_accelerator_device_info();
        global_device_ids[gpu_device_info->stream->parent()->device_ordinal()] =
            device_attributes.xla_global_id();
      }
    }
    gpu_options.set_gpu_global_device_ids(global_device_ids);
  }
  run_options.set_device_assignment(&device_assignment);
  run_options.set_gpu_executable_run_options(&gpu_options);
  return Status::OK();
}

}  // end namespace tensorflow
