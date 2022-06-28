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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc() {
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

#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/op_logger.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt_distributed_interface.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.h"
#include "tensorflow/core/tfrt/eager/virtual_device.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/location.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/metrics/common_metrics.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_type_registration.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

namespace {

using tensorflow::down_cast;

constexpr char kGpuDeviceName[] = "GPU";
constexpr char kEnableNativeOpsAttr[] = "TFRT_TEST_enable_native_ops";
constexpr char kEnableGrapplerAttr[] = "TFRT_TEST_enable_grappler";

TensorMetadata CreateMetadata(DType dtype, absl::Span<const Index> dim_sizes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_0(mht_0_v, 280, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "CreateMetadata");

  return TensorMetadata(
      DType(dtype),
      TensorShape(llvm::ArrayRef<Index>(
          reinterpret_cast<const Index*>(dim_sizes.data()), dim_sizes.size())));
}

tensorflow::DataType ConvertDType(DType kind) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_1(mht_1_v, 290, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ConvertDType");

  switch (kind) {
    case DType::UI8:
      return tensorflow::DT_UINT8;
    case DType::UI16:
      return tensorflow::DT_UINT16;
    case DType::UI32:
      return tensorflow::DT_UINT32;
    case DType::UI64:
      return tensorflow::DT_UINT64;
    case DType::I8:
      return tensorflow::DT_INT8;
    case DType::I16:
      return tensorflow::DT_INT16;
    case DType::I32:
      return tensorflow::DT_INT32;
    case DType::I64:
      return tensorflow::DT_INT64;
    case DType::BF16:
      return tensorflow::DT_BFLOAT16;
    case DType::F16:
      return tensorflow::DT_HALF;
    case DType::F32:
      return tensorflow::DT_FLOAT;
    case DType::F64:
      return tensorflow::DT_DOUBLE;
    case DType::I1:
      return tensorflow::DT_BOOL;
    case DType::Complex64:
      return tensorflow::DT_COMPLEX64;
    case DType::Complex128:
      return tensorflow::DT_COMPLEX128;
    case DType::String:
      return tensorflow::DT_STRING;
    case DType::Resource:
      return tensorflow::DT_RESOURCE;
    case DType::Variant:
      return tensorflow::DT_VARIANT;
    case DType::QUI8:
      return tensorflow::DT_QUINT8;
    case DType::QUI16:
      return tensorflow::DT_QUINT16;
    case DType::QI8:
      return tensorflow::DT_QINT8;
    case DType::QI16:
      return tensorflow::DT_QINT16;
    case DType::QI32:
      return tensorflow::DT_QINT32;
    default:
      LOG(ERROR) << "Unsupported kind " << kind;
      return tensorflow::DT_INVALID;
  }
}

DType ConvertDType(tensorflow::DataType dtype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_2(mht_2_v, 347, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ConvertDType");

  switch (dtype) {
    case tensorflow::DT_UINT8:
      return static_cast<DType>(DType::UI8);
    case tensorflow::DT_UINT16:
      return static_cast<DType>(DType::UI16);
    case tensorflow::DT_UINT32:
      return static_cast<DType>(DType::UI32);
    case tensorflow::DT_UINT64:
      return static_cast<DType>(DType::UI64);
    case tensorflow::DT_INT8:
      return static_cast<DType>(DType::I8);
    case tensorflow::DT_INT16:
      return static_cast<DType>(DType::I16);
    case tensorflow::DT_INT32:
      return static_cast<DType>(DType::I32);
    case tensorflow::DT_INT64:
      return static_cast<DType>(DType::I64);
    case tensorflow::DT_BFLOAT16:
      return static_cast<DType>(DType::BF16);
    case tensorflow::DT_HALF:
      return static_cast<DType>(DType::F16);
    case tensorflow::DT_FLOAT:
      return static_cast<DType>(DType::F32);
    case tensorflow::DT_DOUBLE:
      return static_cast<DType>(DType::F64);
    case tensorflow::DT_BOOL:
      return static_cast<DType>(DType::I1);
    case tensorflow::DT_STRING:
      return static_cast<DType>(DType::String);
    case tensorflow::DT_COMPLEX64:
      return static_cast<DType>(DType::Complex64);
    case tensorflow::DT_COMPLEX128:
      return static_cast<DType>(DType::Complex128);
    case tensorflow::DT_RESOURCE:
      return static_cast<DType>(DType::Resource);
    case tensorflow::DT_VARIANT:
      return static_cast<DType>(DType::Variant);
    case tensorflow::DT_QUINT8:
      return static_cast<DType>(DType::QUI8);
    case tensorflow::DT_QUINT16:
      return static_cast<DType>(DType::QUI16);
    case tensorflow::DT_QINT8:
      return static_cast<DType>(DType::QI8);
    case tensorflow::DT_QINT16:
      return static_cast<DType>(DType::QI16);
    case tensorflow::DT_QINT32:
      return static_cast<DType>(DType::QI32);
    default:
      LOG(FATAL) << "Unsupported dtype " << dtype;
  }
}

OpAttrType ConvertDTypeToOpAttrType(tensorflow::DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_3(mht_3_v, 403, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ConvertDTypeToOpAttrType");

  switch (dtype) {
    case tensorflow::DT_UINT8:
      return OpAttrType::UI8;
    case tensorflow::DT_UINT16:
      return OpAttrType::UI16;
    case tensorflow::DT_UINT32:
      return OpAttrType::UI32;
    case tensorflow::DT_UINT64:
      return OpAttrType::UI64;
    case tensorflow::DT_INT8:
      return OpAttrType::I8;
    case tensorflow::DT_INT16:
      return OpAttrType::I16;
    case tensorflow::DT_INT32:
      return OpAttrType::I32;
    case tensorflow::DT_INT64:
      return OpAttrType::I64;
    case tensorflow::DT_BFLOAT16:
      return OpAttrType::BF16;
    case tensorflow::DT_HALF:
      return OpAttrType::F16;
    case tensorflow::DT_FLOAT:
      return OpAttrType::F32;
    case tensorflow::DT_DOUBLE:
      return OpAttrType::F64;
    case tensorflow::DT_BOOL:
      return OpAttrType::BOOL;
    case tensorflow::DT_COMPLEX64:
      return OpAttrType::COMPLEX64;
    case tensorflow::DT_COMPLEX128:
      return OpAttrType::COMPLEX128;
    default:
      LOG(FATAL) << "Unsupported dtype " << dtype;
  }
}

// This method will first look at the calling op attrs and then look at the
// function def attrs to find the attribute value.
void GetFuncAttr(const OpAttrs& op_attrs, const std::string& op_name,
                 const tensorflow::FunctionLibraryDefinition& func_lib_def,
                 string_view attr_name, bool* value) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_name: \"" + op_name + "\"");
   mht_4_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_4(mht_4_v, 449, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "GetFuncAttr");

  bool success = op_attrs.Get(attr_name, value);
  if (success) {
    DVLOG(2) << "Caller explicitly specifies " << attr_name.str()
             << (value ? "=true " : "=false, ");
    return;
  }

  const tensorflow::FunctionDef* function_def = func_lib_def.Find(op_name);
  if (function_def == nullptr) {
    return;
  }

  tensorflow::Status status =
      GetNodeAttr(tensorflow::AttrSlice(&function_def->attr()),
                  {attr_name.data(), attr_name.size()}, value);
  if (status.ok()) {
    DVLOG(2) << "Function definition explicitly specifies " << attr_name.str()
             << (value ? "=true" : "=false");
    return;
  }
}

int64_t GetNextLocationId() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_5(mht_5_v, 475, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "GetNextLocationId");

  static std::atomic<int64_t> id(0);
  return id.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

tensorflow::DataType TensorInterface::Type() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_6(mht_6_v, 484, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::Type");

  auto kind = tensor_.get().metadata().dtype;
  if (kind == DType::Unsupported) {
    assert(llvm::isa<tensorflow::tfd::RuntimeFallbackTensor>(tensor_.get()));
    return tensor_.get<tensorflow::tfd::RuntimeFallbackTensor>()
        .GetTensorHandle()
        ->DataType();
  }
  return ConvertDType(kind);
}

int TensorInterface::NumDims() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_7(mht_7_v, 498, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::NumDims");
 return tensor_.get().shape().GetRank(); }

int64_t TensorInterface::Dim(int dim_index) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_8(mht_8_v, 503, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::Dim");

  return tensor_.get().shape().GetDimensionSize(dim_index);
}

int64_t TensorInterface::NumElements() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_9(mht_9_v, 510, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::NumElements");

  if (!tensor_) {
    return static_cast<int64_t>(tf_tensor_.NumElements());
  }
  return tensor_.get().shape().GetNumElements();
}

size_t TensorInterface::ByteSize() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_10(mht_10_v, 520, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::ByteSize");

  return tensor_.get().metadata().GetHostSizeInBytes();
}

void* TensorInterface::Data() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_11(mht_11_v, 527, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::Data");

  if (!tensor_) {
    return tensorflow::TensorCApi::Buffer(tf_tensor_)->data();
  } else {
    auto& tensor = tensor_.get<DenseHostTensor>();
    return tensor.data();
  }
}

// TFRT DenseHostTensor is always aligned
bool TensorInterface::IsAligned() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_12(mht_12_v, 540, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::IsAligned");
 return true; }

bool TensorInterface::CanMove() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_13(mht_13_v, 545, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::CanMove");

  // It is safe to move the Tensor if and only if we own the unique reference to
  // the tensor buffer.
  auto& dht = tensor_.get<DenseHostTensor>();
  return tensor_.IsUnique() && dht.buffer()->IsUnique();
}

std::string TensorInterface::SummarizeValue() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_14(mht_14_v, 555, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::SummarizeValue");

  if (!tensor_) {
    return tf_tensor_.SummarizeValue(/*max_entries=*/3, /*print_v2=*/true);
  } else {
    std::string result;
    llvm::raw_string_ostream result_ostream(result);
    tensor_->Print(result_ostream);
    return result;
  }
}

AsyncValueRef<Tensor> TensorInterface::TensorRef() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_15(mht_15_v, 569, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorInterface::TensorRef");

  return tensor_.CopyRef();
}

TensorHandleInterface::TensorHandleInterface(Value&& v, TfrtContext* context)
    : ImmediateExecutionTensorHandle(kTfrt),
      context_(*context),
      value_(std::move(v)) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_16(mht_16_v, 579, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::TensorHandleInterface");
}

TensorHandleInterface::TensorHandleInterface(tensorflow::DataType dtype,
                                             Value&& v, TfrtContext* context)
    : ImmediateExecutionTensorHandle(kTfrt),
      dtype_(dtype),
      context_(*context),
      value_(std::move(v)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_17(mht_17_v, 589, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::TensorHandleInterface");
}

tensorflow::DataType TensorHandleInterface::DataType() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_18(mht_18_v, 594, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::DataType");

  // If dtype_ field is set, use it instead of waiting for the underlying
  // TensorHandle's metadata to be available.
  if (dtype_) {
    return dtype_.getValue();
  }
  auto metadata = Metadata();
  if (!metadata.hasValue()) {
    LOG(ERROR)
        << "Failed to get DataType due to error metadata: "
        << value_.get<TensorHandle>().GetAsyncMetadata().GetError().message;
    return tensorflow::DT_INVALID;
  }
  auto kind = metadata.getValue()->dtype;
  if (kind == DType::Unsupported) {
    AsyncValue* async_tensor = value_.get<TensorHandle>().GetAsyncTensor();
    if (!async_tensor->IsAvailable()) {
      context_.GetHostContext()->Await(FormRef(async_tensor));
    }

    if (async_tensor->IsError()) {
      LOG(ERROR) << "Failed to get DataType from an error tensor "
                 << async_tensor->GetError().message;
      return tensorflow::DT_INVALID;
    }
    assert(async_tensor->IsType<tensorflow::tfd::RuntimeFallbackTensor>());
    return async_tensor->get<tensorflow::tfd::RuntimeFallbackTensor>()
        .GetTensorHandle()
        ->DataType();
  }
  return ConvertDType(kind);
}

tensorflow::Status TensorHandleInterface::TensorHandleStatus() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_19(mht_19_v, 630, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::TensorHandleStatus");

  if (context_.IsAsync()) {
    return tensorflow::Status::OK();
  } else {
    auto metadata = Metadata();
    if (!metadata.hasValue()) {
      LOG(ERROR)
          << "Metadata in the tensor handle is an error metadata: "
          << value_.get<TensorHandle>().GetAsyncMetadata().GetError().message;
      return tensorflow::errors::Internal(
          value_.get<TensorHandle>().GetAsyncMetadata().GetError().message);
    }

    AsyncValue* async_tensor = value_.get<TensorHandle>().GetAsyncTensor();
    if (!async_tensor->IsAvailable()) {
      context_.GetHostContext()->Await(FormRef(async_tensor));
    }

    if (async_tensor->IsError()) {
      LOG(ERROR) << "Async tensor in the tensor handle is an error tensor: "
                 << async_tensor->GetError().message;
      return tensorflow::errors::Internal(async_tensor->GetError().message);
    }

    return tensorflow::Status::OK();
  }
}

tensorflow::Status TensorHandleInterface::Shape(
    tensorflow::PartialTensorShape* shape) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_20(mht_20_v, 662, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::Shape");

  auto metadata = Metadata();
  if (!metadata.hasValue()) {
    return CreateTfErrorStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  int num_dims = metadata.getValue()->shape.GetRank();
  if (num_dims == -1) {
    return tensorflow::Status::OK();
  }
  llvm::SmallVector<Index, 8> dims;
  metadata.getValue()->shape.GetDimensions(&dims);
  TF_RETURN_IF_ERROR(tensorflow::TensorShapeUtils::MakeShape(dims, shape));
  return tensorflow::Status::OK();
}

tensorflow::Status TensorHandleInterface::NumDims(int* num_dims) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_21(mht_21_v, 681, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::NumDims");

  auto metadata = Metadata();
  if (!metadata.hasValue()) {
    return CreateTfErrorStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *num_dims = metadata.getValue()->shape.GetRank();

  return tensorflow::Status::OK();
}

tensorflow::Status TensorHandleInterface::NumElements(
    int64_t* num_elements) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_22(mht_22_v, 696, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::NumElements");

  auto metadata = Metadata();
  if (!metadata.hasValue()) {
    return CreateTfErrorStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *num_elements = metadata.getValue()->shape.GetNumElements();

  return tensorflow::Status::OK();
}

tensorflow::Status TensorHandleInterface::Dim(int dim_index,
                                              int64_t* dim) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_23(mht_23_v, 711, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::Dim");

  auto metadata = Metadata();
  if (!metadata.hasValue()) {
    return CreateTfErrorStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *dim = metadata.getValue()->shape.GetDimensionSize(dim_index);

  return tensorflow::Status::OK();
}

const char* TensorHandleInterface::DeviceName(
    tensorflow::Status* status) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_24(mht_24_v, 726, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::DeviceName");

  auto& th = value_.get<TensorHandle>();
  if (!th.IsDeviceAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncDevice().CopyRCRef());
  }
  if (th.IsDeviceError()) {
    *status = CreateTfErrorStatus(th.GetAsyncDevice().GetError());
    return nullptr;
  }
  return th.GetAvailableDevice()->name().data();
}

const char* TensorHandleInterface::BackingDeviceName(
    tensorflow::Status* status) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_25(mht_25_v, 742, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::BackingDeviceName");

  return DeviceName(status);
}

const char* TensorHandleInterface::DeviceType(
    tensorflow::Status* status) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_26(mht_26_v, 750, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::DeviceType");

  auto& th = value_.get<TensorHandle>();
  if (!th.IsDeviceAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncDevice().CopyRCRef());
  }
  if (th.IsDeviceError()) {
    *status = CreateTfErrorStatus(th.GetAsyncDevice().GetError());
    return nullptr;
  }
  return th.GetAvailableDevice()->type().name().data();
}

tensorflow::AbstractTensorInterface* TensorHandleInterface::Resolve(
    tensorflow::Status* status) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_27(mht_27_v, 766, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::Resolve");

  auto* host_ctx = context_.GetHostContext();
  auto host_device_ref = host_ctx->GetHostDeviceRef();
  auto& th = value_.get<TensorHandle>();

  auto tensor_av = th.GetAsyncTensor();
  if (!tensor_av->IsAvailable()) {
    host_ctx->Await(FormRef(tensor_av));
  }
  if (auto* error = tensor_av->GetErrorIfPresent()) {
    *status = CreateTfErrorStatus(*error);
    return nullptr;
  }
  assert(th.IsMetadataAvailable());

  if (th.GetAsyncTensor()->get<Tensor>().tensor_type() ==
      StringHostTensor::kTensorType) {
    tensorflow::Tensor tf_tensor =
        tensorflow::tfd::CopyShtToTfTensor(tensor_av->get<StringHostTensor>());
    return new tensorflow::TensorInterface(tf_tensor);
  }

  // Convert the tensor to DenseHostTensor.
  auto req_ctx =
      tfrt::RequestContextBuilder(host_ctx, context_.GetResourceContext())
          .build();
  if (!req_ctx) {
    *status = tensorflow::Status(
        tensorflow::error::Code::UNKNOWN,
        StrCat("Failed to build a RequestContext: ", req_ctx.takeError()));
    return nullptr;
  }
  tfrt::ExecutionContext exec_ctx{std::move(*req_ctx)};
  auto target_th = th.TransferTo(exec_ctx, std::move(host_device_ref),
                                 DenseHostTensor::kTensorType);

  auto target_av = target_th.GetAsyncTensor();
  if (!target_av->IsAvailable()) {
    host_ctx->Await(FormRef(target_av));
  }
  if (target_av->IsError()) {
    *status = tensorflow::Status(
        tensorflow::error::Code::UNKNOWN,
        StrCat("Cannot resolve tensor: ", target_av->GetError().message));
    return nullptr;
  }
  auto host_tensor_ref = target_th.ReleaseTensorRef();
  return new TensorInterface(std::move(host_tensor_ref));
}

llvm::Optional<const TensorMetadata*> TensorHandleInterface::Metadata() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_28(mht_28_v, 819, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "TensorHandleInterface::Metadata");

  auto& th = value_.get<TensorHandle>();
  if (!th.IsMetadataAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncMetadata().CopyRCRef());
  }
  if (th.IsMetadataError()) {
    return llvm::None;
  }
  return &th.GetAvailableMetadata();
}

ContextInterface::ContextInterface(
    const tensorflow::SessionOptions& opts,
    tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async, bool use_tfrt_distributed_runtime)
    : ImmediateExecutionContext(kTfrt),
      context_(opts, default_device_placement_policy, is_async),
      use_tfrt_distributed_runtime_(use_tfrt_distributed_runtime) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_29(mht_29_v, 839, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ContextInterface");

  LOG(INFO) << "TFRT Enabled";
  metrics::AddTFRTVersionMetric();

  op_handler_selector_ = std::make_unique<EagerOpHandlerSelector>(
      GetCoreRuntime(), GetEagerContext(), GetFallbackOpHandler(),
      GetEagerContext()->PinSmallOpsToCPU());

  run_metadata_ = std::make_unique<tensorflow::RunMetadata>();
}

ContextInterface::~ContextInterface() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_30(mht_30_v, 853, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::~ContextInterface");
}

AsyncValueRef<Chain>* ContextInterface::GetChain() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_31(mht_31_v, 858, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetChain");

  auto thread_id = std::this_thread::get_id();
  {
    tensorflow::tf_shared_lock l(chain_map_mu_);
    auto it = thread_local_chain_.find(thread_id);
    if (it != thread_local_chain_.end()) {
      return &it->second;
    }
  }
  {
    tensorflow::mutex_lock l(chain_map_mu_);
    if (thread_local_chain_.find(thread_id) == thread_local_chain_.end()) {
      auto chain = GetReadyChain();
      thread_local_chain_[thread_id] = std::move(chain);
    }
    return &thread_local_chain_[thread_id];
  }
}

template <typename T>
static TensorInterface* MakeScalarTensor(T value, HostContext* host) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_32(mht_32_v, 881, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "MakeScalarTensor");

  // The TensorInterface implementation assumes the tensor is a DenseHostTensor,
  // so we need to use a DenseHostTensor to represent a scalar tensor.
  TensorMetadata md(GetDType<T>(), {});
  auto t = DenseHostTensor::CreateUninitialized(md, host);
  if (!t) {
    LOG(ERROR) << "Failed to create DenseHostTensor";
    return nullptr;
  }
  auto& dht = t.getValue();
  MutableDHTArrayView<T> view{&dht};
  view.Elements()[0] = value;

  return new TensorInterface(
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dht)));
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateInt64Scalar(
    int64_t value) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_33(mht_33_v, 902, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateInt64Scalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateUint64Scalar(
    uint64_t value) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_34(mht_34_v, 910, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateUint64Scalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateInt32Scalar(
    int32_t value) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_35(mht_35_v, 918, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateInt32Scalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateFloatScalar(
    float value) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_36(mht_36_v, 926, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateFloatScalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateDoubleScalar(
    double value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_37(mht_37_v, 934, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateDoubleScalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateHalfScalar(
    Eigen::half value) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_38(mht_38_v, 942, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateHalfScalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateStringScalar(
    tensorflow::tstring value) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("value: \"" + (std::string)value + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_39(mht_39_v, 951, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateStringScalar");

  auto* host = GetHostContext();
  TensorMetadata md(DType(DType::String), {});
  auto t = StringHostTensor::MakeConstructedAsyncValueRef(md, host);
  if (t.IsError()) {
    LOG(ERROR) << "Failed to create StringHostTensor";
    return nullptr;
  }
  t->strings()[0] = value;

  t.SetStateConcrete();
  return new TensorInterface(std::move(t));
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateComplex128Scalar(
    tensorflow::complex128 value) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_40(mht_40_v, 969, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateComplex128Scalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateBoolScalar(
    bool value) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_41(mht_41_v, 977, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateBoolScalar");

  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateTensor(
    tensorflow::DataType dtype, absl::Span<const int64_t> dim_sizes) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_42(mht_42_v, 985, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateTensor");

  std::vector<Index> dimvec(dim_sizes.size());
  for (int i = 0; i < dim_sizes.size(); ++i) {
    dimvec[i] = static_cast<int64_t>(dim_sizes[i]);
  }

  TensorMetadata md;
  switch (dtype) {
    case tensorflow::DT_UINT8:
      md = CreateMetadata(DType::UI8, dimvec);
      break;
    case tensorflow::DT_INT8:
      md = CreateMetadata(DType::I8, dimvec);
      break;
    case tensorflow::DT_INT16:
      md = CreateMetadata(DType::I16, dimvec);
      break;
    case tensorflow::DT_INT32:
      md = CreateMetadata(DType::I32, dimvec);
      break;
    case tensorflow::DT_INT64:
      md = CreateMetadata(DType::I64, dimvec);
      break;
    case tensorflow::DT_HALF:
      md = CreateMetadata(DType::F16, dimvec);
      break;
    case tensorflow::DT_FLOAT:
      md = CreateMetadata(DType::F32, dimvec);
      break;
    case tensorflow::DT_DOUBLE:
      md = CreateMetadata(DType::F64, dimvec);
      break;
    case tensorflow::DT_BOOL:
      md = CreateMetadata(DType::I1, dimvec);
      break;
    case tensorflow::DT_COMPLEX64:
      md = CreateMetadata(DType::Complex64, dimvec);
      break;
    case tensorflow::DT_COMPLEX128:
      md = CreateMetadata(DType::Complex128, dimvec);
      break;
    case tensorflow::DT_VARIANT:
      // Note: TF Python API can create variant tensor for ragged tensor.
      md = CreateMetadata(DType::Variant, dimvec);
      break;
    case tensorflow::DT_STRING:
      // No TFRT Metadata needed for non-scalar string tensors.
      break;
    default:
      LOG(ERROR) << "Cannot create tensor with dtype: " << dtype;
      return nullptr;
  }

  if (dtype == tensorflow::DT_STRING) {
    // Create Tensorflow Tensor as a buffer for tstrings.
    return new TensorInterface(
        tensorflow::Tensor(dtype, tensorflow::TensorShape(dim_sizes)));
  } else {
    auto t = DenseHostTensor::CreateUninitialized(md, GetHostContext());
    return new TensorInterface(MakeAvailableAsyncValueRef<DenseHostTensor>(
        GetHostContext(), std::move(t.getValue())));
  }
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateTensor(
    tensorflow::DataType dtype, const int64_t* dims, int num_dims, void* data,
    size_t len, MemoryReleaser memory_releaser, void* memory_releaser_arg) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_43(mht_43_v, 1054, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateTensor");

  TensorMetadata metadata(ConvertDType(dtype),
                          {dims, static_cast<size_t>(num_dims)});
  RCReference<HostBuffer> buffer = HostBuffer::CreateFromExternal(
      data, len,
      [memory_releaser, memory_releaser_arg](void* data, size_t len) {
        memory_releaser(data, len, memory_releaser_arg);
      });
  AsyncValueRef<DenseHostTensor> dht =
      MakeConstructedAsyncValueRef<DenseHostTensor>(GetHostContext(), metadata,
                                                    std::move(buffer));

  dht.SetStateConcrete();
  return new TensorInterface(std::move(dht));
}

bool ContextInterface::UsesTFRT() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_44(mht_44_v, 1073, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::UsesTFRT");
 return true; }

tensorflow::ImmediateExecutionTensorHandle* ContextInterface::CreateLocalHandle(
    tensorflow::AbstractTensorInterface* t) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_45(mht_45_v, 1079, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateLocalHandle");

  auto* tensor_interface = down_cast<TensorInterface*>(t);
  auto* host = GetHostContext();

  // Create RuntimeFallbackTensor from a TF Tensor, and then create
  // the according TensorHandleInterface.
  if (tensor_interface->IsTfTensor()) {
    tensorflow::tfd::OwnedTensorHandle tf_tensor_handle{
        tensorflow::TensorHandle::CreateLocalHandle(
            tensor_interface->TfTensor())};

    auto expected_result_tensor =
        tensorflow::tfd::CreateRuntimeFallbackTensorFromTfTensorHandle(
            std::move(tf_tensor_handle), GetHostContext());

    if (expected_result_tensor) {
      return new TensorHandleInterface(
          Value(TensorHandle(
              host->GetHostDeviceRef(), expected_result_tensor.get().metadata(),
              MakeAvailableAsyncValueRef<
                  tensorflow::tfd::RuntimeFallbackTensor>(
                  host, std::move(expected_result_tensor.get())))),
          GetTfrtContext());
    } else {
      return new TensorHandleInterface(
          Value(TensorHandle::CreateError(MakeErrorAsyncValueRef(
              GetHostContext(), StrCat(expected_result_tensor.takeError())))),
          GetTfrtContext());
    }
  }

  auto tensor_av = tensor_interface->TensorRef();
  const TensorMetadata& md = tensor_av.get<Tensor>().metadata();

  // NOTE(fishx): Following logic is needed to let TF-TFRT fully reach
  // performance parity with current TF. This API is used to by tf.constant
  // to convert Python object to **CPU** Tensor. tf.constant in current TF
  // heavily depends on Tensor Mirroring feature for good performance. However,
  // TFRT does not have Tensor Mirroring feature. In order to use Tensor
  // Mirroring from current TF runtime, we convert the result of tf.constant to
  // Fallback Tensor.

  if (tensor_av.IsAvailable()) {
    if (auto* dht = llvm::dyn_cast<DenseHostTensor>(&tensor_av.get<Tensor>())) {
      return new TensorHandleInterface(
          Value(TensorHandle(
              host->GetHostDeviceRef(), md,
              MakeAvailableAsyncValueRef<
                  tensorflow::tfd::RuntimeFallbackTensor>(
                  host, tensorflow::tfd::CopyRefDHTToRuntimeFallbackTensor(
                            *dht, host)))),
          GetTfrtContext());
    }
  } else {
    auto result_tensor = MakeIndirectAsyncValue(host);
    tensor_av.AndThen([host, result_tensor = result_tensor,
                       tensor_av = tensor_av.CopyRef()]() {
      if (auto* dht =
              llvm::dyn_cast<DenseHostTensor>(&tensor_av.get<Tensor>())) {
        result_tensor->ForwardTo(
            MakeAvailableAsyncValueRef<tensorflow::tfd::RuntimeFallbackTensor>(
                host, tensorflow::tfd::CopyRefDHTToRuntimeFallbackTensor(
                          *dht, host)));
      } else {
        result_tensor->ForwardTo(tensor_av.CopyRef());
      }
    });
    return new TensorHandleInterface(
        Value(TensorHandle(host->GetHostDeviceRef(), md,
                           AsyncValueRef<Tensor>(std::move(result_tensor)))),
        GetTfrtContext());
  }
  return new TensorHandleInterface(
      Value(TensorHandle(host->GetHostDeviceRef(), md, std::move(tensor_av))),
      GetTfrtContext());
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::CreateLocalHandleFromTFTensor(tensorflow::Tensor& t,
                                                const char* d_name) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("d_name: \"" + (d_name == nullptr ? std::string("nullptr") : std::string((char*)d_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_46(mht_46_v, 1162, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateLocalHandleFromTFTensor");

  auto* host = GetHostContext();
  // Create RuntimeFallbackTensor from a TF Tensor, and then create
  // the according TensorHandleInterface.
  tensorflow::tfd::OwnedTensorHandle tf_tensor_handle{
      tensorflow::TensorHandle::CreateLocalHandle(std::move(t))};

  tfrt::Expected<tensorflow::tfd::RuntimeFallbackTensor>
      expected_result_tensor =
          tensorflow::tfd::CreateRuntimeFallbackTensorFromTfTensorHandle(
              std::move(tf_tensor_handle), GetHostContext());

  if (expected_result_tensor) {
    return new TensorHandleInterface(
        Value(TensorHandle(
            host->GetHostDeviceRef(), expected_result_tensor.get().metadata(),
            MakeAvailableAsyncValueRef<tensorflow::tfd::RuntimeFallbackTensor>(
                host, std::move(expected_result_tensor.get())))),
        GetTfrtContext());
  } else {
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(MakeErrorAsyncValueRef(
            GetHostContext(), StrCat(expected_result_tensor.takeError())))),
        GetTfrtContext());
  }
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::TFTensorHandleFromInterface(
    tensorflow::ImmediateExecutionTensorHandle* handle) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_47(mht_47_v, 1194, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::TFTensorHandleFromInterface");

  TensorHandle th = tfrt::tf::TensorHandleFromInterface(handle)->Handle();
  AsyncValue* tensor_av = th.GetAsyncTensor();
  if (tensor_av->IsUnavailable()) GetHostContext()->Await(FormRef(tensor_av));

  auto& tensor = th.GetAsyncTensor()->get<Tensor>();

  if (auto* rtfbt =
          llvm::dyn_cast<tensorflow::tfd::RuntimeFallbackTensor>(&tensor))
    return rtfbt->GetTensorHandle();

  if (auto* dht = llvm::dyn_cast<tfrt::DenseHostTensor>(&tensor)) {
    return tensorflow::TensorHandle::CreateLocalHandle(
        tensorflow::tfd::MoveHostBufferToTfTensor(dht->buffer(), dht->dtype(),
                                                  dht->shape()));
  }

  if (auto* sht = llvm::dyn_cast<tfrt::StringHostTensor>(&tensor)) {
    return tensorflow::TensorHandle::CreateLocalHandle(
        tensorflow::tfd::CopyShtToTfTensor(*sht));
  }

  LOG(ERROR) << "Unsupported tensor type";
  return nullptr;
}

tensorflow::ImmediateExecutionOperation* ContextInterface::CreateOperation() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_48(mht_48_v, 1223, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CreateOperation");

  return new OperationInterface(this);
}

// TODO(srbs): Change this to directly fetch the MLIR function once that is
// supported.
tensorflow::Status ContextInterface::RegisterFunction(
    tensorflow::AbstractFunction* f) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_49(mht_49_v, 1233, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::RegisterFunction");

  tensorflow::FunctionDef* fdef;
  TF_RETURN_IF_ERROR(f->GetFunctionDef(&fdef));
  if (!fdef) {
    return tensorflow::errors::InvalidArgument(
        "GetFunctionDef returned nullptr.");
  }
  return AddFunctionDef(*fdef);
}

void ContextInterface::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* devices) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_50(mht_50_v, 1247, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ListDevices");

  context_.GetEagerContext()->ListDevices(devices);
}

tensorflow::Status ContextInterface::AddDevices(
    std::vector<std::unique_ptr<tensorflow::Device>> devices) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_51(mht_51_v, 1255, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::AddDevices");

  if (!devices.empty() && devices[0]->device_type() != "CPU")
    return tensorflow::errors::InvalidArgument(
        "Device: ", devices[0]->device_type(), " is not allowed to be added ",
        "after the context is initialized. Currently allowed device: CPU. ",
        "May update this API to allow adding more types of devices.");

  for (const auto& d : devices) {
    GetHostContext()->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new CpuDevice(d->name())));
  }
  TF_RETURN_IF_ERROR(GetEagerContext()->AddDevices(std::move(devices)));

  return tensorflow::Status::OK();
}

void ContextInterface::ClearCachesAndThreadExecutors() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_52(mht_52_v, 1274, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ClearCachesAndThreadExecutors");

  GetEagerContext()->ClearCachesAndThreadExecutors();
  GetHostContext()->Quiesce();
}

void ContextInterface::StartStep() {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_53(mht_53_v, 1282, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::StartStep");
 GetEagerContext()->StartStep(); }

void ContextInterface::EndStep() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_54(mht_54_v, 1287, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::EndStep");
 GetEagerContext()->EndStep(); }

tensorflow::Status ContextInterface::EnableCollectiveOps(
    const tensorflow::ServerDef& server_def) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_55(mht_55_v, 1293, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::EnableCollectiveOps");

  if (use_tfrt_distributed_runtime_) {
    return distributed_manager_->EnableCollectiveOps(server_def);
  }
  // Preserve the local virtual device names, since local virtual devices are
  // added by TFRT and we need to add it back after worker server is
  // initialized. Currently one such use case is the TPU_SYSTEM device, which
  // is a virtual device specifically used to initialize TPUs.
  std::vector<std::string> virtual_device_names;

  for (const auto& d :
       GetHostContext()->GetDeviceManager()->ListDevices<Device>()) {
    if (d->IsDeviceType(tfrt::VirtualDevice::kDeviceType)) {
      tensorflow::DeviceNameUtils::ParsedName p;
      if (!tensorflow::DeviceNameUtils::ParseFullName(d->name().str(), &p)) {
        return tensorflow::errors::InvalidArgument(
            "Invalid local virtual device name: ", d->name().str());
      }

      virtual_device_names.push_back(tensorflow::DeviceNameUtils::FullName(
          server_def.job_name(), /*replica=*/0, server_def.task_index(), p.type,
          p.id));
    }
  }

  TF_RETURN_IF_ERROR(GetEagerContext()->EnableCollectiveOps(server_def));

  // Create new devices with updated device name.
  std::vector<std::unique_ptr<tensorflow::Device>> dummy_tf_devices;
  CreateDummyTfDevices(virtual_device_names, &dummy_tf_devices);

  std::string name_prefix =
      absl::StrCat("/job:", server_def.job_name(),
                   "/replica:0/task:", server_def.task_index());

  // Update host device in TFRT HostContext.
  GetHostContext()->ResetHostDevice(
      GetHostContext()
          ->GetDeviceManager()
          ->MaybeAddDevice(TakeRef(
              new CpuDevice(absl::StrCat(name_prefix, "/device:CPU:0"))))
          .release());

  // Update virtual devices in TFRT HostContext.
  AddDummyTfrtDevices(virtual_device_names, GetHostContext());

  // Update eager context's device manager.
  auto* local_device_mgr = dynamic_cast<tensorflow::DynamicDeviceMgr*>(
      GetEagerContext()->local_device_mgr());
  TF_RETURN_IF_ERROR(local_device_mgr->AddDevices(std::move(dummy_tf_devices)));

  return tensorflow::Status::OK();
}

tensorflow::Status ContextInterface::BuildFunctionRequestContext(
    tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
    RCReference<tfrt::RequestContext>* request_context) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_56(mht_56_v, 1352, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::BuildFunctionRequestContext");

  auto* step_container = GetEagerContext()->StepContainer();
  RequestContextBuilder request_context_builder(
      GetHostContext(), GetResourceContext(), step_container->StepId());

  TF_RETURN_IF_ERROR(tensorflow::tfd::SetUpKernelFallbackCompatRequestContext(
      &request_context_builder, runner_table, GetEagerContext()));
  if (distributed_manager_ != nullptr) {
    down_cast<DistributedManagerContextInterface*>(distributed_manager_.get())
        ->UpdateRequestContextBuilder(&request_context_builder);
  }
  auto expected_request_context = std::move(request_context_builder).build();
  if (!expected_request_context) {
    return tensorflow::errors::Internal(
        StrCat(expected_request_context.takeError()));
  }
  *request_context = std::move(expected_request_context.get());
  return tensorflow::Status::OK();
}

tensorflow::Status ContextInterface::BuildOpRequestContext(
    RCReference<tfrt::RequestContext>* request_context) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_57(mht_57_v, 1376, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::BuildOpRequestContext");

  return BuildFunctionRequestContext(/*runner_table=*/nullptr, request_context);
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::CopyTensorHandleToDevice(
    tensorflow::ImmediateExecutionTensorHandle* handle, const char* device_name,
    tensorflow::Status* status) {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_58(mht_58_v, 1387, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::CopyTensorHandleToDevice");

  auto* host_ctx = GetHostContext();

  TensorHandle src_th = tfrt::tf::TensorHandleFromInterface(handle)->Handle();

  auto tfrt_device_name =
      ConvertTfDeviceNameToTfrt(device_name, GetEagerContext());
  if (!tfrt_device_name) {
    *status = tensorflow::errors::InvalidArgument(
        StrCat(tfrt_device_name.takeError()));
    RCReference<AsyncValue> error_av =
        MakeErrorAsyncValueRef(host_ctx, status->error_message());
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(std::move(error_av))),
        GetTfrtContext());
  }
  auto dst_device_ref = host_ctx->GetDeviceManager()->GetDeviceRef<Device>(
      tfrt_device_name.get());
  if (!dst_device_ref) {
    std::string error_message =
        tfrt::StrCat("Failed to find destination device with name: ",
                     tfrt_device_name.get());
    *status = tensorflow::errors::Internal(error_message);
    RCReference<AsyncValue> error_av =
        MakeErrorAsyncValueRef(host_ctx, error_message);
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(std::move(error_av))),
        GetTfrtContext());
  }

  RCReference<RequestContext> request_ctx;
  *status = BuildOpRequestContext(&request_ctx);
  if (!status->ok()) return nullptr;

  ExecutionContext exec_ctx{std::move(request_ctx)};

  auto target_th =
      src_th.TransferToInferredType(exec_ctx, std::move(dst_device_ref));

  auto target_av = target_th.GetAsyncTensor();
  if (target_av->IsError()) {
    *status = tensorflow::errors::Internal(
        tfrt::StrCat("Copying to device <", tfrt_device_name.get(),
                     "> failed: ", target_av->GetError().message));
    return nullptr;
  }
  return new TensorHandleInterface(Value(target_th.CopyRef()),
                                   GetTfrtContext());
}

tensorflow::Status ContextInterface::AddFunctionDef(
    const tensorflow::FunctionDef& fdef) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_59(mht_59_v, 1441, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::AddFunctionDef");

  return GetEagerContext()->AddFunctionDef(fdef);
}

tensorflow::Status ContextInterface::AddFunctionDefWithStackTraces(
    const tensorflow::FunctionDef& fdef,
    const tensorflow::StackTracesMap& stack_traces) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_60(mht_60_v, 1450, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::AddFunctionDefWithStackTraces");

  return GetEagerContext()->AddFunctionDefWithStackTraces(fdef, stack_traces);
}

std::vector<std::string> ContextInterface::ListFunctionNames() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_61(mht_61_v, 1457, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ListFunctionNames");

  return GetEagerContext()->ListFunctionNames();
}

tensorflow::Status ContextInterface::RemoveFunction(const std::string& func) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_62(mht_62_v, 1465, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::RemoveFunction");

  // TODO(tfrt-devs): We need to ensure all invocations of this function is
  // finished before removing it.
  function_cache_.RemoveFunction(func);
  return GetEagerContext()->RemoveFunction(func);
}

const tensorflow::FunctionDef* ContextInterface::FindFunctionDef(
    const std::string& name) const {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_63(mht_63_v, 1477, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::FindFunctionDef");

  return GetEagerContext()->FindFunctionDef(name);
}

const tensorflow::DeviceNameUtils::ParsedName&
ContextInterface::HostCPUParsedName() const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_64(mht_64_v, 1485, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::HostCPUParsedName");

  return context_.HostCPUParsedName();
}

const std::string& ContextInterface::HostCPUName() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_65(mht_65_v, 1492, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::HostCPUName");

  return context_.GetEagerContext()->HostCPUName();
}

tensorflow::CustomDeviceOpHandler&
ContextInterface::GetCustomDeviceOpHandler() {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_66(mht_66_v, 1500, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetCustomDeviceOpHandler");

  return context_.GetEagerContext()->GetCustomDeviceOpHandler();
}

tensorflow::Status ContextInterface::RegisterCustomDevice(
    const std::string& name, std::unique_ptr<tensorflow::CustomDevice> device) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_67(mht_67_v, 1509, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::RegisterCustomDevice");

  return context_.GetEagerContext()->RegisterCustomDevice(name,
                                                          std::move(device));
}

tensorflow::FunctionLibraryDefinition* ContextInterface::FuncLibDef() {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_68(mht_68_v, 1517, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::FuncLibDef");

  return context_.GetEagerContext()->FuncLibDef();
}

void ContextInterface::SetReuseRendezvousForFunctions(
    bool reuse_rendezvous_for_functions) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_69(mht_69_v, 1525, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::SetReuseRendezvousForFunctions");

  // TODO(fishx): This feature doesn't work properly in TFRT yet. Fix it.
  context_.GetEagerContext()->SetReuseRendezvousForFunctions(
      reuse_rendezvous_for_functions);
}

void ContextInterface::ResetGlobalRendezvousForFunction() {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_70(mht_70_v, 1534, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ResetGlobalRendezvousForFunction");

  context_.GetEagerContext()->ResetGlobalRendezvousForFunction();
}

std::vector<std::string> ContextInterface::GetLoggedOpsTestonly() {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_71(mht_71_v, 1541, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetLoggedOpsTestonly");

  const auto& ret = GetHostContext()
                        ->GetOrCreateSharedContext<tensorflow::tfd::OpLogger>()
                        .GetLoggedOps();
  return std::vector<std::string>(ret.begin(), ret.end());
}

HostContext* ContextInterface::GetHostContext() {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_72(mht_72_v, 1551, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetHostContext");

  return GetCoreRuntime()->GetHostContext();
}

tensorflow::EagerContext* ContextInterface::GetEagerContext() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_73(mht_73_v, 1558, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetEagerContext");

  return context_.GetEagerContext();
}

const tensorflow::EagerContext* ContextInterface::GetEagerContext() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_74(mht_74_v, 1565, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetEagerContext");

  return context_.GetEagerContext();
}

CoreRuntime* ContextInterface::GetCoreRuntime() {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_75(mht_75_v, 1572, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetCoreRuntime");

  return context_.GetCoreRuntime();
}

TfrtContext* ContextInterface::GetTfrtContext() {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_76(mht_76_v, 1579, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetTfrtContext");
 return &context_; }

OpHandler* ContextInterface::GetFallbackOpHandler() {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_77(mht_77_v, 1584, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetFallbackOpHandler");

  return context_.GetFallbackOpHandler();
}

ResourceContext* ContextInterface::GetResourceContext() {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_78(mht_78_v, 1591, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::GetResourceContext");

  return context_.GetResourceContext();
}

tensorflow::Status ContextInterface::SelectOpHandlerFromArguments(
    const tensorflow::ImmediateExecutionOperation& op, OpHandler** op_handler) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_79(mht_79_v, 1599, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::SelectOpHandlerFromArguments");

  return op_handler_selector_->SelectFromArguments(op, op_handler);
}

tensorflow::Status ContextInterface::SelectOpHandlerFromNodeDef(
    const tensorflow::ImmediateExecutionOperation& op, const NodeDef* node_def,
    OpHandler** op_handler) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_80(mht_80_v, 1608, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::SelectOpHandlerFromNodeDef");

  return op_handler_selector_->SelectFromNodeDef(op, node_def, op_handler);
}

std::unique_ptr<tensorflow::RunMetadata> ContextInterface::ExportRunMetadata() {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_81(mht_81_v, 1615, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::ExportRunMetadata");

  mutex_lock l(run_metadata_mu_);

  // NOTE(fishx): We need to merge run_metadata from TF Eager Context because
  // right now we still use current TF runtime to execute graph (e.g. tf.data
  // via fallback).
  auto result = GetEagerContext()->ExportRunMetadata();
  result->MergeFrom(*run_metadata_);
  run_metadata_ = std::make_unique<tensorflow::RunMetadata>();

  return result;
}

tensorflow::Status ContextInterface::RunMetadataRecordFunction(
    const std::string& func_name) {
   std::vector<std::string> mht_82_v;
   mht_82_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_82(mht_82_v, 1633, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::RunMetadataRecordFunction");

  const tensorflow::FunctionDef* fdef =
      GetEagerContext()->FindFunctionDef(func_name);
  if (fdef == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "Failed to find function \"", func_name, "\" in function library");
  }
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fdef, tensorflow::AttrSlice(), GetEagerContext()->FuncLibDef(), &fbody));
  tensorflow::GraphDef def;
  fbody->graph->ToGraphDef(&def);
  *def.mutable_library() =
      GetEagerContext()->FuncLibDef()->ReachableDefinitions(def).ToProto();

  mutex_lock l(run_metadata_mu_);
  auto* function_graphs = run_metadata_->add_function_graphs();
  *function_graphs->mutable_pre_optimization_graph() = def;
  // TODO(b/b/171600738): Figure out a way to record the right post optimization
  // graph and partition graph.
  *function_graphs->mutable_post_optimization_graph() = def;
  *function_graphs->add_partition_graphs() = def;
  *run_metadata_->add_partition_graphs() = def;
  return tensorflow::Status::OK();
}

void ContextInterface::SetExecutorForThread(
    tensorflow::EagerExecutor* executor) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_83(mht_83_v, 1663, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "ContextInterface::SetExecutorForThread");

  GetEagerContext()->SetExecutorForThread(executor);
}

tfrt::Location AbortLocationHandler::GetCurrentLocation() {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_84(mht_84_v, 1670, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "AbortLocationHandler::GetCurrentLocation");

  return tfrt::Location(this, GetNextLocationId());
}

void OpAttrsInterface::GetNameAttrList(
    tensorflow::NameAttrList* name_and_attrs) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_85(mht_85_v, 1678, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetNameAttrList");

  fallback_attrs_->FillAttrValueMap(name_and_attrs->mutable_attr());
  name_and_attrs->set_name(fallback_attrs_->op_name());
}

Status OpAttrsInterface::GetTypeList(
    absl::string_view attr_name,
    absl::InlinedVector<tensorflow::DataType, 4>* type_list) const {
   std::vector<std::string> mht_86_v;
   mht_86_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_86(mht_86_v, 1689, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetTypeList");

  return tensorflow::errors::Unimplemented("OpAttrsInterface::GetTypeList");
}

bool OpAttrsInterface::GetInt(absl::string_view attr_name,
                              int64_t* result) const {
   std::vector<std::string> mht_87_v;
   mht_87_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_87(mht_87_v, 1698, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetInt");

  return attrs_->Get<int64_t>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetFloat(absl::string_view attr_name,
                                float* result) const {
   std::vector<std::string> mht_88_v;
   mht_88_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_88(mht_88_v, 1707, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetFloat");

  return attrs_->Get<float>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetBool(absl::string_view attr_name,
                               bool* result) const {
   std::vector<std::string> mht_89_v;
   mht_89_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_89(mht_89_v, 1716, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetBool");

  return attrs_->Get<bool>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetType(absl::string_view attr_name,
                               tensorflow::DataType* result) const {
   std::vector<std::string> mht_90_v;
   mht_90_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_90(mht_90_v, 1725, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OpAttrsInterface::GetType");

  auto optional_type =
      attrs_->GetOptional<OpAttrType>({attr_name.data(), attr_name.size()});
  if (!optional_type.hasValue()) return false;
  *result = tensorflow::tfd::ConvertToTfDataType(optional_type.getValue());
  return true;
}

OperationInterface::OperationInterface(ContextInterface* context)
    : ImmediateExecutionOperation(kTfrt),
      op_attrs_(&attrs_, &fallback_attrs_),
      context_(context) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_91(mht_91_v, 1739, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::OperationInterface");
}

tensorflow::Status OperationInterface::Reset(const char* op,
                                             const char* raw_device_name) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_92_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_92(mht_92_v, 1747, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::Reset");

  op_name_ = op;
  args_.clear();
  attrs_.Reset();
  custom_device_tensor_handle_count_ = 0;
  op_def_ = nullptr;
  fallback_attrs_.Reset(op);
  stack_trace_.reset();
  op_ = nullptr;
  function_state_.reset();
  tensorflow::Status s = tensorflow::OpDefForOp(op_name_, &op_def_);
  is_function_ = !s.ok();
  return SetDeviceName(raw_device_name);
}

tensorflow::Status OperationInterface::Execute(
    absl::Span<tensorflow::AbstractTensorHandle*> retvals, int* num_retvals) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_93(mht_93_v, 1766, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::Execute");

  tensorflow::profiler::TraceMe trace(
      [&] {
        return absl::StrCat("TFRT_Execute:", Name(), " device:", DeviceName());
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
  if (custom_device_tensor_handle_count_ > 0) {
    return tensorflow::errors::InvalidArgument(
        "Cannot execute ops that conntains unsupported arg in TFRT.");
  }

  TF_RETURN_IF_ERROR(Initialize());
  assert(op_ != nullptr || function_state_);
  auto* corert = context_->GetCoreRuntime();
  auto* chain = context_->GetChain();
  auto* host = corert->GetHostContext();
  llvm::SmallVector<TensorHandle, 8> th_args;
  th_args.reserve(args_.size());

  llvm::SmallVector<TensorHandle, 8> result_ths;
  result_ths.resize(*num_retvals);

  if (function_state_) {
    // Set up arguments. Check argument dtype synchronously if available.
    auto arg_types = function_state_->GetArgTypes();
    if (args_.size() != arg_types.size()) {
      return tensorflow::errors::InvalidArgument("Expects ", arg_types.size(),
                                                 " arguments, but ",
                                                 args_.size(), " is provided");
    }
    auto args_size = args_.size();
    for (auto i = 0; i < args_size; ++i) {
      th_args.push_back(down_cast<TensorHandleInterface*>(args_[i].get())
                            ->Handle()
                            .CopyRef());
      // TODO(b/173556766): This dtype check is only needed for corert lowering.
      // In native lowering, compiler should obtain the argument dtype
      // information from FunctionBody directly and lower the op to the native
      // kernel that accepts the specified dtype.
      if (th_args[i].IsMetadataAvailable()) {
        auto arg_dtype = th_args[i].GetAvailableMetadata().dtype;
        if (arg_dtype != arg_types[i]) {
          return tensorflow::errors::InvalidArgument(
              "Expects arg[", i, "] to be ", arg_types[i], " but ", arg_dtype,
              " is provided");
        }
      }
    }

    RCReference<RequestContext> request_ctx;
    TF_RETURN_IF_ERROR(context_->BuildFunctionRequestContext(
        function_state_->GetRunnerTable(), &request_ctx));

    ExecutionContext exec_ctx{std::move(request_ctx),
                              abort_location_handler_.GetCurrentLocation()};

    // Make BEF executor to use TfThreadPoolWorkQueue to dispatch kernels.
    exec_ctx.set_work_queue(
        context_->GetTfrtContext()->GetTfThreadPoolWorkQueue());

    // Execute the function.
    function_state_->GetFunc()(exec_ctx, th_args, OpAttrsRef(attrs_),
                               result_ths, chain);
  } else {
    RCReference<RequestContext> request_ctx;
    TF_RETURN_IF_ERROR(context_->BuildOpRequestContext(&request_ctx));

    ExecutionContext exec_ctx{std::move(request_ctx),
                              abort_location_handler_.GetCurrentLocation()};
    for (auto& arg : args_) {
      th_args.push_back(
          down_cast<TensorHandleInterface*>(arg.get())->Handle().CopyRef());
    }
    // If the CoreRuntimeOp is a native TFRT op, transfer arguments to target
    // device if necessary.
    if (!op_->IsFallback()) {
      // Get the target device of the arguments that we want to implicitly copy
      // to.
      auto dst_device_ref = op_->GetDeviceRef();

      for (auto& th_arg : th_args) {
        th_arg =
            th_arg.TransferTo(exec_ctx, dst_device_ref, op_->GetTensorType());
      }
    }

    (*op_)(exec_ctx, th_args, OpAttrsRef(attrs_), result_ths, chain);
  }

  tensorflow::Status s = tensorflow::Status::OK();

  if (TF_PREDICT_FALSE(!this->context_->IsAsync() && !chain->IsAvailable()))
    host->Await({chain->CopyRCRef()});

  if (TF_PREDICT_FALSE(chain->IsError())) {
    s = CreateTfErrorStatus(chain->GetError());
    // TODO(tfrt-devs): Assess if we need a explicit API to clear error.
    *chain = GetReadyChain();
  }

  for (size_t i = 0, e = result_ths.size(); i != e; ++i) {
    auto& th_ref = result_ths[i];
    if (TF_PREDICT_FALSE(!this->context_->IsAsync() &&
                         !th_ref.GetAsyncTensor()->IsAvailable()))
      host->Await(FormRef(th_ref.GetAsyncTensor()));

    // NOTE(fishx): In async mode, we won't report error synchronously even
    // though it is possible in TFRT. This is intended to match behavior in
    // current TF. However, in the future, we may want to update this
    // behavior since synchronous error may improve user experience in async
    // mode.
    if (TF_PREDICT_FALSE(!this->context_->IsAsync() &&
                         th_ref.GetAsyncTensor()->IsError() && s.ok()))
      s = CreateTfErrorStatus(th_ref.GetAsyncTensor()->GetError());

    if (function_state_ && context_->IsAsync()) {
      retvals[i] = new TensorHandleInterface(function_state_->GetRetTypes()[i],
                                             Value(std::move(result_ths[i])),
                                             context_->GetTfrtContext());
    } else {
      retvals[i] = new TensorHandleInterface(Value(std::move(result_ths[i])),
                                             context_->GetTfrtContext());
    }
  }

  return s;
}

tensorflow::Status OperationInterface::Initialize() {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_94(mht_94_v, 1897, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::Initialize");

  CoreRuntime* corert = context_->GetCoreRuntime();
  if (!is_function_) {
    // Obtain input arguments' dtype attrs as part of the cache key.
    llvm::SmallVector<string_view, 4> dtypes;
    attrs_.IterateEntries([&](const OpAttrsRawEntry& entry) {
      if (entry.type == OpAttrType::DTYPE && !entry.IsArray())
        dtypes.push_back(
            GetNameString(*static_cast<const OpAttrType*>(entry.GetData())));
    });

    OpHandler* op_handler = nullptr;
    TF_RETURN_IF_ERROR(
        context_->SelectOpHandlerFromArguments(*this, &op_handler));
    Expected<CoreRuntimeOp*> expected_op = context_->GetOpCache().GetOrAddOp(
        op_name_, op_handler, device_name_, dtypes, this);
    if (!expected_op) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Cannot obtain CoreRuntimeOp: ", op_name_,
                 " on device: ", device_name_, expected_op.takeError()));
    }
    op_ = expected_op.get();
    // Update device name since op_handler_selecter may choose an op_handler
    // that's different from what the user specifies.
    device_name_ = op_->DeviceName().str();
    return tensorflow::Status::OK();
  }

  bool compile_with_xla = false;
  GetFuncAttr(attrs_, op_name_, *context_->GetEagerContext()->FuncLibDef(),
              tensorflow::kXlaMustCompileAttr, &compile_with_xla);
  // If the function has compile_with_xla==true, we will use RuntimeFallback
  // to execute it, since TFRT does not support xla yet.
  // TODO(tfrt-devs): Native support of compile_with_xla.
  if (compile_with_xla) {
    Expected<CoreRuntimeOp*> expected_op =
        context_->GetOpCache().GetOrAddXlaOp(op_name_, context_);
    if (!expected_op) {
      return tensorflow::errors::NotFound(
          StrCat("Cannot initialize xla function ", op_name_,
                 " on fallback op handler.", expected_op.takeError()));
    }
    op_ = expected_op.get();
    return tensorflow::Status::OK();
  }

  // Note(fishx): We need eager context for now because we need
  // FunctionLibraryDefinition to convert FunctionDef to MLIR TF dialect. In
  // the future, when we can generate MLIR from TF Python, we should get rid of
  // this.
  // FunctionDef -> BEF.
  // Look up the cache. Compile BEF and insert to cache if miss.
  tensorflow::DeviceSet dev_set;
  const DeviceMgr* device_mgr = context_->GetEagerContext()->local_device_mgr();
  if (device_mgr == nullptr)
    return tensorflow::errors::NotFound("Cannot find device manager");
  // TODO(tfrt-devs): support remote devices in TFRT.
  for (auto d : device_mgr->ListDevices()) dev_set.AddDevice(d);
  if (context_->GetDistributedManager() != nullptr &&
      context_->UseTfrtDistributedRuntime()) {
    down_cast<DistributedManagerContextInterface*>(
        context_->GetDistributedManager())
        ->PopulateRemoteDevices(&dev_set);
  }
  FunctionCache::FunctionCacheResult result;

  tensorflow::TfrtFunctionCompileOptions compile_options;

  // Use the host device if the user does not place the function to a specific
  // device.
  compile_options.default_device =
      device_name_.empty() ? context_->GetEagerContext()->HostCPUName()
                           : device_name_;

  // TODO(b/172659131): Do not use TFRT native ops for TF integration for now.
  // Re-enable once we have a concrete plan to implement feature complete
  // TFRT native ops (kernels).
  compile_options.enable_native_ops = false;

  if (fallback_attrs_.NumAttributes() > 0) {
    const auto& ndef = NodeDef();
    // TODO(tfrt-devs): If we are to create more attributes, consider packing
    // them into a proto.
    {
      const auto& it = ndef.attr().find(kEnableNativeOpsAttr);
      if (it != ndef.attr().end()) {
        compile_options.enable_native_ops = it->second.b();
      }
    }

    {
      const auto& it = ndef.attr().find(kEnableGrapplerAttr);
      if (it != ndef.attr().end()) {
        compile_options.enable_grappler = it->second.b();
      }
    }
  }

  llvm::SmallVector<const tfrt::Device*, 4> input_devices;
  input_devices.reserve(args_.size());
  for (auto& arg : args_) {
    auto arg_th = down_cast<TensorHandleInterface*>(arg.get())->Handle();
    if (!arg_th.IsDeviceAvailable()) {
      corert->GetHostContext()->Await(arg_th.GetAsyncDevice().CopyRCRef());
    }
    input_devices.push_back(down_cast<TensorHandleInterface*>(arg.get())
                                ->Handle()
                                .GetAvailableDevice()
                                .get());
  }
  TF_RETURN_IF_ERROR(context_->GetFunctionCache().GetOrAddFunction(
      op_name_, device_name_, dev_set, context_->GetEagerContext(), corert,
      /*request_ctx_fn=*/
      [this](tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
             RCReference<RequestContext>* request_ctx) {
        return context_->BuildFunctionRequestContext(runner_table, request_ctx);
      },
      abort_location_handler_.GetCurrentLocation(), compile_options,
      input_devices, &result));
  // TODO(tfrt-devs): Avoid calling EagerContext::ShouldStoreGraphs().
  if (result.is_cache_miss &&
      context_->GetEagerContext()->ShouldStoreGraphs()) {
    TF_RETURN_IF_ERROR(context_->RunMetadataRecordFunction(op_name_));
  }
  function_state_ = std::move(result.function_state);
  return tensorflow::Status::OK();
}

tensorflow::Status OperationInterface::SetDeviceName(const char* name) {
   std::vector<std::string> mht_95_v;
   mht_95_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_95(mht_95_v, 2029, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetDeviceName");

  if (op_ && name != device_name_) {
    return tensorflow::errors::Internal(
        "Failed to update device name. Right now TFRT cannot update device "
        "name of a fallback op if it is initialized.");
  }
  device_name_ = name ? name : "";
  return tensorflow::Status::OK();
}

tensorflow::Status OperationInterface::AddInput(
    tensorflow::AbstractTensorHandle* input) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_96(mht_96_v, 2043, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::AddInput");

  tensorflow::ImmediateExecutionTensorHandle* h =
      down_cast<tensorflow::ImmediateExecutionTensorHandle*>(input);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(h)) {
    custom_device_tensor_handle_count_++;
  }
  h->Ref();
  args_.push_back(
      tensorflow::core::RefCountPtr<tensorflow::ImmediateExecutionTensorHandle>(
          h));
  return tensorflow::Status::OK();
}

tensorflow::Status OperationInterface::SetInput(
    size_t index, tensorflow::ImmediateExecutionTensorHandle* input) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_97(mht_97_v, 2061, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetInput");

  if (index >= args_.size()) {
    return tensorflow::errors::InvalidArgument("Index >= inputs.size: %d >= %d",
                                               index, args_.size());
  }
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(args_[index].get())) {
    custom_device_tensor_handle_count_--;
  }
  if (tensorflow::CustomDeviceTensorHandle::classof(input)) {
    custom_device_tensor_handle_count_++;
  }
  input->Ref();
  args_[index] =
      tensorflow::core::RefCountPtr<tensorflow::ImmediateExecutionTensorHandle>(
          input);
  return tensorflow::Status::OK();
}

tensorflow::Status OperationInterface::AddInputList(
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_98(mht_98_v, 2084, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::AddInputList");

  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::AddInputList");
}

absl::Span<tensorflow::ImmediateExecutionTensorHandle* const>
OperationInterface::GetInputs() const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_99(mht_99_v, 2093, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::GetInputs");

  return absl::MakeSpan(
      reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle* const*>(
          args_.data()),
      args_.size());
}

tensorflow::Status OperationInterface::SetAttrString(const char* attr_name,
                                                     const char* data,
                                                     size_t length) {
   std::vector<std::string> mht_100_v;
   mht_100_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_100_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_100(mht_100_v, 2107, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrString");

  fallback_attrs_.Set(attr_name, tensorflow::StringPiece(data, length));
  if (attrs_.SetString(attr_name, string_view(data, length)))
    return tensorflow::Status::OK();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrString failed");
}

tensorflow::Status OperationInterface::SetAttrInt(const char* attr_name,
                                                  int64_t value) {
   std::vector<std::string> mht_101_v;
   mht_101_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_101(mht_101_v, 2120, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrInt");

  fallback_attrs_.Set(attr_name, static_cast<int64_t>(value));
  if (attrs_.Set(attr_name, value)) return tensorflow::Status::OK();
  return tensorflow::errors::Internal("OperationInterface::SetAttrInt failed");
}

tensorflow::Status OperationInterface::SetAttrFloat(const char* attr_name,
                                                    float value) {
   std::vector<std::string> mht_102_v;
   mht_102_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_102(mht_102_v, 2131, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrFloat");

  fallback_attrs_.Set(attr_name, value);
  if (attrs_.Set(attr_name, value)) return tensorflow::Status::OK();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFloat failed");
}

tensorflow::Status OperationInterface::SetAttrBool(const char* attr_name,
                                                   bool value) {
   std::vector<std::string> mht_103_v;
   mht_103_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_103(mht_103_v, 2143, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrBool");

  fallback_attrs_.Set(attr_name, value);
  if (attrs_.Set(attr_name, value)) return tensorflow::Status::OK();
  return tensorflow::errors::Internal("OperationInterface::SetAttrBool failed");
}

tensorflow::Status OperationInterface::SetAttrType(const char* attr_name,
                                                   tensorflow::DataType value) {
   std::vector<std::string> mht_104_v;
   mht_104_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_104(mht_104_v, 2154, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrType");

  fallback_attrs_.Set(attr_name, value);
  if (value == tensorflow::DT_INVALID) {
    return tensorflow::errors::InvalidArgument(
        "OperationInterface::SetAttrType failed to set DT_INVALID");
  }
  if (attrs_.Set(attr_name,
                 tfrt::GetOpAttrTypeFromDType(
                     tensorflow::tfd::ConvertTfDataTypeToBefAttrType(value))))
    return tensorflow::Status::OK();
  // TODO(fishx): Remove this workaround once we support all dtype in TF.
  // This is fine for now since attribute "T", "U", "Tidx" is not used by TFRT
  // native ops.
  if (std::strcmp(attr_name, "T") == 0 || std::strcmp(attr_name, "U") == 0 ||
      std::strcmp(attr_name, "Tidx") == 0) {
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::Internal("OperationInterface::SetAttrType failed");
}

tensorflow::Status OperationInterface::SetAttrShape(const char* attr_name,
                                                    const int64_t* dims,
                                                    const int num_dims) {
   std::vector<std::string> mht_105_v;
   mht_105_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_105(mht_105_v, 2180, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrShape");

  // NOTE: This is copied from EagerOperation::SetAttrShape.
  // TODO(b/154554118): Remove the duplication.
  if (num_dims > tensorflow::TensorShape::MaxDimensions()) {
    return tensorflow::errors::InvalidArgument(
        "Value specified for `", attr_name, "` has ", num_dims,
        " dimensions which is over the limit of ",
        tensorflow::TensorShape::MaxDimensions(), ".");
  }

  tensorflow::TensorShapeProto proto;
  size_t offset;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);

    // Set unranked ShapeAttr.
    offset = bef_attr_encoder_.EncodeUnrankedShapeAttr();
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }

    // Set RankedShapeAttr.
    offset = bef_attr_encoder_.EncodeRankedShapeAttr(
        llvm::makeArrayRef(dims, num_dims));
  }
  fallback_attrs_.Set(attr_name, proto);

  auto buf = bef_attr_encoder_.TakeResult();
  tfrt::ShapeAttr shape_attr(buf.data() + offset);
  // TODO(tfrt-devs): Avoid the copy.
  if (attrs_.Set(attr_name, shape_attr)) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrShape failed");
}

tensorflow::Status OperationInterface::SetAttrFunction(
    const char* attr_name, const tensorflow::AbstractOperation* value) {
   std::vector<std::string> mht_106_v;
   mht_106_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_106(mht_106_v, 2222, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrFunction");

  auto* value_operation = down_cast<const OperationInterface*>(value);
  // TODO(b/165412867): Set fallback_attrs_ for eager device placement.
  // Consider removing this and rely on TFRT OpAttrs.
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->Name());
  fallback_attrs_.Set(attr_name, attr_value);

  if (attrs_.SetFunc(attr_name, {string_view(value_operation->Name())}))
    return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunction failed");
}

tensorflow::Status OperationInterface::SetAttrFunctionName(
    const char* attr_name, const char* data, size_t length) {
   std::vector<std::string> mht_107_v;
   mht_107_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_107_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_107(mht_107_v, 2244, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrFunctionName");

  // TODO(b/165412867): Set fallback_attrs_ for eager device placement.
  // Consider removing this and rely on TFRT OpAttrs.
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(data);
  fallback_attrs_.Set(attr_name, attr_value);

  if (attrs_.SetFunc(attr_name, {data})) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunctionName failed");
}

static size_t SerializeTFETensorToDenseAttr(
    tensorflow::AbstractTensorInterface* tensor,
    tfrt::BefAttrEncoder* encoder) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_108(mht_108_v, 2263, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "SerializeTFETensorToDenseAttr");

  std::vector<uint8_t> data;

  const auto element_type =
      tensorflow::tfd::ConvertTfDataTypeToBefAttrType(tensor->Type());
  llvm::SmallVector<int64_t, 4> shape;
  for (int i = 0; i < tensor->NumDims(); ++i) {
    shape.push_back(tensor->Dim(i));
  }
  auto elements = llvm::makeArrayRef(
      reinterpret_cast<const uint8_t*>(tensor->Data()), tensor->ByteSize());
  return encoder->EncodeDenseAttr(static_cast<DType>(element_type), shape,
                                  elements);
}

tensorflow::Status OperationInterface::SetAttrTensor(
    const char* attr_name, tensorflow::AbstractTensorInterface* tensor) {
   std::vector<std::string> mht_109_v;
   mht_109_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_109(mht_109_v, 2283, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrTensor");

  tfrt::BefAttrEncoder encoder;
  const size_t offset = SerializeTFETensorToDenseAttr(tensor, &encoder);
  auto buffer = encoder.TakeResult();
  DenseAttr dense_attr(buffer.data() + offset);
  if (attrs_.Set(attr_name, dense_attr)) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrTensor failed");
}

tensorflow::Status OperationInterface::SetAttrStringList(
    const char* attr_name, const void* const* values, const size_t* lengths,
    int num_values) {
   std::vector<std::string> mht_110_v;
   mht_110_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_110(mht_110_v, 2300, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrStringList");

  std::vector<tensorflow::StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = tensorflow::StringPiece(static_cast<const char*>(values[i]),
                                   lengths[i]);
  }
  fallback_attrs_.Set(attr_name, v);

  tfrt::BefAttrEncoder encoder;
  const size_t offset =
      encoder.EncodeStringListAttr(values, lengths, num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  // TODO(tfrt-devs): Avoid the copy.
  if (attrs_.Set(attr_name, aggr_attr)) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrStringList failed");
}

tensorflow::Status OperationInterface::SetAttrFloatList(const char* attr_name,
                                                        const float* values,
                                                        int num_values) {
   std::vector<std::string> mht_111_v;
   mht_111_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_111(mht_111_v, 2326, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrFloatList");

  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const float>(values, num_values));

  if (attrs_.SetArray(attr_name, tfrt::ArrayRef<float>(values, num_values)))
    return tensorflow::Status::OK();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFloatList failed");
}

tensorflow::Status OperationInterface::SetAttrIntList(const char* attr_name,
                                                      const int64_t* values,
                                                      int num_values) {
   std::vector<std::string> mht_112_v;
   mht_112_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_112(mht_112_v, 2342, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrIntList");

  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));

  if (attrs_.SetArray(attr_name, tfrt::ArrayRef<int64_t>(values, num_values)))
    return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrIntList failed");
}

tensorflow::Status OperationInterface::SetAttrTypeList(
    const char* attr_name, const tensorflow::DataType* values, int num_values) {
   std::vector<std::string> mht_113_v;
   mht_113_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_113(mht_113_v, 2359, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrTypeList");

  fallback_attrs_.Set(attr_name,
                      tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
                          values, num_values));
  // Convert to OpAttrType first.
  llvm::SmallVector<tfrt::DType, 4> tfrt_dtypes;
  tfrt_dtypes.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    tfrt_dtypes.push_back(
        tensorflow::tfd::ConvertTfDataTypeToBefAttrType(values[i]));
  }

  if (attrs_.SetRaw(attr_name, tfrt_dtypes.data(), tfrt::OpAttrType::DTYPE,
                    num_values, OpAttrsRawEntryType::kArray))
    return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrTypeList failed");
}

tensorflow::Status OperationInterface::SetAttrBoolList(
    const char* attr_name, const unsigned char* values, int num_values) {
   std::vector<std::string> mht_114_v;
   mht_114_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_114_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_114(mht_114_v, 2385, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrBoolList");

  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const bool>(b.get(), num_values));

  // Convert to bool first.
  llvm::SmallVector<bool, 4> bool_array;
  bool_array.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    bool_array.push_back(static_cast<bool>((values[i])));
  }
  if (attrs_.SetArray(attr_name,
                      tfrt::ArrayRef<bool>(bool_array.data(), num_values)))
    return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrBoolList failed");
}

tensorflow::Status OperationInterface::SetAttrShapeList(const char* attr_name,
                                                        const int64_t** dims,
                                                        const int* num_dims,
                                                        int num_values) {
   std::vector<std::string> mht_115_v;
   mht_115_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_115(mht_115_v, 2414, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrShapeList");

  std::unique_ptr<tensorflow::TensorShapeProto[]> proto(
      new tensorflow::TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > tensorflow::TensorShape::MaxDimensions()) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Value specified for `", attr_name, "` has ", num_dims_i,
                 " dimensions which is over the limit of ",
                 tensorflow::TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  fallback_attrs_.Set(attr_name,
                      tensorflow::gtl::ArraySlice<tensorflow::TensorShapeProto>(
                          proto.get(), num_values));

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeShapeListAttr(dims, num_dims, num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  if (attrs_.Set(attr_name, aggr_attr)) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrShapeList failed");
}

tensorflow::Status OperationInterface::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
   std::vector<std::string> mht_116_v;
   mht_116_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_116(mht_116_v, 2455, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::SetAttrFunctionList");

  size_t num_values = values.size();
  std::vector<const void*> func_attrs(num_values);
  std::vector<size_t> lengths(num_values);

  for (int i = 0; i < num_values; ++i) {
    auto* value_operation = down_cast<const OperationInterface*>(values[i]);
    lengths[i] = value_operation->Name().length();
    func_attrs[i] = value_operation->Name().c_str();
  }

  // Encode the array of function attributes with BEF typed attribute encoder to
  // an aggregated attribute.
  BefAttrEncoder encoder;
  const size_t offset =
      encoder.EncodeFuncListAttr(func_attrs.data(), lengths.data(), num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  if (attrs_.Set(attr_name, aggr_attr)) return tensorflow::Status::OK();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunctionList failed");
}

tensorflow::Status OperationInterface::InputLength(const char* input_name,
                                                   int* length) {
   std::vector<std::string> mht_117_v;
   mht_117_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_117(mht_117_v, 2484, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::InputLength");

  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::InputLength");
}

tensorflow::Status OperationInterface::OutputLength(const char* output_name,
                                                    int* length) {
   std::vector<std::string> mht_118_v;
   mht_118_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_118(mht_118_v, 2494, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::OutputLength");

  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::OutputLength");
}

const tensorflow::AbstractOpAttrs* OperationInterface::GetOpAttrs() const {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_119(mht_119_v, 2502, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::GetOpAttrs");

  return &op_attrs_;
}

void OperationInterface::AddAttrs(const tensorflow::AbstractOpAttrs* op_attrs) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_120(mht_120_v, 2509, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::AddAttrs");

  auto* tfrt_op_attrs = down_cast<const OpAttrsInterface*>(op_attrs);
  tfrt_op_attrs->GetAttrs()->IterateEntries(
      [this](const OpAttrsRawEntry& entry) {
        attrs_.SetRaw(entry.name, entry.GetData(), entry.type,
                      entry.element_count, entry.entry_type);
      });
  fallback_attrs_.CopyAttributes(*tfrt_op_attrs->GetFallbackAttrs());
}

void OperationInterface::MaybeInferInputAttrs() {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTcc mht_121(mht_121_v, 2522, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.cc", "OperationInterface::MaybeInferInputAttrs");

  if (!op_def_) return;
  for (int i = 0; i < args_.size(); i++) {
    auto* handle = args_[i].get();
    const auto& input_def = op_def_->input_arg(i);
    if (!input_def.number_attr().empty() ||
        !input_def.type_list_attr().empty()) {
      // Some clients that are still setting their input attributes manually are
      // adding input list to their op by calling `TFE_OpAddInput` for each of
      // its elements instead of calling `TFE_OpAddInputList`. When this
      // happens, we cannot detect the end of such list, thus lose track of the
      // input arguments in the op definition. To guarantee backward
      // compatibility with those clients, disable automatic inference in this
      // case.
      return;
    }
    const std::string& type_attr = input_def.type_attr();
    if (!type_attr.empty()) {
      bool success = attrs_.Set(
          type_attr, tfrt::GetOpAttrTypeFromDType(
                         tensorflow::tfd::ConvertTfDataTypeToBefAttrType(
                             handle->DataType())));
      if (success) {
        fallback_attrs_.Set(type_attr, handle->DataType());
      }
    }
  }
}

}  // namespace tf
}  // namespace tfrt
