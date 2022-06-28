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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc() {
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
#include "tensorflow/core/runtime_fallback/util/attr_util.h"

#include <cstdlib>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/logging.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

using ::tensorflow::protobuf::RepeatedFieldBackInserter;
using ::tfrt::AggregateAttr;
using ::tfrt::BEFAttributeType;
using ::tfrt::DenseAttr;
using ::tfrt::DenseHostTensor;
using ::tfrt::HostContext;
using ::tfrt::OpAttrsRawEntry;
using ::tfrt::OpAttrsRef;
using ::tfrt::OpAttrType;
using ::tfrt::string_view;

llvm::Expected<tensorflow::Tensor> DecodeDenseAttrToTfTensor(
    const DenseAttr& dense_attr, HostContext* host) {
  llvm::Expected<DenseHostTensor> dht =
      tfrt::DeserializeDenseHostTensorFromDenseAttr(dense_attr, host);
  if (!dht) {
    return tfrt::MakeStringError(
        "Cannot create DenseHostTensor in DecodeDenseAttrToTensorInterface: ",
        dht.takeError());
  }

  return tfrt::TFRTTensorToTFTensor(*dht, host);
}

llvm::Error FillAttrValueMapUsingArray(const OpAttrsRawEntry& entry,
                                       AttrValue& attr_tmp,
                                       const OpAttrsRef& attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_0(mht_0_v, 238, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "FillAttrValueMapUsingArray");

  attr_tmp.mutable_list()->Clear();
  if (entry.element_count == 0) {
    if (entry.type == OpAttrType::CHAR) {
      // Empty string.
      attr_tmp.set_s("");
    }
    // Empty array of other types.
    return llvm::Error::success();
  }
  switch (entry.type) {
    case OpAttrType::CHAR: {
      string_view attr_value = attrs.GetStringAsserting(entry.name);
      attr_tmp.set_s(attr_value.data(), attr_value.size());
      return llvm::Error::success();
    }

    case OpAttrType::FUNC: {
      string_view attr_value = attrs.GetFuncNameAsserting(entry.name);
      attr_tmp.mutable_func()->set_name(attr_value.data(), attr_value.size());
      return llvm::Error::success();
    }
    case OpAttrType::I64: {
      llvm::ArrayRef<int64_t> int_array =
          attrs.GetArrayAsserting<int64_t>(entry.name);
      auto* mutable_i = attr_tmp.mutable_list()->mutable_i();
      std::copy(int_array.begin(), int_array.end(),
                RepeatedFieldBackInserter(mutable_i));
      return llvm::Error::success();
    }
    case OpAttrType::F32: {
      llvm::ArrayRef<float> float_array =
          attrs.GetArrayAsserting<float>(entry.name);
      auto* mutable_f = attr_tmp.mutable_list()->mutable_f();
      std::copy(float_array.begin(), float_array.end(),
                RepeatedFieldBackInserter(mutable_f));
      return llvm::Error::success();
    }
    case OpAttrType::BOOL: {
      llvm::ArrayRef<bool> bool_array =
          attrs.GetArrayAsserting<bool>(entry.name);
      auto mutable_b = attr_tmp.mutable_list()->mutable_b();
      std::copy(bool_array.begin(), bool_array.end(),
                RepeatedFieldBackInserter(mutable_b));
      return llvm::Error::success();
    }
    case OpAttrType::DTYPE: {
      const auto& op_attr = attrs.GetRawAsserting(entry.name);
      assert(op_attr.IsArray());

      // DTypes in BEF attributes are tfrt::DType enums. So we need
      // to convert then to tensorflow data types first.
      auto bef_dtypes =
          llvm::makeArrayRef(static_cast<const tfrt::DType*>(op_attr.GetData()),
                             op_attr.element_count);

      llvm::SmallVector<tensorflow::DataType, 4> tf_dtypes;
      tf_dtypes.reserve(bef_dtypes.size());
      for (auto bef_dtype : bef_dtypes) {
        tf_dtypes.push_back(ConvertBefAttrTypeToTfDataType(bef_dtype));
      }
      auto* mutable_type = attr_tmp.mutable_list()->mutable_type();
      std::copy(tf_dtypes.begin(), tf_dtypes.end(),
                RepeatedFieldBackInserter(mutable_type));
      return llvm::Error::success();
    }
    default:
      return tfrt::MakeStringError("unsupported array attribute type");
  }
}

llvm::Error FillAttrValueMapUsingAggregate(const OpAttrsRawEntry& entry,
                                           AttrValue& attr_tmp,
                                           const OpAttrsRef& attrs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_1(mht_1_v, 314, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "FillAttrValueMapUsingAggregate");

  AggregateAttr list_attr = attrs.GetAsserting<AggregateAttr>(entry.name);
  int num_values = list_attr.GetNumElements();
  if (num_values == 0) {
    // Create an empty list.
    attr_tmp.mutable_list();
    return llvm::Error::success();
  }
  // It is guaranteed that items in one list attribute have the same
  // type, though their sizes can be different. In particular,
  // list(TensorShape) and list(Tensor) attribute types have to be
  // encoded as AggregateAttr.
  auto attr_base = list_attr.GetAttribute(0);
  auto* mutable_list = attr_tmp.mutable_list();
  mutable_list->Clear();
  if (IsDataTypeAttribute(attr_base.type()) &&
      GetDataType(attr_base.type()) == tfrt::DType::String) {
    // Handle list(string).
    auto* mutable_s = mutable_list->mutable_s();
    mutable_s->Reserve(num_values);
    for (int i = 0; i < num_values; ++i) {
      auto string_attr = list_attr.GetAttributeOfType<tfrt::StringAttr>(i);
      mutable_list->add_s(string_attr.GetValue().data(),
                          string_attr.GetValue().size());
    }
  } else if (attr_base.type() == BEFAttributeType::kFunc) {
    // Handle list(Function).
    auto* mutable_f = mutable_list->mutable_func();
    mutable_f->Reserve(num_values);
    for (int i = 0; i < num_values; ++i) {
      auto func_attr = list_attr.GetAttributeOfType<tfrt::FuncAttr>(i);
      auto mutable_func = mutable_list->add_func();
      mutable_func->set_name(func_attr.GetFunctionName().str());
    }
  } else if (attr_base.type() == BEFAttributeType::kShape) {
    // Handle list(TensorShape).
    auto* mutable_list = attr_tmp.mutable_list();
    auto* mutable_shape = mutable_list->mutable_shape();
    mutable_shape->Reserve(num_values);
    for (int i = 0; i < num_values; ++i) {
      auto shape_attr = list_attr.GetAttributeOfType<tfrt::ShapeAttr>(i);
      auto* added_shape = mutable_list->add_shape();
      if (shape_attr.HasRank()) {
        int rank = shape_attr.GetRank();
        auto shape = shape_attr.GetShape();
        added_shape->mutable_dim()->Reserve(rank);
        for (int d = 0; d < rank; ++d) {
          added_shape->add_dim()->set_size(shape[d]);
        }
      } else {
        added_shape->set_unknown_rank(true);
      }
    }
  } else {
    return tfrt::MakeStringError("unsupported list attribute type");
  }
  return llvm::Error::success();
}

llvm::Error FillAttrValueMapUsingScalar(const OpAttrsRawEntry& entry,
                                        AttrValue& attr_tmp, HostContext* host,
                                        const OpAttrsRef& attrs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_2(mht_2_v, 378, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "FillAttrValueMapUsingScalar");

  switch (entry.type) {
    case OpAttrType::I64: {
      int64_t attr_value = attrs.GetAsserting<int64_t>(entry.name);
      attr_tmp.set_i(attr_value);
      return llvm::Error::success();
    }
    case OpAttrType::F32: {
      float attr_value = attrs.GetAsserting<float>(entry.name);
      attr_tmp.set_f(attr_value);
      return llvm::Error::success();
    }
    case OpAttrType::BOOL: {
      bool attr_value = attrs.GetAsserting<bool>(entry.name);
      attr_tmp.set_b(attr_value);
      return llvm::Error::success();
    }
    case OpAttrType::DTYPE: {
      OpAttrType op_attr_type = attrs.GetAsserting<OpAttrType>(entry.name);
      DataType tf_dtype = ConvertToTfDataType(op_attr_type);
      attr_tmp.set_type(tf_dtype);
      return llvm::Error::success();
    }
    case OpAttrType::SHAPE: {
      auto shape_attr = attrs.GetAsserting<tfrt::ShapeAttr>(entry.name);
      auto* mutable_shape = attr_tmp.mutable_shape();
      if (shape_attr.HasRank()) {
        int rank = shape_attr.GetRank();
        auto shape = shape_attr.GetShape();
        mutable_shape->mutable_dim()->Reserve(rank);
        for (int d = 0; d < rank; ++d) {
          mutable_shape->add_dim()->set_size(shape[d]);
        }
      } else {
        mutable_shape->set_unknown_rank(true);
      }
      return llvm::Error::success();
    }
    case OpAttrType::DENSE: {
      auto dense_attr = attrs.GetAsserting<tfrt::DenseAttr>(entry.name);
      llvm::Expected<tensorflow::Tensor> tf_tensor =
          DecodeDenseAttrToTfTensor(dense_attr, host);
      if (!tf_tensor) return tf_tensor.takeError();
      auto* mutable_tensor = attr_tmp.mutable_tensor();
      if (tf_tensor->NumElements() > 1) {
        tf_tensor->AsProtoTensorContent(mutable_tensor);
      } else {
        tf_tensor->AsProtoField(mutable_tensor);
      }
      return llvm::Error::success();
    }
    case OpAttrType::AGGREGATE: {
      return FillAttrValueMapUsingAggregate(entry, attr_tmp, attrs);
    }
    default:
      LOG(ERROR) << "failure case";
      return tfrt::MakeStringError("unsupported scalar attribute type");
  }
}

}  // namespace

Status ParseTfDataType(absl::string_view dtype, DataType* data_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("dtype: \"" + std::string(dtype.data(), dtype.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_3(mht_3_v, 444, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ParseTfDataType");

  if (dtype == "DT_INT8") {
    *data_type = DataType::DT_INT8;
    return Status::OK();
  } else if (dtype == "DT_INT32") {
    *data_type = DataType::DT_INT32;
    return Status::OK();
  } else if (dtype == "DT_INT64") {
    *data_type = DataType::DT_INT64;
    return Status::OK();
  } else if (dtype == "DT_HALF") {
    *data_type = DataType::DT_HALF;
    return Status::OK();
  } else if (dtype == "DT_FLOAT") {
    *data_type = DataType::DT_FLOAT;
    return Status::OK();
  } else if (dtype == "DT_DOUBLE") {
    *data_type = DataType::DT_DOUBLE;
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported dtype, ", std::string(dtype),
                                   " in ParseTfDataType.");
  }
}

DataType ConvertToTfDataType(tfrt::OpAttrType op_attr_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_4(mht_4_v, 472, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ConvertToTfDataType");

  switch (op_attr_type) {
#define OP_ATTR_TYPE(TFRT_ENUM, DT_ENUM) \
  case tfrt::OpAttrType::TFRT_ENUM:      \
    return DataType::DT_ENUM;
#include "tensorflow/core/runtime_fallback/util/attr_type.def"  // NOLINT
    default:
      TFRT_DLOG(ERROR) << "unsupported dtype" << static_cast<int>(op_attr_type)
                       << " in TFRT fallback kernel.";
      abort();
  }
}

tfrt::OpAttrType ConvertFromTfDataType(DataType data_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_5(mht_5_v, 488, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ConvertFromTfDataType");

  switch (data_type) {
#define OP_ATTR_TYPE(TFRT_ENUM, DT_ENUM) \
  case DataType::DT_ENUM:                \
    return tfrt::OpAttrType::TFRT_ENUM;
#include "tensorflow/core/runtime_fallback/util/attr_type.def"  // NOLINT
    default:
      TFRT_DLOG(ERROR) << "unsupported dtype " << static_cast<int>(data_type)
                       << "in TFRT fallback kernel.";
      abort();
  }
}

DataType ConvertBefAttrTypeToTfDataType(tfrt::DType attr_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_6(mht_6_v, 504, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ConvertBefAttrTypeToTfDataType");

  switch (attr_type) {
    case tfrt::DType::I1:
      return DataType::DT_BOOL;
    case tfrt::DType::I8:
      return DataType::DT_INT8;
    case tfrt::DType::I16:
      return DataType::DT_INT16;
    case tfrt::DType::I32:
      return DataType::DT_INT32;
    case tfrt::DType::I64:
      return DataType::DT_INT64;
    case tfrt::DType::UI8:
      return DataType::DT_UINT8;
    case tfrt::DType::UI16:
      return DataType::DT_UINT16;
    case tfrt::DType::UI32:
      return DataType::DT_UINT32;
    case tfrt::DType::UI64:
      return DataType::DT_UINT64;
    case tfrt::DType::F16:
      return DataType::DT_HALF;
    case tfrt::DType::BF16:
      return DataType::DT_BFLOAT16;
    case tfrt::DType::F32:
      return DataType::DT_FLOAT;
    case tfrt::DType::F64:
      return DataType::DT_DOUBLE;
    case tfrt::DType::Complex64:
      return DataType::DT_COMPLEX64;
    case tfrt::DType::Complex128:
      return DataType::DT_COMPLEX128;
    case tfrt::DType::String:
      return DataType::DT_STRING;
    case tfrt::DType::Resource:
      return DataType::DT_RESOURCE;
    case tfrt::DType::Variant:
      return DataType::DT_VARIANT;
    case tfrt::DType::QUI8:
      return DataType::DT_QUINT8;
    case tfrt::DType::QUI16:
      return DataType::DT_QUINT16;
    case tfrt::DType::QI8:
      return DataType::DT_QINT8;
    case tfrt::DType::QI16:
      return DataType::DT_QINT16;
    case tfrt::DType::QI32:
      return DataType::DT_QINT32;
    default:
      TFRT_DLOG(ERROR) << "unsupported tfrt::DType"
                       << static_cast<int>(attr_type)
                       << " in TFRT fallback kernel.";
      abort();
  }
}

tfrt::DType ConvertTfDataTypeToBefAttrType(DataType data_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_7(mht_7_v, 563, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ConvertTfDataTypeToBefAttrType");

  switch (data_type) {
    case DataType::DT_UINT8:
      return tfrt::DType::UI8;
    case DataType::DT_UINT16:
      return tfrt::DType::UI16;
    case DataType::DT_UINT32:
      return tfrt::DType::UI32;
    case DataType::DT_UINT64:
      return tfrt::DType::UI64;
    case DataType::DT_BOOL:
      return tfrt::DType::I1;
    case DataType::DT_INT8:
      return tfrt::DType::I8;
    case DataType::DT_INT16:
      return tfrt::DType::I16;
    case DataType::DT_INT32:
      return tfrt::DType::I32;
    case DataType::DT_INT64:
      return tfrt::DType::I64;
    case DataType::DT_HALF:
      return tfrt::DType::F16;
    case DataType::DT_BFLOAT16:
      return tfrt::DType::BF16;
    case DataType::DT_FLOAT:
      return tfrt::DType::F32;
    case DataType::DT_DOUBLE:
      return tfrt::DType::F64;
    case DataType::DT_COMPLEX64:
      return tfrt::DType::Complex64;
    case DataType::DT_COMPLEX128:
      return tfrt::DType::Complex128;
    case DataType::DT_STRING:
      return tfrt::DType::String;
    case DataType::DT_RESOURCE:
      return tfrt::DType::Resource;
    case DataType::DT_VARIANT:
      return tfrt::DType::Variant;
    case DataType::DT_QUINT8:
      return tfrt::DType::QUI8;
    case DataType::DT_QUINT16:
      return tfrt::DType::QUI16;
    case DataType::DT_QINT8:
      return tfrt::DType::QI8;
    case DataType::DT_QINT16:
      return tfrt::DType::QI16;
    case DataType::DT_QINT32:
      return tfrt::DType::QI32;
    default:
      TFRT_DLOG(ERROR) << "unsupported DataType " << static_cast<int>(data_type)
                       << " in TFRT fallback kernel.";
      abort();
  }
}

Status ParseBoolAttrValue(absl::string_view attr_value, bool* bool_val) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attr_value: \"" + std::string(attr_value.data(), attr_value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_8(mht_8_v, 622, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ParseBoolAttrValue");

  if (attr_value == "false") {
    *bool_val = false;
    return Status::OK();
  } else if (attr_value == "true") {
    *bool_val = true;
    return Status::OK();
  } else {
    return errors::InvalidArgument("Could not parse bool from \"", attr_value,
                                   "\"");
  }
}

Status ParseIntAttrValue(absl::string_view attr_value, int64_t* int_val) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_value: \"" + std::string(attr_value.data(), attr_value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_9(mht_9_v, 639, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ParseIntAttrValue");

  bool success = absl::SimpleAtoi(attr_value, int_val);
  if (!success) {
    return errors::InvalidArgument("Could not parse int from \"", attr_value,
                                   "\"");
  }
  return Status::OK();
}

Status ParseTensorAttrValue(absl::string_view attr_value,
                            tensorflow::Tensor* tensor) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("attr_value: \"" + std::string(attr_value.data(), attr_value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_10(mht_10_v, 653, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ParseTensorAttrValue");

  if (std::is_base_of<tensorflow::protobuf::Message,
                      tensorflow::TensorProto>()) {
    tensorflow::TensorProto tensor_proto;
    // We use reinterpret_cast here to make sure ParseFromString call
    // below compiles if TensorProto is not a subclass of Message.
    // At run time, we should never get to this point if TensorProto
    // is not a subclass of message due to if-condition above.
    auto* message = reinterpret_cast<protobuf::Message*>(&tensor_proto);
    if (protobuf::TextFormat::ParseFromString(
            static_cast<std::string>(attr_value), message) &&
        tensor->FromProto(tensor_proto)) {
      return Status::OK();
    } else {
      return errors::InvalidArgument("Could not parse tensor value from \"",
                                     attr_value, "\"");
    }
  } else {
    // TextFormat does not work with portable proto implementations.
    return errors::InvalidArgument(
        "Tensor attributes are not supported on mobile.");
  }
}

Status ParseTensorShapeAttrValue(absl::string_view attr_value,
                                 std::vector<int64_t>* shape_val) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_value: \"" + std::string(attr_value.data(), attr_value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_11(mht_11_v, 682, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "ParseTensorShapeAttrValue");

  if (attr_value.size() < 2 || attr_value[0] != '[' ||
      attr_value[attr_value.size() - 1] != ']') {
    return errors::InvalidArgument(
        "Tensor shape attribute must be a string of the form [1,2...], instead "
        "got \"",
        attr_value, "\"");
  }
  absl::string_view attr_value_trunc =
      attr_value.substr(1, attr_value.size() - 2);
  // `container` is an absl::strings_internal::Splitter, which is a
  // lazy-splitting iterable. So we cannot get its size to reserve `dims`.
  auto container = absl::StrSplit(attr_value_trunc, ',');
  for (auto it = container.begin(); it != container.end(); ++it) {
    int64_t int_val;
    if (!ParseIntAttrValue(*it, &int_val).ok()) {
      return errors::InvalidArgument("Failed to parse an integer value from ",
                                     *it, " while parsing shape.");
    }
    shape_val->push_back(int_val);
  }
  return Status::OK();
}

bool IsUnusedAttribute(absl::string_view attr_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_12(mht_12_v, 710, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "IsUnusedAttribute");

  // These are extra attributes added by TF MLIR dialect, and not needed by
  // current TF runtime.
  //
  // TODO(chky): Consider removing this attribute in tf-to-tfrt
  // lowering.
  return absl::StrContains(attr_name, "result_segment_sizes") ||
         absl::StrContains(attr_name, "operand_segment_sizes") ||
         absl::EndsWith(attr_name, "_tf_data_function");
}

llvm::Error FillAttrValueMap(const tfrt::OpAttrsRef& attrs,
                             tfrt::HostContext* host,
                             tensorflow::AttrValueMap* attr_value_map) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_13(mht_13_v, 726, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "FillAttrValueMap");

  AttrValue attr_tmp;
  llvm::Error error = llvm::Error::success();
  attrs.IterateEntries([&error, attr_value_map, &attr_tmp, host,
                        &attrs](const OpAttrsRawEntry& entry) {
    // TFE does not expect a device attribute.
    assert(strcmp(entry.name, "device") != 0);
    if (IsUnusedAttribute(entry.name)) {
      return;
    } else if (entry.IsArray()) {
      error = FillAttrValueMapUsingArray(entry, attr_tmp, attrs);
    } else {
      error = FillAttrValueMapUsingScalar(entry, attr_tmp, host, attrs);
    }
    if (error) return;
    attr_value_map->insert(AttrValueMap::value_type(entry.name, attr_tmp));
  });
  return error;
}

namespace {

tensorflow::Tensor CreateTfTensorFromDenseAttr(tfrt::DenseAttr attr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_14(mht_14_v, 751, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "CreateTfTensorFromDenseAttr");

  tensorflow::TensorShape shape(absl::InlinedVector<int64_t, 4>(
      attr.shape().begin(), attr.shape().end()));
  tensorflow::DataType dtype = ConvertBefAttrTypeToTfDataType(attr.dtype());

  tensorflow::Tensor tensor(dtype, shape);

  std::memcpy(tensor.data(), attr.GetElements(), tensor.TotalBytes());

  return tensor;
}

Status SetUpScalarAttr(tfrt::TypedAttrBase bef_attr,
                       tensorflow::AttrValue* tf_attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_15(mht_15_v, 767, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "SetUpScalarAttr");

  if (auto shape_attr = bef_attr.dyn_cast<tfrt::ShapeAttr>()) {
    if (shape_attr.HasRank()) {
      tensorflow::PartialTensorShape tf_shape(shape_attr.GetShape());
      tf_shape.AsProto(tf_attr->mutable_shape());
    } else {
      tensorflow::PartialTensorShape unranked_shape;
      unranked_shape.AsProto(tf_attr->mutable_shape());
    }
  } else if (auto dense_attr = bef_attr.dyn_cast<tfrt::DenseAttr>()) {
    auto tf_tensor = CreateTfTensorFromDenseAttr(dense_attr);
    tf_tensor.AsProtoTensorContent(tf_attr->mutable_tensor());
  } else if (auto type_attr = bef_attr.dyn_cast<tfrt::TypeAttr>()) {
    tf_attr->set_type(ConvertBefAttrTypeToTfDataType(type_attr.GetValue()));
  } else if (auto i1_attr = bef_attr.dyn_cast<tfrt::I1Attr>()) {
    tf_attr->set_b(i1_attr.GetValue());
  } else if (auto f32_attr = bef_attr.dyn_cast<tfrt::F32Attr>()) {
    tf_attr->set_f(f32_attr.GetValue());
  } else if (auto i64_attr = bef_attr.dyn_cast<tfrt::I64Attr>()) {
    tf_attr->set_i(i64_attr.GetValue());
  } else if (auto string_attr = bef_attr.dyn_cast<tfrt::StringAttr>()) {
    tf_attr->set_s(string_attr.GetValue().data(),
                   string_attr.GetValue().size());
  } else {
    return tensorflow::errors::Internal("Failed to set up attribute.");
  }

  return Status::OK();
}

Status SetUpScalarFunctionAttr(tfrt::StringAttr func_attr,
                               tensorflow::AttrValue& tf_attr) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_16(mht_16_v, 801, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "SetUpScalarFunctionAttr");

  tfrt::string_view func_name = func_attr.GetValue();
  tf_attr.mutable_func()->set_name(func_name.data(), func_name.size());
  return Status::OK();
}

void AddShapeToAttrList(tfrt::ShapeAttr shape,
                        tensorflow::AttrValue::ListValue* list) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_17(mht_17_v, 811, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "AddShapeToAttrList");

  if (shape.HasRank()) {
    tensorflow::PartialTensorShape tf_shape(shape.GetShape());
    tf_shape.AsProto(list->add_shape());
    return;
  }

  tensorflow::PartialTensorShape unranked_shape;
  unranked_shape.AsProto(list->add_shape());
}
void AddTensorToAttrList(tfrt::DenseAttr dense_attr,
                         tensorflow::AttrValue::ListValue* list) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_18(mht_18_v, 825, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "AddTensorToAttrList");

  auto tf_tensor = CreateTfTensorFromDenseAttr(dense_attr);
  tf_tensor.AsProtoTensorContent(list->add_tensor());
}

Status SetUpListAttr(tfrt::AggregateAttr aggregate_attr,
                     tensorflow::AttrValue* tf_attr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_19(mht_19_v, 834, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "SetUpListAttr");

  auto* list = tf_attr->mutable_list();
  for (int i = 0; i < aggregate_attr.GetNumElements(); ++i) {
    auto base = aggregate_attr.GetAttribute(i);
    if (auto shape_attr = base.dyn_cast<tfrt::ShapeAttr>()) {
      AddShapeToAttrList(shape_attr, list);
    } else if (auto dense_attr = base.dyn_cast<tfrt::DenseAttr>()) {
      AddTensorToAttrList(dense_attr, list);
    } else if (auto string_attr = base.dyn_cast<tfrt::StringAttr>()) {
      list->add_s(string_attr.GetValue().data(), string_attr.GetValue().size());
    } else {
      return tensorflow::errors::Internal("Failed to set up list attr.");
    }
  }
  return Status::OK();
}

Status SetUpListAttr(tfrt::ArrayAttr array_attr,
                     tensorflow::AttrValue* tf_attr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_20(mht_20_v, 855, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "SetUpListAttr");

  auto* list = tf_attr->mutable_list();

  // Handle an empty array case.
  if (array_attr.GetNumElements() == 0) {
    return Status::OK();
  }

  tfrt::BEFAttributeType element_type = array_attr.GetElementType();
  if (tfrt::IsDataTypeAttribute(element_type)) {
    tfrt::DType dtype = GetDataType(element_type);
    switch (dtype) {
      case tfrt::DType::I1: {
        for (auto value : array_attr.GetValue<bool>()) {
          list->add_b(value);
        }
        return Status::OK();
      }
      case tfrt::DType::I64: {
        for (auto value : array_attr.GetValue<int64_t>()) {
          list->add_i(value);
        }
        return Status::OK();
      }
      case tfrt::DType::F32: {
        for (auto value : array_attr.GetValue<float>()) {
          list->add_f(value);
        }
        return Status::OK();
      }
      default:
        return tensorflow::errors::Internal(
            StrCat("Failed to set up list attr: unsupported dtype: ",
                   tfrt::DType(dtype)));
    }
  } else if (element_type == tfrt::BEFAttributeType::kType) {
    for (auto value : array_attr.GetValue<tfrt::DType>()) {
      list->add_type(ConvertBefAttrTypeToTfDataType(value));
    }
    return Status::OK();
  }

  return tensorflow::errors::Internal("Failed to set up list attr.");
}

}  // namespace

Status SetUpAttrValueMap(tfrt::AggregateAttr op_attr_array,
                         tfrt::AggregateAttr op_func_attr_array,
                         tensorflow::AttrValueMap* attr_value_map) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTcc mht_21(mht_21_v, 907, "", "./tensorflow/core/runtime_fallback/util/attr_util.cc", "SetUpAttrValueMap");

  auto obtain_name_attr_pair =
      [](tfrt::AggregateAttr attr_array,
         int i) -> std::pair<std::string, tfrt::TypedAttrBase> {
    auto pair = attr_array.GetAttributeOfType<tfrt::AggregateAttr>(i);
    assert(pair.GetNumElements() == 2);
    return {pair.GetAttributeOfType<tfrt::StringAttr>(0).GetValue().str(),
            pair.GetAttribute(1)};
  };

  for (size_t i = 0, e = op_attr_array.GetNumElements(); i != e; ++i) {
    auto name_attr_pair = obtain_name_attr_pair(op_attr_array, i);
    if (IsUnusedAttribute(name_attr_pair.first)) continue;

    AttrValue& tf_attr = (*attr_value_map)[name_attr_pair.first];
    tfrt::TypedAttrBase attr_value = name_attr_pair.second;
    if (auto aggregate_attr = attr_value.dyn_cast<tfrt::AggregateAttr>()) {
      TF_RETURN_IF_ERROR(SetUpListAttr(aggregate_attr, &tf_attr));
    } else if (auto array_attr = attr_value.dyn_cast<tfrt::ArrayAttr>()) {
      TF_RETURN_IF_ERROR(SetUpListAttr(array_attr, &tf_attr));
    } else {
      TF_RETURN_IF_ERROR(SetUpScalarAttr(attr_value, &tf_attr));
    }
  }

  for (size_t i = 0, e = op_func_attr_array.GetNumElements(); i != e; ++i) {
    auto name_attr_pair = obtain_name_attr_pair(op_func_attr_array, i);
    if (IsUnusedAttribute(name_attr_pair.first)) continue;

    AttrValue& tf_attr = (*attr_value_map)[name_attr_pair.first];
    auto attr_value = name_attr_pair.second.dyn_cast<tfrt::StringAttr>();
    TF_RETURN_IF_ERROR(SetUpScalarFunctionAttr(attr_value, tf_attr));
  }

  return Status::OK();
}

}  // namespace tfd
}  // namespace tensorflow
