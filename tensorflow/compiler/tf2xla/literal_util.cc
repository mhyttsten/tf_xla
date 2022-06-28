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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/literal_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

Status HostTensorToBorrowingLiteral(const Tensor& host_tensor,
                                    xla::BorrowingLiteral* literal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "HostTensorToBorrowingLiteral");

  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor.dtype(),
                                           host_tensor.shape(), &xla_shape));
  return HostTensorToBorrowingLiteral(xla_shape, host_tensor, literal);
}

Status HostTensorToBorrowingLiteral(const xla::Shape& xla_shape,
                                    const Tensor& host_tensor,
                                    xla::BorrowingLiteral* literal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "HostTensorToBorrowingLiteral");

  const auto& tshape = host_tensor.shape();
  TF_RET_CHECK(tshape.IsFullyDefined() &&
               tshape.dims() == xla_shape.dimensions_size() &&
               tshape.dim_sizes() == xla_shape.dimensions())
      << "Provided xla::Shape must have the same dims as the Tensor shape.";
  *literal = xla::BorrowingLiteral(
      static_cast<const char*>(DMAHelper::base(&host_tensor)), xla_shape);
  return Status::OK();
}

StatusOr<xla::Literal> HostTensorToLiteral(const Tensor& host_tensor) {
  xla::BorrowingLiteral literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(host_tensor, &literal));
  return literal.Clone();
}

Status HostTensorToMutableBorrowingLiteral(
    Tensor* host_tensor, xla::MutableBorrowingLiteral* literal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "HostTensorToMutableBorrowingLiteral");

  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor->dtype(),
                                           host_tensor->shape(), &xla_shape));
  return HostTensorToMutableBorrowingLiteral(xla_shape, host_tensor, literal);
}

Status HostTensorToMutableBorrowingLiteral(
    const xla::Shape& xla_shape, Tensor* host_tensor,
    xla::MutableBorrowingLiteral* literal) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_3(mht_3_v, 240, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "HostTensorToMutableBorrowingLiteral");

  *literal = xla::MutableBorrowingLiteral(
      static_cast<const char*>(DMAHelper::base(host_tensor)), xla_shape);

  return Status::OK();
}

Status HostTensorsToBorrowingLiteralTuple(absl::Span<const Tensor> host_tensors,
                                          xla::BorrowingLiteral* literal) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_4(mht_4_v, 251, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "HostTensorsToBorrowingLiteralTuple");

  std::vector<const char*> buf_ptrs;
  buf_ptrs.reserve(host_tensors.size());
  std::vector<xla::Shape> tensor_shapes(host_tensors.size());

  for (int i = 0, end = host_tensors.size(); i < end; i++) {
    // Validate runtime shapes and fail if it doesn't match the contract.
    const Tensor* tensor = &host_tensors[i];
    buf_ptrs.emplace_back(static_cast<const char*>(DMAHelper::base(tensor)));
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(tensor->dtype(), tensor->shape(),
                                             &tensor_shapes[i]));
  }

  *literal = xla::BorrowingLiteral(
      buf_ptrs, xla::ShapeUtil::MakeTupleShape(tensor_shapes));

  return Status::OK();
}

Status CopyLiteralToHostTensor(const xla::LiteralSlice& literal,
                               Tensor* host_tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_5(mht_5_v, 274, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "CopyLiteralToHostTensor");

  TF_RET_CHECK(literal.shape().IsArray() &&
               xla::ShapeUtil::ElementsIn(literal.shape()) ==
                   host_tensor->NumElements());
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(
      DataTypeToPrimitiveType(host_tensor->dtype(), &primitive_type));
  if (literal.shape().element_type() != primitive_type) {
    return errors::InvalidArgument(
        "Cannot convert literal of type ",
        xla::PrimitiveType_Name(literal.shape().element_type()),
        " to tensor of type ", DataTypeString(host_tensor->dtype()));
  }
  size_t total_bytes = host_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void* src_ptr = literal.untyped_data();
    void* dst_ptr = DMAHelper::base(host_tensor);
    memcpy(dst_ptr, src_ptr, total_bytes);
  }
  return Status::OK();
}

Status LiteralToHostTensor(const xla::LiteralSlice& literal,
                           DataType target_type, Tensor* host_tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSliteral_utilDTcc mht_6(mht_6_v, 300, "", "./tensorflow/compiler/tf2xla/literal_util.cc", "LiteralToHostTensor");

  TensorShape shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(literal.shape(), &shape));
  *host_tensor = Tensor(target_type, shape);
  return CopyLiteralToHostTensor(literal, host_tensor);
}

}  // namespace tensorflow
