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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc() {
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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/ragged_tensor_variant.h"

namespace tensorflow {

string RaggedTensorVariant::TypeName() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/ragged_tensor_variant.cc", "RaggedTensorVariant::TypeName");
 return "RaggedTensorVariant"; }

string RaggedTensorVariant::DebugString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/kernels/ragged_tensor_variant.cc", "RaggedTensorVariant::DebugString");

  return absl::StrCat(
      "RaggedTensorVariant(dtype=", DataTypeString(values_.dtype()),
      ", ragged_rank=", nested_splits_.size(), ", splits_dtype=",
      DataTypeString(nested_splits_.empty() ? DT_INVALID
                                            : nested_splits_.back().dtype()));
}

void RaggedTensorVariant::Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc mht_2(mht_2_v, 210, "", "./tensorflow/core/kernels/ragged_tensor_variant.cc", "RaggedTensorVariant::Encode");

  data->set_type_name(TypeName());
  for (const auto& splits : nested_splits_) {
    *data->add_tensors() = splits;
  }
  *data->add_tensors() = values_;
}

bool RaggedTensorVariant::Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc mht_3(mht_3_v, 221, "", "./tensorflow/core/kernels/ragged_tensor_variant.cc", "RaggedTensorVariant::Decode");

  if (data.tensors_size() < 1) {
    return false;
  }
  nested_splits_.assign(data.tensors().begin(),
                        std::prev(data.tensors().end()));
  values_ = data.tensors().back();
  return true;
}

namespace {

Status RaggedTensorVariantDeviceCopy(
    const RaggedTensorVariant& from, RaggedTensorVariant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/ragged_tensor_variant.cc", "RaggedTensorVariantDeviceCopy");

  TF_RETURN_IF_ERROR(copy(from.values(), to->mutable_values()));
  // TODO(b/170415165) Should we use `copy` to move splits from device<->host?
  *to->mutable_nested_splits() = from.nested_splits();
  return Status::OK();
}

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_CPU, RaggedTensorVariant,
    RaggedTensorVariantZerosLike<CPUDevice>);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(
    ADD_VARIANT_BINARY_OP, DEVICE_CPU, RaggedTensorVariant,
    RaggedTensorVariantBinaryAdd<CPUDevice>);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(RaggedTensorVariant,
                                       "RaggedTensorVariant");

#define REGISTER_RAGGED_TENSOR_VARIANT_COPY(DIRECTION)  \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      RaggedTensorVariant, DIRECTION, RaggedTensorVariantDeviceCopy)

REGISTER_RAGGED_TENSOR_VARIANT_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_RAGGED_TENSOR_VARIANT_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_RAGGED_TENSOR_VARIANT_COPY(
    VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

}  // namespace tensorflow
