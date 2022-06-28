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
class MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/small_constant_optimization.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/ctstring_internal.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

namespace {

constexpr TF_DataType kAllowedDataType[] = {TF_INT32, TF_INT64, TF_STRING};

void AppendIntValues(const int num_of_elements, const int* int_values,
                     TensorProto* proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/cc/small_constant_optimization.cc", "AppendIntValues");

  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_int_val(int_values[i]);
  }
}

void AppendInt64Values(const int num_of_elements, const int64_t* int64_values,
                       TensorProto* proto) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc mht_1(mht_1_v, 221, "", "./tensorflow/dtensor/cc/small_constant_optimization.cc", "AppendInt64Values");

  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_int64_val(int64_values[i]);
  }
}

void AppendStringValues(const int num_of_elements,
                        const TF_TString* string_values, TensorProto* proto) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc mht_2(mht_2_v, 231, "", "./tensorflow/dtensor/cc/small_constant_optimization.cc", "AppendStringValues");

  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_string_val(
        std::string(TF_TString_GetDataPointer(&string_values[i]),
                    TF_TString_GetSize(&string_values[i])));
  }
}

}  // namespace

absl::optional<NodeDef> ExtractSmallTensorValue(TFE_Context* context,
                                                TFE_TensorHandle* tensor,
                                                const Layout& layout,
                                                TF_Status* status) {
  auto num_elements = TFE_TensorHandleNumElements(tensor, status);
  if (TF_GetCode(status) != TF_OK) return absl::nullopt;

  if (num_elements >= kSmallTensorThreshold) return absl::nullopt;

  // Check the DType before attempting to resolve the tensor so we don't try to
  // copy resource-dtype tensors off the DTensor device. Currently we only
  // extract small int32/int64_t tensors, primarily to catch shapes and axes,
  // and tf_string tensors that are mostly used in save/restore ops.
  const auto& dtype = TFE_TensorHandleDataType(tensor);
  if (absl::c_find(kAllowedDataType, dtype) == std::end(kAllowedDataType)) {
    return absl::nullopt;
  }

  // This is the enum from protobuf, or the following AddNodeAttr will always
  // set the integer field.
  const auto& datatype = static_cast<DataType>(dtype);

  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> value_tensor(
      TFE_TensorHandleResolve(tensor, status), TF_DeleteTensor);
  if (TF_GetCode(status) != TF_OK) return absl::nullopt;

  NodeDef node_def;
  node_def.set_op("Const");
  AddNodeAttr("dtype", datatype, &node_def);

  TensorProto tensor_proto;
  tensor_proto.set_dtype(datatype);
  switch (dtype) {
    case TF_INT32:
      AppendIntValues(num_elements,
                      static_cast<int*>(TF_TensorData(value_tensor.get())),
                      &tensor_proto);
      break;
    case TF_INT64:
      AppendInt64Values(
          num_elements,
          static_cast<const int64_t*>(TF_TensorData(value_tensor.get())),
          &tensor_proto);
      break;
    case TF_STRING:
      AppendStringValues(
          num_elements,
          static_cast<const TF_TString*>(TF_TensorData(value_tensor.get())),
          &tensor_proto);
      break;
    default:
      TF_SetStatus(status, TF_INTERNAL,
                   absl::StrCat("dtype: ", dtype,
                                " fell through the supported extraction list. "
                                "This should not happen.")
                       .c_str());
      return absl::nullopt;
  }

  std::vector<int64_t> dim_list;
  int num_dims = value_tensor->tensor->NumDims();
  dim_list.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dim_list.push_back(value_tensor->tensor->Dim(i));
  }

  TensorShape shape(std::move(dim_list));
  shape.AsProto(tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensor_proto, &node_def);

  AddNodeAttr(kLayoutAttr, {layout.ToString()}, &node_def);
  AddNodeAttr(kMeshAttr, layout.mesh().ToString(), &node_def);
  return node_def;
}

bool ShouldFoldInputArgument(bool is_func, absl::string_view operation_name,
                             int input_index) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("operation_name: \"" + std::string(operation_name.data(), operation_name.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPSsmall_constant_optimizationDTcc mht_3(mht_3_v, 321, "", "./tensorflow/dtensor/cc/small_constant_optimization.cc", "ShouldFoldInputArgument");

  // For function, we never fold small const arguments.
  //
  // - If user presents a python small const to tf.function, it will be embed in
  //   the function, not func argument. For a different (Python) const, a
  //   re-tracing will happen.  so the assumption still holds.
  // - If user passes a TF tensor with small const values, we follow the
  //   tf.function semantics, i.e., treating it as a dynamic input. So, folding
  //   its value should be avoided.
  if (is_func) return false;

  // TODO(xiejw,power): Think about how to generalize this so it does not depend
  // on operation_name. For example, we can check the max abs value of the
  // tensor value.
  if (operation_name == absl::string_view("StatelessRandomUniform") ||
      operation_name == absl::string_view("StatelessRandomUniformFullInt") ||
      operation_name == absl::string_view("StatelessRandomNormal") ||
      operation_name == absl::string_view("StatelessTruncatedNormal")) {
    // For all stateless rng ops, we avoid fold seed (input_index==1) in graph.
    // This is an important optimization to avoid unnecessary MLIR SPMD lowering
    // and TPU compilation during model parameters initialization process.
    // which typically have the same shape for rng ops but different seeds.
    return input_index != 1;
  }

  return true;
}

}  // namespace dtensor
}  // namespace tensorflow
