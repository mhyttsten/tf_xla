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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTh() {
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


#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_encoding.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Converts a TFRT string_view to the Abseil version.
inline absl::string_view ToAbslStringView(tfrt::string_view sv) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("sv: \"" + std::string(sv.data(), sv.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPSattr_utilDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/runtime_fallback/util/attr_util.h", "ToAbslStringView");

  return absl::string_view(sv.data(), sv.size());
}

// Parses the string representation of the DataType in `dtype` into `data_type`.
// Aborts the program for unsupported dtypes.
tensorflow::Status ParseTfDataType(absl::string_view dtype,
                                   DataType* data_type);

// The following 2 functions convert between Tensorflow DataTypes and
// OpAttrTypes. The mapping between OpAttrType and DataType is defined in
// attr_type.def. Aborts on unsupported types.
DataType ConvertToTfDataType(tfrt::OpAttrType op_attr_type);
tfrt::OpAttrType ConvertFromTfDataType(DataType data_type);

// The following 2 functions convert between BEF attribute types and Tensorflow
// DataTypes. Aborts on unsupported datatypes.
DataType ConvertBefAttrTypeToTfDataType(tfrt::DType attr_type);
tfrt::DType ConvertTfDataTypeToBefAttrType(DataType data_type);

// Parses the tensor valued `attr_value` and constructs the tensor with its
// contents in `tensor`. Returns OK status on success, INVALID_ARGUMENT on
// failure.
tensorflow::Status ParseTensorAttrValue(absl::string_view attr_value,
                                        tensorflow::Tensor* tensor);

// Parses a string of the form "[1,2,3,...]" in `attr_value` and returns the
// constituent dimension sizes (shape) in `int_list_val`. Returns
// INVALID_ARGUMENT on invalid input.
tensorflow::Status ParseTensorShapeAttrValue(absl::string_view attr_value,
                                             std::vector<int64_t>* shape_val);

// Parses a boolean from `attr_value` into `bool_val` and returns OK status on
// success. Returns INVALID_ARGUMENT on invalid input.
tensorflow::Status ParseBoolAttrValue(absl::string_view attr_value,
                                      bool* bool_val);

// Parses an int64_t from `attr_value` into `int_val` and returns OK status on
// success. Returns INVLAID_ARGUMENT on invalid input.
tensorflow::Status ParseIntAttrValue(absl::string_view attr_value,
                                     int64_t* int_val);

inline std::vector<absl::string_view> AttrValueSplit(absl::string_view str) {
  return absl::StrSplit(str, absl::MaxSplits('$', 1));
}

// Returns true if `attr_name` is an attribute that is not required by TFRT
// (usually added by stages higher in the lowering process)
bool IsUnusedAttribute(absl::string_view attr_name);

// Fills in the passed in AttrValueMap `attr_value_map` with attributes from
// `attrs`.
llvm::Error FillAttrValueMap(const tfrt::OpAttrsRef& attrs,
                             tfrt::HostContext* host,
                             AttrValueMap* attr_value_map);

// Fills in the passed in AttrValueMap `attr_value_map`.
tensorflow::Status SetUpAttrValueMap(tfrt::AggregateAttr op_attr_array,
                                     tfrt::AggregateAttr op_func_attr_array,
                                     tensorflow::AttrValueMap* attr_value_map);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_
