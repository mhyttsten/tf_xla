/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh() {
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


#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

// Special op types.
TF_CONST_INIT extern const absl::string_view kUnknownOp;
TF_CONST_INIT extern const absl::string_view kDatasetOp;
TF_CONST_INIT extern const absl::string_view kMemcpyHToDOp;
TF_CONST_INIT extern const absl::string_view kMemcpyDToHOp;
TF_CONST_INIT extern const absl::string_view kMemcpyDToDOp;
TF_CONST_INIT extern const absl::string_view kMemcpyHToHOp;

enum class Category {
  kUnknown,
  kTensorFlow,
  kJax,
  kTfData,
  kMemcpyHToD,
  kMemcpyDToH,
  kMemcpyDToD,
  kMemcpyHToH,
};

// Breaks a TensorFlow op fullname into name and type.
struct TfOp {
  Category category = Category::kUnknown;
  absl::string_view name;
  absl::string_view type;
};
TfOp ParseTfOpFullname(absl::string_view tf_op_fullname);

// Returns a vector of TF name scopes extracted from a TF op name.
std::vector<absl::string_view> ParseTfNameScopes(absl::string_view tf_op_name);
std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op);

// Trace event name for TF ops is the op type so they have the same color in
// trace viewer.
std::string TfOpEventName(const TfOp& tf_op);
std::string TfOpEventName(absl::string_view tf_op_fullname);

// Trace event name for dataset ops.
std::string DatasetOpEventName(absl::string_view full_name);

// Returns the iterator name without prefix and parent iterator names.
std::string IteratorName(absl::string_view full_name);

// Returns true if the given name is a TensorFlow Dataset Op.
inline bool IsDatasetOp(absl::string_view tf_op_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tf_op_type: \"" + std::string(tf_op_type.data(), tf_op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_0(mht_0_v, 242, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsDatasetOp");

  return tf_op_type == kDatasetOp;
}
inline bool IsDatasetOp(const TfOp& tf_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_1(mht_1_v, 248, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsDatasetOp");

  return tf_op.category == Category::kTfData;
}

// Returns true if the given name is a TensorFlow Infeed Enqueue Op.
// See: tensorflow/core/tpu/kernels/infeed_ops.h
inline bool IsInfeedEnqueueOp(absl::string_view tf_op_type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tf_op_type: \"" + std::string(tf_op_type.data(), tf_op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_2(mht_2_v, 258, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsInfeedEnqueueOp");

  return absl::StartsWith(tf_op_type, "InfeedEnqueue");
}
inline bool IsInfeedEnqueueOp(const TfOp& tf_op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_3(mht_3_v, 264, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsInfeedEnqueueOp");

  return tf_op.category == Category::kTensorFlow &&
         IsInfeedEnqueueOp(tf_op.type);
}

// Returns true if the given op has XlaSendToHost/XlaRecvFromHost in fullname.
inline bool IsOutsideCompilationOp(absl::string_view tf_op_fullname) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tf_op_fullname: \"" + std::string(tf_op_fullname.data(), tf_op_fullname.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_4(mht_4_v, 274, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsOutsideCompilationOp");

  if (absl::EndsWith(tf_op_fullname, ":XlaSendToHost")) return true;
  if (absl::EndsWith(tf_op_fullname, ":XlaRecvFromHost")) return true;
  return false;
}

// Returns true if the given op is for outside compilation.
inline bool IsOutsideCompilationOp(absl::string_view tf_op_fullname,
                                   absl::string_view hlo_expression) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tf_op_fullname: \"" + std::string(tf_op_fullname.data(), tf_op_fullname.size()) + "\"");
   mht_5_v.push_back("hlo_expression: \"" + std::string(hlo_expression.data(), hlo_expression.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_5(mht_5_v, 287, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsOutsideCompilationOp");

  if (IsOutsideCompilationOp(tf_op_fullname)) return true;
  if (absl::StrContains(hlo_expression, "send-done") &&
      absl::StrContains(hlo_expression, "is_host_transfer=true"))
    return true;
  return false;
}

// Returns true if the given name is a TensorFlow embedding op.
inline bool IsEmbeddingOp(absl::string_view tf_op_fullname) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("tf_op_fullname: \"" + std::string(tf_op_fullname.data(), tf_op_fullname.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_6(mht_6_v, 300, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsEmbeddingOp");

  return absl::StrContains(tf_op_fullname, "Embedding");
}

// Returns true if the given op is for copying data from host to device.
inline bool IsMemcpyHToDOp(absl::string_view tf_op_type) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tf_op_type: \"" + std::string(tf_op_type.data(), tf_op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_7(mht_7_v, 309, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsMemcpyHToDOp");

  return tf_op_type == kMemcpyHToDOp;
}
inline bool IsMemcpyHToDOp(const TfOp& tf_op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_8(mht_8_v, 315, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsMemcpyHToDOp");

  return tf_op.category == Category::kMemcpyHToD;
}

// Returns true if the given op is for copying data from device to host.
inline bool IsMemcpyDToHOp(const TfOp& tf_op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_9(mht_9_v, 323, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsMemcpyDToHOp");

  return tf_op.category == Category::kMemcpyDToH;
}

// Returns true if the given op is for copying data from device to device.
inline bool IsMemcpyDToDOp(const TfOp& tf_op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_10(mht_10_v, 331, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsMemcpyDToDOp");

  return tf_op.category == Category::kMemcpyDToD;
}

// Returns true if the given op is for copying data from host to host.
inline bool IsMemcpyHToHOp(const TfOp& tf_op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTh mht_11(mht_11_v, 339, "", "./tensorflow/core/profiler/utils/tf_op_utils.h", "IsMemcpyHToHOp");

  return tf_op.category == Category::kMemcpyHToH;
}

// Splits a string of tensor shapes in "(shape1;shape2;...)" format, i.e.,
// delimited by '(' and ')' and separated by ';', into the individual shapes.
std::vector<absl::string_view> ParseTensorShapes(
    absl::string_view tensor_shapes);

// Returns true if the given string matches OpDef.name pattern.
bool IsTfOpName(absl::string_view op_name);

// Returns true if the given string matches NodeDef.name pattern.
bool IsTfOpType(absl::string_view op_type);

// Returns true if the given string matches JAX pattern.
bool IsJaxOpType(absl::string_view op_type);

// Returns true if the given strings match JAX pattern.
bool IsJaxOpNameAndType(absl::string_view op_name, absl::string_view op_type);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
