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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc() {
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
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace optimize {
namespace calibration {

LoggingOpResolver::LoggingOpResolver(
    const BuiltinOpsSet& builtin_ops_to_replace,
    const CustomOpsSet& custom_ops_to_replace, const OpResolver& base_resolver,
    KernelEvalFuncPtr logging_eval_fn, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver.cc", "LoggingOpResolver::LoggingOpResolver");

  std::vector<std::string> unresolved_builtin_ops;
  std::vector<std::string> unresolved_custom_ops;

  for (const auto& op_and_version : builtin_ops_to_replace) {
    const TfLiteRegistration* base_registration =
        base_resolver.FindOp(op_and_version.first, op_and_version.second);
    if (!base_registration) {
      unresolved_builtin_ops.push_back(
          EnumNameBuiltinOperator(op_and_version.first));
      continue;
    }
    BuiltinOperatorKey key = op_and_version;
    builtin_op_evalfn_map_[key] = base_registration->invoke;
    auto logging_registration =
        absl::make_unique<TfLiteRegistration>(*base_registration);
    logging_registration->invoke = logging_eval_fn;
    builtin_op_registration_map_[key] = std::move(logging_registration);
  }
  for (const auto& op_and_version : custom_ops_to_replace) {
    const TfLiteRegistration* base_registration = base_resolver.FindOp(
        op_and_version.first.c_str(), op_and_version.second);
    if (!base_registration) {
      if (!IsFlexOp(op_and_version.first.c_str()))
        unresolved_custom_ops.push_back(op_and_version.first.c_str());
      continue;
    }
    CustomOperatorKey key = op_and_version;
    custom_op_evalfn_map_[key] = base_registration->invoke;
    auto logging_registration =
        absl::make_unique<TfLiteRegistration>(*base_registration);
    logging_registration->invoke = logging_eval_fn;
    custom_op_registration_map_[key] = std::move(logging_registration);
  }

  if (!unresolved_builtin_ops.empty() || !unresolved_custom_ops.empty()) {
    if (!error_reporter) return;
    std::string error_message =
        "Failed to initialize op resolver for calibration:";
    if (!unresolved_builtin_ops.empty())
      absl::StrAppend(&error_message, "\nThere are unresolved builtin ops: [",
                      absl::StrJoin(unresolved_builtin_ops, ", "), "]");
    if (!unresolved_custom_ops.empty()) {
      absl::StrAppend(&error_message, "\nThere are unresolved custom ops: [",
                      absl::StrJoin(unresolved_custom_ops, ", "), "]");
    }
    TF_LITE_REPORT_ERROR(error_reporter, error_message.c_str());
  }
}

const TfLiteRegistration* LoggingOpResolver::FindOp(BuiltinOperator op,
                                                    int version) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc mht_1(mht_1_v, 253, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver.cc", "LoggingOpResolver::FindOp");

  BuiltinOperatorKey key = {op, version};
  if (builtin_op_registration_map_.find(key) !=
      builtin_op_registration_map_.end()) {
    return builtin_op_registration_map_.at(key).get();
  }

  return nullptr;
}

KernelEvalFuncPtr LoggingOpResolver::GetWrappedKernelInvoke(BuiltinOperator op,
                                                            int version) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc mht_2(mht_2_v, 267, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver.cc", "LoggingOpResolver::GetWrappedKernelInvoke");

  return builtin_op_evalfn_map_.at({op, version});
}

const TfLiteRegistration* LoggingOpResolver::FindOp(const char* op,
                                                    int version) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc mht_3(mht_3_v, 276, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver.cc", "LoggingOpResolver::FindOp");

  CustomOperatorKey key = {op, version};
  if (custom_op_registration_map_.find(key) !=
      custom_op_registration_map_.end()) {
    return custom_op_registration_map_.at(key).get();
  }

  return nullptr;
}

KernelEvalFuncPtr LoggingOpResolver::GetWrappedKernelInvoke(const char* op,
                                                            int version) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolverDTcc mht_4(mht_4_v, 291, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver.cc", "LoggingOpResolver::GetWrappedKernelInvoke");

  return custom_op_evalfn_map_.at({op, version});
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
