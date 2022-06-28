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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc() {
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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

// The signature contains namespaces (Ex: mlir::TFL::(anonymous namespace)::).
// So only extract the function name as the pass name.
inline std::string extract_pass_name(const std::string &signature) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("signature: \"" + signature + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "extract_pass_name");

  const std::vector<std::string> &v = absl::StrSplit(signature, "::");
  return v.back();
}

// Errors raised by emitOpError start with "'<dialect>.<op>' op". Returns an
// empty string if the pattern is not found or the operator is not in tf or tfl
// dialect.
inline std::string extract_op_name_from_error_message(
    const std::string &error_message) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "extract_op_name_from_error_message");

  int end_pos = error_message.find("' op");
  if ((absl::StartsWith(error_message, "'tf.") ||
       absl::StartsWith(error_message, "'tfl.")) &&
      end_pos != std::string::npos) {
    return error_message.substr(1, end_pos - 1);
  }
  return "";
}

// Only notes with character count smaller than kMaxAcceptedNoteSize will be
// appended to the error message.
const int kMaxAcceptedNoteSize = 1024;
}  // namespace

ErrorCollectorInstrumentation::ErrorCollectorInstrumentation(
    MLIRContext *context)
    : error_collector_(ErrorCollector::GetErrorCollector()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "ErrorCollectorInstrumentation::ErrorCollectorInstrumentation");

  handler_.reset(new ScopedDiagnosticHandler(context, [this](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      Location loc = diag.getLocation();
      std::string error_message = diag.str();
      std::string op_name, error_code;
      if (loc_to_name_.count(loc)) {
        op_name = loc_to_name_[loc];
      } else {
        op_name = extract_op_name_from_error_message(diag.str());
      }

      for (const auto &note : diag.getNotes()) {
        const std::string note_str = note.str();
        if (note_str.rfind(kErrorCodePrefix, 0) == 0) {
          error_code = note_str.substr(sizeof(kErrorCodePrefix) - 1);
        }

        error_message += "\n";
        if (note_str.size() <= kMaxAcceptedNoteSize) {
          error_message += note_str;
        } else {
          error_message += note_str.substr(0, kMaxAcceptedNoteSize);
          error_message += "...";
        }
      }

      ErrorCode error_code_enum = ConverterErrorData::UNKNOWN;
      bool has_valid_error_code =
          ConverterErrorData::ErrorCode_Parse(error_code, &error_code_enum);
      if (!op_name.empty() || has_valid_error_code) {
        error_collector_->ReportError(NewConverterErrorData(
            pass_name_, error_message, error_code_enum, op_name, loc));
      } else {
        common_error_message_ += diag.str();
        common_error_message_ += "\n";
      }
    }
    return failure();
  }));
}

void ErrorCollectorInstrumentation::runBeforePass(Pass *pass,
                                                  Operation *module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_3(mht_3_v, 280, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "ErrorCollectorInstrumentation::runBeforePass");

  // Find the op names with tf or tfl dialect prefix, Ex: "tf.Abs" or "tfl.Abs".
  auto collectOps = [this](Operation *op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "lambda");

    const auto &op_name = op->getName().getStringRef().str();
    if (absl::StartsWith(op_name, "tf.") || absl::StartsWith(op_name, "tfl.")) {
      loc_to_name_.emplace(op->getLoc(), op_name);
    }
  };

  for (auto &region : module->getRegions()) {
    region.walk(collectOps);
  }

  pass_name_ = extract_pass_name(pass->getName().str());
  error_collector_->Clear();
}

void ErrorCollectorInstrumentation::runAfterPass(Pass *pass,
                                                 Operation *module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_5(mht_5_v, 304, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "ErrorCollectorInstrumentation::runAfterPass");

  loc_to_name_.clear();
  pass_name_.clear();
  common_error_message_.clear();
  error_collector_->Clear();
}

void ErrorCollectorInstrumentation::runAfterPassFailed(Pass *pass,
                                                       Operation *module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_instDTcc mht_6(mht_6_v, 315, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst.cc", "ErrorCollectorInstrumentation::runAfterPassFailed");

  // Create a new error if no errors collected yet.
  if (error_collector_->CollectedErrors().empty() &&
      !common_error_message_.empty()) {
    error_collector_->ReportError(NewConverterErrorData(
        pass_name_, common_error_message_, ConverterErrorData::UNKNOWN,
        /*op_name=*/"", module->getLoc()));
  }

  loc_to_name_.clear();
  pass_name_.clear();
  common_error_message_.clear();
}

}  // namespace TFL
}  // namespace mlir
