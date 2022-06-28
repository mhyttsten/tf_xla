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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace mlir {

StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    MLIRContext* context, bool propagate, bool filter_stack)
    : SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_),
      propagate_(propagate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler");

  if (filter_stack) {
    this->shouldShowLocFn = [](Location loc) -> bool {
      // For a Location to be surfaced in the stack, it must evaluate to true.
      // For any Location that is a FileLineColLoc:
      if (FileLineColLoc fileLoc = loc.dyn_cast<FileLineColLoc>()) {
        return !tensorflow::IsInternalFrameForFilename(
            fileLoc.getFilename().str());
      } else {
        // If this is a non-FileLineColLoc, go ahead and include it.
        return true;
      }
    };
  }

  setHandler([this](Diagnostic& diag) { return this->handler(&diag); });
}

StatusScopedDiagnosticHandler::~StatusScopedDiagnosticHandler() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::~StatusScopedDiagnosticHandler");

  // Verify errors were consumed and re-register old handler.
  bool all_errors_produced_were_consumed = ok();
  DCHECK(all_errors_produced_were_consumed) << "Error status not consumed:\n"
                                            << diag_str_;
}

bool StatusScopedDiagnosticHandler::ok() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::ok");
 return diag_str_.empty(); }

Status StatusScopedDiagnosticHandler::ConsumeStatus() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::ConsumeStatus");

  if (ok()) return Status::OK();

  // TODO(jpienaar) This should be combining status with one previously built
  // up.
  Status s = tensorflow::errors::Unknown(diag_str_);
  diag_str_.clear();
  return s;
}

Status StatusScopedDiagnosticHandler::Combine(Status status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_4(mht_4_v, 247, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::Combine");

  if (status.ok()) return ConsumeStatus();

  // status is not-OK here, so if there was no diagnostics reported
  // additionally then return this error.
  if (ok()) return status;

  // Append the diagnostics reported to the status. This repeats the behavior of
  // TensorFlow's AppendToMessage without the additional formatting inserted
  // there.
  status = ::tensorflow::Status(
      status.code(), absl::StrCat(status.error_message(), diag_str_));
  diag_str_.clear();
  return status;
}

LogicalResult StatusScopedDiagnosticHandler::handler(Diagnostic* diag) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_utilDTcc mht_5(mht_5_v, 266, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util.cc", "StatusScopedDiagnosticHandler::handler");

  size_t current_diag_str_size_ = diag_str_.size();

  // Emit the diagnostic and flush the stream.
  emitDiagnostic(*diag);
  diag_stream_.flush();

  // Emit non-errors to VLOG instead of the internal status.
  if (diag->getSeverity() != DiagnosticSeverity::Error) {
    VLOG(1) << diag_str_.substr(current_diag_str_size_);
    diag_str_.resize(current_diag_str_size_);
  }

  // Return failure to signal propagation if necessary.
  return failure(propagate_);
}

}  // namespace mlir
