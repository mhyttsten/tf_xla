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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_util_testDTcc() {
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

#include "llvm/ADT/Twine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace mlir {
namespace {

using testing::HasSubstr;

TEST(ErrorUtilTest, StatusScopedDiagnosticHandler) {
  MLIRContext context;
  auto id = StringAttr::get(&context, "//tensorflow/python/test.py");
  auto loc = FileLineColLoc::get(&context, id, 0, 0);

  // Test OK without diagnostic gets passed through.
  {
    TF_ASSERT_OK(StatusScopedDiagnosticHandler(&context).Combine(Status::OK()));
  }

  // Verify diagnostics are captured as Unknown status.
  {
    StatusScopedDiagnosticHandler handler(&context);
    emitError(loc) << "Diagnostic message";
    ASSERT_TRUE(tensorflow::errors::IsUnknown(handler.ConsumeStatus()));
  }

  // Verify passed in errors are propagated.
  {
    Status err = tensorflow::errors::Internal("Passed in error");
    ASSERT_TRUE(tensorflow::errors::IsInternal(
        StatusScopedDiagnosticHandler(&context).Combine(err)));
  }

  // Verify diagnostic reported are append to passed in error.
  {
    auto function = [&]() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSerror_util_testDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/utils/error_util_test.cc", "lambda");

      emitError(loc) << "Diagnostic message reported";
      emitError(loc) << "Second diagnostic message reported";
      return tensorflow::errors::Internal("Passed in error");
    };
    StatusScopedDiagnosticHandler ssdh(&context);
    Status s = ssdh.Combine(function());
    ASSERT_TRUE(tensorflow::errors::IsInternal(s));
    EXPECT_THAT(s.error_message(), HasSubstr("Passed in error"));
    EXPECT_THAT(s.error_message(), HasSubstr("Diagnostic message reported"));
    EXPECT_THAT(s.error_message(),
                HasSubstr("Second diagnostic message reported"));
  }
}

TEST(ErrorUtilTest, StatusScopedDiagnosticHandlerWithFilter) {
  // Filtering logic is based on tensorflow::IsInternalFrameForFilename()
  // Note we are surfacing the locations that are NOT internal frames
  // so locations that fail IsInternalFrameForFilename() evaluation pass the
  // filter.

  // These locations will fail the IsInternalFrameForFilename() check so will
  // pass the filter.
  MLIRContext context;
  auto id =
      StringAttr::get(&context, "//tensorflow/python/keras/keras_file.py");
  auto loc = FileLineColLoc::get(&context, id, 0, 0);
  auto id2 =
      StringAttr::get(&context, "//tensorflow/python/something/my_test.py");
  auto loc2 = FileLineColLoc::get(&context, id2, 0, 0);
  auto id3 = StringAttr::get(&context, "python/tensorflow/show_file.py");
  auto loc3 = FileLineColLoc::get(&context, id3, 0, 0);

  // These locations will be evalauted as internal frames, passing the
  // IsInternalFramesForFilenames() check so will be filtered out.
  auto id_filtered =
      StringAttr::get(&context, "//tensorflow/python/dir/filtered_file_A.py");
  auto loc_filtered = FileLineColLoc::get(&context, id_filtered, 0, 0);
  auto id_filtered2 =
      StringAttr::get(&context, "dir/tensorflow/python/filtered_file_B.py");
  auto loc_filtered2 = FileLineColLoc::get(&context, id_filtered2, 0, 0);

  // Build a small stack for each error; the MLIR diagnostic filtering will
  // surface a location that would otherwise be filtered if it is the only
  // location associated with an error; therefore we need a combinatination of
  // locations to test.
  auto callsite_loc = mlir::CallSiteLoc::get(loc, loc_filtered);
  auto callsite_loc2 = mlir::CallSiteLoc::get(loc2, loc_filtered2);
  auto callsite_loc3 = mlir::CallSiteLoc::get(loc_filtered2, loc3);

  // Test with filter on.
  StatusScopedDiagnosticHandler ssdh_filter(&context, false, true);
  emitError(callsite_loc) << "Error 1";
  emitError(callsite_loc2) << "Error 2";
  emitError(callsite_loc3) << "Error 3";
  Status s_filtered = ssdh_filter.ConsumeStatus();
  // Check for the files that should not be filtered.
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("keras"));
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("test.py"));
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("show_file"));
  // Verify the filtered files are not present.
  EXPECT_THAT(s_filtered.error_message(), Not(HasSubstr("filtered_file")));
}

TEST(ErrorUtilTest, StatusScopedDiagnosticHandlerWithoutFilter) {
  // Filtering logic should be off so all files should 'pass'.
  MLIRContext context;
  // This file would pass the filter if it was on.
  auto id =
      StringAttr::get(&context, "//tensorflow/python/keras/keras_file.py");
  auto loc = FileLineColLoc::get(&context, id, 0, 0);

  // The '_filtered' locations would be evaluated as internal frames, so would
  // not pass the filter if it was on.
  auto id_filtered =
      StringAttr::get(&context, "//tensorflow/python/dir/filtered_file_A.py");
  auto loc_filtered = FileLineColLoc::get(&context, id_filtered, 0, 0);
  auto id_filtered2 =
      StringAttr::get(&context, "dir/tensorflow/python/filtered_file_B.py");
  auto loc_filtered2 = FileLineColLoc::get(&context, id_filtered2, 0, 0);
  auto id_filtered3 =
      StringAttr::get(&context, "//tensorflow/python/something/my_op.py");
  auto loc_filtered3 = FileLineColLoc::get(&context, id_filtered3, 0, 0);

  // Build a small stack for each error; the MLIR diagnostic filtering will
  // surface a location that would otherwise be filtered if it is the only
  // location associated with an error; therefore we need a combinatination of
  // locations to test.
  auto callsite_loc = mlir::CallSiteLoc::get(loc, loc_filtered);
  auto callsite_loc2 = mlir::CallSiteLoc::get(loc_filtered3, loc_filtered2);

  // Test with filter off.
  StatusScopedDiagnosticHandler ssdh_no_filter(&context, false, false);
  emitError(callsite_loc) << "Error 1";
  emitError(callsite_loc2) << "Error 2";
  Status s_no_filter = ssdh_no_filter.ConsumeStatus();
  // All files should be present, especially the 'filtered' ones.
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("keras"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("my_op"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("filtered_file_A"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("filtered_file_B"));
}

}  // namespace
}  // namespace mlir
