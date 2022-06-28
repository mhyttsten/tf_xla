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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_conv2dDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_conv2dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_conv2dDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv_impl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenConv2DF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs,
    int64_t input_batch, int64_t input_rows, int64_t input_cols,
    int64_t input_channels, int64_t kernel_rows, int64_t kernel_cols,
    int64_t kernel_channels, int64_t kernel_filters, int64_t output_rows,
    int64_t output_cols, int64_t row_stride, int64_t col_stride,
    int64_t padding_top, int64_t padding_bottom, int64_t padding_left,
    int64_t padding_right, int64_t lhs_row_dilation, int64_t lhs_col_dilation,
    int64_t rhs_row_dilation, int64_t rhs_col_dilation,
    int64_t feature_group_count) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tensorflow::xla::EigenConv2DImpl(
      *run_options->intra_op_thread_pool(), out, lhs, rhs, input_batch,
      input_rows, input_cols, input_channels, kernel_rows, kernel_cols,
      kernel_channels, kernel_filters, output_rows, output_cols, row_stride,
      col_stride, padding_top, padding_bottom, padding_left, padding_right,
      lhs_row_dilation, lhs_col_dilation, rhs_row_dilation, rhs_col_dilation,
      feature_group_count);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenConv2DF16(
    const void* run_options_ptr, Eigen::half* out, Eigen::half* lhs,
    Eigen::half* rhs, int64_t input_batch, int64_t input_rows,
    int64_t input_cols, int64_t input_channels, int64_t kernel_rows,
    int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_rows, int64_t output_cols, int64_t row_stride,
    int64_t col_stride, int64_t padding_top, int64_t padding_bottom,
    int64_t padding_left, int64_t padding_right, int64_t lhs_row_dilation,
    int64_t lhs_col_dilation, int64_t rhs_row_dilation,
    int64_t rhs_col_dilation, int64_t feature_group_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_conv2dDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xla/service/cpu/runtime_conv2d.cc", "__xla_cpu_runtime_EigenConv2DF16");

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tensorflow::xla::EigenConv2DImpl(
      *run_options->intra_op_thread_pool(), out, lhs, rhs, input_batch,
      input_rows, input_cols, input_channels, kernel_rows, kernel_cols,
      kernel_channels, kernel_filters, output_rows, output_cols, row_stride,
      col_stride, padding_top, padding_bottom, padding_left, padding_right,
      lhs_row_dilation, lhs_col_dilation, rhs_row_dilation, rhs_col_dilation,
      feature_group_count);
}
