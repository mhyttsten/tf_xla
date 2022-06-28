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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc() {
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

#include "tensorflow/compiler/xla/tests/conv_depthwise_common.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
std::string GetFloatDataType(bool use_bfloat16) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/tests/conv_depthwise_common.cc", "GetFloatDataType");

  return use_bfloat16 ? "bf16" : "f32";
}

std::string DepthwiseConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<DepthwiseConvolution2DSpec, bool>>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xla/tests/conv_depthwise_common.cc", "DepthwiseConvolution2DTestDataToString");

  const auto& spec = ::testing::get<0>(data.param);
  const std::string data_type = GetFloatDataType(::testing::get<1>(data.param));
  std::string str = absl::StrCat(
      "activation_dims_", absl::StrJoin(spec.activation_dims, "x"),
      "_activation_layout_", absl::StrJoin(spec.activation_layout, "_"),
      "_kernel_dims_", absl::StrJoin(spec.kernel_dims, "x"), "_kernel_layout_",
      absl::StrJoin(spec.kernel_layout, "_"), "_output_dims_",
      absl::StrJoin(spec.output_dims, "x"), "_output_layout_",
      absl::StrJoin(spec.output_layout, "_"), data_type);
  // -1 indicates non-existence.
  if (spec.stride != -1) {
    absl::StrAppend(&str, "_lhs_dilation_", spec.lhs_dilate, "x1");
  }

  // Test names are not allowed to contain the '-' character.
  absl::c_replace(str, '-', 'n');
  return str;
}

std::string BuildHloTextDepthwiseConvolution2D(
    const DepthwiseConvolution2DSpec& spec, bool use_bfloat16,
    bool is_scheduled) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_commonDTcc mht_2(mht_2_v, 233, "", "./tensorflow/compiler/xla/tests/conv_depthwise_common.cc", "BuildHloTextDepthwiseConvolution2D");

  const std::string data_type = GetFloatDataType(use_bfloat16);
  const std::string sched_tag = is_scheduled ? ", is_scheduled=true " : "";
  if (spec.activation_dims[1] == 1 && spec.kernel_dims[1] == 2) {
    return absl::StrFormat(
        R"(
    HloModule TensorFlowDepthwiseConv %s
    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d  pad=1_1x%d_%d rhs_dilate=1x%d}, dim_labels=b01f_01io->b01f,
          feature_group_count=%d
    }
    )",
        sched_tag, data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.window, spec.window, spec.window, spec.output_feature);

  } else if (spec.stride == -1) {
    return absl::StrFormat(
        R"(
      HloModule TensorFlowDepthwiseConv %s
      ENTRY main {
        activation = %s[%s]{%s} parameter(0)
        kernel = %s[%s]{%s} parameter(1)
        ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
            window={size=%dx%d}, dim_labels=b01f_01io->b01f,
            feature_group_count=%d
      }
      )",
        sched_tag, data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.output_feature);
  } else {
    return absl::StrFormat(
        R"(
    HloModule TensorFlowDepthwiseConv %s

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d stride=%dx1 pad=%d_%dx0_0 lhs_dilate=%dx1}, 
          dim_labels=b01f_01io->b01f, feature_group_count=%d
    }
    )",
        sched_tag, data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.stride, 0, 0, spec.lhs_dilate, spec.output_feature);
  }
}
}  // namespace xla
