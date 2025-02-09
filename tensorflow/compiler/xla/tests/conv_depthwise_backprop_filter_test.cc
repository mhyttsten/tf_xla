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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc() {
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
namespace {

std::string GetFloatDataType(bool use_bfloat16) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/tests/conv_depthwise_backprop_filter_test.cc", "GetFloatDataType");

  return use_bfloat16 ? "bf16" : "f32";
}

struct BatchGroupedConvolution2DSpec {
  int64_t output_batch, window, window_dilation;
  std::vector<int64_t> activation_dims;
  std::vector<int64_t> kernel_dims;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> activation_and_kernel_layout;
  std::vector<int64_t> output_layout;
};

class BatchGroupedConvolution2DTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<BatchGroupedConvolution2DSpec, bool>> {};

class BatchGroupedConvolution2DDepthTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<BatchGroupedConvolution2DSpec, bool>> {};

static std::vector<BatchGroupedConvolution2DSpec> GetConv2DTestCases(
    bool use_depth_multiplier) {
  std::vector<BatchGroupedConvolution2DSpec> config_set;
  std::vector<std::vector<int64_t>> config_options = {
      {129, 10, 3, 2}, {4, 3, 3, 258}, {8, 4, 2, 128},
      {8, 3, 2, 256},  {256, 7, 5, 4}, {128, 6, 6, 4},
      {32, 5, 2, 129}, {16, 4, 3, 2},  {16, 3, 2, 64}};

  int64_t counter = 2;
  for (auto option : config_options) {
    int64_t feature = option[3];
    int64_t activation_size = option[1];
    int64_t kernel_size = option[2];
    int64_t batch = option[0];

    BatchGroupedConvolution2DSpec config;
    config.window_dilation = 1;
    config.output_batch = feature;
    config.window = kernel_size;

    config.activation_dims = {batch, activation_size, activation_size, feature};

    const int64_t depthwise_multiplier = use_depth_multiplier ? counter++ : 1;
    config.kernel_dims = {batch, kernel_size, kernel_size,
                          feature * depthwise_multiplier};
    // Don't let the counter grow too much, else the compute demand will grow.
    if (counter == 4) {
      counter = 2;
    }
    int64_t output_space_size = 3 + activation_size - kernel_size;
    config.output_dims = {output_space_size, output_space_size,
                          feature * depthwise_multiplier, 1};

    config.activation_and_kernel_layout = {0, 3, 1, 2};
    config.output_layout = {2, 3, 0, 1};
    config_set.push_back(config);

    BatchGroupedConvolution2DSpec different_layout_config = config;
    different_layout_config.activation_and_kernel_layout = {3, 0, 1, 2};
    config_set.push_back(different_layout_config);

    // Add configurations for window dilation cases.
    if (activation_size % 2 == 0 && activation_size == kernel_size) {
      BatchGroupedConvolution2DSpec config;
      config.window_dilation = 2;
      config.output_batch = feature;
      config.window = kernel_size / 2;
      config.activation_dims = {batch, activation_size, activation_size,
                                feature};
      config.kernel_dims = {batch, kernel_size / 2, kernel_size / 2, feature};
      config.activation_and_kernel_layout = {0, 3, 1, 2};
      config.output_layout = {2, 3, 0, 1};

      int64_t output_space_size = 5;
      config.output_dims = {output_space_size, output_space_size, feature, 1};

      config_set.push_back(config);

      BatchGroupedConvolution2DSpec different_layout_config = config;
      different_layout_config.activation_and_kernel_layout = {3, 0, 1, 2};
      config_set.push_back(different_layout_config);
    }
  }

  return config_set;
}

std::string BatchGroupedConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<BatchGroupedConvolution2DSpec, bool>>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc mht_1(mht_1_v, 294, "", "./tensorflow/compiler/xla/tests/conv_depthwise_backprop_filter_test.cc", "BatchGroupedConvolution2DTestDataToString");

  const auto& spec = ::testing::get<0>(data.param);
  const std::string data_type = GetFloatDataType(::testing::get<1>(data.param));
  std::string str = absl::StrCat(
      "activation_dims_", absl::StrJoin(spec.activation_dims, "x"),
      "_kernel_dims_", absl::StrJoin(spec.kernel_dims, "x"),
      "_activation_layout_",
      absl::StrJoin(spec.activation_and_kernel_layout, "_"), "_output_dims_",
      absl::StrJoin(spec.output_dims, "x"), data_type, "_output_layout_",
      absl::StrJoin(spec.output_layout, "_"));

  // Test names are not allowed to contain the '-' character.
  absl::c_replace(str, '-', 'n');
  return str;
}

std::string BuildHloTextBatchGroupedConvolution2D(
    const BatchGroupedConvolution2DSpec& spec, bool use_bfloat16,
    bool scheduled = false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconv_depthwise_backprop_filter_testDTcc mht_2(mht_2_v, 315, "", "./tensorflow/compiler/xla/tests/conv_depthwise_backprop_filter_test.cc", "BuildHloTextBatchGroupedConvolution2D");

  const std::string data_type = GetFloatDataType(use_bfloat16);
  const std::string scheduled_tag = scheduled ? ",is_scheduled=true" : "";
  return absl::StrFormat(
      R"(
    HloModule TensorFlowDepthwiseConv %s

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d pad=1_%dx1_%d rhs_dilate=%dx%d}, dim_labels=f01b_i01o->01fb,
          batch_group_count=%d
    }
    )",
      scheduled_tag, data_type, absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.output_dims, ","),
      absl::StrJoin(spec.output_layout, ","), data_type,
      absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), spec.window,
      spec.window, spec.window_dilation, spec.window_dilation,
      spec.window_dilation, spec.window_dilation, spec.output_batch);
}

XLA_TEST_P(BatchGroupedConvolution2DTest, DoIt) {
  const BatchGroupedConvolution2DSpec& spec = ::testing::get<0>(GetParam());
  bool use_bfloat16 = ::testing::get<1>(GetParam());

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_BFLOAT16
  if (use_bfloat16) {
    return;
  }
#endif

  const std::string hlo_text = BuildHloTextBatchGroupedConvolution2D(
      spec, use_bfloat16, /*scheduled=*/false);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0.01, 0.01}));
}

INSTANTIATE_TEST_CASE_P(
    BatchGroupedConvolution2DTestWithRandomIndices,
    BatchGroupedConvolution2DTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetConv2DTestCases(/*use_depth_multiplier=*/false)),
        ::testing::Bool()),
    BatchGroupedConvolution2DTestDataToString);

INSTANTIATE_TEST_CASE_P(
    BatchGroupedConvolution2DDepthMultiplierTestWithRandomIndices,
    BatchGroupedConvolution2DTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetConv2DTestCases(/*use_depth_multiplier=*/true)),
        ::testing::Bool()),
    BatchGroupedConvolution2DTestDataToString);

}  // namespace
}  // namespace xla
