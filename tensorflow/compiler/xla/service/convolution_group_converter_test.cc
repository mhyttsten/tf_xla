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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc() {
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

#include "tensorflow/compiler/xla/service/convolution_group_converter.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using ConvolutionGroupConverterTest = HloTestBase;
namespace op = testing::opcode_matchers;

TEST_F(ConvolutionGroupConverterTest,
       ConvertFeatureGroupCountEqualToInputFeatureDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,2], filter: f32[1,1,2]) -> f32[1,2,2] {
  %input = f32[1,2,2]{2,1,0} parameter(0)
  %copy = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
  %filter = f32[1,1,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy, f32[1,1,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  auto cost_model = [](HloInstruction* conv) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  ConvolutionGroupConverter converter(should_expand, cost_model,
                                      /*convert_batch_groups_only=*/false);
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  // Make sure the convolution is converted to one with feature_group_count = 1.
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->feature_group_count(), 1);
  // Verify that the filter operand has been replaced.
  EXPECT_THAT(root->operand(1),
              op::Select(op::Eq(op::Broadcast(op::Constant()),
                                op::Broadcast(op::Constant())),
                         op::Broadcast(op::Reshape(op::Parameter())),
                         op::Broadcast(op::Constant())));
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertFeatureGroupCountDivisorOfInputFeatureDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,4], filter: f32[1,2,2]) -> f32[1,2,2] {
  %input = f32[1,2,4]{2,1,0} parameter(0)
  %copy = f32[1,2,4]{2,0,1} copy(f32[1,2,4]{2,1,0} %input)
  %filter = f32[1,2,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,4]{2,0,1} %copy, f32[1,2,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  auto cost_model = [](HloInstruction* conv) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      false);
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  // Make sure the convolution is replaced with a reshape.
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->operand(0)->feature_group_count(), 1);
  EXPECT_EQ(root->operand(0)->shape().rank(), 4);
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertBatchGroupCountEqualToInputBatchDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[16,19,19,512]{3,2,1,0}, filter: f32[16,19,19,512]{3,2,1,0}) -> f32[3,3,512,1]{3,2,1,0} {
  %input = f32[16,19,19,512]{3,2,1,0} parameter(0)
  %filter = f32[16,19,19,512]{3,2,1,0} parameter(1)
  ROOT %convolution = f32[3,3,512,1]{3,2,1,0} convolution(f32[16,19,19,512]{3,2,1,0} %input, f32[16,19,19,512]{3,2,1,0} %filter), window={size=19x19 pad=1_1x1_1}, dim_labels=f01b_i01o->01fb, batch_group_count=512
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_4(mht_4_v, 294, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  auto cost_model = [](HloInstruction* conv) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return false; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      true);
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();

  // Verify that the convolution is replaced by a convert.
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  // Make sure the convert is being fed by a reduce window.
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kReduceWindow);
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertBatchGroupCountNotEqualToInputBatchDim) {
  std::string hlo_string = R"(HloModule m
  ENTRY main {
  %input = f32[1,1,1,4] parameter(0)
  %filter = f32[1,1,1,2] parameter(1)
  ROOT %convolution = f32[1,1,2,2] convolution(%input,%filter),
      window={size=1x1}, dim_labels=f01b_i01o->01fb, batch_group_count=2
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_6(mht_6_v, 329, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return true; };
  auto cost_model = [](HloInstruction* conv) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_group_converter_testDTcc mht_7(mht_7_v, 333, "", "./tensorflow/compiler/xla/service/convolution_group_converter_test.cc", "lambda");
 return false; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      true);
  // Make sure that batch group count is rewritten even if
  // batch_group_count == output_feature but not input_batch
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace xla
