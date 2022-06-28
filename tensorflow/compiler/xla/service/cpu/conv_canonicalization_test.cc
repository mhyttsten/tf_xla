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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalization_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalization_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalization_testDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"

#include <vector>

#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

#include "tensorflow/compiler/xla/test_helpers.h"

namespace xla {
namespace cpu {

using ::testing::ElementsAre;

class ConvCanonicalizationTest : public HloTestBase {
 public:
  ConvCanonicalizationTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalization_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/cpu/conv_canonicalization_test.cc", "ConvCanonicalizationTest");

    for (int i = 0; i < 2; ++i) {
      auto dim = conv_window_.add_dimensions();
      dim->set_size(kWindowSize);
      dim->set_stride(1);
      dim->set_padding_low(0);
      dim->set_padding_high(0);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
    }
  }

 protected:
  Window conv_window_;

  static constexpr int kBatchSize = 50;
  static constexpr int kInputSize = 28;
  static constexpr int kWindowSize = 5;
  static constexpr int kInputFeatureCount = 32;
  static constexpr int kOutputFeatureCount = 64;
};

TEST_F(ConvCanonicalizationTest, NonCanonicalToCanonical) {
  auto builder = HloComputation::Builder(TestName());
  // The input dimensions are in CNHW order.
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kInputFeatureCount, kBatchSize, kInputSize, kInputSize))));
  // The kernel dimensions are in OIHW order.
  auto kernel = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kOutputFeatureCount, kInputFeatureCount, kWindowSize, kWindowSize))));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(1);
  dnums.set_output_batch_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_output_feature_dimension(0);
  auto output_size = kInputSize - kWindowSize + 1;
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(
          F32, {kOutputFeatureCount, kBatchSize, output_size, output_size}),
      input, kernel, /*feature_group_count=*/1, /*batch_group_count=*/1,
      conv_window_, dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  ConvCanonicalization conv_canonicalization(&target_machine_features);
  EXPECT_TRUE(conv_canonicalization.Run(module.get()).ValueOrDie());

  const HloInstruction* output_reshape = entry_computation->root_instruction();
  EXPECT_EQ(HloOpcode::kTranspose, output_reshape->opcode());
  const HloInstruction* canonical_conv = output_reshape->operand(0);
  EXPECT_EQ(HloOpcode::kConvolution, canonical_conv->opcode());
  const HloInstruction* input_reshape = canonical_conv->operand(0);
  EXPECT_EQ(HloOpcode::kTranspose, input_reshape->opcode());
  const HloInstruction* kernel_reshape = canonical_conv->operand(1);
  EXPECT_EQ(HloOpcode::kTranspose, kernel_reshape->opcode());

  // The input is in CNHW order. input_reshape should produce
  // NHWC for the convolution to hit the Eigen fast path.
  EXPECT_THAT(input_reshape->dimensions(), ElementsAre(1, 2, 3, 0));
  // The kernel is in OIHW order. kernel_reshape should produce
  // HWIO for the convolution to hit the Eigen fast path.
  EXPECT_THAT(kernel_reshape->dimensions(), ElementsAre(2, 3, 1, 0));
  // The output of the canonical convolution is in NHWC order (the same as
  // input_reshape's order). output_reshape should restore that order to the
  // order of the computation root (CNHW).
  EXPECT_THAT(output_reshape->dimensions(), ElementsAre(3, 0, 1, 2));
}

TEST_F(ConvCanonicalizationTest, CanonicalStaysTheSame) {
  auto builder = HloComputation::Builder(TestName());
  // The input dimensions are in NHWC order.
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kBatchSize, kInputSize, kInputSize, kInputFeatureCount))));
  // The kernel dimensions are in HWIO order.
  auto kernel = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kWindowSize, kWindowSize, kInputFeatureCount, kOutputFeatureCount))));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  auto output_size = kInputSize - kWindowSize + 1;
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(
          F32, {kBatchSize, output_size, output_size, kOutputFeatureCount}),
      input, kernel, /*feature_group_count=*/1, /*batch_group_count=*/1,
      conv_window_, dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  ConvCanonicalization conv_canonicalization(&target_machine_features);
  EXPECT_FALSE(conv_canonicalization.Run(module.get()).ValueOrDie());
}

}  // namespace cpu
}  // namespace xla
