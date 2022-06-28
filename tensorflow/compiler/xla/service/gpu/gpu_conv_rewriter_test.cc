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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriter_testDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"

#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;

class GpuConvRewriterTest : public HloTestBase {
 public:
  GpuConvRewriterTest()
      : HloTestBase(/*layout_sensitive=*/true,
                    /*allow_mixed_precision=*/false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriter_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter_test.cc", "GpuConvRewriterTest");

    for (int i = 0; i < 2; ++i) {
      WindowDimension* window_dim = default_conv_window_.add_dimensions();
      window_dim->set_size(1);
      window_dim->set_stride(1);
      window_dim->set_padding_low(0);
      window_dim->set_padding_high(0);
      window_dim->set_window_dilation(1);
      window_dim->set_base_dilation(1);
    }
    // TF data shapes are by default in the NHWC order, and filter shape is by
    // default in HWIO order. For backward filter convolution, we need to swap
    // the batch and feature dimension in the activations, and treat the batch
    // dimension in gradients as the input feature dimension in the filter.
    //
    // TODO(jingyue): Add more tests on NCHW input order, which TF also
    // supports.
    tf_default_dnums_for_backward_filter_.set_input_batch_dimension(3);
    tf_default_dnums_for_backward_filter_.set_input_feature_dimension(0);
    tf_default_dnums_for_backward_filter_.add_input_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.add_input_spatial_dimensions(2);
    tf_default_dnums_for_backward_filter_.set_kernel_input_feature_dimension(0);
    tf_default_dnums_for_backward_filter_.set_kernel_output_feature_dimension(
        3);
    tf_default_dnums_for_backward_filter_.add_kernel_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.add_kernel_spatial_dimensions(2);
    tf_default_dnums_for_backward_filter_.add_output_spatial_dimensions(0);
    tf_default_dnums_for_backward_filter_.add_output_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.set_output_batch_dimension(2);
    tf_default_dnums_for_backward_filter_.set_output_feature_dimension(3);

    tf_default_dnums_for_backward_input_.set_input_batch_dimension(0);
    tf_default_dnums_for_backward_input_.set_output_batch_dimension(0);
    tf_default_dnums_for_backward_input_.set_input_feature_dimension(3);
    tf_default_dnums_for_backward_input_.set_output_feature_dimension(3);
    tf_default_dnums_for_backward_input_.add_input_spatial_dimensions(1);
    tf_default_dnums_for_backward_input_.add_output_spatial_dimensions(1);
    tf_default_dnums_for_backward_input_.add_input_spatial_dimensions(2);
    tf_default_dnums_for_backward_input_.add_output_spatial_dimensions(2);
    tf_default_dnums_for_backward_input_.set_kernel_input_feature_dimension(3);
    tf_default_dnums_for_backward_input_.set_kernel_output_feature_dimension(2);
    tf_default_dnums_for_backward_input_.add_kernel_spatial_dimensions(0);
    tf_default_dnums_for_backward_input_.add_kernel_spatial_dimensions(1);
  }

 protected:
  bool RunPass(HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriter_testDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter_test.cc", "RunPass");

    return GpuConvRewriter().Run(module).ValueOrDie();
  }

  // A convolution window with stride 1 and zero padding. The size fields are
  // not set.
  Window default_conv_window_;
  ConvolutionDimensionNumbers tf_default_dnums_for_backward_filter_;
  ConvolutionDimensionNumbers tf_default_dnums_for_backward_input_;
};

TEST_F(GpuConvRewriterTest, BackwardFilterConvolve) {
  HloComputation::Builder builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 2, 1}), "gradients"));
  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_size(2);
  conv_window.mutable_dimensions(1)->set_window_dilation(2);
  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          activations->shape(), gradients->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_filter_,
          /*preferred_element_type=*/absl::nullopt)
          .ConsumeValueOrDie(),
      activations, gradients, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  OpMetadata metadata;
  metadata.set_op_name("foo");
  conv->set_metadata(metadata);

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardFilterCallTarget), 0));

  // Check that metadata was preserved.
  const auto& md_after_opt =
      entry_computation->root_instruction()->operand(0)->metadata();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(md_after_opt, metadata))
      << md_after_opt.DebugString() << " vs " << metadata.DebugString();
}

TEST_F(GpuConvRewriterTest,
       BackwardFilterConvolveEquivalentToForwardConvolution) {
  HloComputation::Builder builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "gradients"));
  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_size(3);
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          activations->shape(), gradients->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_filter_,
          /*preferred_element_type=*/absl::nullopt)
          .ConsumeValueOrDie(),
      activations, gradients, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::GetTupleElement(op::CustomCall(kCudnnConvForwardCallTarget), 0));
}

// Extracted from block35 training.
TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithPaddedActivations) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "gradients"));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(35);
    conv_window.mutable_dimensions(i)->set_padding_low(1);
    conv_window.mutable_dimensions(i)->set_padding_high(1);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {32, 3, 3, 32}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardFilterCallTarget), 0));
}

// Extracted from inception v3 training.
TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithPaddedGradients) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {20, 4, 4, 320}), "gradients"));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(4);
    conv_window.mutable_dimensions(i)->set_padding_high(-1);
    conv_window.mutable_dimensions(i)->set_window_dilation(2);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {320, 3, 3, 192}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardFilterCallTarget), 0));
}

TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithUnevenPadding) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "gradients"));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(35);
    // Uneven padding: padding_low=0, padding_high=1
    conv_window.mutable_dimensions(i)->set_padding_high(1);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {32, 2, 2, 32}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardFilterCallTarget), 0));
}

TEST_F(GpuConvRewriterTest, BackwardInputConvolveEvenPadding) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {4, 5, 16, 16}), "output"));
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {5, 3, 7, 7}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {2, 3}));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(7);
    conv_window.mutable_dimensions(i)->set_padding_low(3);
    conv_window.mutable_dimensions(i)->set_padding_high(3);
  }
  ConvolutionDimensionNumbers conv_dnums;
  conv_dnums.set_input_batch_dimension(0);
  conv_dnums.set_output_batch_dimension(0);
  conv_dnums.set_input_feature_dimension(1);
  conv_dnums.set_output_feature_dimension(1);
  conv_dnums.add_input_spatial_dimensions(2);
  conv_dnums.add_output_spatial_dimensions(2);
  conv_dnums.add_input_spatial_dimensions(3);
  conv_dnums.add_output_spatial_dimensions(3);
  conv_dnums.set_kernel_input_feature_dimension(0);
  conv_dnums.set_kernel_output_feature_dimension(1);
  conv_dnums.add_kernel_spatial_dimensions(2);
  conv_dnums.add_kernel_spatial_dimensions(3);

  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {4, 3, 16, 16}), /*lhs=*/output,
      /*rhs=*/reverse_kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window, conv_dnums,
      DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(),
      ShapeInference::InferConvolveShape(
          output->shape(), reverse_kernel->shape(),
          /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
          conv_dnums, /*preferred_element_type=*/absl::nullopt)
          .ValueOrDie()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));

  ASSERT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardInputCallTarget), 0));
  const HloInstruction* custom_call =
      entry_computation->root_instruction()->operand(0);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = custom_call->window().dimensions(i);
    // Low padding of the backward input convolution
    //   = kernel_size - 1 - low padding on gradients.
    EXPECT_EQ(3, window_dim.padding_low());
    EXPECT_EQ(3, window_dim.padding_high());
    EXPECT_EQ(1, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
}

// Convolve([abc], [x], base_dilation=2)
//   = Convolve([abc], Reverse([x]), base_dilation=2)
//   = BackwardInputConvolve([abc], [x], stride=2)
TEST_F(GpuConvRewriterTest, BackwardInputConvolve1x1Filter) {
  auto builder = HloComputation::Builder(TestName());
  // NHWC dimension order.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // HWOI dimension order.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), "kernel"));

  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_base_dilation(2);

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          output->shape(), kernel->shape(),
          /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_input_,
          /*preferred_element_type=*/absl::nullopt)
          .ConsumeValueOrDie(),
      /*lhs=*/output, /*rhs=*/kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardInputCallTarget), 0));
}

// BackwardInputConvolve([abc], [x], stride=1) is equivalent to
// ForwardConvolve([abc], [x], stride=1). No need to fold it into backward input
// convolution.
TEST_F(GpuConvRewriterTest,
       BackwardInputConvolve1x1FilterEquivalentToForwardConvolve) {
  auto builder = HloComputation::Builder(TestName());
  // NHWC dimension order.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // HWOI dimension order.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), "kernel"));

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          output->shape(), kernel->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, default_conv_window_,
          tf_default_dnums_for_backward_input_,
          /*preferred_element_type=*/absl::nullopt)
          .ConsumeValueOrDie(),
      /*lhs=*/output, /*rhs=*/kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, default_conv_window_,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::GetTupleElement(op::CustomCall(kCudnnConvForwardCallTarget), 0));
}

// Extracted from Inception V3 training.
//
//                                  filter(HWIO)
//                                  3x3x192x320
//                                      |
//                                      v
//      gradients(NHWC)              reverse
//        20x4x4x320               3x3x192x320
//                    \            /
//                     \          /
//  conv (NHWC) with padding (low=2,high=3,interior=1)
//                     20x10x10x192
//
// Gradients are padded unevenly.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveUnevenPaddingOnGradients) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 4, 4, 320}), "output"));
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {3, 3, 192, 320}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(3);
    conv_window.mutable_dimensions(i)->set_padding_low(2);
    conv_window.mutable_dimensions(i)->set_padding_high(3);
    // Interior padding = 1.
    conv_window.mutable_dimensions(i)->set_base_dilation(2);
  }
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/absl::nullopt)
                         .ValueOrDie()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardInputCallTarget), 0));
  const HloInstruction* custom_call =
      entry_computation->root_instruction()->operand(0);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = custom_call->window().dimensions(i);
    EXPECT_EQ(0, window_dim.padding_low());
    EXPECT_EQ(0, window_dim.padding_high());
    EXPECT_EQ(2, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
}

// Similar to BackwardInputConvolveUnevenPadding, but the low padding of the
// gradients exceeds kernel_size - 1. Therefore, this pattern cannot be fused.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveLowPaddingTooLarge) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 4, 4, 320}), "output"));
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {3, 3, 192, 320}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(3);
    conv_window.mutable_dimensions(i)->set_padding_low(3);
    conv_window.mutable_dimensions(i)->set_padding_high(2);
    conv_window.mutable_dimensions(i)->set_base_dilation(2);
  }
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/absl::nullopt)
                         .ValueOrDie()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::GetTupleElement(op::CustomCall(kCudnnConvForwardCallTarget), 0));
}

// Extracted from Resnet-50.
//
// For simplicity, we focus on the column dimension and ignore other dimensions.
// We use [?] to represent the shape instead of the content.
//
// Suppose operator FC does
//   [4] = conv([14], [3], stride=2, padding_high=1)  // Padding::kSame
//
// BC = BackwardInput(FC) does:
//   [14] = conv([7], reverse([3]),
//               padding_low=2, padding_high=1, base_dilation=2)
//
// We should fuse BC even though padding on activations is uneven, because
// GpuConvPaddingLegalization will canonicalize the fusion HLO.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveUnevenPaddingOnActivations) {
  auto builder = HloComputation::Builder(TestName());
  // The gradients are in NCHW layout.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 7, 1}), "output"));
  // The kernel is in HWIO layout.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 3, 1, 1}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  WindowDimension* forward_conv_col_dim = conv_window.mutable_dimensions(1);
  forward_conv_col_dim->set_size(3);
  forward_conv_col_dim->set_padding_low(2);
  forward_conv_col_dim->set_padding_high(1);
  forward_conv_col_dim->set_base_dilation(2);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {1, 1, 14, 1}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/absl::nullopt)
                         .ValueOrDie()));

  auto module = CreateNewVerifiedModule();
  const HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardInputCallTarget), 0));
  const WindowDimension& backward_conv_col_dim =
      entry_computation->root_instruction()->operand(0)->window().dimensions(1);
  EXPECT_EQ(0, backward_conv_col_dim.padding_low());
  EXPECT_EQ(1, backward_conv_col_dim.padding_high());
}

// For simplicity, we focus on the column dimension and ignore other dimensions.
// We use [?] to represent the shape instead of the content.
//
// Suppose operator FC does
//   [3] = conv([4], [2], padding_low=1, padding_high=-1)
//
// BC = BackwardInput(FC) does:
//   [4] = conv([3], reverse([2]), padding_high=2)
//
// We currently don't fuse BC because GpuConvPaddingLegalization
// doesn't support negative padding on the gradients of backward convolution
// (b/32744257).
TEST_F(GpuConvRewriterTest,
       BackwardInputConvolveNegativePaddingHighOnActivations) {
  auto builder = HloComputation::Builder(TestName());
  // The gradients are in NCHW layout.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // The kernel is in HWIO layout.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 2, 1, 1}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  WindowDimension* forward_conv_col_dim = conv_window.mutable_dimensions(1);
  forward_conv_col_dim->set_size(2);
  forward_conv_col_dim->set_padding_high(2);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {1, 1, 4, 1}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/absl::nullopt)
                         .ValueOrDie()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::GetTupleElement(op::CustomCall(kCudnnConvForwardCallTarget), 0));
}

// Check that we will materialize a reversed version of a constant in order to
// pattern-match a backwards input convolution.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveConstantFilter) {
  Array4D<float> constant_arr(4, 4, 2, 2);
  constant_arr.FillIota(0);
  std::string constant_str =
      LiteralUtil::CreateR4FromArray4D(constant_arr).ToStringWithoutShape();

  const std::string module_str = absl::StrFormat(R"(
    HloModule test

    ENTRY entry_computation {
      param0 = f32[128,2,16,16]{3,2,1,0} parameter(0)
      constant = f32[4,4,2,2]{3,2,1,0} constant(%s)
      ROOT convolution = f32[128,2,32,32]{3,2,1,0} convolution(param0, constant),
          window={size=4x4 pad=2_2x2_2 lhs_dilate=2x2},
          dim_labels=bf01_01oi->bf01, feature_group_count=1
    })",
                                                 constant_str);
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      op::GetTupleElement(op::CustomCall(kCudnnConvBackwardInputCallTarget, _,
                                         op::Reverse(op::Constant())),
                          0));
}

TEST_F(GpuConvRewriterTest, TestBackwardFilterPattern) {
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test {
      input = f32[8,120,256,256] parameter(0)
      filter = f32[8,120,256,256] parameter(1)

      ROOT conv = f32[120,120,3,3] convolution(input, filter), window={size=256x256 pad=1_1x1_1}, dim_labels=fb01_io01->fb01
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::GetTupleElement(
                  op::CustomCall(kCudnnConvBackwardFilterCallTarget, _, _), 0));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
