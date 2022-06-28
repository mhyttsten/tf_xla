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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemms_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemms_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemms_testDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace gpu {
namespace {

class CublasGemmPadForTensorCoresTest : public HloTestBase {
 protected:
  bool PadForF16Gemms(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemms_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms_test.cc", "PadForF16Gemms");

    return CublasPadForGemms(PrimitiveType::F16, 8).Run(module).ValueOrDie();
  }
};

TEST_F(CublasGemmPadForTensorCoresTest, OneDotRootComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33708] parameter(1)
    ROOT %dot.2309 = f16[2048,33708]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
                })")
                    .ValueOrDie();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Shape("f16[2048, 33708]"),
          op::Slice(AllOf(
              op::Shape("f16[2048, 33712]"),
              op::Dot(AllOf(op::Shape("f16[2048, 1024]"),
                            op::Pad(AllOf(op::Shape("f16[2048, 1024]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("f16[]"), op::Constant()))),
                      AllOf(op::Shape("f16[1024, 33712]"),
                            op::Pad(AllOf(op::Shape("f16[1024, 33708]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("f16[]"), op::Constant()))),
                      /*lhs_contracting_dim=*/1,
                      /*rhs_contracting_dim=*/0)))));
}

TEST_F(CublasGemmPadForTensorCoresTest, OneDotS8RootComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = s8[2047,1023] parameter(0)
    %param2 = s8[1023,33707] parameter(1)
    ROOT %dot.2309 = s32[2047,33707]{1,0} dot(s8[2047,1023]{1,0} %param1,
                s8[1023,33707]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
                })")
                    .ValueOrDie();

  EXPECT_TRUE(
      CublasPadForGemms(PrimitiveType::S8, 4).Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Shape("s32[2047, 33707]"),
          op::Slice(AllOf(
              op::Shape("s32[2048, 33708]"),
              op::Dot(AllOf(op::Shape("s8[2048, 1024]"),
                            op::Pad(AllOf(op::Shape("s8[2047, 1023]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("s8[]"), op::Constant()))),
                      AllOf(op::Shape("s8[1024, 33708]"),
                            op::Pad(AllOf(op::Shape("s8[1023, 33707]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("s8[]"), op::Constant()))),
                      /*lhs_contracting_dim=*/1,
                      /*rhs_contracting_dim=*/0)))));
}

TEST_F(CublasGemmPadForTensorCoresTest, TwoDotsComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048, 1024] parameter(0)
    %param2 = f16[1024, 33708] parameter(1)
    %param3 = f16[33708, 1] parameter(2)
    %dot1 = f16[2048, 33708]{1,0} dot(f16[2048, 1024]{1,0} %param1,
                f16[1024, 33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT %dot2 = f16[2048, 1]{1,0} dot(f16[2048, 33708]{1,0} %dot1,
                f16[33708, 1]{0,1} %param3),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })")
                    .ValueOrDie();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Shape("f16[2048, 1]"),
          op::Slice(AllOf(
              op::Shape("f16[2048, 8]"),
              op::Dot(
                  AllOf(
                      op::Shape("f16[2048, 33712]"),
                      AllOf(
                          op::Shape("f16[2048, 33712]"),
                          AllOf(
                              op::Shape("f16[2048, 33712]"),
                              op::Pad(
                                  AllOf(op::Shape("f16[2048, 33708]"),
                                        op::Slice(AllOf(
                                            op::Shape("f16[2048, 33712]"),
                                            op::Dot(
                                                AllOf(op::Shape(
                                                          "f16[2048, 1024]"),
                                                      op::Pad()),
                                                AllOf(op::Shape(
                                                          "f16[1024, 33712]"),
                                                      op::Pad()),
                                                1, 0)))),
                                  AllOf(op::Shape("f16[]"), op::Constant()))))),
                  AllOf(op::Shape("f16[33712, 8]"),
                        AllOf(op::Shape("f16[33712, 8]"),
                              op::Pad(
                                  AllOf(op::Shape("f16[33708, 1]"),
                                        op::Parameter()),
                                  AllOf(op::Shape("f16[]"), op::Constant())))),
                  /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/0)))));

  auto* dot2 = root->operand(0)->operand(0)->operand(0)->operand(0);
  EXPECT_THAT(
      dot2,
      AllOf(op::Dot(
          AllOf(op::Shape("f16[2048, 1024]"),
                op::Pad(AllOf(op::Shape("f16[2048, 1024]"), op::Parameter()),
                        AllOf(op::Shape("f16[]"), op::Constant()))),
          AllOf(op::Shape("f16[1024, 33712]"),
                op::Pad(AllOf(op::Shape("f16[1024, 33708]"), op::Parameter()),
                        AllOf(op::Shape("f16[]"), op::Constant()))),
          /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/0)));
}

TEST_F(CublasGemmPadForTensorCoresTest, DotWithBatchDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[3, 5, 2048, 1024] parameter(0)
    %param2 = f16[3, 5, 1024, 33708] parameter(1)
    ROOT %dot.2309 = f16[3, 5, 2048, 33708]{3, 2, 1,0} dot(f16[3, 5, 2048, 1024]{3, 2, 1,0} %param1,
                f16[3, 5, 1024, 33708]{2, 3, 0,1} %param2), lhs_batch_dims={0, 1}, rhs_batch_dims={0, 1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}})")
                    .ValueOrDie();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Shape("f16[3, 5, 2048, 33708]"),
          op::Slice(AllOf(
              op::Shape("f16[3, 5, 2048, 33712]"),
              op::Dot(AllOf(op::Shape("f16[3, 5, 2048, 1024]"),
                            op::Pad(AllOf(op::Shape("f16[3, 5, 2048, 1024]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("f16[]"), op::Constant()))),
                      AllOf(op::Shape("f16[3, 5, 1024, 33712]"),
                            op::Pad(AllOf(op::Shape("f16[3, 5, 1024, 33708]"),
                                          op::Parameter()),
                                    AllOf(op::Shape("f16[]"), op::Constant()))),
                      /*lhs_contracting_dim=*/3,
                      /*rhs_contracting_dim=*/2)))));
}

TEST_F(CublasGemmPadForTensorCoresTest, NoDotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %x = f32[] parameter(0)
    %y = f32[] parameter(1)
    ROOT %maximum = f32[] maximum(f32[] %x, f32[] %y)
  })")
                    .ValueOrDie();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, F32DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f32[2048,1024] parameter(0)
    %param2 = f32[1024,33708] parameter(1)
    ROOT %dot.2309 = f32[2048,33708]{1,0} dot(f32[2048,1024]{1,0} %param1,
                f32[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .ValueOrDie();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, F64DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f64[2048,1024] parameter(0)
    %param2 = f64[1024,33708] parameter(1)
    ROOT %dot.2309 = f64[2048,33708]{1,0} dot(f64[2048,1024]{1,0} %param1,
                f64[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .ValueOrDie();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, MultiplesOf8DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33712] parameter(1)
    ROOT %dot.2309 = f16[2048,33712]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33712]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .ValueOrDie();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, CheckSavingMetadata) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33708] parameter(1)
    ROOT %dot.2309 = f16[2048,33708]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                metadata={op_type="MatMul" op_name="transformer_v2/Transformer/decode/embedding_shared_weights_1/presoftmax_linear/MatMul"}
                })")
                    .ValueOrDie();

  SCOPED_TRACE(module->ToString());

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  auto metadata = module->entry_computation()->root_instruction()->metadata();
  EXPECT_EQ("MatMul", metadata.op_type());
  EXPECT_EQ(
      "transformer_v2/Transformer/decode/embedding_shared_weights_1/"
      "presoftmax_linear/MatMul",
      metadata.op_name());
}

TEST_F(CublasGemmPadForTensorCoresTest, NotCanonicalizedDot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[3, 5, 2048, 1024] parameter(0)
    %param2 = f16[3, 5, 1024, 33708] parameter(1)
    ROOT %dot.2309 = f16[3,2048, 33708]{2, 1, 0} dot(f16[3, 5, 2048, 1024]{3, 2, 1, 0} %param1, f16[3, 5, 1024, 33708]{3, 2, 1, 0} %param2), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={3, 1}, rhs_contracting_dims={2, 1}})")
                    .ValueOrDie();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
