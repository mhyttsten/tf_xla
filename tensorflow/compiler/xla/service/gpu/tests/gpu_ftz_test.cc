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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc() {
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

#include <memory>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

// Check that the ftz (flush denormals to zero) flag is reflected in PTX as
// expected.

namespace xla {
namespace gpu {
namespace {

class GpuFtzTest : public GpuCodegenTest {
 public:
  explicit GpuFtzTest(bool ftz) : ftz_(ftz) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_ftz_test.cc", "GpuFtzTest");
}

  // Creates an HLO module that performs the given binary operation on some
  // data.
  std::unique_ptr<VerifiedHloModule> CreateBinaryOpModule(HloOpcode op) {
    HloComputation::Builder builder(TestName());

    Shape param_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{100, 100}, /*minor_to_major=*/{1, 0});
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/0, param_shape, "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/1, param_shape, "y"));
    builder.AddInstruction(HloInstruction::CreateBinary(param_shape, op, x, y));

    auto hlo_module = CreateNewVerifiedModuleWithFTZ(ftz_);
    hlo_module->AddEntryComputation(builder.Build());
    return hlo_module;
  }

  // Creates an HLO module that performs the given unary operation on some data.
  std::unique_ptr<VerifiedHloModule> CreateUnaryOpModule(HloOpcode op) {
    HloComputation::Builder builder(TestName());

    Shape param_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{100, 100}, /*minor_to_major=*/{1, 0});
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/0, param_shape, "x"));
    builder.AddInstruction(HloInstruction::CreateUnary(param_shape, op, x));

    auto hlo_module = CreateNewVerifiedModuleWithFTZ(ftz_);
    hlo_module->AddEntryComputation(builder.Build());
    return hlo_module;
  }

  bool ftz_;
};

class GpuFtzEnabledTest : public GpuFtzTest {
 public:
  GpuFtzEnabledTest() : GpuFtzTest(/*ftz=*/true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_ftz_test.cc", "GpuFtzEnabledTest");
}
};

class GpuFtzDisabledTest : public GpuFtzTest {
 public:
  GpuFtzDisabledTest() : GpuFtzTest(/*ftz=*/false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_ftz_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_ftz_test.cc", "GpuFtzDisabledTest");
}
};

// Check that we emit mul.ftz.f32 when in ftz mode, and plain mul.f32 otherwise.
TEST_F(GpuFtzEnabledTest, MultiplyFtz) {
  CompileAndOptionallyVerifyPtx(CreateBinaryOpModule(HloOpcode::kMultiply), R"(
    CHECK-NOT: mul.rn.f32
    CHECK: mul.rn.ftz.f32
    CHECK-NOT: mul.rn.f32
  )");
}
TEST_F(GpuFtzDisabledTest, MultiplyFtz) {
  CompileAndOptionallyVerifyPtx(CreateBinaryOpModule(HloOpcode::kMultiply), R"(
    CHECK-NOT: mul.rn.ftz.f32
    CHECK: mul.rn.f32
    CHECK-NOT: mul.rn.ftz.f32
  )");
}

// In NVPTX, exp(float) is implemented in libdevice, and consults __nvvm_reflect
// to determine whether or not ftz is enabled.
// The implementation in CUDA 11 uses one ex2.approx.ftz, irrespective of ftz
// being enabled or not. In previous CUDA versions, there is a leading
// ex2.approx that does obey the ftz setting.
// Instead of pattern matching implementation details, it might be better to
// value-test the actual result instead. TODO(csigg): change to value-test.
TEST_F(GpuFtzEnabledTest, ExpFtz) {
  CompileAndOptionallyVerifyPtx(CreateUnaryOpModule(HloOpcode::kExp), R"(
    CHECK-NOT: ex2.approx.f32
    CHECK:     ex2.approx.ftz.f32
    CHECK-NOT: ex2.approx.f32
  )");
}

TEST_F(GpuFtzDisabledTest, ExpFtz) {
  CompileAndOptionallyVerifyPtx(CreateUnaryOpModule(HloOpcode::kExp), R"(
    CHECK:     ex2.approx.ftz.f32
    CHECK-NOT: ex2.approx.f32
    CHECK-NOT: ex2.approx.ftz.f32
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
