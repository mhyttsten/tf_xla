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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_convolution_regression_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_convolution_regression_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_convolution_regression_testDTcc() {
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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class GpuConvolutionRegressionTest : public HloTestBase {
 public:
  // RunHloPasses goes through convolution autotuning, which performs
  // correctness cross-checking.
  void CheckForHloText(absl::string_view hlo_string) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_convolution_regression_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_convolution_regression_test.cc", "CheckForHloText");

    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    (void)backend().compiler()->RunHloPasses(
        ParseAndReturnVerifiedModule(hlo_string, config).ConsumeValueOrDie(),
        backend().default_stream_executor(), backend().memory_allocator());
  }
};

TEST_F(GpuConvolutionRegressionTest, Computation1) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation1 (param_0: f32[1,20,257], param_1: f32[31,257,136]) -> (f32[1,23,136], u8[0]) {
  %param_0 = f32[1,20,257]{2,1,0} parameter(0)
  %copy.3 = f32[1,20,257]{1,2,0} copy(f32[1,20,257]{2,1,0} %param_0)
  %param_1 = f32[31,257,136]{2,1,0} parameter(1)
  %copy.4 = f32[31,257,136]{0,2,1} copy(f32[31,257,136]{2,1,0} %param_1)
  %custom-call.1 = (f32[1,23,136]{1,2,0}, u8[0]{0}) custom-call(f32[1,20,257]{1,2,0} %copy.3, f32[31,257,136]{0,2,1} %copy.4), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[1,23,136]{1,2,0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %copy.5 = f32[1,23,136]{2,1,0} copy(f32[1,23,136]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[1,23,136]{2,1,0}, u8[0]{0}) tuple(f32[1,23,136]{2,1,0} %copy.5, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, Computation2) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation3 (param_0: f32[138,20,1], param_1: f32[31,1,1]) -> (f32[138,23,1], u8[0]) {
  %param_0 = f32[138,20,1]{2,1,0} parameter(0)
  %bitcast = f32[138,20,1]{1,2,0} bitcast(f32[138,20,1]{2,1,0} %param_0)
  %param_1 = f32[31,1,1]{2,1,0} parameter(1)
  %bitcast.1 = f32[31,1,1]{0,2,1} bitcast(f32[31,1,1]{2,1,0} %param_1)
  %custom-call.1 = (f32[138,23,1]{1,2,0}, u8[0]{0}) custom-call(f32[138,20,1]{1,2,0} %bitcast, f32[31,1,1]{0,2,1} %bitcast.1), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[138,23,1]{1,2,0} get-tuple-element((f32[138,23,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %bitcast.2 = f32[138,23,1]{2,1,0} bitcast(f32[138,23,1]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[138,23,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[138,23,1]{2,1,0}, u8[0]{0}) tuple(f32[138,23,1]{2,1,0} %bitcast.2, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, Computation3) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation5 (param_0: f32[138,100,136], param_1: f32[31,136,1]) -> (f32[138,183,1], u8[0]) {
  %param_0 = f32[138,100,136]{2,1,0} parameter(0)
  %copy.3 = f32[138,100,136]{1,2,0} copy(f32[138,100,136]{2,1,0} %param_0)
  %param_1 = f32[31,136,1]{2,1,0} parameter(1)
  %copy.4 = f32[31,136,1]{0,2,1} copy(f32[31,136,1]{2,1,0} %param_1)
  %custom-call.1 = (f32[138,183,1]{1,2,0}, u8[0]{0}) custom-call(f32[138,100,136]{1,2,0} %copy.3, f32[31,136,1]{0,2,1} %copy.4), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[138,183,1]{1,2,0} get-tuple-element((f32[138,183,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %bitcast = f32[138,183,1]{2,1,0} bitcast(f32[138,183,1]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[138,183,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[138,183,1]{2,1,0}, u8[0]{0}) tuple(f32[138,183,1]{2,1,0} %bitcast, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, BackwardFilterAlgo0Incorrect) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY %TestComputation {
  %param_0 = f16[7680,96,6,6]{1,3,2,0} parameter(0)
  %param_1 = f16[7680,64,4,4]{1,3,2,0} parameter(1)
  ROOT %custom-call.1 = (f16[64,96,3,3]{1,3,2,0}, u8[0]{0}) custom-call(f16[7680,96,6,6]{1,3,2,0} %param_0, f16[7680,64,4,4]{1,3,2,0} %param_1), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardFilter", backend_config="{conv_result_scale:1}"
})");
}

// See b/135429938.
TEST_F(GpuConvolutionRegressionTest, RedzoneCheckerFailure1) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY %TestComputation {
  %param_0 = f32[2,128,1,378]{3,2,1,0} parameter(0)
  %param_1 = f32[1,5,128,128]{1,0,2,3} parameter(1)
  ROOT %custom-call.1 = (f32[2,128,1,378]{3,2,1,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=1x5 pad=0_0x2_2}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})");
}

TEST_F(GpuConvolutionRegressionTest, Conv0D) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY TestComputation {
  %parameter.1 = f32[10,5]{1,0} parameter(0)
  %parameter.2 = f32[5,7]{0,1} parameter(1)
  ROOT %custom-call.1 = (f32[10,7]{1,0}, u8[0]{0}) custom-call(f32[10,5]{1,0} %parameter.1, f32[5,7]{0,1} %parameter.2), window={}, dim_labels=bf_io->bf, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
