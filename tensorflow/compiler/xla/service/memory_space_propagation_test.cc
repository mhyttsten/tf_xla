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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagation_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagation_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_propagation.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class MemorySpacePropagationTest : public HloTestBase {
 public:
  MemorySpacePropagationTest()
      : HloTestBase(),
        verifier_(/*layout_sensitive=*/false, /*allow_mixed_precision*/ false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagation_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/memory_space_propagation_test.cc", "MemorySpacePropagationTest");

  }

  Status Verify(HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagation_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/xla/service/memory_space_propagation_test.cc", "Verify");
 return verifier_.Run(module).status(); }

 private:
  HloVerifier verifier_;
};

TEST_F(MemorySpacePropagationTest, NoMemorySpace) {
  absl::string_view hlo_string = R"(
  HloModule NoMemorySpace

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)} copy(%param2)
    %fusion = s32[6]{0:T(128)} fusion(s32[6]{0:T(128)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_FALSE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK_AND_ASSIGN(auto ref, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NonTupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, TupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedInputFusion) {
  // Tests propagating the memory space to nested fusions on the input side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)S(1)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)S(1)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedOutputFusion) {
  // Tests propagating the memory space to nested fusions on the output side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)S(1)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)S(1)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, BitcastInFusion) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)S(1)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)S(1)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

}  // namespace
}  // namespace xla
