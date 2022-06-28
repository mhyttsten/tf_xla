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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_support_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_support_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_support_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace tflite {
namespace optimize {
namespace utils {
namespace {

class ReducedPrecisionSupportTest : public testing::Test {
 protected:
  tflite::TestErrorReporter error_reporter_;
};

TEST_F(ReducedPrecisionSupportTest, BitwiseOps) {
  ReducedPrecisionSupport mask0 = ReducedPrecisionSupport::None;
  ReducedPrecisionSupport mask1 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  EXPECT_EQ(mask0, mask0 & mask1);
  EXPECT_EQ(mask1, mask0 | mask1);
  mask0 |= fp16;
  EXPECT_EQ(true, SupportsFP16Inference(mask0));
  mask0 |= bf16;
  EXPECT_EQ(true, SupportsBfloat16Inference(mask0));
  ReducedPrecisionSupport mask2 = ReducedPrecisionSupport::Float16Accumulation;
  mask2 &= fp16;
  EXPECT_EQ(mask2, ReducedPrecisionSupport::None);
}

TEST_F(ReducedPrecisionSupportTest, SupportTests) {
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport mask = bf16 | fp16;
  EXPECT_EQ(true, SupportsFP16Inference(mask));
  EXPECT_EQ(true, SupportsBfloat16Inference(mask));
  EXPECT_EQ(false, SupportsFP16Accumulation(mask));
  EXPECT_EQ(false, SupportsFP32Accumulation(mask));
  EXPECT_EQ(true, SupportsReducedPrecisionInference(mask));
  EXPECT_EQ(true, SupportsReducedPrecisionInference(mask));
  EXPECT_EQ(false, SupportsEitherFP16OrFP32Accumulation(mask));
  mask = mask | ReducedPrecisionSupport::Float16Accumulation;
  EXPECT_EQ(true, SupportsFP16Accumulation(mask));
  EXPECT_EQ(false, SupportsFP32Accumulation(mask));
  EXPECT_EQ(true, SupportsEitherFP16OrFP32Accumulation(mask));
}

TEST_F(ReducedPrecisionSupportTest, MetadataStrings) {
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport accfp32 =
      ReducedPrecisionSupport::Float32Accumulation;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport maskA = bf16 | fp16 | accfp32;
  std::pair<std::string, std::string> ans =
      MetadataForReducedPrecisionSupport(maskA);
  EXPECT_EQ("fp16bf16accfp32", ans.second);
  EXPECT_EQ("reduced_precision_support", ans.first);
  ReducedPrecisionSupport maskB = fp16 | accfp16;
  EXPECT_EQ("fp16accfp16", MetadataForReducedPrecisionSupport(maskB).second);
}

TEST_F(ReducedPrecisionSupportTest, ReadStringsIntoMasks) {
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport maskfp16 = fp16;
  ReducedPrecisionSupport maskfp16accfp16 = fp16 | accfp16;
  ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
  size_t idx = 0;
  std::string metadata = "fp16accfp16";
  EXPECT_EQ(true, ReadInferenceType(metadata, &idx, &mask));
  EXPECT_EQ(maskfp16, mask);
  EXPECT_EQ(idx, 4);
  idx = 7;
  EXPECT_EQ(true, ReadAccumulationType(metadata, &idx, &mask));
  EXPECT_EQ(maskfp16accfp16, mask);
  EXPECT_EQ(idx, 11);
}

TEST_F(ReducedPrecisionSupportTest, SetMasks) {
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport accfp32 =
      ReducedPrecisionSupport::Float32Accumulation;
  ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true, SetMaskFromReducedPrecisionMetadata("bf16accfp32", &mask));
  EXPECT_EQ(mask, bf16 | accfp32);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true, SetMaskFromReducedPrecisionMetadata("fp16accfp16", &mask));
  EXPECT_EQ(mask, fp16 | accfp16);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true,
            SetMaskFromReducedPrecisionMetadata("fp16bf16accfp32", &mask));
  EXPECT_EQ(mask, fp16 | bf16 | accfp32);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("accfp32", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("qwerwer", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false,
            SetMaskFromReducedPrecisionMetadata("fp16accfp32fp16", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("fp16accbf16", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_support_testDTcc mht_0(mht_0_v, 309, "", "./tensorflow/lite/tools/optimize/reduced_precision_support_test.cc", "main");

  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
