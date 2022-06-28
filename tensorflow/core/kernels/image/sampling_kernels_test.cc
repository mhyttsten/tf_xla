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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernels_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernels_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernels_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/image/sampling_kernels.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace functor {
namespace {

class KernelsTest : public ::testing::Test {
 protected:
  template <typename KernelType>
  void TestKernelValues(const KernelType& kernel, const std::vector<float>& x,
                        const std::vector<float>& expected) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernels_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/image/sampling_kernels_test.cc", "TestKernelValues");

    ASSERT_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); ++i) {
      constexpr float kTolerance = 1e-3;
      EXPECT_NEAR(kernel(x[i]), expected[i], kTolerance);
      EXPECT_NEAR(kernel(-x[i]), expected[i], kTolerance);
    }
  }
};

TEST_F(KernelsTest, TestKernelValues) {
  // Tests kernel values against a set of known golden values
  TestKernelValues(CreateLanczos1Kernel(), {0.0f, 0.5f, 1.0f, 1.5},
                   {1.0f, 0.4052f, 0.0f, 0.0f});
  TestKernelValues(CreateLanczos3Kernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5f, 3.5},
                   {1.0f, 0.6079f, 0.0f, -0.1351f, 0.0243f, 0.0f});
  TestKernelValues(
      CreateLanczos5Kernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5},
      {1.0f, 0.6262f, 0.0f, -0.1822f, 0.0810569f, -0.0334f, 0.0077f, 0.0f});
  TestKernelValues(CreateGaussianKernel(), {0.0f, 0.5f, 1.0f, 1.5},
                   {1.0f, 0.6065f, 0.1353f, 0.0f});

  TestKernelValues(CreateBoxKernel(), {0.0f, 0.25f, 0.5f, 1.0f},
                   {1.0f, 1.0f, 0.5f, 0.0f});
  TestKernelValues(CreateTriangleKernel(), {0.0f, 0.5f, 1.0f},
                   {1.0f, 0.5f, 0.0f});

  TestKernelValues(CreateKeysCubicKernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5},
                   {1.0f, 0.5625f, 0.0f, -0.0625f, 0.0f});
  TestKernelValues(CreateMitchellCubicKernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5},
                   {0.8889f, 0.5347f, 0.0556f, -0.0347f, 0.0f});
}

TEST(SamplingKernelTypeFromStringTest, Works) {
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos1"), Lanczos1Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos3"), Lanczos3Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos5"), Lanczos5Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("gaussian"), GaussianKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("box"), BoxKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("triangle"), TriangleKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("mitchellcubic"), MitchellCubicKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("keyscubic"), KeysCubicKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("not a kernel"),
            SamplingKernelTypeEnd);
}

}  // namespace
}  // namespace functor
}  // namespace tensorflow
