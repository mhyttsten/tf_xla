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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

namespace {

void TestPadOperation(const HWC& prepend, const HWC& append,
                      const BHWC& output_shape, std::vector<float>&& expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/gpu/gl/kernels/pad_test.cc", "TestPadOperation");

  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = output_shape;

  PadAttributes attr;
  attr.prepended = BHWC(0, prepend.h, prepend.w, prepend.c);
  attr.appended = BHWC(0, append.h, append.w, append.c);
  attr.type = PaddingContentType::ZEROS;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0}));
  ASSERT_OK(model.Invoke(*NewPadNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), expected));
}

void TestPrepending(const HWC& prepend, const BHWC& output_shape,
                    std::vector<float>&& expected) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/delegates/gpu/gl/kernels/pad_test.cc", "TestPrepending");

  TestPadOperation(prepend, HWC(0, 0, 0), output_shape, std::move(expected));
}

void TestAppending(const HWC& append, const BHWC& output_shape,
                   std::vector<float>&& expected) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpad_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/delegates/gpu/gl/kernels/pad_test.cc", "TestAppending");

  TestPadOperation(HWC(0, 0, 0), append, output_shape, std::move(expected));
}

}  // namespace

TEST(PadTest, PrependH) {
  TestPrepending(/*prepend=*/HWC(1, 0, 0),
                 /*output_shape=*/BHWC(1, 2, 1, 1), /*expected=*/{0, 1});
}

TEST(PadTest, PrependW) {
  TestPrepending(/*prepend=*/HWC(0, 1, 0), /*output_shape=*/BHWC(1, 1, 2, 1),
                 /*expected=*/{0, 1});
}

TEST(PadTest, PrependC) {
  TestPrepending(/*prepend=*/HWC(0, 0, 1), /*output_shape=*/BHWC(1, 1, 1, 2),
                 /*expected=*/{0, 1});
}

TEST(PadTest, PrependCx4) {
  TestPrepending(/*prepend=*/HWC(0, 0, 4), /*output_shape=*/BHWC(1, 1, 1, 5),
                 /*expected=*/{0, 0, 0, 0, 1});
}

TEST(PadTest, PrependHWC) {
  TestPrepending(/*prepend=*/HWC(1, 1, 1), /*output_shape=*/BHWC(1, 2, 2, 2),
                 /*expected=*/{0, 0, 0, 0, 0, 0, 0, 1});
}

TEST(PadTest, AppendH) {
  TestAppending(/*append=*/HWC(1, 0, 0), /*output_shape=*/BHWC(1, 2, 1, 1),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendW) {
  TestAppending(/*append=*/HWC(0, 1, 0), /*output_shape=*/BHWC(1, 1, 2, 1),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendC) {
  TestAppending(/*append=*/HWC(0, 0, 1), /*output_shape=*/BHWC(1, 1, 1, 2),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendHWC) {
  TestAppending(/*append=*/HWC(1, 1, 1), /*output_shape=*/BHWC(1, 2, 2, 2),
                /*expected=*/{1, 0, 0, 0, 0, 0, 0, 0});
}

TEST(PadTest, PrependHWCAppendHWC) {
  TestPadOperation(/*prepend=*/HWC(1, 1, 1), /*append=*/HWC(1, 1, 1),
                   /*output_shape=*/BHWC(1, 3, 3, 3),
                   /*expected=*/{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

TEST(MirrorPadWidthTest, Smoke) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 3, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 7, 1);

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 2, 0);
  attr.appended = BHWC(0, 0, 2, 0);
  attr.type = PaddingContentType::REFLECT;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewPadNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0}));
}

TEST(MirrorPadChannelsTest, Smoke) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 3);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 7);

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 2);
  attr.appended = BHWC(0, 0, 0, 2);
  attr.type = PaddingContentType::REFLECT;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewPadNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
