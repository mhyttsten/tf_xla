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
class MHTracer_DTPStensorflowPSlitePStocoPStooling_util_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStooling_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStooling_util_testDTcc() {
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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

enum class Agreement { kBroadcast, kExtend, kBroadcastNotExtend, kNeither };

// A pair of Shapes and whether they should agree up to broadcasting, extending
// or neither.
struct ShapePair {
  Shape left;
  Shape right;
  Agreement agreement;
};

std::vector<ShapePair> CreateShapePairs() {
  return std::vector<ShapePair>(
      {// These agree up to broadcast.
       {Shape({3}), Shape({3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcast},
       {Shape({8, 1, 6, 1}), Shape({7, 1, 5}), Agreement::kBroadcast},
       {Shape({}), Shape({3}), Agreement::kBroadcast},
       {Shape({}), Shape({3, 1}), Agreement::kBroadcast},

       // These extend (and therefore broadcast).
       {Shape({3}), Shape({3}), Agreement::kExtend},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 1, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 3}), Agreement::kExtend},

       // These strictly broadcast and do not extend.
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({1}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({4}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({15, 1, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 1}), Agreement::kBroadcastNotExtend},
       {Shape({3, 1}), Shape({}), Agreement::kBroadcastNotExtend},

       // These do not broadcast (and therefore also do not extend).
       {Shape({3}), Shape({4}), Agreement::kNeither},
       {Shape({2, 1}), Shape({8, 4, 3}), Agreement::kNeither}});
}

// ShapeTest is an empty parameterized test fixture since there is no state.
class ShapeTest : public ::testing::TestWithParam<ShapePair> {};

TEST_P(ShapeTest, Agrees) {
  const ShapePair& param = GetParam();

  switch (param.agreement) {
    case Agreement::kBroadcast: {
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kExtend: {
      EXPECT_TRUE(ShapesAgreeUpToExtending(param.left, param.right));
      // Anything that extends should also broadcast.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kBroadcastNotExtend: {
      // Verify that it strictly broadcasts but does not extend.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      break;
    }
    case Agreement::kNeither: {
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AgreeBroadcast, ShapeTest,
                         ::testing::ValuesIn(CreateShapePairs()));

static const char kNegativeValuesMessage[] =
    "Tensor shape should not include negative values";
static const char kLargeTensorMessage[] = "Tensor shape is too large";

TEST(NumElementsTest, Int) {
  int count;
  tensorflow::Status status = tensorflow::Status::OK();

  status = NumElements(std::vector<int>{1024, 1024, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 2146435072);

  status = NumElements(std::vector<int>{1024, 0, 2048}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 0);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.error_message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int>{1024, 1024, 2048}, &count);
  EXPECT_EQ(status.error_message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Int32) {
  int32_t count;
  tensorflow::Status status = tensorflow::Status::OK();

  status = NumElements(std::vector<int32_t>{1024, 1024, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 2146435072);

  status = NumElements(std::vector<int32_t>{1, 2, -3}, &count);
  EXPECT_EQ(status.error_message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int32_t>{1024, 1024, 2048}, &count);
  EXPECT_EQ(status.error_message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Int64) {
  int64_t count;
  tensorflow::Status status = tensorflow::Status::OK();

  status = NumElements(std::vector<int64_t>{16777216, 16777216, 32767}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 9223090561878065152LL);

  status = NumElements(std::vector<int64_t>{1, 2, -3}, &count);
  EXPECT_EQ(status.error_message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int64_t>{16777216, 16777216, 32768}, &count);
  EXPECT_EQ(status.error_message(), kLargeTensorMessage);
}

TEST(NumElementsTest, UnsignedInt32) {
  uint32_t count;
  tensorflow::Status status = tensorflow::Status::OK();

  status = NumElements(std::vector<uint32_t>{1024, 2048, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 4292870144);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.error_message(), kNegativeValuesMessage);

  status = NumElements(std::vector<uint32_t>{1024, 2048, 2048}, &count);
  EXPECT_EQ(status.error_message(), kLargeTensorMessage);
}

TEST(NumElementsTest, UnsignedInt64) {
  uint64_t count;
  tensorflow::Status status = tensorflow::Status::OK();

  status =
      NumElements(std::vector<uint64_t>{16777216, 16777216, 65535}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 18446462598732840960ULL);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.error_message(), kNegativeValuesMessage);

  status =
      NumElements(std::vector<uint64_t>{16777216, 16777216, 65536}, &count);
  EXPECT_EQ(status.error_message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Scalar) {
  tensorflow::Status status = tensorflow::Status::OK();

  int32_t count;
  status = NumElements(std::vector<int32_t>{}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 1);

  uint64_t countu64;
  status = NumElements(std::vector<uint64_t>{}, &countu64);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(countu64, 1ULL);
}

TEST(FusedActivationTest, DefaultsToUnfused) {
  EXPECT_TRUE(OperatorSupportsFusedActivation(OperatorType::kAdd));
  EXPECT_FALSE(OperatorSupportsFusedActivation(OperatorType::kNone));
  EXPECT_FALSE(OperatorSupportsFusedActivation(static_cast<OperatorType>(255)));
}

}  // namespace toco

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_util_testDTcc mht_0(mht_0_v, 378, "", "./tensorflow/lite/toco/tooling_util_test.cc", "main");

  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
