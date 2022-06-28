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
class MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc() {
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

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class LSHProjectionOpModel : public SingleOpModel {
 public:
  LSHProjectionOpModel(LSHProjectionType type,
                       std::initializer_list<int> hash_shape,
                       std::initializer_list<int> input_shape,
                       std::initializer_list<int> weight_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/lsh_projection_test.cc", "LSHProjectionOpModel");

    hash_ = AddInput(TensorType_FLOAT32);
    input_ = AddInput(TensorType_INT32);
    if (weight_shape.size() > 0) {
      weight_ = AddInput(TensorType_FLOAT32);
    }
    output_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_LSH_PROJECTION,
                 BuiltinOptions_LSHProjectionOptions,
                 CreateLSHProjectionOptions(builder_, type).Union());
    if (weight_shape.size() > 0) {
      BuildInterpreter({hash_shape, input_shape, weight_shape});
    } else {
      BuildInterpreter({hash_shape, input_shape});
    }

    output_size_ = 1;
    for (int i : hash_shape) {
      output_size_ *= i;
      if (type == LSHProjectionType_SPARSE) {
        break;
      }
    }
  }
  void SetInput(std::initializer_list<int> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/kernels/lsh_projection_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  void SetHash(std::initializer_list<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/kernels/lsh_projection_test.cc", "SetHash");

    PopulateTensor(hash_, data);
  }

  void SetWeight(std::initializer_list<float> f) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projection_testDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/kernels/lsh_projection_test.cc", "SetWeight");
 PopulateTensor(weight_, f); }

  std::vector<int> GetOutput() { return ExtractVector<int>(output_); }

 private:
  int input_;
  int hash_;
  int weight_;
  int output_;

  int output_size_;
};

TEST(LSHProjectionOpTest2, Dense1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_DENSE, {3, 2}, {5}, {5});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({1.0, 1.0, 1.0, 1.0, 1.0});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 1, 1, 1, 0));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 1, 0, 0));
#endif
}

TEST(LSHProjectionOpTest2, Sparse1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5}, {});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 3, 8 + 2));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 1, 8 + 0));
#endif
}

TEST(LSHProjectionOpTest2, Sparse3DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5, 2, 2}, {5});

  m.SetInput({1234, 2345, 3456, 1234, 4567, 5678, 6789, 4567, 7891, 8912,
              9123, 7890, -987, -876, -765, -987, -543, -432, -321, -543});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({0.12, 0.34, 0.56, 0.67, 0.78});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 3, 8 + 2));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 2, 4 + 1, 8 + 1));
#endif
}

}  // namespace
}  // namespace tflite
