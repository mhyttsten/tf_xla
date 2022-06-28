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
class MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc() {
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
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the
License.
==============================================================================*/
// Unit test for TFLite Lookup op.

#include <stdint.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

float kTestTolerance = 7.41e-03;

using ::testing::ElementsAreArray;

class BaseEmbeddingLookupOpModel : public SingleOpModel {
 public:
  BaseEmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                             std::initializer_list<int> weight_shape,
                             TensorType weight_type = TensorType_FLOAT32,
                             TensorType output_type = TensorType_FLOAT32) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "BaseEmbeddingLookupOpModel");

    input_ = AddInput(TensorType_INT32);
    weight_ = AddInput(weight_type);
    output_ = AddOutput(output_type);
    SetBuiltinOp(BuiltinOperator_EMBEDDING_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({index_shape, weight_shape});
  }

  void SetInput(std::initializer_list<int> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int weight_;
  int output_;
};

class EmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  using BaseEmbeddingLookupOpModel::BaseEmbeddingLookupOpModel;

  template <typename T>
  void Set3DWeightMatrix(const std::function<T(int, int, int)>& function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "Set3DWeightMatrix");

    TfLiteTensor* tensor = interpreter_->tensor(weight_);
    int rows = tensor->dims->data[0];
    int columns = tensor->dims->data[1];
    int features = tensor->dims->data[2];
    T* data = GetTensorData<T>(tensor);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        for (int k = 0; k < features; k++) {
          data[(i * columns + j) * features + k] = function(i, j, k);
        }
      }
    }
  }
};

class HybridEmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  HybridEmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                               std::initializer_list<int> weight_shape,
                               TensorType type)
      : BaseEmbeddingLookupOpModel(index_shape, weight_shape, type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_3(mht_3_v, 271, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "HybridEmbeddingLookupOpModel");
}

  void SetWeight(std::initializer_list<float> data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_4(mht_4_v, 276, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "SetWeight");

    SymmetricQuantizeAndPopulate(weight_, data);
  }

  void SetSignedWeight(std::initializer_list<float> data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookup_testDTcc mht_5(mht_5_v, 283, "", "./tensorflow/lite/kernels/embedding_lookup_test.cc", "SetSignedWeight");

    SignedSymmetricQuantizeAndPopulate(weight_, data);
  }
};

// TODO(ahentz): write more tests that exercise the details of the op, such as
// lookup errors and variable input shapes.
TEST(EmbeddingLookupOpTest, SimpleTest) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4});
  m.SetInput({1, 0, 2});
  m.Set3DWeightMatrix<float>(
      [](int i, int j, int k) -> float { return i + j / 10.0f + k / 100.0f; });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                  0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                  2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
              })));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple2DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 8}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple3DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple4DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple2DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 8}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple3DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple4DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(EmbeddingLookupHybridOpTest, Simple3DTestQuantized) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_UINT8, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.Set3DWeightMatrix<uint8_t>(
      [](int i, int j, int k) -> uint8_t { return 100 * i + 10 * j + k; });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({
                  100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
                  0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
                  200, 201, 202, 203, 210, 211, 212, 213,  // Row 2
              }));
}

}  // namespace
}  // namespace tflite
