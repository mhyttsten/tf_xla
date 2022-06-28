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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc() {
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
#include "tensorflow/lite/delegates/flex/buffer_map.h"

#include <sys/types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {
namespace {

using ::testing::ElementsAre;

// A bit of RAII to simplify handling of TfLiteTensors in the tests.
using UniqueTfLiteTensor =
    std::unique_ptr<TfLiteTensor, std::function<void(TfLiteTensor*)>>;

template <typename T>
UniqueTfLiteTensor MakeLiteTensor(const std::vector<int>& shape,
                                  const std::vector<T>& data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/flex/buffer_map_test.cc", "MakeLiteTensor");

  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(T), tensor.get());
  memcpy(tensor->data.raw, data.data(), data.size() * sizeof(T));
  return tensor;
}

template <>
UniqueTfLiteTensor MakeLiteTensor<string>(const std::vector<int>& shape,
                                          const std::vector<string>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/delegates/flex/buffer_map_test.cc", "MakeLiteTensor<string>");

  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<string>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(string), tensor.get());

  DynamicBuffer b;
  for (const string& s : data) {
    b.AddString(s.data(), s.size());
  }
  b.WriteToTensor(tensor.get(), ConvertVectorToTfLiteIntArray(shape));
  return tensor;
}

template <typename T>
tensorflow::Tensor MakeTensor(const std::vector<int>& shape,
                              const std::vector<T>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/lite/delegates/flex/buffer_map_test.cc", "MakeTensor");

  BufferMap buffer_map;  // BufferMap is the easiest way to build the tensor.
  UniqueTfLiteTensor t1 = MakeLiteTensor<T>(shape, data);
  buffer_map.SetFromTfLite(0, t1.get());
  return buffer_map.GetTensor(0);
}

std::vector<int64_t> GetTensorShape(const tensorflow::Tensor& t) {
  std::vector<int64_t> shape(t.dims());
  for (int i = 0; i < t.dims(); ++i) {
    shape[i] = t.dim_size(i);
  }
  return shape;
}

template <typename T>
std::vector<T> GetTensorData(const tensorflow::Tensor& t) {
  const T* data = t.flat<T>().data();
  return std::vector<T>(data, data + t.NumElements());
}

TEST(BufferMapTest, EmptyBuffer) {
  BufferMap buffer_map;
  EXPECT_FALSE(buffer_map.HasTensor(0));
}

TEST(BufferMapTest, SetFromTfLite) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteString) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<string>({1, 2, 1, 3}, {"", "", "", "str1", "", ""});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<tensorflow::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "str1", "", ""));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_STRING);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, SetFromTfLiteStringTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<string>({1, 2, 4}, {"", "", "", "s3", "", "", "s1", "s2"});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<tensorflow::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "s3", "", "", "s1", "s2"));
}

TEST(BufferMapTest, SetFromTfLiteBuiltinResource) {
  BufferMap buffer_map;

  // Constructs a fake resource tensor.
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = kTfLiteResource;
  tensor->dims = ConvertVectorToTfLiteIntArray({1});
  TfLiteTensorRealloc(sizeof(int32_t), tensor.get());
  tensor->delegate = nullptr;
  tensor->data.i32[0] = 1;

  buffer_map.SetFromTfLite(0, tensor.get());
  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_RESOURCE);
  ASSERT_EQ(out_tensor.NumElements(), 1);
  tensorflow::ResourceHandle handle =
      out_tensor.flat<tensorflow::ResourceHandle>()(0);
  EXPECT_EQ(handle.name(), "tflite_resource_variable:1");
}

TEST(BufferMapTest, SetFromTensorFlow) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTensorFlowTwice) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  tensorflow::Tensor t2 = MakeTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTensorFlow(0, t2);

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TfLiteOverwritesTensorFlow) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_FALSE(buffer_map.IsTensorFlowTensor(0));
  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TensorFlowOverwritesTfLite) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t2.get());
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_TRUE(buffer_map.IsTensorFlowTensor(0));
  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));
}

}  // namespace
}  // namespace flex
}  // namespace tflite
