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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variable_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variable_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variable_testDTcc() {
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
#include "tensorflow/lite/experimental/resource/resource_variable.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace resource {
// Helper util that initialize 'tensor'.
void InitTensor(const std::vector<int>& shape, TfLiteAllocationType alloc_type,
                float default_value, TfLiteTensor* tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variable_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/experimental/resource/resource_variable_test.cc", "InitTensor");

  memset(tensor, 0, sizeof(TfLiteTensor));
  int num_elements = 1;
  for (auto dim : shape) num_elements *= dim;
  if (shape.empty()) num_elements = 0;
  float* buf = static_cast<float*>(malloc(sizeof(float) * num_elements));
  for (int i = 0; i < num_elements; ++i) buf[i] = default_value;
  const int bytes = num_elements * sizeof(buf[0]);
  auto* dims = ConvertArrayToTfLiteIntArray(shape.size(), shape.data());
  TfLiteTensorReset(TfLiteType::kTfLiteFloat32, nullptr, dims, {},
                    reinterpret_cast<char*>(buf), bytes, alloc_type, nullptr,
                    false, tensor);
}

TEST(ResourceTest, NonDynamicTensorAssign) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  TfLiteTensor tensor;
  std::vector<int> shape = {1};
  InitTensor(shape, kTfLiteArenaRw, 1.0f, &tensor);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();

  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Cleanup
  // For non dynamic tensors we need to delete the buffers manually.
  free(tensor.data.raw);
  TfLiteTensorFree(&tensor);
}

TEST(ResourceTest, DynamicTensorAssign) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  TfLiteTensor tensor;
  std::vector<int> shape = {1};
  InitTensor(shape, kTfLiteDynamic, 1.0f, &tensor);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();

  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor);
}

TEST(ResourceTest, AssignSameSizeTensor) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  // We create 2 tensors and make 2 calls for Assign.
  // The second Assign call should trigger the case of assign with same size.
  TfLiteTensor tensor_a, tensor_b;
  std::vector<int> shape_a = {1};
  std::vector<int> shape_b = {1};
  InitTensor(shape_a, kTfLiteDynamic, 1.0, &tensor_a);
  InitTensor(shape_b, kTfLiteDynamic, 4.0, &tensor_b);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_a));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Second AssignFrom but now tensor_b has same size as the variable.
  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_b));
  EXPECT_TRUE(var.IsInitialized());
  value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(4.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor_a);
  TfLiteTensorFree(&tensor_b);
}

TEST(ResourceTest, AssignDifferentSizeTensor) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  // We create 2 tensors and make 2 calls for Assign.
  // The second Assign call should trigger the case of assign with different
  // size.
  TfLiteTensor tensor_a, tensor_b;
  std::vector<int> shape_a = {1};
  std::vector<int> shape_b = {2};
  InitTensor(shape_a, kTfLiteDynamic, 1.0, &tensor_a);
  InitTensor(shape_b, kTfLiteDynamic, 4.0, &tensor_b);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_a));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Second AssignFrom but now tensor_b has different size from the variable.
  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_b));
  EXPECT_TRUE(var.IsInitialized());
  value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float) * 2, value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(2, value->dims->data[0]);
  EXPECT_EQ(4.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor_a);
  TfLiteTensorFree(&tensor_b);
}

TEST(IsBuiltinResource, IsBuiltinResourceTest) {
  TfLiteTensor tensor;
  tensor.type = kTfLiteResource;
  tensor.delegate = nullptr;
  // Resource type and not delegate output.
  EXPECT_TRUE(IsBuiltinResource(&tensor));

  // Not valid tensor.
  EXPECT_FALSE(IsBuiltinResource(nullptr));

  // Not a resource type.
  tensor.type = kTfLiteFloat32;
  EXPECT_FALSE(IsBuiltinResource(&tensor));

  // Resource but coming from a delegate.
  tensor.type = kTfLiteResource;
  TfLiteDelegate delegate;
  tensor.delegate = &delegate;
  EXPECT_FALSE(IsBuiltinResource(&tensor));
}

}  // namespace resource
}  // namespace tflite
