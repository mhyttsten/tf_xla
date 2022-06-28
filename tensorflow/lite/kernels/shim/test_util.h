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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh() {
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


#include <initializer_list>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {

// A wrapper around TfLiteTensor which frees it in its dtor.
class UniqueTfLiteTensor {
 public:
  explicit UniqueTfLiteTensor(TfLiteTensor* tensor) : tensor_(tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/shim/test_util.h", "UniqueTfLiteTensor");
}

  UniqueTfLiteTensor() = default;

  // Returns the underlying pointer

  TfLiteTensor* get();

  TfLiteTensor& operator*();

  TfLiteTensor* operator->();

  const TfLiteTensor* get() const;

  const TfLiteTensor& operator*() const;

  const TfLiteTensor* operator->() const;

  // Resets the underlying pointer
  void reset(TfLiteTensor* tensor);

  // Deallocates the tensor as well
  ~UniqueTfLiteTensor();

 private:
  TfLiteTensor* tensor_ = nullptr;
};

// Prints a debug string for the given tensor.
std::string TfliteTensorDebugString(const ::TfLiteTensor* tensor,
                                    const std::size_t max_values = 30);

// Calculate the total number of elements given the shape.
std::size_t NumTotalFromShape(const std::initializer_list<int>& shape);

template <typename T>
void ReallocDynamicTensor(const std::initializer_list<int> shape,
                          TfLiteTensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/shim/test_util.h", "ReallocDynamicTensor");

  TfLiteTensorFree(tensor);
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
  // Populate Shape
  TfLiteIntArray* shape_arr = TfLiteIntArrayCreate(shape.size());
  int i = 0;
  const std::size_t num_total = NumTotalFromShape(shape);
  for (const int dim : shape) shape_arr->data[i++] = dim;
  tensor->dims = shape_arr;
  if (tensor->type != kTfLiteString) {
    TfLiteTensorRealloc(num_total * sizeof(T), tensor);
  }
}

// Populates a tensor with the given values
template <typename T>
void PopulateTfLiteTensorValue(const std::initializer_list<T> values,
                               TfLiteTensor* tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh mht_2(mht_2_v, 260, "", "./tensorflow/lite/kernels/shim/test_util.h", "PopulateTfLiteTensorValue");

  T* buffer = reinterpret_cast<T*>(tensor->data.raw);
  int i = 0;
  for (const auto v : values) {
    buffer[i++] = v;
  }
}

template <>
void PopulateTfLiteTensorValue<std::string>(
    const std::initializer_list<std::string> values, TfLiteTensor* tensor);

template <typename T>
void PopulateTfLiteTensor(const std::initializer_list<T> values,
                          const std::initializer_list<int> shape,
                          TfLiteTensor* tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTh mht_3(mht_3_v, 278, "", "./tensorflow/lite/kernels/shim/test_util.h", "PopulateTfLiteTensor");

  const std::size_t num_total = NumTotalFromShape(shape);
  CHECK_EQ(num_total, values.size());
  // Populate Shape
  ReallocDynamicTensor<T>(shape, tensor);
  // Value allocation
  PopulateTfLiteTensorValue<T>(values, tensor);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_UTILS_H_
