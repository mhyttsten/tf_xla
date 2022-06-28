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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh() {
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


#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

inline RuntimeShape GetTensorShape(std::vector<int32_t> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_0(mht_0_v, 195, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "GetTensorShape");

  return RuntimeShape(data.size(), data.data());
}

// A list of tensors in a format that can be used by kernels like split and
// concatenation.
template <typename T>
class VectorOfTensors {
 public:
  // Build with the tensors in 'tensor_list'.
  VectorOfTensors(const TfLiteContext& context,
                  const TfLiteIntArray& tensor_list) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_1(mht_1_v, 209, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "VectorOfTensors");

    int num_tensors = tensor_list.size;

    all_data_.reserve(num_tensors);
    all_shape_.reserve(num_tensors);
    all_shape_ptr_.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
      TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
      all_data_.push_back(GetTensorData<T>(t));
      all_shape_.push_back(GetTensorShape(t));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_shape in the previous loop and then we
    // are free to grab iterators here.
    for (int i = 0; i < num_tensors; ++i) {
      all_shape_ptr_.push_back(&all_shape_[i]);
    }
  }
  // Return a pointer to the data pointers of all tensors in the list. For
  // example:
  //   float* const* f = v.data();
  //   f[0][1] is the second element of the first tensor.
  T* const* data() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_2(mht_2_v, 236, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "data");
 return all_data_.data(); }

  // Return a pointer the shape pointers of all tensors in the list. For
  // example:
  //   const RuntimeShape* const* d = v.dims();
  //   dims[1] are the dimensions of the second tensor in the list.
  const RuntimeShape* const* shapes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_3(mht_3_v, 245, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "shapes");
 return all_shape_ptr_.data(); }

 private:
  std::vector<T*> all_data_;
  std::vector<RuntimeShape> all_shape_;
  std::vector<RuntimeShape*> all_shape_ptr_;
};

// A list of quantized tensors in a format that can be used by kernels like
// split and concatenation.
class VectorOfQuantizedTensors : public VectorOfTensors<uint8_t> {
 public:
  // Build with the tensors in 'tensor_list'.
  VectorOfQuantizedTensors(const TfLiteContext& context,
                           const TfLiteIntArray& tensor_list)
      : VectorOfTensors<uint8_t>(context, tensor_list) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_4(mht_4_v, 263, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "VectorOfQuantizedTensors");

    for (int i = 0; i < tensor_list.size; ++i) {
      TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
      zero_point_.push_back(t->params.zero_point);
      scale_.push_back(t->params.scale);
    }
  }

  const float* scale() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_5(mht_5_v, 274, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "scale");
 return scale_.data(); }
  const int32_t* zero_point() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_6(mht_6_v, 278, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "zero_point");
 return zero_point_.data(); }

 private:
  std::vector<int32_t> zero_point_;
  std::vector<float> scale_;
};

// Writes randomly accessed values from `input` sequentially into `output`.
template <typename T>
class SequentialTensorWriter {
 public:
  SequentialTensorWriter(const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_7(mht_7_v, 292, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "SequentialTensorWriter");

    input_data_ = GetTensorData<T>(input);
    output_ptr_ = GetTensorData<T>(output);
  }
  SequentialTensorWriter(const T* input_data, T* output_data)
      : input_data_(input_data), output_ptr_(output_data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_8(mht_8_v, 300, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "SequentialTensorWriter");
}

  void Write(int position) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_9(mht_9_v, 305, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "Write");
 *output_ptr_++ = input_data_[position]; }
  void WriteN(int position, int len) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSportable_tensorDTh mht_10(mht_10_v, 309, "", "./tensorflow/lite/kernels/internal/portable_tensor.h", "WriteN");

    memcpy(output_ptr_, &input_data_[position], sizeof(T) * len);
    output_ptr_ += len;
  }

 private:
  const T* input_data_;
  T* output_ptr_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
