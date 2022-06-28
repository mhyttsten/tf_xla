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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
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
class MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh() {
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


#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/cc/experimental/base/public/status.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// Tensor represents an n-dimensional array of values.
class Tensor {
 public:
  using DeleterCallback = std::function<void(void*, size_t)>;

  // Constructs a Tensor from user provided buffer.
  //
  // Params:
  //  dtype - The dtype of the tensor's data.
  //  shape - A shape vector, where each element corresponds to the size of
  //          the tensor's corresponding dimension.
  //  data - Pointer to a buffer of memory to construct a Tensor out of.
  //  len - The length (in bytes) of `data`
  //  deleter - A std::function to be called when the Tensor no longer needs the
  //            memory in `data`. This can be used to free `data`, or
  //            perhaps decrement a refcount associated with `data`, etc.
  //  status - Set to OK on success and an error on failure.
  // Returns:
  // If an error occurred, status->ok() will be false, and the returned
  // Tensor must not be used.
  // TODO(bmzhao): Add Runtime as an argument to this function so we can swap to
  // a TFRT backed tensor.
  // TODO(bmzhao): Add benchmarks on overhead for this function; we can
  // consider using int64_t* + length rather than vector.
  static Tensor FromBuffer(TF_DataType dtype, const std::vector<int64_t>& shape,
                           void* data, size_t len, DeleterCallback deleter,
                           Status* status);

  // TODO(bmzhao): In the case we construct a tensor from non-owned memory,
  // we should offer a way to deep copy the tensor into a new tensor, which
  // owns the underlying memory. This could be a .deepcopy()/clone() method.

  // TODO(bmzhao): In the future, we want to relax the non-copyability
  // constraint. To do so, we can add a C API function that acts like
  // CopyFrom:
  // https://github.com/tensorflow/tensorflow/blob/08931c1e3e9eb2e26230502d678408e66730826c/tensorflow/core/framework/tensor.h#L301-L311

  // Tensor is movable, but not copyable
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;

  // Returns the number of dimensions in the tensor. Can be -1, which represents
  // unknown rank.
  int dims() const;

  // Returns the number of elements in dimension `d`.
  // REQUIRES: `0 <= d < dims()`
  int64_t dim_size(int d) const;

  // Returns a pointer to the underlying data buffer.
  void* data() const;

  // Returns the data type of the tensor.
  TF_DataType dtype() const;

  // Returns the number of elements in the tensor. For a tensor with a partially
  // defined shape, -1 means not fully defined.
  int64_t num_elements() const;

  // Returns the size of the underlying data in bytes.
  size_t num_bytes() const;

 private:
  friend class TensorHandle;
  friend class Runtime;

  // Wraps a TF_Tensor. Takes ownership of handle.
  explicit Tensor(TF_Tensor* tensor) : tensor_(tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_0(mht_0_v, 270, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor");
}

  // Tensor is not copyable
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // Returns the underlying TF_Tensor that this object wraps.
  // This object retains ownership of the pointer.
  TF_Tensor* GetTFTensor() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_1(mht_1_v, 281, "", "./tensorflow/cc/experimental/base/public/tensor.h", "GetTFTensor");
 return tensor_.get(); }

  struct DeleterStruct {
    std::function<void(void*, size_t)> deleter;
  };

  static void DeleterFunction(void* memory, size_t len, void* deleter_struct) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_2(mht_2_v, 290, "", "./tensorflow/cc/experimental/base/public/tensor.h", "DeleterFunction");

    DeleterStruct* deleter = reinterpret_cast<DeleterStruct*>(deleter_struct);
    deleter->deleter(memory, len);
    delete deleter;
  }

  struct TFTensorDeleter {
    void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
  };
  std::unique_ptr<TF_Tensor, TFTensorDeleter> tensor_;
};

inline void* Tensor::data() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_3(mht_3_v, 305, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::data");
 return TF_TensorData(tensor_.get()); }

inline int Tensor::dims() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_4(mht_4_v, 310, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::dims");
 return TF_NumDims(tensor_.get()); }

inline int64_t Tensor::dim_size(int d) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_5(mht_5_v, 315, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::dim_size");

  return TF_Dim(tensor_.get(), d);
}

inline TF_DataType Tensor::dtype() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_6(mht_6_v, 322, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::dtype");

  return TF_TensorType(tensor_.get());
}

inline int64_t Tensor::num_elements() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_7(mht_7_v, 329, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::num_elements");

  return TF_TensorElementCount(tensor_.get());
}

inline size_t Tensor::num_bytes() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_8(mht_8_v, 336, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::num_bytes");

  return TF_TensorByteSize(tensor_.get());
}

inline Tensor Tensor::FromBuffer(TF_DataType dtype,
                                 const std::vector<int64_t>& shape, void* data,
                                 size_t len, DeleterCallback deleter,
                                 Status* status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPStensorDTh mht_9(mht_9_v, 346, "", "./tensorflow/cc/experimental/base/public/tensor.h", "Tensor::FromBuffer");

  // Credit to apassos@ for this technique:
  // Despite the fact that our API takes a std::function deleter, we are able
  // to maintain ABI stability because:
  // 1. Only a function pointer is sent across the C API (&DeleterFunction)
  // 2. DeleterFunction is defined in the same build artifact that constructed
  //    the std::function (so there isn't confusion about std::function ABI).
  // Note that 2. is satisfied by the fact that this is a header-only API, where
  // the function implementations are inline.

  DeleterStruct* deleter_struct = new DeleterStruct{deleter};
  TF_Tensor* tensor = TF_NewTensor(dtype, shape.data(), shape.size(), data, len,
                                   &DeleterFunction, deleter_struct);
  if (tensor == nullptr) {
    status->SetStatus(TF_INVALID_ARGUMENT,
                      "Failed to create tensor for input buffer");
    return Tensor(nullptr);
  }
  return Tensor(tensor);
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
