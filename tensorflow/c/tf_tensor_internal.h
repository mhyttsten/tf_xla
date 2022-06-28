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

#ifndef TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
#define TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPStf_tensor_internalDTh {
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
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPStf_tensor_internalDTh() {
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


#include <memory>

#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/casts.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

// This struct forms part of the C API's public interface. It must strictly be
// passed to or returned from C functions *by pointer*. Otherwise, changes to
// its internal structure will break the C API's binary interface.
typedef struct TF_Tensor {
  tensorflow::AbstractTensorInterface* tensor;
} TF_Tensor;

class TF_ManagedBuffer : public tensorflow::TensorBuffer {
 public:
  TF_ManagedBuffer(void* data, size_t len,
                   void (*deallocator)(void* data, size_t len, void* arg),
                   void* deallocator_arg, bool owns_memory)
      : TensorBuffer(data),
        len_(len),
        deallocator_(deallocator),
        deallocator_arg_(deallocator_arg),
        owns_memory_(owns_memory) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_0(mht_0_v, 216, "", "./tensorflow/c/tf_tensor_internal.h", "TF_ManagedBuffer");
}

  ~TF_ManagedBuffer() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_1(mht_1_v, 221, "", "./tensorflow/c/tf_tensor_internal.h", "~TF_ManagedBuffer");

    (*deallocator_)(data(), len_, deallocator_arg_);
  }

  size_t size() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_2(mht_2_v, 228, "", "./tensorflow/c/tf_tensor_internal.h", "size");
 return len_; }
  TensorBuffer* root_buffer() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_3(mht_3_v, 232, "", "./tensorflow/c/tf_tensor_internal.h", "root_buffer");
 return this; }
  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_4(mht_4_v, 237, "", "./tensorflow/c/tf_tensor_internal.h", "FillAllocationDescription");

    int64_t rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }

  bool OwnsMemory() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_5(mht_5_v, 246, "", "./tensorflow/c/tf_tensor_internal.h", "OwnsMemory");
 return owns_memory_; }

 private:
  const size_t len_;
  void (*const deallocator_)(void* data, size_t len, void* arg);
  void* const deallocator_arg_;
  bool owns_memory_;
};

namespace tensorflow {

class TensorCApi {
 public:
  static TensorBuffer* Buffer(const Tensor& tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_6(mht_6_v, 262, "", "./tensorflow/c/tf_tensor_internal.h", "Buffer");
 return tensor.buf_; }
  static Tensor MakeTensor(TF_DataType type, const TensorShape& shape,
                           TensorBuffer* buf) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_7(mht_7_v, 267, "", "./tensorflow/c/tf_tensor_internal.h", "MakeTensor");

    return Tensor(static_cast<DataType>(type), shape, buf);
  }
};

// Allocates tensor data buffer using specified allocator.
// `operation` is a name for this operation.
void* allocate_tensor(const char* operation, size_t len, Allocator* allocator);

// Deallocates tensor data buffer.
// Defaults to deallocating using CPU allocator. You can pass pointer to
// a different Allocator as `arg`.
void deallocate_buffer(void* data, size_t len, void* arg);

class TensorInterface : public AbstractTensorInterface {
 public:
  TensorInterface() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_8(mht_8_v, 286, "", "./tensorflow/c/tf_tensor_internal.h", "TensorInterface");
}
  explicit TensorInterface(tensorflow::Tensor t) : tensor_(std::move(t)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_9(mht_9_v, 290, "", "./tensorflow/c/tf_tensor_internal.h", "TensorInterface");
}
  ~TensorInterface() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_10(mht_10_v, 294, "", "./tensorflow/c/tf_tensor_internal.h", "~TensorInterface");
}

  void Release() override;

  DataType Type() const override;
  int NumDims() const override;
  int64_t Dim(int dim_index) const override;
  int64_t NumElements() const override;
  size_t ByteSize() const override;
  void* Data() const override;
  bool IsAligned() const override;
  bool CanMove() const override;
  std::string SummarizeValue() const override;

  Status ToTensor(tensorflow::Tensor* dst) const;
  Status BitcastFrom(const TensorInterface& from, DataType type,
                     const int64_t* new_dims, int num_new_dims);
  Status FromProto(const tensorflow::TensorProto& from);

  tensorflow::Tensor& Tensor() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_11(mht_11_v, 316, "", "./tensorflow/c/tf_tensor_internal.h", "Tensor");
 return tensor_; }

 private:
  tensorflow::Tensor tensor_;
};

inline Tensor& TensorFromInterface(AbstractTensorInterface* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPStf_tensor_internalDTh mht_12(mht_12_v, 325, "", "./tensorflow/c/tf_tensor_internal.h", "TensorFromInterface");

  return down_cast<TensorInterface*>(tensor)->Tensor();
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

TF_Tensor* TF_TensorFromTensor(const Tensor& src, Status* status);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
