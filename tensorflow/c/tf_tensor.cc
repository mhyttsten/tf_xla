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
class MHTracer_DTPStensorflowPScPStf_tensorDTcc {
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
   MHTracer_DTPStensorflowPScPStf_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPStf_tensorDTcc() {
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

#include "tensorflow/c/tf_tensor.h"

#include <memory>
#include <vector>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/casts.h"

using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {
void* allocate_tensor(const char* operation, size_t len, Allocator* allocator) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("operation: \"" + (operation == nullptr ? std::string("nullptr") : std::string((char*)operation)) + "\"");
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_0(mht_0_v, 209, "", "./tensorflow/c/tf_tensor.cc", "allocate_tensor");

  void* data = allocator->AllocateRaw(EIGEN_MAX_ALIGN_BYTES, len);
  if (LogMemory::IsEnabled() && data != nullptr) {
    LogMemory::RecordRawAllocation(
        operation, LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, len, data,
        allocator);
  }
  return data;
}

void* allocate_tensor(const char* operation, size_t len) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("operation: \"" + (operation == nullptr ? std::string("nullptr") : std::string((char*)operation)) + "\"");
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_1(mht_1_v, 223, "", "./tensorflow/c/tf_tensor.cc", "allocate_tensor");

  return allocate_tensor(operation, len, cpu_allocator());
}

void deallocate_buffer(void* data, size_t len, void* arg) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_2(mht_2_v, 230, "", "./tensorflow/c/tf_tensor.cc", "deallocate_buffer");

  Allocator* allocator = nullptr;
  if (arg == nullptr) {
    allocator = cpu_allocator();
  } else {
    allocator = reinterpret_cast<Allocator*>(arg);
  }
  if (LogMemory::IsEnabled() && data != nullptr) {
    LogMemory::RecordRawDeallocation(
        "TensorFlow C Api", LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data,
        allocator, false);
  }
  allocator->DeallocateRaw(data);
}
}  // namespace tensorflow

namespace {
TF_Tensor* CreateTensor(TF_ManagedBuffer* buf, TF_DataType dtype,
                        const int64_t* dims, int num_dims, size_t len) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_3(mht_3_v, 251, "", "./tensorflow/c/tf_tensor.cc", "CreateTensor");

  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  // TODO(gjn): Make the choice of interface a compile-time configuration.
  tensorflow::TensorInterface ret(
      Tensor(static_cast<tensorflow::DataType>(dtype),
             tensorflow::TensorShape(dimvec), buf));
  buf->Unref();
  size_t elem_size = TF_DataTypeSize(dtype);
  if (elem_size > 0 && len < (elem_size * ret.NumElements())) {
    return nullptr;
  }
  return new TF_Tensor{new tensorflow::TensorInterface(ret)};
}
}  // namespace

TF_Tensor* TF_AllocateTensor(TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_4(mht_4_v, 274, "", "./tensorflow/c/tf_tensor.cc", "TF_AllocateTensor");

  void* data = tensorflow::allocate_tensor("TF_AllocateTensor", len,
                                           tensorflow::cpu_allocator());
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, tensorflow::deallocate_buffer,
                           tensorflow::cpu_allocator(), /*owns_memory=*/true);
  return CreateTensor(buf, dtype, dims, num_dims, len);
}

TF_Tensor* TF_NewTensor(TF_DataType dtype, const int64_t* dims, int num_dims,
                        void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_5(mht_5_v, 289, "", "./tensorflow/c/tf_tensor.cc", "TF_NewTensor");

  TF_ManagedBuffer* buf = nullptr;
  if (dtype != TF_STRING && dtype != TF_RESOURCE &&
      tensorflow::DataTypeCanUseMemcpy(
          static_cast<tensorflow::DataType>(dtype)) &&
      reinterpret_cast<intptr_t>(data) % std::max(1, EIGEN_MAX_ALIGN_BYTES) !=
          0) {
    // TF_STRING and TF_RESOURCE tensors have a different representation in
    // TF_Tensor than they do in tensorflow::Tensor. So a copy here is a waste
    // (any alignment requirements will be taken care of by TF_TensorToTensor
    // and TF_TensorFromTensor).
    //
    // Other types have the same representation, so copy only if it is safe to
    // do so.
    buf = new TF_ManagedBuffer(tensorflow::allocate_tensor("TF_NewTensor", len),
                               len, tensorflow::deallocate_buffer, nullptr,
                               /*owns_memory=*/true);
    std::memcpy(buf->data(), data, len);
    // Free the original buffer.
    deallocator(data, len, deallocator_arg);
  } else {
    buf = new TF_ManagedBuffer(data, len, deallocator, deallocator_arg,
                               /*owns_memory=*/false);
  }

  return CreateTensor(buf, dtype, dims, num_dims, len);
}

TF_Tensor* TF_TensorMaybeMove(TF_Tensor* t) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_6(mht_6_v, 320, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorMaybeMove");

  return t->tensor->CanMove() ? t : nullptr;
}

void TF_DeleteTensor(TF_Tensor* t) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_7(mht_7_v, 327, "", "./tensorflow/c/tf_tensor.cc", "TF_DeleteTensor");

  if (t == nullptr) {
    return;
  }

  if (t->tensor) {
    t->tensor->Release();
  }

  delete t;
}

TF_DataType TF_TensorType(const TF_Tensor* t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_8(mht_8_v, 342, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorType");

  return static_cast<TF_DataType>(t->tensor->Type());
}

int TF_NumDims(const TF_Tensor* t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_9(mht_9_v, 349, "", "./tensorflow/c/tf_tensor.cc", "TF_NumDims");
 return t->tensor->NumDims(); }

int64_t TF_Dim(const TF_Tensor* t, int dim_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_10(mht_10_v, 354, "", "./tensorflow/c/tf_tensor.cc", "TF_Dim");

  return t->tensor->Dim(dim_index);
}

size_t TF_TensorByteSize(const TF_Tensor* t) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_11(mht_11_v, 361, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorByteSize");
 return t->tensor->ByteSize(); }

void* TF_TensorData(const TF_Tensor* t) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_12(mht_12_v, 366, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorData");
 return t->tensor->Data(); }

int64_t TF_TensorElementCount(const TF_Tensor* t) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_13(mht_13_v, 371, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorElementCount");

  int64_t result = 1;
  int rank = TF_NumDims(t);
  for (int dim = 0; dim < rank; ++dim) {
    result *= TF_Dim(t, dim);
  }
  return result;
}

void TF_TensorBitcastFrom(const TF_Tensor* from, TF_DataType type,
                          TF_Tensor* to, const int64_t* new_dims,
                          int num_new_dims, TF_Status* status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_14(mht_14_v, 385, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorBitcastFrom");

  TF_SetStatus(status, TF_OK, "");
  Status cc_status(
      tensorflow::down_cast<tensorflow::TensorInterface*>(to->tensor)
          ->BitcastFrom(
              *tensorflow::down_cast<const tensorflow::TensorInterface*>(
                  from->tensor),
              static_cast<tensorflow::DataType>(type), new_dims, num_new_dims));
  Set_TF_Status_from_Status(status, cc_status);
}

namespace tensorflow {

void TensorInterface::Release() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_15(mht_15_v, 401, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::Release");

  if (Type() == DT_STRING && NumElements() > 0) {
    TF_TString* data = static_cast<TF_TString*>(Data());
    if (CanMove() && data != nullptr) {
      for (int64_t i = 0; i < NumElements(); ++i) {
        TF_TString_Dealloc(&data[i]);
      }
    }
  }
  delete this;
}

bool TensorInterface::CanMove() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_16(mht_16_v, 416, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::CanMove");

  // It is safe to move the Tensor if and only if we own the unique reference to
  // it. In that case, we might as well not delete and reallocate, but a future
  // implementation might need to do so.
  TensorBuffer* buf = tensorflow::TensorCApi::Buffer(tensor_);
  if (buf->RefCountIsOne() && buf->root_buffer()->RefCountIsOne() &&
      buf->OwnsMemory()) {
    return true;
  }
  return false;
}

std::string TensorInterface::SummarizeValue() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_17(mht_17_v, 431, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::SummarizeValue");

  return tensor_.SummarizeValue(/*max_entries=*/3, /*print_v2=*/true);
}

DataType TensorInterface::Type() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_18(mht_18_v, 438, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::Type");
 return tensor_.dtype(); }

int TensorInterface::NumDims() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_19(mht_19_v, 443, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::NumDims");
 return tensor_.dims(); }

int64_t TensorInterface::Dim(int dim_index) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_20(mht_20_v, 448, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::Dim");

  return static_cast<int64_t>(tensor_.dim_size(dim_index));
}

int64_t TensorInterface::NumElements() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_21(mht_21_v, 455, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::NumElements");

  return static_cast<int64_t>(tensor_.NumElements());
}

size_t TensorInterface::ByteSize() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_22(mht_22_v, 462, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::ByteSize");

  return tensorflow::TensorCApi::Buffer(tensor_)->size();
}

void* TensorInterface::Data() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_23(mht_23_v, 469, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::Data");

  return tensorflow::TensorCApi::Buffer(tensor_)->data();
}

Status TensorInterface::BitcastFrom(const TensorInterface& from, DataType type,
                                    const int64_t* new_dims, int num_new_dims) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_24(mht_24_v, 477, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::BitcastFrom");

  tensorflow::TensorShape s;
  for (int i = 0; i < num_new_dims; ++i) {
    s.AddDim(new_dims[i]);
  }
  return tensor_.BitcastFrom(from.tensor_, type, s);
}

Status TensorInterface::FromProto(const tensorflow::TensorProto& from) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_25(mht_25_v, 488, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::FromProto");

  bool success = tensor_.FromProto(from);
  if (success) return Status::OK();
  return errors::InvalidArgument("Unparseable tensor proto");
}

}  // namespace tensorflow

// --------------------------------------------------------------------------

static void DeleteArray(void* data, size_t size, void* arg) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_26(mht_26_v, 501, "", "./tensorflow/c/tf_tensor.cc", "DeleteArray");

  DCHECK_EQ(data, arg);
  delete[] reinterpret_cast<char*>(arg);
}

// Create an empty tensor of type 'dtype'. 'shape' can be arbitrary, but has to
// result in a zero-sized tensor.
static TF_Tensor* EmptyTensor(TF_DataType dtype,
                              const tensorflow::TensorShape& shape) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_27(mht_27_v, 512, "", "./tensorflow/c/tf_tensor.cc", "EmptyTensor");

  static char empty;
  int64_t nelems = 1;
  std::vector<int64_t> dims;
  auto shape_dims = shape.dims();
  dims.reserve(shape_dims);
  for (int i = 0; i < shape_dims; ++i) {
    dims.push_back(shape.dim_size(i));
    nelems *= shape.dim_size(i);
  }
  CHECK_EQ(nelems, 0);
  return TF_NewTensor(
      dtype, reinterpret_cast<const int64_t*>(dims.data()), shape.dims(),
      reinterpret_cast<void*>(&empty), 0, [](void*, size_t, void*) {}, nullptr);
}

namespace tensorflow {

// Non-static for testing.
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_28(mht_28_v, 534, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorFromTensor");

  *status = tensorflow::Status::OK();
  if (!src.IsInitialized()) {
    *status = FailedPrecondition(
        "attempt to use a tensor with an uninitialized value");
    return nullptr;
  }
  if (src.NumElements() == 0) {
    return EmptyTensor(static_cast<TF_DataType>(src.dtype()), src.shape());
  }

  Tensor tensor;
  if (!tensor.CopyFrom(src, src.shape())) {
    return nullptr;
  }
  return new TF_Tensor{new tensorflow::TensorInterface(std::move(tensor))};
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_29(mht_29_v, 555, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorToTensor");

  return tensorflow::down_cast<const tensorflow::TensorInterface*>(src->tensor)
      ->ToTensor(dst);
}

Status TensorInterface::ToTensor(tensorflow::Tensor* dst) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_30(mht_30_v, 563, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::ToTensor");

  *dst = tensor_;
  return Status::OK();
}

bool TensorInterface::IsAligned() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_31(mht_31_v, 571, "", "./tensorflow/c/tf_tensor.cc", "TensorInterface::IsAligned");
 return tensor_.IsAligned(); }

}  // namespace tensorflow

bool TF_TensorIsAligned(const TF_Tensor* t) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPStf_tensorDTcc mht_32(mht_32_v, 578, "", "./tensorflow/c/tf_tensor.cc", "TF_TensorIsAligned");
 return t->tensor->IsAligned(); }
