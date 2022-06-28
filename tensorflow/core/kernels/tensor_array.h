/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh() {
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


#include <limits.h>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

// Full implementations are in tensor_array.cc
template <typename Device, typename T>
Status AddToTensor(OpKernelContext* ctx, Tensor* sum, const Tensor* current,
                   const Tensor* add) {
  return errors::InvalidArgument(
      "tensor_array::AddToTensor type not supported: ",
      DataTypeString(DataTypeToEnum<T>::value));
}

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                         \
  template <>                                                        \
  Status AddToTensor<Device, T>(OpKernelContext * ctx, Tensor * sum, \
                                const Tensor* current, const Tensor* add);

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_WRITE_OR_ADD

template <typename Device, typename T>
Status TensorSetZero(OpKernelContext* ctx, Tensor* value) {
  return errors::InvalidArgument(
      "tensor_array::TensorSetZero type not supported: ",
      DataTypeString(DataTypeToEnum<T>::value));
}

#define TENSOR_ARRAY_SET_ZERO(Device, T) \
  template <>                            \
  Status TensorSetZero<Device, T>(OpKernelContext * ctx, Tensor * value);

#define TENSOR_ARRAY_SET_ZERO_CPU(T) TENSOR_ARRAY_SET_ZERO(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_CPU);
TF_CALL_bool(TENSOR_ARRAY_SET_ZERO_CPU);
#undef TENSOR_ARRAY_SET_ZERO_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_SET_ZERO_GPU(T) TENSOR_ARRAY_SET_ZERO(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
#undef TENSOR_ARRAY_SET_ZERO_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_SET_ZERO

}  // namespace tensor_array

// The TensorArray object keeps an array of Tensors.  It allows reading from the
// array and writing to the array.
//
// Important properties:
//   * Usually, writing to a particular index in the TensorArray is allowed at
//     most once per index.  In a special case, writes with the flag
//     multiple_writes_aggregate allow multiple writes to the same
//     index.  In this case, the writes are summed.
//   * Multiple reads are supported.
//   * Deep copies of Tensors are rarely made.  The only time they are made is
//     when WriteOrAggregate is called at least twice on the same index with the
//     flag multiple_writes_aggregate = True.
//   * Reading and Writing to the array is protected by a mutex.
//     All operations on a TensorArray are thread-safe.
//   * A TensorArray may be preemptively closed, which releases all
//     memory associated with it.
//
// These properties together allow the TensorArray to work as a
// functional object and makes gradient computation easy.  For
// example:
//   * Write-Once semantics mean the gradient of a TensorArray Read never has to
//     worry which of multiple writes to that index the gradient value
//     is meant for.
//   * Read-Many semantics (when using clear_after_read=false) allow the
//     TensorArray to be read, packed, or concatenated multiple times;
//     and the gradient operations use the multiple_writes_aggregate
//     flag to aggregate the backprop writes.  Multiple backprop writes to
//     the same index are partial gradients corresponding to the
//     multiple reads of that index in the forward phase.
//
class TensorArray : public ResourceBase {
 public:
  static std::atomic<int64_t> tensor_array_counter;

  // Construct a TensorArray for holding Tensors of type 'dtype' with
  // 'N' elements.  While the underlying storage is a std::vector and
  // can hold more than MAX_INT entries, in practice we do not expect
  // users to construct this many Tensors for storage in a TensorArray.
  TensorArray(const string& key, const DataType& dtype, const Tensor& handle,
              int32_t N, const PartialTensorShape& element_shape,
              bool identical_element_shapes, bool dynamic_size,
              bool multiple_writes_aggregate, bool is_grad, int32_t marked_size,
              bool clear_after_read)
      : key_(key),
        dtype_(dtype),
        handle_(handle),
        closed_(false),
        dynamic_size_(dynamic_size),
        multiple_writes_aggregate_(multiple_writes_aggregate),
        gradients_disallowed_(false),
        clear_after_read_(clear_after_read),
        is_grad_(is_grad),
        marked_size_(marked_size),
        element_shape_(element_shape),
        identical_element_shapes_(identical_element_shapes),
        tensors_(N) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_0(mht_0_v, 325, "", "./tensorflow/core/kernels/tensor_array.h", "TensorArray");
}

  // Write Tensor 'value' to index 'index'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * If the array has dynamic size:
  //      The index is >= 0
  //    Otherwise:
  //      The index is in [0, N) where N == Size()
  //  * The dtype of the Tensor in 'value' matches the TensorArray's dtype.
  //  * If multiple_writes_aggregate is false:
  //    The Tensor at 'index' has not yet been written to.
  //  * If multiple_writes_aggregate is true:
  //    The Tensor at 'index' has the same shape as value.
  //
  // Side effects:
  //  * On the first write to 'index':
  //    - The underlying Tensor in 'value' has a new reference to it.
  //    - The index 'index' is marked as written.
  //  * If multiple_writes_aggregate is false, subsequent writes to 'index'
  //    raise an InvalidArgument error.
  //  * If multiple_writes_aggregate is true, subsequent writes to 'index':
  //    - The underlying Tensors in 'value' and from the first write
  //      are released and a local Tensor is created.
  //    - Index 'index' is also marked as local_copy.
  //    - The gradients_disallowed flag is set true (GradientsAllowed()
  //      will now return false).
  //
  // Note, value is passed as a pointer because we its underlying
  // Tensor's shape is accessed.  Otherwise it is not modified.
  template <typename Device, typename T>
  Status WriteOrAggregate(OpKernelContext* ctx, const int32_t index,
                          const Tensor* value) {
    mutex_lock l(mu_);
    return LockedWriteOrAggregate<Device, T>(ctx, index, value);
  }

  template <typename Device, typename T>
  Status WriteOrAggregateMany(OpKernelContext* ctx,
                              const std::vector<int32>& indices,
                              std::vector<Tensor>* values) {
    mutex_lock l(mu_);
    int32_t i = 0;
    for (const int32_t ix : indices) {
      Status s = LockedWriteOrAggregate<Device, T>(ctx, ix, &(*values)[i]);
      ++i;
      TF_RETURN_IF_ERROR(s);
    }
    return Status::OK();
  }

  // Read from index 'index' into Tensor 'value'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * The index is in [0, N)
  //  * The Tensor at 'index' has been written to.
  //  * The Tensor at 'index' has not been read from with flag
  //    clear_after_read = true.
  //
  // Side effects:
  //  * If clear_after_read is true, the reference to the underlying
  //    Tensor is deleted.
  //  * The reference to the underlying Tensor at 'index' is copied to
  //    the returned '*value'.
  //  * The index is marked as read (it cannot be rewritten to).
  template <typename Device, typename T>
  Status Read(OpKernelContext* ctx, const int32_t index, Tensor* value) {
    mutex_lock l(mu_);
    return LockedRead<Device, T>(ctx, index, value);
  }

  template <typename Device, typename T>
  Status ReadMany(OpKernelContext* ctx, const std::vector<int32>& indices,
                  std::vector<Tensor>* values) {
    mutex_lock l(mu_);
    values->clear();
    values->resize(indices.size());
    int32_t i = 0;
    for (const int32_t ix : indices) {
      Status s = LockedRead<Device, T>(ctx, ix, &(*values)[i]);
      ++i;
      if (!s.ok()) return s;
    }
    return Status::OK();
  }

  DataType ElemType() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_1(mht_1_v, 416, "", "./tensorflow/core/kernels/tensor_array.h", "ElemType");
 return dtype_; }

  PartialTensorShape ElemShape() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_2(mht_2_v, 421, "", "./tensorflow/core/kernels/tensor_array.h", "ElemShape");

    mutex_lock l(mu_);
    return element_shape_;
  }

  Status SetElemShape(const PartialTensorShape& candidate) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_3(mht_3_v, 429, "", "./tensorflow/core/kernels/tensor_array.h", "SetElemShape");

    mutex_lock l(mu_);
    PartialTensorShape new_element_shape_;
    Status s = element_shape_.MergeWith(candidate, &new_element_shape_);
    if (!s.ok()) {
      return s;
    }
    element_shape_ = new_element_shape_;
    return Status::OK();
  }

  string DebugString() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_4(mht_4_v, 443, "", "./tensorflow/core/kernels/tensor_array.h", "DebugString");

    mutex_lock l(mu_);
    CHECK(!closed_);
    return strings::StrCat("TensorArray[", tensors_.size(), "]");
  }

  bool IsClosed() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_5(mht_5_v, 452, "", "./tensorflow/core/kernels/tensor_array.h", "IsClosed");

    mutex_lock l(mu_);
    return closed_;
  }

  // Return the size of the TensorArray.
  Status Size(int32* size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_6(mht_6_v, 461, "", "./tensorflow/core/kernels/tensor_array.h", "Size");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = tensors_.size();
    return Status::OK();
  }

  // Record the size of the TensorArray after an unpack or split.
  Status SetMarkedSize(int32_t size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_7(mht_7_v, 472, "", "./tensorflow/core/kernels/tensor_array.h", "SetMarkedSize");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    if (!is_grad_) {
      marked_size_ = size;
    }
    return Status::OK();
  }

  // Return the marked size of the TensorArray.
  Status MarkedSize(int32* size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_8(mht_8_v, 485, "", "./tensorflow/core/kernels/tensor_array.h", "MarkedSize");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = marked_size_;
    return Status::OK();
  }

  // Return the size that should be used by pack or concat op.
  Status PackOrConcatSize(int32* size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_9(mht_9_v, 496, "", "./tensorflow/core/kernels/tensor_array.h", "PackOrConcatSize");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = is_grad_ ? marked_size_ : tensors_.size();
    return Status::OK();
  }

  // Once a TensorArray is being used for gradient calculations, it
  // should be marked as no longer resizeable.
  void DisableDynamicSize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_10(mht_10_v, 508, "", "./tensorflow/core/kernels/tensor_array.h", "DisableDynamicSize");

    mutex_lock l(mu_);
    dynamic_size_ = false;
  }

  bool HasDynamicSize() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_11(mht_11_v, 516, "", "./tensorflow/core/kernels/tensor_array.h", "HasDynamicSize");

    mutex_lock l(mu_);
    return dynamic_size_;
  }

  bool GradientsAllowed() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_12(mht_12_v, 524, "", "./tensorflow/core/kernels/tensor_array.h", "GradientsAllowed");

    mutex_lock l(mu_);
    return !gradients_disallowed_;
  }

  bool HasIdenticalElementShapes() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_13(mht_13_v, 532, "", "./tensorflow/core/kernels/tensor_array.h", "HasIdenticalElementShapes");
 return identical_element_shapes_; }

  // Copy the TensorShapes from another TensorArray into this one.
  // If `shapes_to_prepend` is set, expands the rank of the copied shape by
  // prepending the passed in shape prefix to the shape values in `rhs`.
  // The sizes of the two TensorArrays must match and this one
  // may not have any entries filled in.  This performs a "soft copy",
  // essentially filling the current TensorArray with virtual
  // zero-tensors, which will be replaced by future aggregate writes,
  // or instantiated by future reads.  Requires a non-const pointer
  // to the rhs to access its mutex.
  Status CopyShapesFrom(TensorArray* rhs, const TensorShape* shape_to_prepend);

  // Clear the TensorArray, including any Tensor references, and mark as closed.
  void ClearAndMarkClosed() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_14(mht_14_v, 549, "", "./tensorflow/core/kernels/tensor_array.h", "ClearAndMarkClosed");

    mutex_lock l(mu_);
    tensors_.clear();
    closed_ = true;
  }

  mutex* mu() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_15(mht_15_v, 558, "", "./tensorflow/core/kernels/tensor_array.h", "mu");
 return &mu_; }
  Tensor* handle() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_16(mht_16_v, 562, "", "./tensorflow/core/kernels/tensor_array.h", "handle");
 return &handle_; }

  ResourceHandle resource_handle(OpKernelContext* ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_17(mht_17_v, 567, "", "./tensorflow/core/kernels/tensor_array.h", "resource_handle");

    return ctx->step_container()->MakeResourceHandle<TensorArray>(
        key_, *ctx->device());
  }

 private:
  Status LockedWrite(OpKernelContext* ctx, const int32_t index, Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename Device, typename T>
  Status LockedWriteOrAggregate(OpKernelContext* ctx, const int32_t index,
                                const Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename Device, typename T>
  Status LockedRead(OpKernelContext* ctx, const int32_t index, Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status LockedReturnIfClosed() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (closed_) {
      return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                     " has already been closed.");
    }
    return Status::OK();
  }

  const string key_;

  const DataType dtype_;
  Tensor handle_;

  mutable mutex mu_;

  // Marks that the tensor_array_ has been cleared.
  bool closed_ TF_GUARDED_BY(mu_);

  // Writes are allowed to grow the array.
  bool dynamic_size_;

  // Multiple writes to the same index will result in summation of the
  // values (used by backprop)
  const bool multiple_writes_aggregate_;

  // If multiple Writes were attempted (e.g. via attribute
  // multiple_writes_aggregate), then gradients are disallowed.
  bool gradients_disallowed_ TF_GUARDED_BY(mu_);

  // After a read at an index, clear away its Tensor to release memory.
  const bool clear_after_read_;

  // True iff this is a gradient tensor array.
  const bool is_grad_;

  // The size of the TensorArray after a (legacy) unpack or split is performed.
  // -1 if there has been no unpack or split performed on the TensorArray.
  int32 marked_size_;

  // The shape of each element in the TensorArray, may be partially known or not
  // known at all.
  PartialTensorShape element_shape_ TF_GUARDED_BY(mu_);

  // Whether all elements in the TensorArray have identical shapes.
  // This allows certain behaviors, like dynamically checking for
  // consistent shapes on write, and being able to fill in properly
  // shaped zero tensors on stack -- even if the initial element_shape
  // was not fully defined.
  const bool identical_element_shapes_;

  // TensorAndState is used to keep track of the Tensors stored in the
  // TensorArray, along with their shapes, and a boolean that determines whether
  // they have already been read or not.
  struct TensorAndState {
    TensorAndState()
        : written(false), read(false), cleared(false), local_copy(false) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTh mht_18(mht_18_v, 643, "", "./tensorflow/core/kernels/tensor_array.h", "TensorAndState");
}
    Tensor tensor;
    TensorShape shape;
    bool written;  // True if a Tensor has been written to the index.
    bool read;  // True if a Tensor has been written to and read from the index.
    bool cleared;  // True if a tensor has been read with
                   // clear_after_read = true;

    // Used by writes when multiple_writes_aggregate is true.  In this
    // case, the first time a value is written, it is a shallow copy.
    // The second time a value is written, it is aggregated.  However,
    // in this case a new Tensor must be constructed to hold the
    // aggregated value.  This flag marks that such a Tensor is being
    // used.  All future writes will aggregate to the existing local Tensor.
    bool local_copy;
  };
  // The list of underlying Tensors and states.
  std::vector<TensorAndState> tensors_ TF_GUARDED_BY(mu_);
};

template <typename Device, typename T>
Status TensorArray::LockedWriteOrAggregate(OpKernelContext* ctx,
                                           const int32_t index,
                                           const Tensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  size_t index_size = static_cast<size_t>(index);
  if (index < 0 || (!dynamic_size_ && index_size >= tensors_.size())) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1), ": Tried to write to index ",
        index, " but array is not resizeable and size is: ", tensors_.size());
  }
  if (dynamic_size_) {
    // We must grow the internal TensorArray
    if (index_size >= tensors_.capacity()) {
      tensors_.reserve(2 * (index_size + 1));
    }
    if (index_size >= tensors_.size()) {
      tensors_.resize(index_size + 1);
    }
  }
  TensorAndState& t = tensors_[index];

  if (value->dtype() != dtype_) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1),
        ": Could not write to TensorArray index ", index,
        " because the value dtype is ", DataTypeString(value->dtype()),
        " but TensorArray dtype is ", DataTypeString(dtype_), ".");
  }
  if (!element_shape_.IsCompatibleWith(value->shape())) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1),
        ": Could not write to TensorArray index ", index,
        " because the value shape is ", value->shape().DebugString(),
        " which is incompatible with the TensorArray's inferred element "
        "shape: ",
        element_shape_.DebugString(), " (consider setting infer_shape=False).");
  } else if (identical_element_shapes_ && !element_shape_.IsFullyDefined()) {
    element_shape_ = PartialTensorShape(value->shape().dim_sizes());
  }

  if (t.read) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not write to TensorArray index ",
                                   index, " because it has already been read.");
  }

  if (!multiple_writes_aggregate_ && t.written) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not write to TensorArray index ",
                                   index,
                                   " because it has already been written to.");
  }

  if (t.written) {
    DCHECK(multiple_writes_aggregate_);

    // Check that value shape matches t.shape
    if (value->shape() != t.shape) {
      return errors::InvalidArgument(
          "TensorArray ", handle_.vec<tstring>()(1),
          ": Could not aggregate to TensorArray index ", index,
          " because the existing shape is ", t.shape.DebugString(),
          " but the new input shape is ", value->shape().DebugString(), ".");
    }

    if (!t.tensor.IsInitialized() || t.tensor.NumElements() == 0) {
      // If existing_t == nullptr but written == true, then what was stored
      // was just a shape, which just means zeros.  So all we must do in this
      // case is copy the reference over and return early.
      t.tensor = *value;
      return Status::OK();
    }

    Tensor* existing_t = &t.tensor;

    if (t.local_copy) {
      Status s = tensor_array::AddToTensor<Device, T>(ctx, existing_t,
                                                      existing_t, value);
      TF_RETURN_IF_ERROR(s);
    } else {
      Tensor local_tensor;
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(dtype_, existing_t->shape(), &local_tensor));
      Status s = tensor_array::AddToTensor<Device, T>(ctx, &local_tensor,
                                                      existing_t, value);
      TF_RETURN_IF_ERROR(s);
      t.tensor = local_tensor;
      t.local_copy = true;
    }

    // We've aggregated the values, so disallow backprop on this
    // TensorArray.
    gradients_disallowed_ = true;
  } else {
    t.tensor = *value;
    t.shape = value->shape();
    t.written = true;
  }
  return Status::OK();
}

template <typename Device, typename T>
Status TensorArray::LockedRead(OpKernelContext* ctx, const int32_t index,
                               Tensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  if ((index < 0) ||
      (!is_grad_ && (static_cast<size_t>(index) >= tensors_.size()))) {
    return errors::InvalidArgument("Tried to read from index ", index,
                                   " but array size is: ", tensors_.size());
  }
  size_t index_t = static_cast<size_t>(index);
  if ((is_grad_ && (index_t >= tensors_.size() || !tensors_[index].written)) ||
      (!is_grad_ && (index_t < tensors_.size() && !tensors_[index].written))) {
    // Special case returning zeros if this is a gradient read that happens
    // after a stop_gradients call with dynamic forward TensorArrays.
    // There is sometimes a race condition where the gradient is not
    // written due to stop_gradients, but is later read.
    TensorShape element_shape;
    if (is_grad_ && index_t < tensors_.size() &&
        tensors_[index].shape.dims() > 0) {
      // A gradient TensorArray has more specific gradient information
      // available for each entry.  A forward TensorArray must rely on
      // the global element_shape_ to fill in zeros on read.
      element_shape = tensors_[index].shape;
    } else if (!element_shape_.IsFullyDefined()) {
      return errors::InvalidArgument(
          "TensorArray ", handle_.vec<tstring>()(1),
          ": Could not read from TensorArray index ", index,
          ".  Furthermore, the element shape is not fully defined: ",
          element_shape_.DebugString(),
          ".  It is possible you are working with a resizeable TensorArray and "
          "stop_gradients is not allowing the gradients to be written.  If you "
          "set the full "
          "element_shape property on the forward TensorArray, the proper "
          "all-zeros tensor "
          "will be returned instead of incurring this error.");
    } else {
      element_shape_.AsTensorShape(&element_shape);  // Always succeeds.
    }
    if (index_t >= tensors_.size()) {
      // Fill in tensors_ up to index to have known shape.
      size_t old_tensors_size = tensors_.size();
      tensors_.resize(index + 1);
      for (size_t i = old_tensors_size; i < index + 1; ++i) {
        tensors_[i].shape = element_shape;
        tensors_[i].written = true;
      }
    } else {
      tensors_[index].shape = element_shape;
      tensors_[index].written = true;
    }
  }

  TensorAndState& t = tensors_[index];

  if (t.cleared) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not read index ", index,
                                   " twice because it was cleared after a "
                                   "previous read (perhaps try setting "
                                   "clear_after_read = false?).");
  }

  if (!t.tensor.IsInitialized() || t.tensor.NumElements() == 0) {
    // We stored just a shape, but no value.  This means create and
    // return zeros of the appropriate shape.
    TF_RETURN_IF_ERROR(ctx->allocate_temp(dtype_, t.shape, &t.tensor));
    if (t.shape.num_elements() > 0) {
      Status s = tensor_array::TensorSetZero<Device, T>(ctx, &t.tensor);
      if (!s.ok()) return s;
    }
  }

  // Data is available inside the tensor, copy the reference over.
  *value = t.tensor;

  if (clear_after_read_) {
    t.tensor = Tensor();
    t.cleared = true;
  }
  t.read = true;
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_
