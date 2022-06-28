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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh() {
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


#include <cstdint>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Forward declarations.  In particular, we forward declare protos so that their
// symbols can be removed from .so exports.
class AllocationDescription;
class Allocator;
class OpKernelContext;
class Tensor;
class TensorBuffer;
class TensorCApi;
class TensorCord;
class TensorDescription;
class TensorProto;
class Var;

namespace batch_util {
Status CopyElementToSlice(Tensor element, Tensor* parent, int64_t index);
Status CopySliceToElement(const Tensor& parent, Tensor* element, int64_t index);
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64_t index);
Status CopyContiguousSlices(const Tensor& src, int64_t src_offset,
                            int64_t dst_offset, int64_t num_slices,
                            Tensor* dst);
}  // namespace batch_util

/// @ingroup core

/// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
 public:
  explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/framework/tensor.h", "TensorBuffer");
}
  ~TensorBuffer() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_1(mht_1_v, 239, "", "./tensorflow/core/framework/tensor.h", "~TensorBuffer");
}

  /// \brief data() points to a memory region of size() bytes.
  ///
  /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
  /// It can be called multiple times when the contents of a `Tensor` are
  /// accessed, and so making it non-virtual allows the body to be inlined.
  void* data() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_2(mht_2_v, 249, "", "./tensorflow/core/framework/tensor.h", "data");
 return data_; }

  /// \brief Size (in bytes) of the buffer.
  virtual size_t size() const = 0;

  /// \brief If this TensorBuffer is sub-buffer of another TensorBuffer,
  /// returns that TensorBuffer. Otherwise, returns this.
  virtual TensorBuffer* root_buffer() = 0;

  /// \brief Fills metadata about the allocation into the proto.
  virtual void FillAllocationDescription(
      AllocationDescription* proto) const = 0;

  virtual bool GetAllocatedBytes(size_t* out_bytes) const;

  /// \brief Helper method to reinterpret the buffer as an array of `T`.
  template <typename T>
  T* base() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_3(mht_3_v, 269, "", "./tensorflow/core/framework/tensor.h", "base");

    return reinterpret_cast<T*>(data());
  }

  /// \brief Whether this TensorBuffer owns the underlying memory.
  virtual bool OwnsMemory() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_4(mht_4_v, 277, "", "./tensorflow/core/framework/tensor.h", "OwnsMemory");
 return true; }

  /// \brief The type of the underlying memory.
  virtual AllocatorMemoryType GetMemoryType() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_5(mht_5_v, 283, "", "./tensorflow/core/framework/tensor.h", "GetMemoryType");

    return AllocatorMemoryType::kUnknown;
  }

 private:
  void* const data_;
};

/// Represents an n-dimensional array of values.
class Tensor {
 public:
  /// \brief Creates a 1-dimensional, 0-element float tensor.
  ///
  /// The returned Tensor is not a scalar (shape {}), but is instead
  /// an empty one-dimensional Tensor (shape {0}, NumElements() ==
  /// 0). Since it has no elements, it does not need to be assigned a
  /// value and is initialized by default (IsInitialized() is
  /// true). If this is undesirable, consider creating a one-element
  /// scalar which does require initialization:
  ///
  /// ```c++
  ///
  ///     Tensor(DT_FLOAT, TensorShape({}))
  ///
  /// ```
  Tensor();

  /// \brief Creates a Tensor of the given `type` and `shape`.  If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  Tensor(DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` to allocate the underlying buffer. If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` and the specified "allocation_attr" to
  /// allocate the underlying buffer. If the kernel and step are known
  /// allocation_attr.allocation_will_be_logged should be set to true
  /// and LogMemory::RecordTensorAllocation should be called after the
  /// tensor is constructed. Calling the Tensor constructor directly
  /// from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape,
         const AllocationAttributes& allocation_attr);

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Acquires a ref on buf that belongs to this Tensor.
  Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Takes an ownership of the bufffer from the reference counted pointer.
  Tensor(DataType type, TensorShape shape, core::RefCountPtr<TensorBuffer> buf);

  /// \brief Creates an empty Tensor of the given data type.
  ///
  /// Like Tensor(), returns a 1-dimensional, 0-element Tensor with
  /// IsInitialized() returning True. See the Tensor() documentation
  /// for details.
  explicit Tensor(DataType type);

  /// \brief Initializes a tensor with the input `type` and `shape`, or returns
  /// an error and leaves `out_tensor` unmodified. This factory method should be
  /// used instead of the corresponding constructor if calling code cannot
  /// validate that the `DataType` is valid and supported.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  static Status BuildTensor(DataType type, const TensorShape& shape,
                            Tensor* out_tensor);

 private:
  // A tag type for selecting the `Tensor` constructor overload that creates a
  // scalar tensor in host memory.
  struct host_scalar_tag {};

  class HostScalarTensorBufferBase;
  template <typename T>
  struct ValueAndTensorBuffer;

  // Creates a tensor with the given scalar `value` in CPU memory.
  template <typename T>
  Tensor(T value, host_scalar_tag tag);

 public:
  // A series of specialized constructors for scalar tensors in host memory.
  //
  // NOTE: The `Variant` host-scalar constructor is not defined, because Variant
  // is implicitly constructible from many different types, and this causes
  // ambiguities with some compilers.
  explicit Tensor(float scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_6(mht_6_v, 394, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(double scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_7(mht_7_v, 399, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(int32_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_8(mht_8_v, 404, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(uint32 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_9(mht_9_v, 409, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(uint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_10(mht_10_v, 414, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(uint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_11(mht_11_v, 419, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(int16_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_12(mht_12_v, 424, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(int8_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_13(mht_13_v, 429, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(tstring scalar_value)
      : Tensor(std::move(scalar_value), host_scalar_tag{}) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("scalar_value: \"" + (std::string)scalar_value + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_14(mht_14_v, 435, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(complex64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_15(mht_15_v, 440, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(complex128 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_16(mht_16_v, 445, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(int64_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_17(mht_17_v, 450, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(uint64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_18(mht_18_v, 455, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(bool scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_19(mht_19_v, 460, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(qint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_20(mht_20_v, 465, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(quint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_21(mht_21_v, 470, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(qint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_22(mht_22_v, 475, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(quint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_23(mht_23_v, 480, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(qint32 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_24(mht_24_v, 485, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(bfloat16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_25(mht_25_v, 490, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(Eigen::half scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_26(mht_26_v, 495, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}
  explicit Tensor(ResourceHandle scalar_value)
      : Tensor(std::move(scalar_value), host_scalar_tag{}) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_27(mht_27_v, 500, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}

  // NOTE: The `const char*` host-scalar constructor is provided as a
  // convenience because otherwise passing a string literal would surprisingly
  // construct a DT_BOOL tensor.
  explicit Tensor(const char* scalar_value)
      : Tensor(tstring(scalar_value), host_scalar_tag{}) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("scalar_value: \"" + (scalar_value == nullptr ? std::string("nullptr") : std::string((char*)scalar_value)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_28(mht_28_v, 510, "", "./tensorflow/core/framework/tensor.h", "Tensor");
}

  /// Copy constructor.
  Tensor(const Tensor& other);

  /// \brief Move constructor. After this call, <other> is safely destructible
  /// can be assigned to, and IsInitialized() can be called and will return
  /// false. Other calls on <other> (e.g. shape manipulation) are not valid.
  Tensor(Tensor&& other);

  // Explicitly delete constructor that take a pointer (except char*)
  // so that the pointer doesn't get implicitly cast to bool.
  template <typename T, typename std::enable_if<!std::is_same<T, char>::value,
                                                T>::type* = nullptr>
  explicit Tensor(T* t) = delete;

  ~Tensor();

  /// Returns the data type.
  DataType dtype() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_29(mht_29_v, 532, "", "./tensorflow/core/framework/tensor.h", "dtype");
 return shape_.data_type(); }

  /// Returns the shape of the tensor.
  const TensorShape& shape() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_30(mht_30_v, 538, "", "./tensorflow/core/framework/tensor.h", "shape");
 return shape_; }

  /// \brief Convenience accessor for the tensor shape.
  ///
  /// For all shape accessors, see comments for relevant methods of
  /// `TensorShape` in `tensor_shape.h`.
  int dims() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_31(mht_31_v, 547, "", "./tensorflow/core/framework/tensor.h", "dims");
 return shape().dims(); }

  /// Convenience accessor for the tensor shape.
  int64_t dim_size(int d) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_32(mht_32_v, 553, "", "./tensorflow/core/framework/tensor.h", "dim_size");
 return shape().dim_size(d); }

  /// Convenience accessor for the tensor shape.
  int64_t NumElements() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_33(mht_33_v, 559, "", "./tensorflow/core/framework/tensor.h", "NumElements");
 return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_34(mht_34_v, 564, "", "./tensorflow/core/framework/tensor.h", "IsSameSize");

    return shape().IsSameSize(b.shape());
  }

  // True iff the two tensors use the same underlying refcounted storage
  bool SharesBufferWith(const Tensor& b) const;

  /// \brief If necessary, has this Tensor been initialized?
  ///
  /// Zero-element Tensors are always considered initialized, even if they
  /// have never been assigned to and do not have any memory allocated.
  bool IsInitialized() const;

  /// Returns the estimated memory usage of this tensor.
  size_t TotalBytes() const;

  // Returns the size of allocated memory for this tensor.
  size_t AllocatedBytes() const;

  /// Returns true iff this tensor is aligned.
  bool IsAligned() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_35(mht_35_v, 587, "", "./tensorflow/core/framework/tensor.h", "IsAligned");

#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    void* ptr = base<void>();
    return dtype() == DT_STRING || NumElements() == 0 ||
           (reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0);
#endif
  }

  /// Assign operator. This tensor shares other's underlying storage.
  Tensor& operator=(const Tensor& other) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_36(mht_36_v, 601, "", "./tensorflow/core/framework/tensor.h", "=");

    CopyFromInternal(other, other.shape());
    return *this;
  }

  /// Move operator.  See move constructor for details.
  Tensor& operator=(Tensor&& other);

  /// \brief Copy the other tensor into this tensor and reshape it.
  ///
  /// This tensor shares other's underlying storage. Returns `true`
  /// iff `other.shape()` has the same number of elements of the given
  /// `shape`.
  bool CopyFrom(const Tensor& other,
                const TensorShape& shape) TF_MUST_USE_RESULT {
    if (other.NumElements() != shape.num_elements()) return false;
    CopyFromInternal(other, shape);
    return true;
  }

  /// \brief Slice this tensor along the 1st dimension.

  /// I.e., the returned tensor satisfies
  ///     returned[i, ...] == this[dim0_start + i, ...].
  /// The returned tensor shares the underlying tensor buffer with this
  /// tensor.
  ///
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// NOTE: When fed with an N-dimensional tensor, this method returns a tensor
  /// also with N dimensions. If you want to select a sub tensor, see SubSlice.
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`
  Tensor Slice(int64_t dim0_start, int64_t dim0_limit) const;

  /// \brief Select a subslice from this tensor along the 1st dimension.
  ///
  /// When fed with an N-dimensional tensor, this method returns a tensor with
  /// N-1 dimensions, where the returned tensor is a subslice of the input
  /// tensor along the first dimension. The N-1 dimensions of the returned
  /// tensor are the last N-1 dimensions of the input tensor.
  ///
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= index < dim_size(0)`
  Tensor SubSlice(int64_t index) const;

  /// \brief Parse `other` and construct the tensor.

  /// Returns `true` iff the parsing succeeds. If the parsing fails,
  /// the state of `*this` is unchanged.
  bool FromProto(const TensorProto& other) TF_MUST_USE_RESULT;
  bool FromProto(Allocator* a, const TensorProto& other) TF_MUST_USE_RESULT;

  /// \brief Fills in `proto` with `*this` tensor's content.
  ///
  /// `AsProtoField()` fills in the repeated field for `proto.dtype()`, while
  /// `AsProtoTensorContent()` encodes the content in `proto.tensor_content()`
  /// in a compact form.
  void AsProtoField(TensorProto* proto) const;
  void AsProtoTensorContent(TensorProto* proto) const;

  /// \brief Return the tensor data as an `Eigen::Tensor` with the type and
  /// sizes of this `Tensor`.
  ///
  /// Use these methods when you know the data type and the number of
  /// dimensions of the Tensor and you want an `Eigen::Tensor`
  /// automatically sized to the `Tensor` sizes. The implementation check
  /// fails if either type or sizes mismatch.
  ///
  /// Example:
  ///
  /// ```c++
  ///
  ///     typedef float T;
  ///     Tensor my_mat(...built with Shape{rows: 3, cols: 5}...);
  ///     auto mat = my_mat.matrix<T>();    // 2D Eigen::Tensor, 3 x 5.
  ///     auto mat = my_mat.tensor<T, 2>(); // 2D Eigen::Tensor, 3 x 5.
  ///     auto vec = my_mat.vec<T>();       // CHECK fails as my_mat is 2D.
  ///     auto vec = my_mat.tensor<T, 3>(); // CHECK fails as my_mat is 2D.
  ///     auto mat = my_mat.matrix<int32>();// CHECK fails as type mismatch.
  ///
  /// ```
  template <typename T>
  typename TTypes<T>::Vec vec() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_37(mht_37_v, 696, "", "./tensorflow/core/framework/tensor.h", "vec");

    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_38(mht_38_v, 704, "", "./tensorflow/core/framework/tensor.h", "matrix");

    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor();

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_tensor();

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor reinterpret_last_dimension();

  /// \brief Return the tensor data as an `Eigen::Tensor` of the data type and a
  /// specified shape.
  ///
  /// These methods allow you to access the data with the dimensions
  /// and sizes of your choice.  You do not need to know the number of
  /// dimensions of the Tensor to call them.  However, they `CHECK` that
  /// the type matches and the dimensions requested creates an
  /// `Eigen::Tensor` with the same number of elements as the tensor.
  ///
  /// Example:
  ///
  /// ```c++
  ///
  ///     typedef float T;
  ///     Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
  ///     // 1D Eigen::Tensor, size 60:
  ///     auto flat = my_ten.flat<T>();
  ///     // 2D Eigen::Tensor 12 x 5:
  ///     auto inner = my_ten.flat_inner_dims<T>();
  ///     // 2D Eigen::Tensor 4 x 15:
  ///     auto outer = my_ten.shaped<T, 2>({4, 15});
  ///     // CHECK fails, bad num elements:
  ///     auto outer = my_ten.shaped<T, 2>({4, 8});
  ///     // 3D Eigen::Tensor 6 x 5 x 2:
  ///     auto weird = my_ten.shaped<T, 3>({6, 5, 2});
  ///     // CHECK fails, type mismatch:
  ///     auto bad   = my_ten.flat<int32>();
  ///
  /// ```
  template <typename T>
  typename TTypes<T>::Flat flat() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_39(mht_39_v, 762, "", "./tensorflow/core/framework/tensor.h", "flat");

    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedFlat unaligned_flat() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_40(mht_40_v, 770, "", "./tensorflow/core/framework/tensor.h", "unaligned_flat");

    return unaligned_shaped<T, 1>({NumElements()});
  }

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the last NDIMS-1 into the first dimension of the
  /// result. If NDIMS > dims() then leading dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_inner_dims();

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the first NDIMS-1 into the last dimension of the
  /// result. If NDIMS > dims() then trailing dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_outer_dims();

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing the
  /// first 'begin' Tensor dimensions into the first dimension of the result and
  /// the Tensor dimensions of the last dims() - 'begin' - NDIMS into the last
  /// dimension of the result. If 'begin' < 0 then the |'begin'| leading
  /// dimensions of size 1 will be added. If 'begin' + NDIMS > dims() then
  /// 'begin' + NDIMS - dims() trailing dimensions of size 1 will be added.
  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::Tensor flat_inner_outer_dims(int64_t begin);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64_t> new_sizes);

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_shaped(
      gtl::ArraySlice<int64_t> new_sizes);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(
      gtl::ArraySlice<int64_t> new_sizes);

  /// \brief Return the Tensor data as a `TensorMap` of fixed size 1:
  /// `TensorMap<TensorFixedSize<T, 1>>`.

  /// Using `scalar()` allows the compiler to perform optimizations as
  /// the size of the tensor is known at compile time.
  template <typename T>
  typename TTypes<T>::Scalar scalar();

  /// Const versions of all the methods above.
  template <typename T>
  typename TTypes<T>::ConstVec vec() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_41(mht_41_v, 826, "", "./tensorflow/core/framework/tensor.h", "vec");

    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_42(mht_42_v, 834, "", "./tensorflow/core/framework/tensor.h", "matrix");

    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_tensor() const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor reinterpret_last_dimension() const;

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_43(mht_43_v, 863, "", "./tensorflow/core/framework/tensor.h", "flat");

    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedConstFlat unaligned_flat() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_44(mht_44_v, 871, "", "./tensorflow/core/framework/tensor.h", "unaligned_flat");

    return unaligned_shaped<T, 1>({NumElements()});
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor shaped(
      gtl::ArraySlice<int64_t> new_sizes) const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_shaped(
      gtl::ArraySlice<int64_t> new_sizes) const;

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(
      gtl::ArraySlice<int64_t> new_sizes) const;

  template <typename T>
  typename TTypes<T>::ConstScalar scalar() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_dims() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_outer_dims() const;

  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_outer_dims(
      int64_t begin) const;

  /// Render the first `max_entries` values in `*this` into a string.
  std::string SummarizeValue(int64_t max_entries, bool print_v2 = false) const;

  /// A human-readable summary of the tensor suitable for debugging.
  // `num_values` is the number of actual data values in the tensor
  // included in the message. If the tensor might be resident in
  // GPU/TPU memory use DeviceSafeDebugString instead.
  std::string DebugString(int num_values) const;
  std::string DebugString() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_45(mht_45_v, 916, "", "./tensorflow/core/framework/tensor.h", "DebugString");
 return DebugString(3); }

  // Variant of DebugString() that should be used for possibly non-CPU tensors.
  // If the tensor is not resident on CPU, we can't read its values as
  // DebugString() does.
  std::string DeviceSafeDebugString() const;

  /// Fill in the `TensorDescription` proto with metadata about the
  /// tensor that is useful for monitoring and debugging.
  void FillDescription(TensorDescription* description) const;

  /// \brief Returns a `StringPiece` mapping the current tensor's buffer.
  ///
  /// The returned `StringPiece` may point to memory location on devices
  /// that the CPU cannot address directly.
  ///
  /// NOTE: The underlying tensor buffer is refcounted, so the lifetime
  /// of the contents mapped by the `StringPiece` matches the lifetime of
  /// the buffer; callers should arrange to make sure the buffer does
  /// not get destroyed while the `StringPiece` is still used.
  ///
  /// REQUIRES: `DataTypeCanUseMemcpy(dtype())`.
  StringPiece tensor_data() const;
  void* data() const;

  /// Copy the other tensor into this tensor, reshape it and reinterpret the
  /// buffer's datatype. If Status::OK() is returned, the two tensors now share
  /// the same underlying storage.
  ///
  /// This call requires that the `other` tensor and the given type and shape
  /// are "compatible" (i.e. they occupy the same number of bytes).
  ///
  /// Specifically:
  ///
  /// shape.num_elements() * DataTypeSize(type)
  ///
  /// must equal
  ///
  /// other.num_elements() * DataTypeSize(other.dtype())
  ///
  /// In addition, this function requires:
  ///   * DataTypeSize(other.dtype()) != 0
  ///   * DataTypeSize(type) != 0
  ///
  /// If any of the requirements are not met, errors::InvalidArgument is
  /// returned.
  Status BitcastFrom(const Tensor& other, DataType dtype,
                     const TensorShape& shape);

  /// Like BitcastFrom, but CHECK fails if any preconditions are not met.
  ///
  /// Deprecated. Use BitcastFrom instead and check the returned Status.
  void UnsafeCopyFromInternal(const Tensor& other, DataType dtype,
                              const TensorShape& shape) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_46(mht_46_v, 972, "", "./tensorflow/core/framework/tensor.h", "UnsafeCopyFromInternal");

    TF_CHECK_OK(BitcastFrom(other, dtype, shape));
  }

  // Returns true if the refcount on buf_ and any possible underlying root
  // buffer is one.
  bool RefCountIsOne() const;

  // Returns the type of the underlying memory.
  AllocatorMemoryType GetMemoryType() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_47(mht_47_v, 984, "", "./tensorflow/core/framework/tensor.h", "GetMemoryType");
 return buf_->GetMemoryType(); }

 private:
  void CheckType(DataType expected_dtype) const;
  void CheckTypeAndIsAligned(DataType expected_dtype) const;
  void CheckIsAlignedAndSingleElement() const;
  void set_dtype(DataType t) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_48(mht_48_v, 993, "", "./tensorflow/core/framework/tensor.h", "set_dtype");
 shape_.set_data_type(t); }

  // TensorShape's InlineVector.
  static gtl::InlinedVector<int64_t, 4> ComputeFlatInnerDims(
      gtl::ArraySlice<int64_t> orig, int64_t num_out_dims);
  static gtl::InlinedVector<int64_t, 4> ComputeFlatOuterDims(
      gtl::ArraySlice<int64_t> orig, int64_t num_out_dims);

  TensorShape shape_;
  TensorBuffer* buf_;

  friend class DMAHelper;             // For access to buf_.
  friend class TensorCApi;            // For access to buf_.
  friend class TensorCord;            // For access to buf_.
  friend class TensorReference;       // For access to buf_.
  friend class VariableOp;            // For access to set_shape.
  friend class AutoReloadVariableOp;  // For access to set_shape.
  friend class TensorTestHelper;      // For access to set_shape.
  friend class CastOpBase;            // For access to set_dtype.
  friend class ScopedAllocator;       // For access to buf_.
  friend Status batch_util::CopyElementToSlice(
      Tensor element, Tensor* parent,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::CopySliceToElement(
      const Tensor& parent, Tensor* element,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::MaybeMoveSliceToElement(
      Tensor* parent, Tensor* element,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::CopyContiguousSlices(
      const Tensor& src, int64_t src_offset, int64_t dst_offset,
      int64_t num_slices,
      Tensor* dst);  // For access to base<T>().

  bool CanUseDMA() const;

  // Only needed by variable op to set the shape of an uninitialized
  // Tensor.
  // TODO: Remove this when we have a better story for detecting
  // uninitialized tensors.
  void set_shape(const TensorShape& shape) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_49(mht_49_v, 1036, "", "./tensorflow/core/framework/tensor.h", "set_shape");

    DataType dt = dtype();
    shape_ = shape;
    set_dtype(dt);
  }

  inline void CopyFromInternal(const Tensor& other, const TensorShape& shape) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_50(mht_50_v, 1045, "", "./tensorflow/core/framework/tensor.h", "CopyFromInternal");

    DCHECK_EQ(shape.num_elements(), other.NumElements());
    // Data type will be overwritten if this == &other, since dtype is part of
    // shape.
    DataType other_dtype = other.dtype();
    shape_ = shape;
    set_dtype(other_dtype);
    if (buf_ != other.buf_) {
      if (buf_) buf_->Unref();
      buf_ = other.buf_;
      if (buf_) buf_->Ref();
    }
  }

  template <typename T>
  T* base() const;

  template <size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64_t> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;

  template <typename T, size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64_t> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;
};

// Implementation details

// START_SKIP_DOXYGEN

template <typename T>
T* Tensor::base() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_51(mht_51_v, 1081, "", "./tensorflow/core/framework/tensor.h", "Tensor::base");

  return buf_ == nullptr ? nullptr : buf_->base<T>();
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_52(mht_52_v, 1089, "", "./tensorflow/core/framework/tensor.h", "Tensor::tensor");

  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::tensor() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_53(mht_53_v, 1099, "", "./tensorflow/core/framework/tensor.h", "Tensor::tensor");

  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_tensor() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_54(mht_54_v, 1109, "", "./tensorflow/core/framework/tensor.h", "Tensor::bit_casted_tensor");

  CHECK(IsAligned());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_tensor() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_55(mht_55_v, 1119, "", "./tensorflow/core/framework/tensor.h", "Tensor::bit_casted_tensor");

  CHECK(IsAligned());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::reinterpret_last_dimension() {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_56(mht_56_v, 1129, "", "./tensorflow/core/framework/tensor.h", "Tensor::reinterpret_last_dimension");

  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  CHECK(IsAligned());
  CHECK_EQ(NDIMS, dims() - 1);
  CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::reinterpret_last_dimension()
    const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_57(mht_57_v, 1148, "", "./tensorflow/core/framework/tensor.h", "Tensor::reinterpret_last_dimension");

  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  CHECK(IsAligned());
  CHECK_EQ(NDIMS, dims() - 1);
  CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(), dims);
}

template <size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64_t> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_58(mht_58_v, 1168, "", "./tensorflow/core/framework/tensor.h", "Tensor::FillDimsAndValidateCompatibleShape");

  CHECK_EQ(NDIMS, new_sizes.size());
  int64_t new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
}

template <typename T, size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64_t> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  CHECK_EQ(NDIMS, new_sizes.size());
  int64_t new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  const int element_size = DataTypeSize(BaseType(dtype()));
  if (element_size > 0) {
    CHECK_EQ(new_num_elements * sizeof(T), NumElements() * element_size);
  } else {
    // DataTypeSize() returns 0 for some data types. In this case, assume that T
    // has the same size as the buffer type.
    // NOTE: If we can be sure that DataTypeSize() does not return 0 for all POD
    // types, then we should check DataTypeToEnum<T>::v() == dtype(). Or simply
    // check if `element_size > 0` to err when bit cast is attempted on Tensor
    // of unknown data type size.
    CHECK_EQ(new_num_elements, NumElements());
  }
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_59(mht_59_v, 1207, "", "./tensorflow/core/framework/tensor.h", "Tensor::shaped");

  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_60(mht_60_v, 1219, "", "./tensorflow/core/framework/tensor.h", "Tensor::bit_casted_shaped");

  CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_61(mht_61_v, 1231, "", "./tensorflow/core/framework/tensor.h", "Tensor::unaligned_shaped");

  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_62(mht_62_v, 1243, "", "./tensorflow/core/framework/tensor.h", "Tensor::shaped");

  CheckType(DataTypeToEnum<T>::v());
  CHECK(IsAligned()) << "ptr = " << base<void>();
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_63(mht_63_v, 1256, "", "./tensorflow/core/framework/tensor.h", "Tensor::bit_casted_shaped");

  CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedConstTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_64(mht_64_v, 1268, "", "./tensorflow/core/framework/tensor.h", "Tensor::unaligned_shaped");

  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedConstTensor(base<T>(), dims);
}

template <typename T>
typename TTypes<T>::Scalar Tensor::scalar() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_65(mht_65_v, 1279, "", "./tensorflow/core/framework/tensor.h", "Tensor::scalar");

  static_assert(
      !std::is_same<T, std::string>::value,
      "std::string is no longer a scalar type, use tensorflow::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::Scalar(base<T>());
}

template <typename T>
typename TTypes<T>::ConstScalar Tensor::scalar() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_66(mht_66_v, 1291, "", "./tensorflow/core/framework/tensor.h", "Tensor::scalar");

  static_assert(
      !std::is_same<T, std::string>::value,
      "std::string is no longer a scalar type, use tensorflow::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::ConstScalar(base<T>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_dims() {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_67(mht_67_v, 1303, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_inner_dims");

  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_outer_dims() {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_68(mht_68_v, 1311, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_outer_dims");

  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_outer_dims(int64_t begin) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_69(mht_69_v, 1319, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_inner_outer_dims");

  gtl::InlinedVector<int64_t, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_dims() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_70(mht_70_v, 1329, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_inner_dims");

  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_outer_dims() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_71(mht_71_v, 1337, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_outer_dims");

  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_outer_dims(
    int64_t begin) const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_72(mht_72_v, 1346, "", "./tensorflow/core/framework/tensor.h", "Tensor::flat_inner_outer_dims");

  gtl::InlinedVector<int64_t, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}

inline Tensor::Tensor(const Tensor& other)
    : shape_(other.shape()), buf_(other.buf_) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_73(mht_73_v, 1356, "", "./tensorflow/core/framework/tensor.h", "Tensor::Tensor");

  if (buf_) buf_->Ref();
}

inline Tensor::Tensor(Tensor&& other)
    : shape_(std::move(other.shape_)), buf_(other.buf_) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_74(mht_74_v, 1364, "", "./tensorflow/core/framework/tensor.h", "Tensor::Tensor");

  other.buf_ = nullptr;
}

class Tensor::HostScalarTensorBufferBase : public TensorBuffer {
 public:
  using TensorBuffer::TensorBuffer;
  bool GetAllocatedBytes(size_t* out_bytes) const final;
  void FillAllocationDescription(AllocationDescription* proto) const final;
};

// A packed representation for a single scalar value of type `T`, and a
// `TensorBuffer` implementation that describes (and manages the lifetime of)
// that value.
template <typename T>
struct Tensor::ValueAndTensorBuffer {
  class HostScalarTensorBuffer : public Tensor::HostScalarTensorBufferBase {
   public:
    explicit HostScalarTensorBuffer(void* data)
        : HostScalarTensorBufferBase(data) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_75(mht_75_v, 1386, "", "./tensorflow/core/framework/tensor.h", "HostScalarTensorBuffer");
}
    size_t size() const final {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_76(mht_76_v, 1390, "", "./tensorflow/core/framework/tensor.h", "size");
 return sizeof(T); }
    TensorBuffer* root_buffer() final {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_77(mht_77_v, 1394, "", "./tensorflow/core/framework/tensor.h", "root_buffer");
 return this; }

    // Override `operator delete` so that calling `delete this` in
    // `core::Refcounted::Unref()` for an object of this type will free
    // the enclosing `ValueAndTensorBuffer` for the tensor buffer.
    //
    // NOTE(mrry): The definition of this method must be outside the class
    // definition in order to satisfy some compilers.
    static void operator delete(void* ptr);

    static void operator delete(void*, void*) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_78(mht_78_v, 1407, "", "./tensorflow/core/framework/tensor.h", "delete");

      // Some compilers require an overridden class-specific deallocation
      // function, which will be called if placement `new` throws an
      // exception.
    }

   private:
    ~HostScalarTensorBuffer() override {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_79(mht_79_v, 1417, "", "./tensorflow/core/framework/tensor.h", "~HostScalarTensorBuffer");
 static_cast<T*>(data())->~T(); }
  };

  T value;
  HostScalarTensorBuffer tensor_buffer;
};

/* static */
template <typename T>
void Tensor::ValueAndTensorBuffer<T>::HostScalarTensorBuffer::operator delete(
    void* ptr) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_80(mht_80_v, 1430, "", "./tensorflow/core/framework/tensor.h", "delete");

  // Use a dummy object to compute to offset of
  // `ValueAndTensorBuffer::tensor_buffer`, because `offsetof()` is not
  // necessarily defined on this non-POD type (until C++17).
  //
  // NOTE(mrry): Using `sizeof(Tensor::ValueAndTensorBuffer<T>)` here requires
  // us to define this method outside the class definition, so that it is not
  // considered an incomplete type.
  typename std::aligned_storage<sizeof(Tensor::ValueAndTensorBuffer<T>),
                                alignof(Tensor::ValueAndTensorBuffer<T>)>::type
      dummy_storage_;
  Tensor::ValueAndTensorBuffer<T>* dummy_object =
      reinterpret_cast<Tensor::ValueAndTensorBuffer<T>*>(&dummy_storage_);
  intptr_t offset = reinterpret_cast<intptr_t>(&dummy_object->tensor_buffer) -
                    reinterpret_cast<intptr_t>(dummy_object);

  port::AlignedFree(static_cast<char*>(ptr) - offset);
}

template <typename T>
Tensor::Tensor(T value, host_scalar_tag tag) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_81(mht_81_v, 1453, "", "./tensorflow/core/framework/tensor.h", "Tensor::Tensor");

  auto* value_and_buf = static_cast<Tensor::ValueAndTensorBuffer<T>*>(
      port::AlignedMalloc(sizeof(typename Tensor::ValueAndTensorBuffer<T>),
                          EIGEN_MAX_ALIGN_BYTES));
  new (&value_and_buf->value) T(std::move(value));
  new (&value_and_buf->tensor_buffer)
      typename Tensor::ValueAndTensorBuffer<T>::HostScalarTensorBuffer(
          value_and_buf);
  buf_ = &value_and_buf->tensor_buffer;
  set_dtype(DataTypeToEnum<T>::value);
}

inline Tensor& Tensor::operator=(Tensor&& other) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensorDTh mht_82(mht_82_v, 1468, "", "./tensorflow/core/framework/tensor.h", "=");

  // Avoid self-assignment, since we might destroy our underlying buffer.
  if (&other != this) {
    shape_ = std::move(other.shape_);
    if (buf_) buf_->Unref();
    buf_ = other.buf_;
    other.buf_ = nullptr;
  }
  return *this;
}

// END_SKIP_DOXYGEN

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
