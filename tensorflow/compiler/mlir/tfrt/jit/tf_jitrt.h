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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh() {
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


#include <string>
#include <utility>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {

// Record JitRt kernel compilation time for a given session name.
void RecordCompileTime(const std::string& model_name, const std::string& kernel,
                       absl::optional<size_t> specialization,
                       absl::Duration compile_time);

// A set of helper classes to convert results returned from the compiled
// functions (memrefs or async memrefs) to the Tensorflow Tensors that can be
// seamlessly passed to the Tensorflow fallback kernels.

// MemrefTensorBuffer wraps a memref returned from the compiled kernel into
// the Tensorflow `TensorBuffer` that can be used to construct a `Tensor`.
class MemrefTensorBuffer : public TensorBuffer {
 public:
  MemrefTensorBuffer(void* base_ptr, void* data, size_t size, bool owner)
      : TensorBuffer(data), base_ptr_(base_ptr), size_(size), owner_(owner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "MemrefTensorBuffer");
}

  ~MemrefTensorBuffer() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "~MemrefTensorBuffer");

    if (owner_) free(base_ptr_);
  }

  void FillAllocationDescription(AllocationDescription* proto) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_2(mht_2_v, 232, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "FillAllocationDescription");

    proto->set_requested_bytes(size());
    proto->set_allocator_name("tf_jitrt");
  }

  size_t size() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_3(mht_3_v, 240, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "size");
 return size_; }
  bool OwnsMemory() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_4(mht_4_v, 244, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "OwnsMemory");
 return owner_; }
  TensorBuffer* root_buffer() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_5(mht_5_v, 248, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "root_buffer");
 return this; }

 private:
  void* base_ptr_;
  size_t size_;
  bool owner_;
};

// Reuse conversion context as a kernel context for convenience, can be a
// separate allocation if needed.
struct TensorflowConversionContext
    : public tfrt::jitrt::Executable::KernelContext {
  // Keep track of compiled kernel operands to detect input to output
  // forwarding, and tensors returned multiple times.
  using TensorOrBuffer = llvm::PointerUnion<const Tensor*, TensorBuffer*>;

  TensorflowConversionContext(size_t num_operands, size_t num_results)
      : num_pending_results(num_results) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_6(mht_6_v, 268, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "TensorflowConversionContext");

    runtime_tensors.reserve(num_operands + num_results - 1);
  }

  // Ensure that the context is always moved around instead of copying.
  TensorflowConversionContext(const TensorflowConversionContext&) = delete;
  TensorflowConversionContext(TensorflowConversionContext&&) = default;

  void* forward(size_t size, size_t alignment,
                llvm::ArrayRef<unsigned> candidates) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrtDTh mht_7(mht_7_v, 280, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h", "forward");

    // TODO(ecg): Do the real buffer forwarding here.
    return nullptr;
  }

  // Memrefs that are already materialized as runtime tensors:
  //   1. Tensor operands that we got from the caller.
  //   2. Tensor buffers that we constructed for newly allocated memrefs.
  llvm::SmallDenseMap<const void*, TensorOrBuffer> runtime_tensors;

  // The number of results that are waiting for the conversion.
  size_t num_pending_results;
};

namespace internal {
// The returned memref can point into statically allocated memory that we can't
// pass to `free` (memref.global). The LLVM lowering of `memref.global` sets the
// allocated pointer to the magic value 0xDEADBEEF.
template <typename T, int rank>
inline bool IsStaticStorageDuration(StridedMemRefType<T, rank>* memref) {
  return reinterpret_cast<std::intptr_t>(memref->basePtr) == 0xDEADBEEF;
}
}  // namespace internal

// Converts StridedMemrefType to the Tensor. This struct satisfies
// ReturnStridedMemref's concept (see jitrt.h).
struct ConvertTensor {
  using ResultType = tfrt_stub::FallbackTensor;
  using ConversionContext = TensorflowConversionContext;

  template <typename T, int rank>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
    return memref->sizes;
  }

  template <typename T>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
    return {};
  }

  template <typename T, int rank>
  static Tensor Convert(ConversionContext& ctx, void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    auto memref_sizes = Sizes(memref);

    // Convert TFRT data type into Tensorflow data type.
    auto dtype = tfd::GetTfDataType(tfrt::GetDType<T>());

    // Build a Tensorflow TensorShape from memref sizes.
    TensorShape shape(memref_sizes);

    // Check if returned memref already has corresponding runtime tensor.
    auto it = ctx.runtime_tensors.find(memref->data);
    ConversionContext::TensorOrBuffer runtime_tensor =
        it != ctx.runtime_tensors.end() ? it->second : nullptr;

    // Forward operand tensor to the result.
    if (auto* operand = runtime_tensor.dyn_cast<const Tensor*>()) {
      Tensor result;
      auto st = result.BitcastFrom(*operand, dtype, shape);
      assert(st.ok() && "failed to bitcast from forwarded tensor");
      (void)st;
      return result;
    }

    // The same memref returned multiple times.
    if (auto* buffer = runtime_tensor.dyn_cast<TensorBuffer*>()) {
      buffer->Ref();
      auto ptr = core::RefCountPtr<TensorBuffer>(buffer);
      return Tensor(dtype, std::move(shape), std::move(ptr));
    }

    // This is a newly allocated memref, and we need to wrap it into the runtime
    // tensor buffer to pass it back to the caller as a Tensor.
    size_t size = sizeof(T);
    for (int64_t dim : memref_sizes) size *= dim;

    // Create a TensorBuffer from the returned memref.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(memref->data, size);
    auto* buffer = new MemrefTensorBuffer(
        memref->basePtr, memref->data, size,
        /*owner=*/!internal::IsStaticStorageDuration(memref));

    // Construct a tensor from the memory buffer.
    auto ptr = core::RefCountPtr<MemrefTensorBuffer>(buffer);
    Tensor tensor(dtype, std::move(shape), std::move(ptr));

    // Keep track of memrefs already used to construct runtime tensors.
    if (--ctx.num_pending_results > 0)
      ctx.runtime_tensors.try_emplace(memref->data, buffer);

    // Incorrect alignment will lead to a segfault in the downstream Tensorflow
    // kernels, check it before returning to the runtime.
    if (internal::IsStaticStorageDuration(memref)) {
      DCHECK(tensor.IsAligned()) << "global memref is not aligned";
    } else {
      DCHECK(tensor.IsAligned()) << "allocated memref is not aligned";
    }

    return tensor;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_H_
