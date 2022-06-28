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

// Compatibility layer for calling directly into a TensorFlow kernel via TFRT,
// bypassing the existing TensorFlow runtime. This file defines:
//
//   TFRTOpKernel
//   TFRTOpKernelConstruction
//   TFRTOpKernelContext
//
// Note that these are standalone objects that do not share a base class with
// TF's corresponding OpKernel, OpKernelConstruction, and OpKernelContext types.
// There is no common base class to avoid virtual call overhead. Kernels that
// support these fallback types must be templated: see
// core/kernels/aggregate_ops.h for an example.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TFRT_OP_KERNEL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TFRT_OP_KERNEL_H_
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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh() {
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

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tfrt/common/compat/eigen/thread_pool_device.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime

namespace tfrt {
class AsyncKernelFrame;
}  // namespace tfrt

namespace tensorflow {

class Status;
class TFRTOpKernel;
class TFRTOpMeta;
class Tensor;
class TensorShape;

//////////////////////////////////////////////////////////////////////
// OpKernel interface.
//////////////////////////////////////////////////////////////////////
class TFRTOpKernelConstruction {
 public:
  explicit TFRTOpKernelConstruction(const tfrt::OpAttrsRef& attributes);

  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const;

  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_0(mht_0_v, 244, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "MatchSignature");

    // TODO(annarev): Move MatchSignatureHelper out of op_kernel.h
    // and call it here.
    return Status::OK();
  }

  const llvm::Optional<std::string>& error();

 private:
  const tfrt::OpAttrsRef& attributes_;
  // If an error occurs, the error message is stored here.
  llvm::Optional<std::string> error_;
};

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::string* value) const;

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         DataType* value) const;

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         Padding* value) const;

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::vector<int32>* value) const;

Status MissingAttributeError(StringPiece attr_name);

template <class T>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         T* value) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_1(mht_1_v, 281, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "TFRTOpKernelConstruction::GetAttr");

  bool success = attributes_.Get<T>(
      llvm::StringRef(attr_name.data(), attr_name.size()), value);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  return Status::OK();
}

// An implementation of OpKernelContext that fetches inputs from a
// tfrt::AsyncKernelFrame. Outputs and errors are stored internally.
class TFRTOpKernelContext {
 public:
  explicit TFRTOpKernelContext(
      llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs,
      int num_outputs, const TFRTOpMeta* op_meta, tfrt::HostContext* host);
  const Tensor& output(int index);
  const llvm::Optional<std::string>& error();

  // OpKernelContext interface implementation.
  bool ValidateInputsAreSameShape(TFRTOpKernel* op);
  const Tensor& input(int index);
  int num_inputs() const;
  void set_output(int index, const Tensor& tensor);
  int num_outputs() const;
  bool forward_input_to_output_with_shape(int input_index, int output_index,
                                          const TensorShape& output_shape,
                                          Tensor** output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_2(mht_2_v, 311, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "forward_input_to_output_with_shape");

    return false;
  }
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp);
  Status allocate_output(int index, const TensorShape& shape, Tensor** tensor);
  DataType expected_output_dtype(int i) const;

  template <typename EigenDeviceType>
  const EigenDeviceType& eigen_device() const;

  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

 private:
  llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs_;
  const TFRTOpMeta* op_meta_;

  // The kernel's outputs are kept here. We can't directly store outputs in the
  // AsyncKernelFrame because we must temporarily store allocate_output's Tensor
  // somewhere until the Tensor is initialized. If we stored the allocated
  // Tensor directly in the AsyncKernelFrame, the frame's output becomes
  // available and downstream kernels may use the allocated (but uninitialized)
  // Tensor.
  std::vector<Tensor> outputs_;

  // If an error occurs, the error message is stored here.
  llvm::Optional<std::string> error_;

  tfrt::compat::EigenHostContext eigen_host_context_;
};

class TFRTOpKernel {
 public:
  explicit TFRTOpKernel(TFRTOpKernelConstruction* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_3(mht_3_v, 350, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "TFRTOpKernel");
}
  virtual ~TFRTOpKernel() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_4(mht_4_v, 354, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "~TFRTOpKernel");
}
  virtual void Compute(TFRTOpKernelContext* context) = 0;
};

inline void CheckNotInComputeAsync(TFRTOpKernelConstruction*, const char*) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_5(mht_5_v, 361, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "CheckNotInComputeAsync");
}
inline void CheckNotInComputeAsync(TFRTOpKernelContext*, const char*) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_6(mht_6_v, 365, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "CheckNotInComputeAsync");
}

//////////////////////////////////////////////////////////////////////
// Forwarding op metadata.
//////////////////////////////////////////////////////////////////////

// Op metadata. For now TFRTOpMeta only stores the op's output types.
class TFRTOpMeta {
 public:
  explicit TFRTOpMeta(std::vector<DataType> output_types);
  DataType output_type(int index) const;

 private:
  std::vector<DataType> output_types_;
};

// Construct a TFRTOpMeta from .Input(), .Output(), and .Attr()
// specifications. This supports the same syntax as TF's REGISTER_OP macro, but
// this implementation only supports a subset of the full language.
//
// Currently, this only supports single-tensor outputs with fixed type.
// TODO(lauj) Support attribute outputs and compound attribute types as used by
// AddN.
class TFRTOpMetaBuilder {
 public:
  explicit TFRTOpMetaBuilder(StringPiece op_name);
  TFRTOpMetaBuilder& Output(StringPiece output_spec);
  TFRTOpMetaBuilder& Input(StringPiece input_spec);
  TFRTOpMetaBuilder& Attr(StringPiece attr_spec);

  const string& op_name() const;
  TFRTOpMeta BuildMeta() const;

 private:
  string op_name_;
  std::vector<DataType> output_types_;
};

// Map from op name to TFRTOpMeta.
class TFRTOpMetaMap {
 public:
  TFRTOpMetaMap();
  void RegisterOpMeta(const TFRTOpMetaBuilder& op_builder);

  // Returns nullptr if there is no metadata for op_name.
  const TFRTOpMeta* GetOpMeta(StringPiece op_name) const;

 private:
  llvm::StringMap<TFRTOpMeta> op_metas_;
};

extern llvm::ManagedStatic<TFRTOpMetaMap> tfrt_forwarding_op_meta_map;

// Implementation detail for REGISTER_KERNEL_FALLBACK_OP. This helps with
// evaluating the .Input()/.Output()/.Attr() clauses in the REGISTER_OP syntax
// before calling BuildMeta().
class TFRTOpRegisterer {
 public:
  TFRTOpRegisterer(  // NOLINT(google-explicit-constructor)
      const TFRTOpMetaBuilder& op_builder);
};

#define REGISTER_KERNEL_FALLBACK_OP(name) \
  REGISTER_KERNEL_FALLBACK_OP_UNIQ_HELPER(__COUNTER__, name)

#define REGISTER_KERNEL_FALLBACK_OP_UNIQ_HELPER(ctr, name) \
  REGISTER_KERNEL_FALLBACK_OP_UNIQ(ctr, name)

#define REGISTER_KERNEL_FALLBACK_OP_UNIQ(ctr, name)                         \
  static TFRTOpRegisterer global_tfrt_forwarding_op_meta_builder_##ctr##_ = \
      TFRTOpMetaBuilder(name)

//////////////////////////////////////////////////////////////////////
// Forwarding kernel registration.
//////////////////////////////////////////////////////////////////////

// Represents Kernel Fallback kernel registration information.
struct TFRTOpKernelReg {
  using CallbackT =
      std::unique_ptr<TFRTOpKernel> (*)(TFRTOpKernelConstruction*);

  explicit TFRTOpKernelReg(CallbackT callback) : callback(callback) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTh mht_7(mht_7_v, 449, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h", "TFRTOpKernelReg");
}

  // Callback that creates a kernel.
  CallbackT callback;
  // Map from attribute names to type it must match.
  // For e.g. foo: DT_FLOAT indicates that foo attribute
  // must be a tfdtype attribute with type float.
  llvm::StringMap<DataType> type_constraints;
};

class TFRTOpKernelFactories {
 public:
  TFRTOpKernelFactories();
  void RegisterFactory(StringPiece kernel_class_name,
                       TFRTOpKernelReg kernel_info);

  // Creates a kernel with the given name and passes op_kernel_construction
  // to kernel constructor.
  // Returns the constructed kernel on success.
  // In case of failure, returns a nullptr. Kernel creation can fail in one
  // of the following cases:
  //   1. Kernel with the given name is not found.
  //   2. Attributes in op_kernel_construction don't match type constraints
  //      for any of the kernels with this name.
  //      Note that we consider a constraint to be "not matched" if attribute
  //      it applies to is not in op_kernel_construction.
  std::unique_ptr<TFRTOpKernel> CreateKernel(
      StringPiece kernel_class_name,
      TFRTOpKernelConstruction* op_kernel_construction) const;

 private:
  llvm::StringMap<std::vector<TFRTOpKernelReg>> factories_;
};

// TODO(lauj) Should we move these kernel registrations to tfrt::KernelRegistry?
extern llvm::ManagedStatic<TFRTOpKernelFactories>
    tfrt_forwarding_kernel_factories;

#define REGISTER_KERNEL_FALLBACK_KERNEL(name, ...) \
  REGISTER_KERNEL_FALLBACK_KERNEL_UNIQ_HELPER(__COUNTER__, name, __VA_ARGS__)

#define REGISTER_KERNEL_FALLBACK_KERNEL_UNIQ_HELPER(ctr, name, ...) \
  REGISTER_KERNEL_FALLBACK_KERNEL_UNIQ(ctr, name, __VA_ARGS__)

#define REGISTER_KERNEL_FALLBACK_KERNEL_UNIQ(ctr, name, ...)             \
  static bool global_tfrt_forwarding_kernel_##ctr##_registered_ = []() { \
    ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(     \
        name, TFRTOpKernelReg([](TFRTOpKernelConstruction* construction) \
                                  -> std::unique_ptr<TFRTOpKernel> {     \
          return std::make_unique<__VA_ARGS__>(construction);            \
        }));                                                             \
    return true;                                                         \
  }();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TFRT_OP_KERNEL_H_
