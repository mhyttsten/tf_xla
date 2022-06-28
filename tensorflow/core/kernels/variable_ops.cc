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
class MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc() {
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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// Makes a unique name for a temporary variable inside a while loop body,
// because loop can be executed in multiple iterations in parallel.
string TemporaryVariableName(const string& var_name,
                             const FrameAndIter& control_frame) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("var_name: \"" + var_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/variable_ops.cc", "TemporaryVariableName");

  if (control_frame.frame_id != kIllegalFrameId &&
      control_frame.iter_id != kIllegalIterId) {
    return strings::StrCat(var_name, "/frame:", control_frame.frame_id,
                           "/iter:", control_frame.iter_id);
  }
  return var_name;
}

}  // namespace

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
class LegacyVar : public ResourceBase {
 public:
  explicit LegacyVar(DataType dtype) : tensor_(dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/variable_ops.cc", "LegacyVar");
}
  // Not copyable or movable.
  LegacyVar(const LegacyVar&) = delete;
  LegacyVar& operator=(const LegacyVar&) = delete;

  mutex* mu() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/kernels/variable_ops.cc", "mu");
 return &mu_; }
  Tensor* tensor() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/kernels/variable_ops.cc", "tensor");
 return &tensor_; }

  string DebugString() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/variable_ops.cc", "DebugString");

    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;

  ~LegacyVar() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/kernels/variable_ops.cc", "~LegacyVar");
}
};

VariableOp::VariableOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_6(mht_6_v, 256, "", "./tensorflow/core/kernels/variable_ops.cc", "VariableOp::VariableOp");

  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  dtype_ = RemoveRefType(context->output_type(0));
  OP_REQUIRES_OK(context, cinfo_.Init(context->resource_manager(), def(),
                                      true /* use name() */));
}

void VariableOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/kernels/variable_ops.cc", "VariableOp::Compute");

  auto creator = [this](LegacyVar** var) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_8(mht_8_v, 270, "", "./tensorflow/core/kernels/variable_ops.cc", "lambda");

    *var = new LegacyVar(dtype_);
    (*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  LegacyVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<LegacyVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));
  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
}

class TemporaryVariableOp : public OpKernel {
 public:
  explicit TemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_9(mht_9_v, 295, "", "./tensorflow/core/kernels/variable_ops.cc", "TemporaryVariableOp");

    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    // Variable name defaults to op name if not specified explicitly.
    if (var_name_.empty()) var_name_ = name();
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_10(mht_10_v, 306, "", "./tensorflow/core/kernels/variable_ops.cc", "Compute");

    Status s;
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    auto unique_name = TemporaryVariableName(var_name_, context->frame_iter());
    auto* tmp_var = new TmpVar;
    OP_REQUIRES(context, tmp_var,
                errors::ResourceExhausted("Could not allocate TmpVar."));
    tmp_var->name = unique_name;
    s = context->allocate_temp(dtype_, shape_, &tmp_var->val);
    if (!s.ok()) tmp_var->Unref();
    OP_REQUIRES_OK(context, s);
    OP_REQUIRES_OK(context,
                   context->step_container()->Create(rm, unique_name, tmp_var));
    context->set_output_ref(0, &tmp_var->mu, &tmp_var->val);
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          tmp_var->val.AllocatedBytes());
    }
  }

 private:
  // Refcounted temporary variable resource.
  friend class DestroyTemporaryVariableOp;
  struct TmpVar : public ResourceBase {
    mutex mu;
    Tensor val;
    string name;
    string DebugString() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_11(mht_11_v, 337, "", "./tensorflow/core/kernels/variable_ops.cc", "DebugString");
 return name; }
    ~TmpVar() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_12(mht_12_v, 341, "", "./tensorflow/core/kernels/variable_ops.cc", "~TmpVar");
 VLOG(3) << "TmpVar " << name << " deleted"; }
  };

  TensorShape shape_;
  DataType dtype_;
  string var_name_;
};

class DestroyTemporaryVariableOp : public OpKernel {
 public:
  explicit DestroyTemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_13(mht_13_v, 355, "", "./tensorflow/core/kernels/variable_ops.cc", "DestroyTemporaryVariableOp");

    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES(context, !var_name_.empty(),
                errors::InvalidArgument("Missing var_name attribute"));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_14(mht_14_v, 366, "", "./tensorflow/core/kernels/variable_ops.cc", "Compute");

    // NOTE(pbar): All other mutators of the Tensor Ref *must* have completed
    // their execution before this DestroyTemporaryVariable op executes.
    // This is typically achieved using control dependencies.
    CHECK(IsRefType(context->input_dtype(0)));
    Tensor tmpvar = context->mutable_input(0, false);
    context->set_output(0, tmpvar);
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    auto unique_name = TemporaryVariableName(var_name_, context->frame_iter());
    OP_REQUIRES_OK(
        context, context->step_container()->Delete<TemporaryVariableOp::TmpVar>(
                     rm, unique_name));
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          -static_cast<int64_t>(tmpvar.AllocatedBytes()));
    }
  }

 private:
  string var_name_;
};

class IsVariableInitializedOp : public OpKernel {
 public:
  explicit IsVariableInitializedOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_15(mht_15_v, 395, "", "./tensorflow/core/kernels/variable_ops.cc", "IsVariableInitializedOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSvariable_opsDTcc mht_16(mht_16_v, 400, "", "./tensorflow/core/kernels/variable_ops.cc", "Compute");

    // Get a mutable input tensor of the Ref input.
    const Tensor& input_tensor = context->mutable_input(0, false);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    bool result = input_tensor.IsInitialized();
    output_tensor() = result;
  }
};

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("VariableV2").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("TemporaryVariable").Device(DEVICE_CPU),
                        TemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable").Device(DEVICE_CPU),
                        DestroyTemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(DEVICE_CPU),
                        IsVariableInitializedOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Only register 'Variable' on GPU for the subset of types also supported by
// 'Assign' (see dense_update_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("VariableV2").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                        \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype"),              \
                          TemporaryVariableOp);                            \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                 \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T"),                  \
                          DestroyTemporaryVariableOp);                     \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                    \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype")               \
                              .HostMemory("is_initialized"),               \
                          IsVariableInitializedOp);

TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_uint32(REGISTER_GPU_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Variable").Device(DEVICE_DEFAULT).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("VariableV2").Device(DEVICE_DEFAULT).TypeConstraint<type>("dtype"), \
      VariableOp);                                                             \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                            \
                              .Device(DEVICE_DEFAULT)                          \
                              .TypeConstraint<type>("dtype"),                  \
                          TemporaryVariableOp);                                \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                     \
                              .Device(DEVICE_DEFAULT)                          \
                              .TypeConstraint<type>("T"),                      \
                          DestroyTemporaryVariableOp);                         \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                        \
                              .Device(DEVICE_DEFAULT)                          \
                              .TypeConstraint<type>("dtype")                   \
                              .HostMemory("is_initialized"),                   \
                          IsVariableInitializedOp);

TF_CALL_int64(REGISTER_DEFAULT_KERNELS);
TF_CALL_uint32(REGISTER_DEFAULT_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_DEFAULT_KERNELS);
#undef REGISTER_DEFAULT_KERNELS

}  // namespace tensorflow
