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
class MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc() {
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

#include "tensorflow/core/kernels/control_flow_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

void SwitchOp::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/control_flow_ops.cc", "SwitchOp::Compute");

  const Tensor& outputPorts = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(outputPorts.shape()),
              errors::InvalidArgument("The second input must be a scalar, "
                                      "but it has shape ",
                                      outputPorts.shape().DebugString()));

  bool pred = outputPorts.scalar<bool>()();
  int port = (pred) ? 1 : 0;
  if (context->input_is_ref(0)) {
    context->forward_ref_input_to_ref_output(0, port);
  } else {
    context->set_output(port, context->input(0));
  }
}

void SwitchNOp::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/control_flow_ops.cc", "SwitchNOp::Compute");

  const Tensor& output_index_t = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(output_index_t.shape()),
              errors::InvalidArgument("The second input must be a scalar, "
                                      "but it has shape ",
                                      output_index_t.shape().DebugString()));
  int output_index = output_index_t.scalar<int>()();
  if (output_index < 0 || output_index >= num_outputs()) {
    output_index = num_outputs() - 1;
  }
  context->set_output(output_index, context->input(0));
}

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(DEVICE_DEFAULT).HostMemory("pred"), SwitchOp);
REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(DEVICE_TPU_SYSTEM).HostMemory("pred"), SwitchOp);

REGISTER_KERNEL_BUILDER(
    Name("_SwitchN").Device(DEVICE_DEFAULT).HostMemory("output_index"),
    SwitchNOp);

#define REGISTER_CPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("output_index") \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)

#define REGISTER_CPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_GPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("output_index") \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)

#define REGISTER_GPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

TF_CALL_ALL_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_REF_SWITCH);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_SWITCH);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_SWITCH);
TF_CALL_variant(REGISTER_GPU_SWITCH);
TF_CALL_bool(REGISTER_GPU_SWITCH);
TF_CALL_bool(REGISTER_GPU_REF_SWITCH);

#undef REGISTER_CPU_SWITCH
#undef REGISTER_CPU_REF_SWITCH
#undef REGISTER_GPU_SWITCH
#undef REGISTER_GPU_REF_SWITCH

// Special GPU kernels for int32, string & resource handles. Requiring all
// inputs and outputs to be in host memory.
// TODO(b/25387198): Also enable int32 in device memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output_index") \
                              .HostMemory("outputs")      \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_REF_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL


class RefSelectOp : public OpKernel {
 public:
  explicit RefSelectOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_2(mht_2_v, 337, "", "./tensorflow/core/kernels/control_flow_ops.cc", "RefSelectOp");

    OP_REQUIRES_OK(context, context->GetAttr("N", &num_ref_inputs_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_3(mht_3_v, 344, "", "./tensorflow/core/kernels/control_flow_ops.cc", "Compute");

    const Tensor& index_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(index_tensor.shape()),
                errors::InvalidArgument("Index must be a scalar, "
                                        "but it has shape ",
                                        index_tensor.shape().DebugString()));

    int32_t index = index_tensor.scalar<int32>()();

    OP_REQUIRES(context, index >= 0 && index < num_ref_inputs_,
                errors::InvalidArgument("Index must be in the range [0, ",
                                        num_ref_inputs_, ") but got ", index));
    context->forward_ref_input_to_ref_output(index + 1, 0);
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_4(mht_4_v, 362, "", "./tensorflow/core/kernels/control_flow_ops.cc", "IsExpensive");
 return false; }

  ~RefSelectOp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_5(mht_5_v, 367, "", "./tensorflow/core/kernels/control_flow_ops.cc", "~RefSelectOp");
}

  TF_DISALLOW_COPY_AND_ASSIGN(RefSelectOp);

 private:
  int num_ref_inputs_;
};

#define REGISTER_CPU_REF_SELECT(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSelect")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("index")        \
                              .TypeConstraint<type>("T"), \
                          RefSelectOp)
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SELECT);

#undef REGISTER_CPU_REF_SWITCH

MergeOp::MergeOp(OpKernelConstruction* context) : OpKernel(context) {
  const DataType dt = context->input_type(0);
  const int num_in = context->num_inputs();
  OP_REQUIRES_OK(context, context->MatchSignature(DataTypeVector(num_in, dt),
                                                  {dt, DT_INT32}));
}

void MergeOp::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_6(mht_6_v, 395, "", "./tensorflow/core/kernels/control_flow_ops.cc", "MergeOp::Compute");

  bool input_seen = false;
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (context->has_input(i)) {
      if (input_seen) {
        context->SetStatus(
            errors::Internal("Merge can not have more than one valid input."));
        return;
      }
      input_seen = true;

      if (IsRefType(context->input_dtype(i))) {
        context->forward_ref_input_to_ref_output(i, 0);
      } else {
        context->set_output(0, context->input(i));
      }
      // The value_index output is typically used only in gradient calculations,
      // so we can avoid allocating in many inference workloads.
      if (context->output_required(1)) {
        Tensor* value_index = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                         &value_index));
        value_index->scalar<int32>()() = i;
      }
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("Merge").Device(DEVICE_CPU), MergeOp);
REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(DEVICE_DEFAULT).HostMemory("value_index"), MergeOp);
REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(DEVICE_TPU_SYSTEM).HostMemory("value_index"), MergeOp);
REGISTER_KERNEL_BUILDER(Name("RefMerge").Device(DEVICE_CPU), MergeOp);

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

#define REGISTER_GPU_REF_KERNEL(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL


// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp);                       \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL


void EnterOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_DEFAULT), EnterOp);
REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_TPU_SYSTEM), EnterOp);
REGISTER_KERNEL_BUILDER(Name("RefEnter").Device(DEVICE_CPU), EnterOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Enter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefEnter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL


// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Enter")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefEnter")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_REF_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

void ExitOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_DEFAULT), ExitOp);
REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_TPU_SYSTEM), ExitOp);
REGISTER_KERNEL_BUILDER(Name("RefExit").Device(DEVICE_CPU), ExitOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Exit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefExit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL


// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Exit")                    \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefExit")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

void NextIterationOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_DEFAULT),
                        NextIterationOp);
REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_TPU_SYSTEM),
                        NextIterationOp);
REGISTER_KERNEL_BUILDER(Name("RefNextIteration").Device(DEVICE_CPU),
                        NextIterationOp);

#define REGISTER_GPU_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("NextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      NextIterationOp);                                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("RefNextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      NextIterationOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("NextIteration")           \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp);               \
  REGISTER_KERNEL_BUILDER(Name("RefNextIteration")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL


LoopCondOp::LoopCondOp(OpKernelConstruction* context) : OpKernel(context) {}
LoopCondOp::~LoopCondOp() = default;

void LoopCondOp::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_7(mht_7_v, 650, "", "./tensorflow/core/kernels/control_flow_ops.cc", "LoopCondOp::Compute");

  CancellationManager* cm = context->cancellation_manager();
  if (cm != nullptr) {
    bool already_cancelled = cm->IsCancelled();
    OP_REQUIRES(context, !already_cancelled,
                errors::Cancelled("Loop execution was cancelled."));
  }

  context->set_output(0, context->input(0));
}

bool LoopCondOp::IsExpensive() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_8(mht_8_v, 664, "", "./tensorflow/core/kernels/control_flow_ops.cc", "LoopCondOp::IsExpensive");
 return false; }

REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_CPU), LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);

// ControlTrigger kernel
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_DEFAULT),
                        ControlTriggerOp);
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_TPU_SYSTEM),
                        ControlTriggerOp);

// When called, abort op will abort the current process. This can be used to
// abort remote PSs when needed.
class AbortOp : public OpKernel {
 public:
  explicit AbortOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_9(mht_9_v, 691, "", "./tensorflow/core/kernels/control_flow_ops.cc", "AbortOp");

    OP_REQUIRES_OK(context, context->GetAttr("error_msg", &error_msg_));
    OP_REQUIRES_OK(
        context, context->GetAttr("exit_without_error", &exit_without_error_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTcc mht_10(mht_10_v, 700, "", "./tensorflow/core/kernels/control_flow_ops.cc", "Compute");

    if (!exit_without_error_) {
      LOG(FATAL) << "Abort_op intentional failure; " << error_msg_;
    } else {
      LOG(WARNING) << "Exiting the process: " << error_msg_;
      exit(0);
    }
  }

 private:
  string error_msg_;
  bool exit_without_error_;
};

REGISTER_KERNEL_BUILDER(Name("Abort").Device(DEVICE_CPU), AbortOp);

}  // namespace tensorflow
