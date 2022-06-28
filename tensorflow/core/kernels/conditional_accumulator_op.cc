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
class MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc() {
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

#include "tensorflow/core/kernels/conditional_accumulator.h"
#include "tensorflow/core/kernels/conditional_accumulator_base_op.h"

namespace tensorflow {

/**
 * Defines a ConditionalAccumulatorOp, which constructs a ConditionalAccumulator
 * and returns its handle.
 */
template <typename Device, typename T>
class ConditionalAccumulatorOp : public ConditionalAccumulatorBaseOp {
 public:
  explicit ConditionalAccumulatorOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseOp(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "ConditionalAccumulatorOp");
}

 protected:
  Creator GetCreator() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetCreator");

    return [this](ConditionalAccumulatorBase** ret) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_2(mht_2_v, 210, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "lambda");

      ConditionalAccumulator<Device, T>* accumulator =
          new ConditionalAccumulator<Device, T>(dtype_, shape_, cinfo_.name(),
                                                reduction_type_);
      *ret = accumulator;
      return Status::OK();
    };
  }

  Status CheckSignature(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "CheckSignature");

    TF_RETURN_IF_ERROR(ctx->MatchSignature({}, {DT_STRING_REF}));
    return Status::OK();
  }

  void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) override {
    ctx->set_output_ref(0, &mu_, &accumulator_);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ConditionalAccumulatorOp);
};

#define REGISTER_KERNELS(type, dev)                           \
  REGISTER_KERNEL_BUILDER(Name("ConditionalAccumulator")      \
                              .Device(DEVICE_##dev)           \
                              .TypeConstraint<type>("dtype"), \
                          ConditionalAccumulatorOp<dev##Device, type>)

// Resource conditional accumulator
template <typename Device, typename T>
class ResourceConditionalAccumulatorOp : public ConditionalAccumulatorBaseOp {
 public:
  explicit ResourceConditionalAccumulatorOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseOp(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "ResourceConditionalAccumulatorOp");
}

 protected:
  Creator GetCreator() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetCreator");

    return [this](ConditionalAccumulatorBase** ret) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_6(mht_6_v, 259, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "lambda");

      ConditionalAccumulator<Device, T>* accumulator =
          new ConditionalAccumulator<Device, T>(dtype_, shape_, cinfo_.name(),
                                                reduction_type_);
      *ret = accumulator;
      return Status::OK();
    };
  }

  Status CheckSignature(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_7(mht_7_v, 271, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "CheckSignature");

    TF_RETURN_IF_ERROR(ctx->MatchSignature({}, {DT_RESOURCE}));
    return Status::OK();
  }

  void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) override {
    auto h = accumulator_.template flat<tstring>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<ConditionalAccumulatorBase>()));
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceConditionalAccumulatorOp);
};

#define REGISTER_RESOURCE_KERNELS(type, dev)                     \
  REGISTER_KERNEL_BUILDER(Name("ResourceConditionalAccumulator") \
                              .Device(DEVICE_##dev)              \
                              .TypeConstraint<type>("dtype"),    \
                          ResourceConditionalAccumulatorOp<dev##Device, type>)

// End of Resource conditional accumulator

#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS(type, CPU)

TF_CALL_half(REGISTER_KERNELS_CPU);
TF_CALL_float(REGISTER_KERNELS_CPU);
TF_CALL_double(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS

#define REGISTER_RESOURCE_KERNELS_CPU(type) REGISTER_RESOURCE_KERNELS(type, CPU)

TF_CALL_half(REGISTER_RESOURCE_KERNELS_CPU);
TF_CALL_float(REGISTER_RESOURCE_KERNELS_CPU);
TF_CALL_double(REGISTER_RESOURCE_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS

/**
 * Defines a AccumulateGradientOp, the execution of which adds a gradient to the
 * given ConditionalAccumulator.
 */
class AccumulatorApplyGradientOp
    : public ConditionalAccumulatorBaseApplyGradientOp {
 public:
  explicit AccumulatorApplyGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseApplyGradientOp(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "AccumulatorApplyGradientOp");
}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetExpectedInputs");

    DataTypeVector expected_inputs;
    expected_inputs = {DT_STRING_REF, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    return expected_inputs;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorApplyGradientOp);
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorApplyGradient").Device(DEVICE_CPU),
                        AccumulatorApplyGradientOp);

class ResourceAccumulatorApplyGradientOp
    : public ConditionalAccumulatorBaseApplyGradientOp {
 public:
  explicit ResourceAccumulatorApplyGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseApplyGradientOp(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_10(mht_10_v, 353, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "ResourceAccumulatorApplyGradientOp");
}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_11(mht_11_v, 359, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetExpectedInputs");

    DataTypeVector expected_inputs;
    expected_inputs = {DT_RESOURCE, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    return expected_inputs;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ResourceAccumulatorApplyGradientOp);
};

REGISTER_KERNEL_BUILDER(
    Name("ResourceAccumulatorApplyGradient").Device(DEVICE_CPU),
    ResourceAccumulatorApplyGradientOp);

/**
 * Defines a ConditionalAccumulatorBaseTakeGradientOp, the execution of which
 * returns the average gradient accumulated by the given ConditionalAccumulator.
 */
class AccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit AccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_12(mht_12_v, 385, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "AccumulatorTakeGradientOp");
}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_13(mht_13_v, 391, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetExpectedInputs");

    return {DT_STRING_REF, DT_INT32};
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorTakeGradientOp);
};
REGISTER_KERNEL_BUILDER(Name("AccumulatorTakeGradient").Device(DEVICE_CPU),
                        AccumulatorTakeGradientOp);

class ResourceAccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit ResourceAccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_14(mht_14_v, 408, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "ResourceAccumulatorTakeGradientOp");
}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_opDTcc mht_15(mht_15_v, 414, "", "./tensorflow/core/kernels/conditional_accumulator_op.cc", "GetExpectedInputs");

    return {DT_RESOURCE, DT_INT32};
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ResourceAccumulatorTakeGradientOp);
};

REGISTER_KERNEL_BUILDER(
    Name("ResourceAccumulatorTakeGradient").Device(DEVICE_CPU),
    ResourceAccumulatorTakeGradientOp);

}  // namespace tensorflow
