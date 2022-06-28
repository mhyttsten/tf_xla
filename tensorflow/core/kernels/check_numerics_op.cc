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
class MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

// clang-format off
#include "tensorflow/core/platform/bfloat16.h"

#include <math.h>  // NOLINT
#include <algorithm>  // NOLINT
#include <numeric>  // NOLINT
// clang-format on

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[2]);
};

extern template struct CheckNumericsLaunch<Eigen::half>;
extern template struct CheckNumericsLaunch<float>;
extern template struct CheckNumericsLaunch<double>;

template <typename T>
struct CheckNumericsLaunchV2 {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[3]);
};

extern template struct CheckNumericsLaunchV2<Eigen::half>;
extern template struct CheckNumericsLaunchV2<float>;
extern template struct CheckNumericsLaunchV2<double>;
#endif

namespace {

const int kInfBit = 0x01;
const int kNaNBit = 0x02;
const int kNegativeInfBit = 0x04;
const int kPositiveInfBit = 0x08;

template <typename Device, typename T>
class CheckNumericsOp;

// Partial specialization for CPU
// TODO(jeff,rmlarsen): We should make this variant be an AsyncOpKernel, as
// was done for the GPU case below.
template <typename T>
class CheckNumericsOp<CPUDevice, T> : public OpKernel {
 public:
  explicit CheckNumericsOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_0(mht_0_v, 253, "", "./tensorflow/core/kernels/check_numerics_op.cc", "CheckNumericsOp");

    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/kernels/check_numerics_op.cc", "Compute");

    // pass along the input to the output
    context->set_output(0, context->input(0));

    auto in = context->input(0).flat<T>();
    const T* data = in.data();
    const int64_t size = in.size();
    // Check to see if any element of the tensor is NaN or Inf.
    int fp_props = std::accumulate(
        data, data + size, 0,
        [this](const int x, const T& y) { return checkFloatingElement(x, y); });
    if (fp_props != 0) {
      const string& status = getErrorString(fp_props);
      if (!status.empty()) {
        context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                   status, " values"));
      }
    }
  }

 protected:
  virtual int checkFloatingElement(const int x, const T& y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/kernels/check_numerics_op.cc", "checkFloatingElement");

    int result = x;
    if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
      // Do nothing: common case.
    } else {
      if (Eigen::numext::isinf(y)) {
        result |= kInfBit;
      } else if (Eigen::numext::isnan(y)) {
        result |= kNaNBit;
      }
    }
    return result;
  }

  virtual const string getErrorString(const int fp_props) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/kernels/check_numerics_op.cc", "getErrorString");

    string status;
    if ((fp_props & kInfBit) && (fp_props & kNaNBit)) {
      status = "Inf and NaN";
    } else {
      if (fp_props & kInfBit) {
        status = "Inf";
      }
      if (fp_props & kNaNBit) {
        status = "NaN";
      }
    }
    return status;
  }

 private:
  string message_;
};

template <typename Device, typename T>
class CheckNumericsV2Op;

// Partial specialization for CPU: v2.
// The v2 op differs from the v1 in that it distinguishes -inf and +inf.
template <typename T>
class CheckNumericsV2Op<CPUDevice, T> : public CheckNumericsOp<CPUDevice, T> {
 public:
  explicit CheckNumericsV2Op(OpKernelConstruction* context)
      : CheckNumericsOp<CPUDevice, T>(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_4(mht_4_v, 334, "", "./tensorflow/core/kernels/check_numerics_op.cc", "CheckNumericsV2Op");
}

 protected:
  int checkFloatingElement(const int x, const T& y) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/kernels/check_numerics_op.cc", "checkFloatingElement");

    int result = x;
    if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
      // Do nothing: common case.
    } else {
      if (Eigen::numext::isinf(y)) {
        result |= y < static_cast<T>(0.) ? kNegativeInfBit : kPositiveInfBit;
      } else if (Eigen::numext::isnan(y)) {
        result |= kNaNBit;
      }
    }
    return result;
  }

  const string getErrorString(const int fp_props) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_6(mht_6_v, 357, "", "./tensorflow/core/kernels/check_numerics_op.cc", "getErrorString");

    std::vector<string> anomalies;
    if (fp_props & kNegativeInfBit) {
      anomalies.push_back("-Inf");
    }
    if (fp_props & kPositiveInfBit) {
      anomalies.push_back("+Inf");
    }
    if (fp_props & kNaNBit) {
      anomalies.push_back("NaN");
    }
    if (anomalies.size() == 3) {
      return strings::StrCat(anomalies[0], ", ", anomalies[1], ", and ",
                             anomalies[2]);
    } else if (anomalies.size() == 2) {
      return strings::StrCat(anomalies[0], " and ", anomalies[1]);
    } else {
      return anomalies[0];
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Partial specialization for GPU
template <typename T>
class CheckNumericsOp<GPUDevice, T> : public AsyncOpKernel {
 public:
  typedef GPUDevice Device;

  explicit CheckNumericsOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_7(mht_7_v, 390, "", "./tensorflow/core/kernels/check_numerics_op.cc", "CheckNumericsOp");

    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_8(mht_8_v, 399, "", "./tensorflow/core/kernels/check_numerics_op.cc", "ComputeAsync");

    // pass along the input to the output
    context->set_output(0, context->input(0));
    if (context->input(0).NumElements() == 0) {
      done();
      return;
    }
    auto input = context->input(0).flat<T>();

    // Allocate and initialize the elements to hold the check results
    Tensor abnormal_detected;
    const int abnormal_detected_size = getAnomalyIndicatorSize();
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream != nullptr,
                      errors::Internal("No GPU stream available."), done);

    se::DeviceMemoryBase abnormal_detected_ptr(
        abnormal_detected.flat<int>().data(),
        abnormal_detected.flat<int>().size());
    stream->ThenMemset32(&abnormal_detected_ptr, 0,
                         abnormal_detected.flat<int>().size() * sizeof(int));

    // Call the GPU kernels for the numerical checks
    const Device& d = context->eigen_device<Device>();
    RunKernel(d, input.data(), input.size(),
              abnormal_detected.flat<int>().data());

    // Copy the results from device to host
    AllocatorAttributes attr;
    attr.set_on_host(true);
    attr.set_gpu_compatible(true);
    Tensor abnormal_detected_host;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DT_INT32, TensorShape({abnormal_detected_size}),
                               &abnormal_detected_host, attr),
        done);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(abnormal_detected_host.flat<int>().data(),
                         abnormal_detected_ptr,
                         abnormal_detected_size * sizeof(int))
            .ok(),
        errors::Internal("GPU memcpy from device to host failed"), done);

    // We have observed crashes on some network stacks when not holding
    // this tensor reference.
    TensorReference abnormal_detected_ref(abnormal_detected);
    auto check_cb = [this, stream, abnormal_detected_ref,
                     abnormal_detected_host, context, done]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_9(mht_9_v, 456, "", "./tensorflow/core/kernels/check_numerics_op.cc", "lambda");

#if GOOGLE_CUDA
      se::cuda::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};
#elif TENSORFLOW_USE_ROCM
      se::rocm::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};
#endif
      TTypes<const int>::Vec abnormal_detected_host_flat =
          abnormal_detected_host.flat<int>();
      abnormal_detected_ref.Unref();
      checkForAnomalies(context, abnormal_detected_host_flat);
      done();
    };
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, std::move(check_cb));
  }

 protected:
  virtual int getAnomalyIndicatorSize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_10(mht_10_v, 479, "", "./tensorflow/core/kernels/check_numerics_op.cc", "getAnomalyIndicatorSize");
 return 2; }

  virtual void RunKernel(const GPUDevice& d, const T* data, int size,
                         int* abnormal_detected) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_11(mht_11_v, 485, "", "./tensorflow/core/kernels/check_numerics_op.cc", "RunKernel");

    CheckNumericsLaunch<T>().Run(d, data, size, abnormal_detected);
  }

  virtual void checkForAnomalies(
      OpKernelContext* context,
      const TTypes<const int>::Vec& abnormality_indicators) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_12(mht_12_v, 494, "", "./tensorflow/core/kernels/check_numerics_op.cc", "checkForAnomalies");

    const int is_nan = abnormality_indicators(0);
    const int is_inf = abnormality_indicators(1);
    if (is_nan || is_inf) {
      LOG(ERROR) << "abnormal_detected_host @" << abnormality_indicators.data()
                 << " = {" << is_nan << ", " << is_inf << "} " << message_;

      string anomalies;
      if (is_nan && is_inf) {
        anomalies = "Inf and NaN";
      } else if (is_nan) {
        anomalies = "NaN";
      } else if (is_inf) {
        anomalies = "Inf";
      }
      context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                 anomalies, " values"));
    }
  }

  string message_;
};

template <typename T>
class CheckNumericsV2Op<GPUDevice, T> : public CheckNumericsOp<GPUDevice, T> {
 public:
  CheckNumericsV2Op(OpKernelConstruction* context)
      : CheckNumericsOp<GPUDevice, T>(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_13(mht_13_v, 524, "", "./tensorflow/core/kernels/check_numerics_op.cc", "CheckNumericsV2Op");
}

 protected:
  int getAnomalyIndicatorSize() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_14(mht_14_v, 530, "", "./tensorflow/core/kernels/check_numerics_op.cc", "getAnomalyIndicatorSize");
 return 3; }

  void RunKernel(const GPUDevice& d, const T* data, int size,
                 int* abnormal_detected) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_15(mht_15_v, 536, "", "./tensorflow/core/kernels/check_numerics_op.cc", "RunKernel");

    CheckNumericsLaunchV2<T>().Run(d, data, size, abnormal_detected);
  }

  void checkForAnomalies(
      OpKernelContext* context,
      const TTypes<const int>::Vec& abnormality_indicators) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheck_numerics_opDTcc mht_16(mht_16_v, 545, "", "./tensorflow/core/kernels/check_numerics_op.cc", "checkForAnomalies");

    const int is_nan = abnormality_indicators(0);
    const int is_negative_inf = abnormality_indicators(1);
    const int is_positive_inf = abnormality_indicators(2);
    if (is_negative_inf || is_positive_inf || is_nan) {
      std::vector<string> anomalies;
      if (is_negative_inf) {
        anomalies.push_back("-Inf");
      }
      if (is_positive_inf) {
        anomalies.push_back("+Inf");
      }
      if (is_nan) {
        anomalies.push_back("NaN");
      }
      string all_anomalies;
      if (anomalies.size() == 3) {
        all_anomalies = strings::StrCat(anomalies[0], ", ", anomalies[1],
                                        ", and ", anomalies[2]);
      } else if (anomalies.size() == 2) {
        all_anomalies = strings::StrCat(anomalies[0], " and ", anomalies[1]);
      } else {
        all_anomalies = anomalies[0];
      }
      context->SetStatus(errors::InvalidArgument(
          this->message_, " : Tensor had ", all_anomalies, " values"));
    }
  }

  static constexpr int abnormal_detected_size = 3;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace

#define REGISTER_CPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("CheckNumerics").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CheckNumericsOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_bfloat16(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);

#define REGISTER_V2_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("CheckNumericsV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CheckNumericsV2Op<CPUDevice, T>);
TF_CALL_half(REGISTER_V2_CPU_KERNEL);
TF_CALL_bfloat16(REGISTER_V2_CPU_KERNEL);
TF_CALL_float(REGISTER_V2_CPU_KERNEL);
TF_CALL_double(REGISTER_V2_CPU_KERNEL);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsV2Op<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsV2Op<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsV2Op<GPUDevice, double>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
