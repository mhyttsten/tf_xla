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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/kernels/image/adjust_hue_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class AdjustHueOpBase : public OpKernel {
 protected:
  explicit AdjustHueOpBase(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "AdjustHueOpBase");
}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* delta;
    Tensor* output;
    int64_t channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& delta = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta.shape()),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64_t channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.delta = &delta;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device, typename T>
class AdjustHueOp;

namespace internal {

// Helper function to convert a RGB color to H-and-V-range. H is in the range
// of [0, 6] instead of the normal [0, 1]
static void rgb_to_hv_range(float r, float g, float b, float* h, float* v_min,
                            float* v_max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_2(mht_2_v, 268, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "rgb_to_hv_range");

  float v_mid;
  int h_category;
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  // For the conditions, we don't care about the case where two components are
  // equal. It is okay to count it in either side in that case.
  if (r < g) {
    if (b < r) {
      // b < r < g
      *v_max = g;
      v_mid = r;
      *v_min = b;
      h_category = 1;
    } else if (b > g) {
      // r < g < b
      *v_max = b;
      v_mid = g;
      *v_min = r;
      h_category = 3;
    } else {
      // r < b < g
      *v_max = g;
      v_mid = b;
      *v_min = r;
      h_category = 2;
    }
  } else {
    // g < r
    if (b < g) {
      // b < g < r
      *v_max = r;
      v_mid = g;
      *v_min = b;
      h_category = 0;
    } else if (b > r) {
      // g < r < b
      *v_max = b;
      v_mid = r;
      *v_min = g;
      h_category = 4;
    } else {
      // g < b < r
      *v_max = r;
      v_mid = b;
      *v_min = g;
      h_category = 5;
    }
  }
  if (*v_max == *v_min) {
    *h = 0;
    return;
  }
  auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
  bool increase = ((h_category & 0x1) == 0);
  *h = h_category + (increase ? ratio : (1 - ratio));
}

// Helper function to convert from H-and-V-range to RGB.
static void hv_range_to_rgb(float h, float v_min, float v_max, float* r,
                            float* g, float* b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_3(mht_3_v, 331, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "hv_range_to_rgb");

  int h_category = static_cast<int>(h);
  float ratio = h - h_category;
  bool increase = ((h_category & 0x1) == 0);
  if (!increase) {
    ratio = 1 - ratio;
  }
  float v_mid = v_min + ratio * (v_max - v_min);
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  switch (h_category) {
    case 0:
      *r = v_max;
      *g = v_mid;
      *b = v_min;
      break;
    case 1:
      *r = v_mid;
      *g = v_max;
      *b = v_min;
      break;
    case 2:
      *r = v_min;
      *g = v_max;
      *b = v_mid;
      break;
    case 3:
      *r = v_min;
      *g = v_mid;
      *b = v_max;
      break;
    case 4:
      *r = v_mid;
      *g = v_min;
      *b = v_max;
      break;
    case 5:
    default:
      *r = v_max;
      *g = v_min;
      *b = v_mid;
  }
}
}  // namespace internal

template <>
class AdjustHueOp<CPUDevice, float> : public AdjustHueOpBase {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context)
      : AdjustHueOpBase(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_4(mht_4_v, 383, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "AdjustHueOp");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_5(mht_5_v, 389, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "DoCompute");

    const Tensor* input = options.input;
    const Tensor* delta = options.delta;
    Tensor* output = options.output;
    const int64_t channel_count = options.channel_count;
    static const int kChannelSize = 3;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float delta_h = delta->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [&input_data, &output_data, delta_h](int64_t start_channel,
                                               int64_t end_channel) {
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              float h, v_min, v_max;
              // Convert the RGB color to Hue/V-range.
              internal::rgb_to_hv_range(p[0], p[1], p[2], &h, &v_min, &v_max);
              static const int kChannelRange = 6;
              // Adjust the hue value. And adjust the hue back into the valid
              // range of [0, 6). It is faster than a fmod by avoiding
              // a float-point division since h is often very close to this
              // range.
              h += delta_h * kChannelRange;
              while (h < 0) {
                h += kChannelRange;
              }
              while (h >= kChannelRange) {
                h -= kChannelRange;
              }
              // Convert the hue and v-range back into RGB.
              internal::hv_range_to_rgb(h, v_min, v_max, q, q + 1, q + 2);
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustHue").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustHueOp<CPUDevice, float>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
class AdjustHueOp<GPUDevice, T> : public AdjustHueOpBase {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context)
      : AdjustHueOpBase(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_6(mht_6_v, 444, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "AdjustHueOp");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_hue_opDTcc mht_7(mht_7_v, 450, "", "./tensorflow/core/kernels/image/adjust_hue_op.cc", "DoCompute");

    const Tensor* input = options.input;
    const Tensor* delta = options.delta;
    Tensor* output = options.output;
    const int64_t number_of_elements = input->NumElements();
    GPUDevice device = context->eigen_gpu_device();
    const auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    if (number_of_elements > 0) {
      const T* input_data = input->flat<T>().data();
      const float* delta_h = delta->flat<float>().data();
      T* const output_data = output->flat<T>().data();
      functor::AdjustHueGPU<T>()(&device, number_of_elements, input_data,
                                 delta_h, output_data);
    }
  }
};

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AdjustHue").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustHueOp<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)

#undef REGISTER_GPU

#endif

//} // namespace functor
}  // namespace tensorflow
