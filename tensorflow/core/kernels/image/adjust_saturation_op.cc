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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc() {
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

#include "tensorflow/core/kernels/image/adjust_saturation_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class AdjustSaturationOpBase : public OpKernel {
 protected:
  explicit AdjustSaturationOpBase(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "AdjustSaturationOpBase");
}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* scale;
    Tensor* output;
    int64_t channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& scale = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale.shape()),
                errors::InvalidArgument("scale must be scalar: ",
                                        scale.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64_t channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.scale = &scale;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device, typename T>
class AdjustSaturationOp;

namespace internal {
static void rgb_to_hsv(float r, float g, float b, float* h, float* s,
                       float* v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "rgb_to_hsv");

  float vv = std::max(r, std::max(g, b));
  float range = vv - std::min(r, std::min(g, b));
  if (vv > 0) {
    *s = range / vv;
  } else {
    *s = 0;
  }
  float norm = 1.0f / (6.0f * range);
  float hh;
  if (r == vv) {
    hh = norm * (g - b);
  } else if (g == vv) {
    hh = norm * (b - r) + 2.0 / 6.0;
  } else {
    hh = norm * (r - g) + 4.0 / 6.0;
  }
  if (range <= 0.0) {
    hh = 0;
  }
  if (hh < 0.0) {
    hh = hh + 1;
  }
  *v = vv;
  *h = hh;
}

// Algorithm from wikipedia, https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
static void hsv_to_rgb(float h, float s, float v, float* r, float* g,
                       float* b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_3(mht_3_v, 298, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "hsv_to_rgb");

  float c = s * v;
  float m = v - c;
  float dh = h * 6;
  float rr, gg, bb;
  int h_category = static_cast<int>(dh);
  float fmodu = dh;
  while (fmodu <= 0) {
    fmodu += 2.0f;
  }
  while (fmodu >= 2.0f) {
    fmodu -= 2.0f;
  }
  float x = c * (1 - std::abs(fmodu - 1));
  switch (h_category) {
    case 0:
      rr = c;
      gg = x;
      bb = 0;
      break;
    case 1:
      rr = x;
      gg = c;
      bb = 0;
      break;
    case 2:
      rr = 0;
      gg = c;
      bb = x;
      break;
    case 3:
      rr = 0;
      gg = x;
      bb = c;
      break;
    case 4:
      rr = x;
      gg = 0;
      bb = c;
      break;
    case 5:
      rr = c;
      gg = 0;
      bb = x;
      break;
    default:
      rr = 0;
      gg = 0;
      bb = 0;
  }
  *r = rr + m;
  *g = gg + m;
  *b = bb + m;
}

}  // namespace internal

template <>
class AdjustSaturationOp<CPUDevice, float> : public AdjustSaturationOpBase {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : AdjustSaturationOpBase(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_4(mht_4_v, 362, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "AdjustSaturationOp");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_5(mht_5_v, 368, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "DoCompute");

    const Tensor* input = options.input;
    const Tensor* scale = options.scale;
    Tensor* output = options.output;
    const int64_t channel_count = options.channel_count;
    static const int kChannelSize = 3;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float scale_h = scale->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [&input_data, &output_data, scale_h](int64_t start_channel,
                                               int64_t end_channel) {
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              float h, s, v;
              // Convert the RGB color to Hue/V-range.
              internal::rgb_to_hsv(p[0], p[1], p[2], &h, &s, &v);
              s = std::min(1.0f, std::max(0.0f, s * scale_h));
              // Convert the hue and v-range back into RGB.
              internal::hsv_to_rgb(h, s, v, q, q + 1, q + 2);
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustSaturation").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustSaturationOp<CPUDevice, float>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
class AdjustSaturationOp<GPUDevice, T> : public AdjustSaturationOpBase {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : AdjustSaturationOpBase(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_6(mht_6_v, 412, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "AdjustSaturationOp");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_saturation_opDTcc mht_7(mht_7_v, 418, "", "./tensorflow/core/kernels/image/adjust_saturation_op.cc", "DoCompute");

    const Tensor* input = options.input;
    const Tensor* scale = options.scale;
    Tensor* output = options.output;
    const int64_t number_of_elements = input->NumElements();
    GPUDevice device = context->eigen_gpu_device();
    const auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    if (number_of_elements > 0) {
      const T* input_data = input->flat<T>().data();
      const float* scale_data = scale->flat<float>().data();
      T* const output_data = output->flat<T>().data();
      functor::AdjustSaturationGPU<T>()(&device, number_of_elements, input_data,
                                        scale_data, output_data);
    }
  }
};

#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("AdjustSaturation").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustSaturationOp<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)

#undef REGISTER_GPU

#endif

}  // namespace tensorflow
