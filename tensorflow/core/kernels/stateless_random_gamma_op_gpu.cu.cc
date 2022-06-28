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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_op_gpuDTcuDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/stateless_random_gamma_op.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {
typedef Eigen::GpuDevice GPUDevice;

// Each attempt to generate a new draw from the Gamma distribution is 95+%
// successful, and requires 1-2 normal + 1 uniform sample.
static constexpr int kReservedSamplesPerOutput = 256;

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

template <typename T>
__global__ void __launch_bounds__(1024)
    FillKernel(int64 num_samples, int64 num_alphas, int64 samples_per_alpha,
               random::PhiloxRandom random, T* samples_flat,
               const T* alpha_flat) {
  using Eigen::numext::exp;
  using Eigen::numext::log;
  using Eigen::numext::log1p;
  using Eigen::numext::pow;

  typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
  typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;

  Normal normal;
  Uniform uniform;

  RandomSampleBuffer<Normal> normal_buffer(&normal);
  RandomSampleBuffer<Uniform> uniform_buffer(&uniform);

  for (int64 output_idx : GpuGridRangeX(num_samples)) {
    int64 alpha_idx = output_idx / samples_per_alpha;
    int64 sample_idx = output_idx % samples_per_alpha;

    const double alpha = static_cast<double>(alpha_flat[alpha_idx]);

    DISABLE_FLOAT_EQUALITY_WARNING
    if (alpha == 1.0) {
      ENABLE_FLOAT_EQUALITY_WARNING
      // Sample from an exponential distribution.
      // As we want data stable regardless of sharding, we skip on a per-sample
      // basis.
      random::PhiloxRandom gen = random;
      gen.Skip(kReservedSamplesPerOutput * output_idx);
      double u = uniform(&gen)[Uniform::kResultElementCount - 1];
      const double res = -log1p(-u);
      // We use alpha_idx + sample_idx * num_alphas instead of output_idx
      // to generate numbers in the right order (CPU and GPU kernels
      // must generate numbers in the same order).
      samples_flat[alpha_idx + sample_idx * num_alphas] = static_cast<T>(res);
    } else {  // if alpha != 1.0
      // Transformation-rejection from pairs of uniform and normal random
      // variables. http://dl.acm.org/citation.cfm?id=358414
      //
      // The algorithm has an acceptance rate of ~95% for small alpha (~1),
      // and higher accept rates for higher alpha, so runtime is
      // O(NumAlphas * NumSamples * k) with k ~ 1 / 0.95.
      //
      // For alpha<1, we add one to d=alpha-1/3, and multiply the final
      // result by uniform()^(1/alpha)
      const bool alpha_less_than_one = alpha < 1.0;
      const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
      const double c = 1.0 / 3 / sqrt(d);

      // Since each sample may use a variable number of normal/uniform
      // samples, and we want data stable regardless of sharding, we skip on a
      // per-sample basis.
      random::PhiloxRandom gen = random;
      gen.Skip(kReservedSamplesPerOutput * output_idx);

      // To prevent overwriting SampleBuffer's underlying array with
      // zeros (in tensorflow::random::Array constructor), we just mark
      // the buffer as empty instead of initializing a new SampleBuffer
      // object here. The next call to operator() will fill the buffer
      // with new numbers.
      normal_buffer.Clear();
      uniform_buffer.Clear();

      // Keep trying until we don't reject a sample. In practice, we will
      // only reject ~5% at worst, for low alpha near 1.
      while (true) {
        const double x = normal_buffer(&gen);
        double v = 1 + c * x;
        if (v <= 0) {
          continue;
        }
        v = v * v * v;
        double u = uniform_buffer(&gen);
        // The first option in the if is a "squeeze" short-circuit to
        // dodge the two logs. Magic constant sourced from the paper
        // linked above. Upward of .91 of the area covered by the log
        // inequality is covered by the squeeze as well (larger coverage
        // for smaller values of alpha).
        if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
            (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
          double res = d * v;
          if (alpha_less_than_one) {
            double b = uniform_buffer(&gen);
            res *= pow(b, 1 / alpha);
          }
          // We use alpha_idx + sample_idx * num_alphas instead of output_idx
          // to generate numbers in the right order (CPU and GPU kernels
          // must generate numbers in the same order).
          samples_flat[alpha_idx + sample_idx * num_alphas] =
              static_cast<T>(res);
          break;
        }
      }  // while: true
    }    // if (alpha == 1.0)
  }      // for: output_idx
}

}  // namespace

namespace functor {

template <typename T>
struct StatelessRandomGammaFunctor<GPUDevice, T> {
  static Status Fill(OpKernelContext* ctx, const T* alpha_flat,
                     int64 num_samples, int64 num_alphas,
                     int64 samples_per_alpha,
                     const random::PhiloxRandom& random, T* samples_flat) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_op_gpuDTcuDTcc mht_0(mht_0_v, 323, "", "./tensorflow/core/kernels/stateless_random_gamma_op_gpu.cu.cc", "Fill");

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig cfg = GetGpuLaunchConfig(num_samples, d);

    TF_CHECK_OK(GpuLaunchKernel(FillKernel<T>, cfg.block_count,
                                cfg.thread_per_block, 0, d.stream(),
                                num_samples, num_alphas, samples_per_alpha,
                                random, samples_flat, alpha_flat));
    return Status::OK();
  }
};

}  // namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::StatelessRandomGammaFunctor<GPUDevice, type>;

TF_CALL_half(REGISTER_GPU_SPEC);
TF_CALL_bfloat16(REGISTER_GPU_SPEC);
TF_CALL_float(REGISTER_GPU_SPEC);
TF_CALL_double(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
