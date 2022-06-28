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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc() {
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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/stateless_random_gamma_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/work_sharder.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {

namespace {

// Each attempt to generate a new draw from the Gamma distribution is 95+%
// successful, and requires 1-2 normal + 1 uniform sample.
static constexpr int kReservedSamplesPerOutput = 256;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

};  // namespace

namespace functor {

template <typename T>
struct StatelessRandomGammaFunctor<CPUDevice, T> {
  static Status Fill(OpKernelContext* ctx, const T* alpha_flat,
                     int64_t num_samples, int64_t num_alphas,
                     int64_t samples_per_alpha,
                     const random::PhiloxRandom& random, T* samples_flat) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/kernels/stateless_random_gamma_op.cc", "Fill");

    typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
    typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;

    // We partition work first across alphas then across samples-per-alpha to
    // avoid a couple flops which can be done on a per-alpha basis.

    auto DoWork = [samples_per_alpha, num_alphas, &random, samples_flat,
                   alpha_flat](int64_t start_output, int64_t limit_output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/stateless_random_gamma_op.cc", "lambda");

      // Capturing "random" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "random" by reference and explicitly do a copy assignment.

      using Eigen::numext::exp;
      using Eigen::numext::log;
      using Eigen::numext::log1p;
      using Eigen::numext::pow;

      Normal normal;
      Uniform uniform;

      RandomSampleBuffer<Normal> normal_buffer(&normal);
      RandomSampleBuffer<Uniform> uniform_buffer(&uniform);

      for (int64_t output_idx = start_output; output_idx < limit_output;
           /* output_idx incremented within inner loop below */) {
        int64_t alpha_idx = output_idx / samples_per_alpha;

        // Instead of +alpha_idx for each sample, we offset the pointer once.
        T* const samples_alpha_offset = samples_flat + alpha_idx;

        // Several calculations can be done on a per-alpha basis.
        const double alpha = static_cast<double>(alpha_flat[alpha_idx]);

        DISABLE_FLOAT_EQUALITY_WARNING
        if (alpha == 1.0) {
          ENABLE_FLOAT_EQUALITY_WARNING
          // Sample from an exponential distribution.
          for (int64_t sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // As we want data stable regardless of sharding, we skip on a
            // per-sample basis.
            random::PhiloxRandom gen = random;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            double u = uniform(&gen)[Uniform::kResultElementCount - 1];
            const double res = -log1p(-u);
            samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
          }       // for (sample_idx)
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

          // Compute the rest of the samples for the current alpha value.
          for (int64_t sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // Since each sample may use a variable number of normal/uniform
            // samples, and we want data stable regardless of sharding, we skip
            // on a per-sample basis.
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
                samples_alpha_offset[sample_idx * num_alphas] =
                    static_cast<T>(res);
                break;
              }
            }  // while: true
          }    // for: sample_idx
        }      // if (alpha == 1.0)
      }        // for: output_idx
    };         // DoWork

    // Two calls to log only occur for ~10% of samples reaching the log line.
    //   2 x 100 (64-bit cycles per log) x 0.10 = ~20.
    // Other ops: sqrt, +, *, /, %... something like 15 of these, at 3-6 cycles
    // each = ~60.
    // All of this /0.95 (expected value of geometric distribution is 1/p) due
    // to the rejection possibility = ~85.
    static const int kElementCost = 85 + 2 * Normal::kElementCost +
                                    Uniform::kElementCost +
                                    3 * random::PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_samples,
          kElementCost, DoWork);
    return Status::OK();
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
class StatelessRandomGammaOp : public OpKernel {
 public:
  explicit StatelessRandomGammaOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc mht_2(mht_2_v, 373, "", "./tensorflow/core/kernels/stateless_random_gamma_op.cc", "StatelessRandomGammaOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc mht_3(mht_3_v, 378, "", "./tensorflow/core/kernels/stateless_random_gamma_op.cc", "Compute");

    // Sanitize input
    const Tensor& shape_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    TensorShape shape;
    OP_REQUIRES_OK(context, tensor::MakeShape(shape_t, &shape));
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(context, GenerateKey(seed_t, &key, &counter));

    // Fill in the random numbers
    Fill(context, random::PhiloxRandom(counter, key), output);
  }

 private:
  void Fill(OpKernelContext* ctx, random::PhiloxRandom random, Tensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_gamma_opDTcc mht_4(mht_4_v, 405, "", "./tensorflow/core/kernels/stateless_random_gamma_op.cc", "Fill");

    const Tensor& alpha_t = ctx->input(2);

    TensorShape samples_shape = output->shape();
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, alpha_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    const int64_t num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));

    const int64_t num_samples = samples_shape.num_elements();
    const int64_t samples_per_alpha = num_samples / num_alphas;
    const auto alpha_flat = alpha_t.flat<T>().data();
    auto samples_flat = output->flat<T>().data();

    OP_REQUIRES_OK(ctx, functor::StatelessRandomGammaFunctor<Device, T>::Fill(
                            ctx, alpha_flat, num_samples, num_alphas,
                            samples_per_alpha, random, samples_flat));
  }

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomGammaOp);
};

// Register CPU kernels for stateless gamma op.
#define REGISTER_GAMMA_CPU(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGammaV2")      \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .HostMemory("alpha")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomGammaOp<CPUDevice, TYPE>)

TF_CALL_half(REGISTER_GAMMA_CPU);
TF_CALL_bfloat16(REGISTER_GAMMA_CPU);
TF_CALL_float(REGISTER_GAMMA_CPU);
TF_CALL_double(REGISTER_GAMMA_CPU);

#undef REGISTER_GAMMA_CPU

// Register GPU kernels for stateless gamma op.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GAMMA_GPU(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGammaV2")      \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomGammaOp<GPUDevice, TYPE>)

TF_CALL_half(REGISTER_GAMMA_GPU);
TF_CALL_bfloat16(REGISTER_GAMMA_GPU);
TF_CALL_float(REGISTER_GAMMA_GPU);
TF_CALL_double(REGISTER_GAMMA_GPU);

#undef REGISTER_GAMMA_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace
}  // namespace tensorflow
