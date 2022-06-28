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
class MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Based on "Notes on generating Sobol sequences. August 2008" by Joe and Kuo.
// [1] https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "third_party/eigen3/Eigen/Core"
#include "sobol_data.h"  // from @sobol_data
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/platform_strings.h"

namespace tensorflow {

// Embed the platform strings in this binary.
TF_PLATFORM_STRINGS()

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {

// Each thread will calculate at least kMinBlockSize points in the sequence.
constexpr int kMinBlockSize = 512;

// Returns number of digits in binary representation of n.
// Example: n=13. Binary representation is 1101. NumBinaryDigits(13) -> 4.
int NumBinaryDigits(int n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/sobol_op.cc", "NumBinaryDigits");

  return static_cast<int>(std::log2(n) + 1);
}

// Returns position of rightmost zero digit in binary representation of n.
// Example: n=13. Binary representation is 1101. RightmostZeroBit(13) -> 1.
int RightmostZeroBit(int n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/sobol_op.cc", "RightmostZeroBit");

  int k = 0;
  while (n & 1) {
    n >>= 1;
    ++k;
  }
  return k;
}

// Returns an integer representation of point `i` in the Sobol sequence of
// dimension `dim` using the given direction numbers.
Eigen::VectorXi GetFirstPoint(int i, int dim,
                              const Eigen::MatrixXi& direction_numbers) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/kernels/sobol_op.cc", "GetFirstPoint");

  // Index variables used in this function, consistent with notation in [1].
  // i - point in the Sobol sequence
  // j - dimension
  // k - binary digit
  Eigen::VectorXi integer_sequence = Eigen::VectorXi::Zero(dim);
  // go/wiki/Sobol_sequence#A_fast_algorithm_for_the_construction_of_Sobol_sequences
  int gray_code = i ^ (i >> 1);
  int num_digits = NumBinaryDigits(i);
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < num_digits; ++k) {
      if ((gray_code >> k) & 1) integer_sequence(j) ^= direction_numbers(j, k);
    }
  }
  return integer_sequence;
}

// Calculates `num_results` Sobol points of dimension `dim` starting at the
// point `start_point + skip` and writes them into `output` starting at point
// `start_point`.
template <typename T>
void CalculateSobolSample(int32_t dim, int32_t num_results, int32_t skip,
                          int32_t start_point,
                          typename TTypes<T>::Flat output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/kernels/sobol_op.cc", "CalculateSobolSample");

  // Index variables used in this function, consistent with notation in [1].
  // i - point in the Sobol sequence
  // j - dimension
  // k - binary digit
  const int num_digits =
      NumBinaryDigits(skip + start_point + num_results + 1);
  Eigen::MatrixXi direction_numbers(dim, num_digits);

  // Shift things so we can use integers everywhere. Before we write to output,
  // divide by constant to convert back to floats.
  const T normalizing_constant = 1./(1 << num_digits);
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < num_digits; ++k) {
      direction_numbers(j, k) = sobol_data::kDirectionNumbers[j][k]
                                << (num_digits - k - 1);
    }
  }

  // If needed, skip ahead to the appropriate point in the sequence. Otherwise
  // we start with the first column of direction numbers.
  Eigen::VectorXi integer_sequence =
      (skip + start_point > 0)
          ? GetFirstPoint(skip + start_point + 1, dim, direction_numbers)
          : direction_numbers.col(0);

  for (int j = 0; j < dim; ++j) {
    output(start_point * dim + j) = integer_sequence(j) * normalizing_constant;
  }
  // go/wiki/Sobol_sequence#A_fast_algorithm_for_the_construction_of_Sobol_sequences
  for (int i = start_point + 1; i < num_results + start_point; ++i) {
    // The Gray code for the current point differs from the preceding one by
    // just a single bit -- the rightmost bit.
    int k = RightmostZeroBit(i + skip);
    // Update the current point from the preceding one with a single XOR
    // operation per dimension.
    for (int j = 0; j < dim; ++j) {
      integer_sequence(j) ^= direction_numbers(j, k);
      output(i * dim + j) = integer_sequence(j) * normalizing_constant;
    }
  }
}

}  // namespace

template <typename Device, typename T>
class SobolSampleOp : public OpKernel {
 public:
  explicit SobolSampleOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/kernels/sobol_op.cc", "SobolSampleOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsobol_opDTcc mht_5(mht_5_v, 320, "", "./tensorflow/core/kernels/sobol_op.cc", "Compute");

    int32_t dim = context->input(0).scalar<int32_t>()();
    int32_t num_results = context->input(1).scalar<int32_t>()();
    int32_t skip = context->input(2).scalar<int32_t>()();

    OP_REQUIRES(context, dim >= 1,
                errors::InvalidArgument("dim must be at least one"));
    OP_REQUIRES(context, dim <= sobol_data::kMaxSobolDim,
                errors::InvalidArgument("dim must be at most ",
                                        sobol_data::kMaxSobolDim));
    OP_REQUIRES(context, num_results >= 1,
                errors::InvalidArgument("num_results must be at least one"));
    OP_REQUIRES(context, skip >= 0,
                errors::InvalidArgument("skip must be non-negative"));
    OP_REQUIRES(context,
                num_results < std::numeric_limits<int32_t>::max() - skip,
                errors::InvalidArgument("num_results+skip must be less than ",
                                        std::numeric_limits<int32_t>::max()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_results, dim}), &output));
    auto output_flat = output->flat<T>();
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    int num_threads = worker_threads.num_threads;
    int block_size = std::max(
        kMinBlockSize, static_cast<int>(std::ceil(
                           static_cast<float>(num_results) / num_threads)));
    worker_threads.workers->TransformRangeConcurrently(
        block_size, num_results /* total */,
        [&dim, &skip, &output_flat](const int start, const int end) {
          CalculateSobolSample<T>(dim, end - start /* num_results */, skip,
                                  start, output_flat);
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SobolSample").Device(DEVICE_CPU).TypeConstraint<double>("dtype"),
    SobolSampleOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("SobolSample").Device(DEVICE_CPU).TypeConstraint<float>("dtype"),
    SobolSampleOp<CPUDevice, float>);

}  // namespace tensorflow
