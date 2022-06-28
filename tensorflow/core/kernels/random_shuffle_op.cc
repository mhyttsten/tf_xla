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
class MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc() {
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

// See docs in ../ops/random_ops.cc.

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

// TODO(irving): If performance is critical, generate output directly instead
// of an in-place shuffle using a pseudorandom permutation like
//
//   https://github.com/otherlab/geode/blob/master/geode/random/permute.cpp
//
// This is probably also the right thing if we want a GPU version of shuffling.

// We use our own version of std::random_shuffle to guarantee that exactly
// size - 1 samples are used.
template <class Iter, class Random>
static inline void RandomShuffle(Iter first, Iter last, Random& uniform) {
  if (first == last) return;
  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}

template <class IntT, class InT, class OutT, class Random>
static void IndexedShuffle(const int64_t size, const InT& input_mat,
                           OutT output_mat, Random& uniform) {
  std::vector<IntT> permutation(size);
  for (IntT i = 0; i < size; i++) {
    permutation[i] = i;
  }
  RandomShuffle(permutation.begin(), permutation.end(), uniform);
  for (IntT i = 0; i < size; i++) {
    output_mat.template chip<0>(i) = input_mat.template chip<0>(permutation[i]);
  }
}

template <typename T>
class RandomShuffleOp : public OpKernel {
 public:
  explicit RandomShuffleOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc mht_0(mht_0_v, 234, "", "./tensorflow/core/kernels/random_shuffle_op.cc", "RandomShuffleOp");

    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/random_shuffle_op.cc", "Compute");

    const Tensor& input = context->input(0);

    if (input.NumElements() <= 1 || input.dim_size(0) <= 1) {
      // No shuffling is required, so copy input directly to output
      context->set_output(0, input);
    } else {
      // Reserve enough random samples for shuffling
      const int64_t size = input.dim_size(0);
      const int64_t samples = size - 1;
      auto local_gen = generator_.ReserveSamples32(samples);
      random::SingleSampleAdapter<random::PhiloxRandom> single(&local_gen);
      const auto uniform = [&single](uint32 n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_shuffle_opDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/kernels/random_shuffle_op.cc", "lambda");
 return single() % n; };

      if (input.dims() == 1) {
        // For 1D data, copy and then shuffle in place
        context->set_output(0, tensor::DeepCopy(input));
        auto vec = context->mutable_output(0)->vec<T>();
        RandomShuffle(vec.data(), vec.data() + size, uniform);
      } else {
        // For >= 2D, shuffle indices and then copy across
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        const auto input_mat = input.flat_outer_dims<T>();
        auto output_mat = output->flat_outer_dims<T>();
        if (size < kint32max) {
          IndexedShuffle<int32>(size, input_mat, output_mat, uniform);
        } else {
          IndexedShuffle<int64_t>(size, input_mat, output_mat, uniform);
        }
      }
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RandomShuffle").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RandomShuffleOp<T>);
TF_CALL_ALL_TYPES(REGISTER)

}  // namespace tensorflow
