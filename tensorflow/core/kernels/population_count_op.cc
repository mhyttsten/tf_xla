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
class MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc

#define EIGEN_USE_THREADS

#include <bitset>

#include "tensorflow/core/kernels/population_count_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class PopulationCountOp : public OpKernel {
 public:
  explicit PopulationCountOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/population_count_op.cc", "PopulationCountOp");
}

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/population_count_op.cc", "Compute");

    const Tensor& input_t = c->input(0);
    Tensor* output_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, input_t.shape(), &output_t));

    auto input = input_t.flat<T>();
    auto output = output_t->flat<uint8>();

    functor::PopulationCount<Device, T> popcnt;
    popcnt(c, input, output);
  }
};

#define REGISTER_POPULATION_COUNT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PopulationCountOp<CPUDevice, type>);

TF_CALL_uint8(REGISTER_POPULATION_COUNT);
TF_CALL_int8(REGISTER_POPULATION_COUNT);
TF_CALL_uint16(REGISTER_POPULATION_COUNT);
TF_CALL_int16(REGISTER_POPULATION_COUNT);
TF_CALL_int32(REGISTER_POPULATION_COUNT);
TF_CALL_uint32(REGISTER_POPULATION_COUNT);
TF_CALL_int64(REGISTER_POPULATION_COUNT);
TF_CALL_uint64(REGISTER_POPULATION_COUNT);

#undef REGISTER_POPULATION_COUNT

namespace functor {

namespace {

template <typename T>
inline uint8 PopCnt(const T v);

#define POPCNT(T, N)                  \
  template <>                         \
  uint8 PopCnt<T>(const T v) {        \
    return std::bitset<N>(v).count(); \
  }

POPCNT(int8_t, 8);
POPCNT(uint8, 8);
POPCNT(int16_t, 16);
POPCNT(uint16, 16);
POPCNT(int32_t, 32);
POPCNT(uint32, 32);
POPCNT(int64_t, 64);
POPCNT(uint64, 64);

#undef POPCNT

}  // namespace

template <typename T>
struct PopulationCount<CPUDevice, T> {
  void operator()(OpKernelContext* c, typename TTypes<T>::ConstFlat input,
                  TTypes<uint8>::Flat output) {
    const T* input_ptr = input.data();
    uint8* output_ptr = output.data();
    auto shard = [input_ptr, output_ptr](int64_t start, int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpopulation_count_opDTcc mht_2(mht_2_v, 281, "", "./tensorflow/core/kernels/population_count_op.cc", "lambda");

      for (int64_t i = start; i < limit; ++i) {
        output_ptr[i] = PopCnt<T>(input_ptr[i]);
      }
    };
    int64_t total_shards = input.size();
    // Approximating cost of popcnt: convert T to int64
    // (std::bitset constructor) and convert int64 to uint8
    // (bitset.count() -> output).  The .count() itself is relatively cheap.
    const double total_cost = (Eigen::TensorOpCost::CastCost<T, uint8>() +
                               Eigen::TensorOpCost::CastCost<int64_t, uint8>());
    const int64_t shard_cost = (total_cost >= static_cast<double>(kint64max))
                                   ? kint64max
                                   : static_cast<int64_t>(total_cost);

    auto worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_POPULATION_COUNT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PopulationCountOp<GPUDevice, type>)

TF_CALL_uint8(REGISTER_POPULATION_COUNT);
TF_CALL_int8(REGISTER_POPULATION_COUNT);
TF_CALL_uint16(REGISTER_POPULATION_COUNT);
TF_CALL_int16(REGISTER_POPULATION_COUNT);
TF_CALL_int32(REGISTER_POPULATION_COUNT);
TF_CALL_int64(REGISTER_POPULATION_COUNT);

#undef REGISTER_POPULATION_COUNT

namespace functor {

#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  void PopulationCount<GPUDevice, T>::operator()(              \
      OpKernelContext* c, typename TTypes<T>::ConstFlat input, \
      TTypes<uint8>::Flat output);                             \
  extern template struct PopulationCount<GPUDevice, T>

TF_CALL_uint8(DECLARE_GPU_SPEC);
TF_CALL_int8(DECLARE_GPU_SPEC);
TF_CALL_uint16(DECLARE_GPU_SPEC);
TF_CALL_int16(DECLARE_GPU_SPEC);
TF_CALL_int32(DECLARE_GPU_SPEC);
TF_CALL_int64(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
