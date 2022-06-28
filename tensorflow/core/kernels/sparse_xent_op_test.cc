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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_op_testDTcc() {
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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/xent_op.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <class T>
static Graph* SparseXent(int batch_size, int num_classes, DataType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_op_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/kernels/sparse_xent_op_test.cc", "SparseXent");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor logits(type, TensorShape({batch_size, num_classes}));
  logits.flat<T>().setRandom();
  Tensor labels(DT_INT64, TensorShape({batch_size}));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, num_classes - 1);
  auto labels_t = labels.flat<int64_t>();
  for (int i = 0; i < batch_size; ++i) {
    labels_t(i) = dist(gen);
  }
  test::graph::Binary(g, "SparseSoftmaxCrossEntropyWithLogits",
                      test::graph::Constant(g, logits),
                      test::graph::Constant(g, labels));
  return g;
}

#define BM_SparseXentDev(BATCH, CLASS, DEVICE, C_TYPE, TF_TYPE)         \
  static void BM_SparseXent##_##BATCH##_##CLASS##_##DEVICE##_##C_TYPE(  \
      ::testing::benchmark::State& state) {                             \
    test::Benchmark(#DEVICE, SparseXent<C_TYPE>(BATCH, CLASS, TF_TYPE), \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    const int64_t tot =                                                 \
        static_cast<int64_t>(state.iterations()) * BATCH * CLASS;       \
    state.SetItemsProcessed(tot);                                       \
    state.SetBytesProcessed(tot * sizeof(C_TYPE));                      \
  }                                                                     \
  BENCHMARK(BM_SparseXent##_##BATCH##_##CLASS##_##DEVICE##_##C_TYPE);

/// The representative tests for ptb_word on GPU
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_SparseXentDev(8, 1000000, gpu, float, DT_FLOAT);

BM_SparseXentDev(16, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(16, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(16, 100000, gpu, float, DT_FLOAT);

BM_SparseXentDev(32, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(32, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(32, 100000, gpu, float, DT_FLOAT);

BM_SparseXentDev(64, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(64, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(64, 100000, gpu, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// CPU
#define BM_SparseXentDev_CPU(C_TYPE, TF_TYPE)         \
  BM_SparseXentDev(8, 1000000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(16, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(16, 100000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(32, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(32, 100000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(64, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(64, 100000, cpu, C_TYPE, TF_TYPE);

BM_SparseXentDev_CPU(float, DT_FLOAT);
BM_SparseXentDev_CPU(bfloat16, DT_BFLOAT16);

}  // end namespace tensorflow
