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
class MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENTE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#define EIGEN_USE_CUSTOM_THREAD_POOL
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_benchmark.h"
#include "tensorflow/core/platform/test_benchmark.h"

#define CREATE_THREAD_POOL(threads) \
  Eigen::ThreadPool tp(threads);    \
  Eigen::ThreadPoolDevice device(&tp, threads)

// -------------------------------------------------------------------------- //
// Spatial Convolutions                                                       //
// -------------------------------------------------------------------------- //

void SpatialConvolution(::testing::benchmark::State& state, int num_threads,
                        /* Input dimensions: */
                        int input_batches, int input_height, int input_width,
                        int input_depth,
                        /* Filter (kernel) dimensions: */
                        int filter_count, int filter_height, int filter_width) {
  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      SpatialConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(input_batches, input_height,
                                            input_width, input_depth);
  typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
                                             input_depth, filter_count);

  benchmark.SpatialConvolution(input_dims, filter_dims);

  auto num_computed_elements =
      (input_dims.TotalSize() / input_depth) * filter_count;
  auto flops =
      num_computed_elements * (input_depth * filter_height * filter_width);
  state.SetItemsProcessed(flops * state.iterations());
}

void SpatialConvolutionBackwardInput(::testing::benchmark::State& state,
                                     int num_threads,
                                     /* Input dimensions: */
                                     int input_batches, int input_height,
                                     int input_width, int input_depth,
                                     /* Filter (kernel) dimensions: */
                                     int filter_count, int filter_height,
                                     int filter_width) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc mht_0(mht_0_v, 232, "", "./tensorflow/core/kernels/eigen_benchmark_cpu_test.cc", "SpatialConvolutionBackwardInput");

  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      SpatialConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(input_batches, input_height,
                                            input_width, input_depth);
  typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
                                             input_depth, filter_count);

  benchmark.SpatialConvolutionBackwardInput(input_dims, filter_dims);

  auto num_computed_elements = input_dims.TotalSize();
  auto flops =
      num_computed_elements * (input_depth * filter_height * filter_width);
  state.SetItemsProcessed(flops * state.iterations());
}

void SpatialConvolutionBackwardKernel(::testing::benchmark::State& state,
                                      int num_threads,
                                      /* Input dimensions: */
                                      int input_batches, int input_height,
                                      int input_width, int input_depth,
                                      /* Filter (kernel) dimensions: */
                                      int filter_count, int filter_height,
                                      int filter_width) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/kernels/eigen_benchmark_cpu_test.cc", "SpatialConvolutionBackwardKernel");

  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      SpatialConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(input_batches, input_height,
                                            input_width, input_depth);
  typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
                                             input_depth, filter_count);

  benchmark.SpatialConvolutionBackwardKernel(input_dims, filter_dims);

  auto num_computed_elements = filter_dims.TotalSize();
  auto flops =
      num_computed_elements * (input_batches * input_height * input_width);
  state.SetItemsProcessed(flops * state.iterations());
}

// Macro arguments names: --------------------------------------------------- //
//   NT: num threads
//    N: batch size
//    H: height
//    W: width
//    C: channels
//   FC: filter count
//   FH: filter height
//   FW: filter width

#define BM_SPATIAL_NAME(prefix, NT, N, H, W, C, FC, FH, FW) \
  BM_##prefix##_CPU_##NT##T_in_##N##_##H##_##W##_##C##_f_##FC##_##FH##_##FW

#define BM_SpatialConvolution(NT, N, H, W, C, FC, FH, FW, LABEL)          \
  static void BM_SPATIAL_NAME(SpatialConvolution, NT, N, H, W, C, FC, FH, \
                              FW)(::testing::benchmark::State & state) {  \
    state.SetLabel(LABEL);                                                \
    SpatialConvolution(state, NT, N, H, W, C, FC, FH, FW);                \
  }                                                                       \
  BENCHMARK(BM_SPATIAL_NAME(SpatialConvolution, NT, N, H, W, C, FC, FH, FW))

#define BM_SpatialConvolutionBwdInput(NT, N, H, W, C, FC, FH, FW, LABEL)      \
  static void BM_SPATIAL_NAME(SpatialConvolutionBwdInput, NT, N, H, W, C, FC, \
                              FH, FW)(::testing::benchmark::State & state) {  \
    state.SetLabel(LABEL);                                                    \
    SpatialConvolutionBackwardInput(state, NT, N, H, W, C, FC, FH, FW);       \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_SPATIAL_NAME(SpatialConvolutionBwdInput, NT, N, H, W, C, FC, FH, FW))

#define BM_SpatialConvolutionBwdKernel(NT, N, H, W, C, FC, FH, FW, LABEL)      \
  static void BM_SPATIAL_NAME(SpatialConvolutionBwdKernel, NT, N, H, W, C, FC, \
                              FH, FW)(::testing::benchmark::State & state) {   \
    state.SetLabel(LABEL);                                                     \
    SpatialConvolutionBackwardKernel(state, NT, N, H, W, C, FC, FH, FW);       \
  }                                                                            \
  BENCHMARK(BM_SPATIAL_NAME(SpatialConvolutionBwdKernel, NT, N, H, W, C, FC,   \
                            FH, FW))

#define BM_SpatialConvolutions(N, H, W, C, FC, FH, FW, LABEL) \
  BM_SpatialConvolution(2, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolution(4, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolution(8, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolution(16, N, H, W, C, FC, FH, FW, LABEL);

#define BM_SpatialConvolutionsBwdInput(N, H, W, C, FC, FH, FW, LABEL) \
  BM_SpatialConvolutionBwdInput(2, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdInput(4, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdInput(8, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdInput(16, N, H, W, C, FC, FH, FW, LABEL);

#define BM_SpatialConvolutionsBwdKernel(N, H, W, C, FC, FH, FW, LABEL) \
  BM_SpatialConvolutionBwdKernel(2, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdKernel(4, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdKernel(8, N, H, W, C, FC, FH, FW, LABEL);    \
  BM_SpatialConvolutionBwdKernel(16, N, H, W, C, FC, FH, FW, LABEL);

// ImageNet Forward Convolutions -------------------------------------------- //

BM_SpatialConvolutions(32,          // batch size
                       56, 56, 64,  // input: height, width, depth
                       192, 3, 3,   // filter: count, height, width
                       "conv2_00");

BM_SpatialConvolutions(32, 28, 28, 96, 128, 3, 3, "conv3a_00_3x3");
BM_SpatialConvolutions(32, 28, 28, 16, 32, 5, 5, "conv3a_00_5x5");
BM_SpatialConvolutions(32, 28, 28, 128, 192, 3, 3, "conv3_00_3x3");
BM_SpatialConvolutions(32, 28, 28, 32, 96, 5, 5, "conv3_00_5x5");
BM_SpatialConvolutions(32, 14, 14, 96, 204, 3, 3, "conv4a_00_3x3");
BM_SpatialConvolutions(32, 14, 14, 16, 48, 5, 5, "conv4a_00_5x5");
BM_SpatialConvolutions(32, 14, 14, 112, 224, 3, 3, "conv4b_00_3x3");
BM_SpatialConvolutions(32, 14, 14, 24, 64, 5, 5,
                       "conv4b_00_5x5 / conv4c_00_5x5");
BM_SpatialConvolutions(32, 14, 14, 128, 256, 3, 3, "conv4c_00_3x3");
BM_SpatialConvolutions(32, 14, 14, 144, 288, 3, 3, "conv4d_00_3x3");
BM_SpatialConvolutions(32, 14, 14, 32, 64, 5, 5, "conv4d_00_5x5");
BM_SpatialConvolutions(32, 14, 14, 160, 320, 3, 3, "conv4_00_3x3");
BM_SpatialConvolutions(32, 14, 14, 32, 128, 5, 5, "conv4_00_5x5");
BM_SpatialConvolutions(32, 7, 7, 160, 320, 3, 3, "conv5a_00_3x3");
BM_SpatialConvolutions(32, 7, 7, 48, 128, 5, 5, "conv5a_00_5x5 / conv5_00_5x5");
BM_SpatialConvolutions(32, 7, 7, 192, 384, 3, 3, "conv5_00_3x3");

// Benchmarks from https://github.com/soumith/convnet-benchmarks
BM_SpatialConvolutions(128, 128, 128, 3, 96, 11, 11, "convnet-layer1");
BM_SpatialConvolutions(128, 64, 64, 64, 128, 9, 9, "convnet-layer2");
BM_SpatialConvolutions(128, 32, 32, 128, 128, 9, 9, "convnet-layer3");
BM_SpatialConvolutions(128, 16, 16, 128, 128, 7, 7, "convnet-layer4");
BM_SpatialConvolutions(128, 13, 13, 384, 384, 3, 3, "convnet-layer5");

// ImageNet BackwardInput Convolutions -------------------------------------- //

BM_SpatialConvolutionsBwdInput(32, 56, 56, 64, 192, 3, 3, "conv2_00");
BM_SpatialConvolutionsBwdInput(32, 28, 28, 96, 128, 3, 3, "conv3a_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 28, 28, 16, 32, 5, 5, "conv3a_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 28, 28, 128, 192, 3, 3, "conv3_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 28, 28, 32, 96, 5, 5, "conv3_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 96, 204, 3, 3, "conv4a_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 16, 48, 5, 5, "conv4a_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 112, 224, 3, 3, "conv4b_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 24, 64, 5, 5,
                               "conv4b_00_5x5 / conv4c_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 128, 256, 3, 3, "conv4c_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 144, 288, 3, 3, "conv4d_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 32, 64, 5, 5, "conv4d_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 160, 320, 3, 3, "conv4_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 14, 14, 32, 128, 5, 5, "conv4_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 7, 7, 160, 320, 3, 3, "conv5a_00_3x3");
BM_SpatialConvolutionsBwdInput(32, 7, 7, 48, 128, 5, 5,
                               "conv5a_00_5x5 / conv5_00_5x5");
BM_SpatialConvolutionsBwdInput(32, 7, 7, 192, 384, 3, 3, "conv5_00_3x3");

// ImageNet BackwardKernel Convolutions ------------------------------------- //

BM_SpatialConvolutionsBwdKernel(32, 56, 56, 64, 192, 3, 3, "conv2_00");
BM_SpatialConvolutionsBwdKernel(32, 28, 28, 96, 128, 3, 3, "conv3a_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 28, 28, 16, 32, 5, 5, "conv3a_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 28, 28, 128, 192, 3, 3, "conv3_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 28, 28, 32, 96, 5, 5, "conv3_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 96, 204, 3, 3, "conv4a_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 16, 48, 5, 5, "conv4a_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 112, 224, 3, 3, "conv4b_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 24, 64, 5, 5,
                                "conv4b_00_5x5 / conv4c_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 128, 256, 3, 3, "conv4c_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 144, 288, 3, 3, "conv4d_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 32, 64, 5, 5, "conv4d_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 160, 320, 3, 3, "conv4_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 14, 14, 32, 128, 5, 5, "conv4_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 7, 7, 160, 320, 3, 3, "conv5a_00_3x3");
BM_SpatialConvolutionsBwdKernel(32, 7, 7, 48, 128, 5, 5,
                                "conv5a_00_5x5 / conv5_00_5x5");
BM_SpatialConvolutionsBwdKernel(32, 7, 7, 192, 384, 3, 3, "conv5_00_3x3");

// -------------------------------------------------------------------------- //
// Cuboid Convolutions                                                        //
// -------------------------------------------------------------------------- //

void CuboidConvolution(::testing::benchmark::State& state, int num_threads,
                       /* Input dimensions: */
                       int input_batches, int input_height, int input_width,
                       int input_planes, int input_depth,
                       /* Filter (kernel) dimensions: */
                       int filter_count, int filter_height, int filter_width,
                       int filter_planes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc mht_2(mht_2_v, 428, "", "./tensorflow/core/kernels/eigen_benchmark_cpu_test.cc", "CuboidConvolution");

  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      CuboidConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(
      input_batches, input_height, input_width, input_planes, input_depth);
  typename Benchmark::Dimensions filter_dims(
      filter_height, filter_width, filter_planes, input_depth, filter_count);

  benchmark.CuboidConvolution(input_dims, filter_dims);

  auto num_computed_elements =
      (input_dims.TotalSize() / input_depth) * filter_count;
  auto flops = num_computed_elements *
               (input_depth * filter_height * filter_width * filter_planes);
  state.SetItemsProcessed(flops * state.iterations());
}

void CuboidConvolutionBackwardInput(::testing::benchmark::State& state,
                                    int num_threads,
                                    /* Input dimensions: */
                                    int input_batches, int input_height,
                                    int input_width, int input_planes,
                                    int input_depth,
                                    /* Filter (kernel) dimensions: */
                                    int filter_count, int filter_height,
                                    int filter_width, int filter_planes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc mht_3(mht_3_v, 460, "", "./tensorflow/core/kernels/eigen_benchmark_cpu_test.cc", "CuboidConvolutionBackwardInput");

  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      CuboidConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(
      input_batches, input_height, input_width, input_planes, input_depth);
  typename Benchmark::Dimensions filter_dims(
      filter_height, filter_width, filter_planes, input_depth, filter_count);

  benchmark.CuboidConvolutionBackwardInput(input_dims, filter_dims);

  auto num_computed_elements = input_dims.TotalSize();
  auto flops = num_computed_elements *
               (input_depth * filter_height * filter_width * filter_planes);
  state.SetItemsProcessed(flops * state.iterations());
}

void CuboidConvolutionBackwardKernel(::testing::benchmark::State& state,
                                     int num_threads,
                                     /* Input dimensions: */
                                     int input_batches, int input_height,
                                     int input_width, int input_planes,
                                     int input_depth,
                                     /* Filter (kernel) dimensions: */
                                     int filter_count, int filter_height,
                                     int filter_width, int filter_planes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_benchmark_cpu_testDTcc mht_4(mht_4_v, 491, "", "./tensorflow/core/kernels/eigen_benchmark_cpu_test.cc", "CuboidConvolutionBackwardKernel");

  CREATE_THREAD_POOL(num_threads);

  using Benchmark =
      CuboidConvolutionBenchmarksSuite<float, Eigen::ThreadPoolDevice>;
  auto benchmark = Benchmark(state, device);

  typename Benchmark::Dimensions input_dims(
      input_batches, input_height, input_width, input_planes, input_depth);
  typename Benchmark::Dimensions filter_dims(
      filter_height, filter_width, filter_planes, input_depth, filter_count);

  benchmark.CuboidConvolutionBackwardKernel(input_dims, filter_dims);

  auto num_computed_elements = filter_dims.TotalSize();
  auto flops = num_computed_elements *
               (input_batches * input_height * input_width * input_planes);
  state.SetItemsProcessed(flops * state.iterations());
}

// The multiple #'s in the function names + the `::testing::benchmark::State&`
// as parameters apparently confuses clang if they are not on the same line. So
// we need to turn off LINT and clang-format for this block.
//
// clang-format off
// NOLINTBEGIN

// Macro arguments names: --------------------------------------------------- //
//   NT: num threads
//    N: batch size
//    H: height
//    W: width
//    P: panes
//    C: channels
//   FC: filter count
//   FH: filter height
//   FW: filter width
//   FP: filter panes

#define BM_CONCAT(a, b) a##b

#define BM_CUBOID_NAME(p, NT, N, H, W, P, C, FC, FH, FW, FP)     \
  BM_CONCAT(BM_##p##_CPU_##NT##T_in_##N##_##H##_##W##_##P##_##C, \
            _f_##FC##_##FH##_##FW##_##FP)

#define BM_CuboidConvolution(NT, N, H, W, P, C, FC, FH, FW, FP, LABEL)         \
  static void BM_CUBOID_NAME(CuboidConvolution, NT, N, H, W, P, C, FC, FH, FW, FP)(::testing::benchmark::State & state) {                   \
    state.SetLabel(LABEL);                                    \
    CuboidConvolution(state, NT, N, H, W, P, C, FC, FH, FW, FP);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_CUBOID_NAME(CuboidConvolution, NT, N, H, W, P, C, FC, FH, FW, FP))

#define BM_CuboidConvolutionBwdInput(NT, N, H, W, P, C, FC, FH, FW, FP, LABEL) \
  static void BM_CUBOID_NAME(CuboidConvolutionBwdInput, NT, N, H, W, P, C, FC, FH, FW, FP)(::testing::benchmark::State & state) {           \
    state.SetLabel(LABEL);                                    \
    CuboidConvolutionBackwardInput(state, NT, N, H, W, P, C, FC, FH, FW, FP);  \
  }                                                                            \
  BENCHMARK(BM_CUBOID_NAME(CuboidConvolutionBwdInput, NT, N, H, W, P, C, FC,   \
                           FH, FW, FP))

#define BM_CuboidConvolutionBwdKernel(NT, N, H, W, P, C, FC, FH, FW, FP,       \
                                      LABEL)                                   \
  static void BM_CUBOID_NAME(CuboidConvolutionBwdKernel, NT, N, H, W, P, C, FC, FH, FW, FP)(::testing::benchmark::State & state) {       \
    state.SetLabel(LABEL);                                    \
    CuboidConvolutionBackwardKernel(state, NT, N, H, W, P, C, FC, FH, FW, FP); \
  }                                                                            \
  BENCHMARK(BM_CUBOID_NAME(CuboidConvolutionBwdKernel, NT, N, H, W, P, C, FC,  \
                           FH, FW, FP))

// NOLINTEND
// clang-format on

#define BM_CuboidConvolutions(N, H, W, P, C, FC, FH, FW, FP, LABEL) \
  BM_CuboidConvolution(2, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolution(4, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolution(8, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolution(16, N, H, W, P, C, FC, FH, FW, FP, LABEL);

#define BM_CuboidConvolutionsBwdInput(N, H, W, P, C, FC, FH, FW, FP, LABEL) \
  BM_CuboidConvolutionBwdInput(2, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdInput(4, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdInput(8, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdInput(16, N, H, W, P, C, FC, FH, FW, FP, LABEL);

#define BM_CuboidConvolutionsBwdKernel(N, H, W, P, C, FC, FH, FW, FP, LABEL) \
  BM_CuboidConvolutionBwdKernel(2, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdKernel(4, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdKernel(8, N, H, W, P, C, FC, FH, FW, FP, LABEL);    \
  BM_CuboidConvolutionBwdKernel(16, N, H, W, P, C, FC, FH, FW, FP, LABEL);

// Random Cuboid Convolutions ----------------------------------------------- //
// TODO(ezhulenev): find representative dims for cuboid convolutions (find
// models using Conv3D ops).

BM_CuboidConvolutions(8,              // batch size
                      25, 25, 25, 4,  // input: height, width, panes, depth
                      16, 5, 5, 5,    // filter: count, height, width, panes
                      "conv3d_depth4");
BM_CuboidConvolutions(8, 25, 25, 25, 8, 16, 5, 5, 5, "conv3d_depth8");
BM_CuboidConvolutions(2, 9, 31, 31, 64, 64, 5, 5, 5, "b2_conv3d_1");
BM_CuboidConvolutions(2, 5, 27, 27, 64, 64, 5, 5, 5, "b2_conv3d_2");

BM_CuboidConvolutionsBwdInput(8, 25, 25, 25, 4, 16, 5, 5, 5, "conv3d_depth4");
BM_CuboidConvolutionsBwdInput(8, 25, 25, 25, 8, 16, 5, 5, 5, "conv3d_depth8");
BM_CuboidConvolutionsBwdInput(2, 9, 31, 31, 64, 64, 5, 5, 5, "b2_conv3d_1");
BM_CuboidConvolutionsBwdInput(2, 5, 27, 27, 64, 64, 5, 5, 5, "b2_conv3d_2");

BM_CuboidConvolutionsBwdKernel(8, 25, 25, 25, 4, 16, 5, 5, 5, "conv3d_depth4");
BM_CuboidConvolutionsBwdKernel(8, 25, 25, 25, 8, 16, 5, 5, 5, "conv3d_depth8");
BM_CuboidConvolutionsBwdKernel(2, 9, 31, 31, 64, 64, 5, 5, 5, "b2_conv3d_1");
BM_CuboidConvolutionsBwdKernel(2, 5, 27, 27, 64, 64, 5, 5, 5, "b2_conv3d_2");
