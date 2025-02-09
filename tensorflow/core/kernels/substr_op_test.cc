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
class MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc() {
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

#include <string>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Test data from the TensorFlow README.md.
const char* ascii_lines[] = {
    "**TensorFlow** is an open source software library for numerical "
    "computation using data flow graphs.",
    "The graph nodes represent mathematical operations, while the graph edges "
    "represent the multidimensional data arrays (tensors) that flow between "
    "them.",
    "This flexible architecture enables you to deploy computation to one or "
    "more CPUs or GPUs in a desktop, server, or mobile device without "
    "rewriting code.",
    "TensorFlow also includes "
    "[TensorBoard](https://www.tensorflow.org/guide/"
    "summaries_and_tensorboard), a data visualization toolkit.",
    "TensorFlow was originally developed by researchers and engineers working "
    "on the Google Brain team within Google's Machine Intelligence Research "
    "organization for the purposes of conducting machine learning and deep "
    "neural networks research.",
    "The system is general enough to be applicable in a wide variety of other "
    "domains, as well.",
    "TensorFlow provides stable Python API and C APIs as well as without API "
    "backwards compatibility guarantee like C++, Go, Java, JavaScript and "
    "Swift."};

const char* unicode_lines[] = {
    "TensorFlow\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe4\xbd\xbf\xe7\x94\xa8\xe6"
    "\x95\xb0\xe6\x8d\xae\xe6\xb5\x81\xe5\x9b\xbe\xe8\xbf\x9b\xe8\xa1\x8c\xe6"
    "\x95\xb0\xe5\x80\xbc\xe8\xae\xa1\xe7\xae\x97\xe7\x9a\x84\xe5\xbc\x80\xe6"
    "\xba\x90\xe8\xbd\xaf\xe4\xbb\xb6\xe5\xba\x93\xe3\x80\x82",
    "\xe5\x9b\xbe\xe5\xbd\xa2\xe8\x8a\x82\xe7\x82\xb9\xe8\xa1\xa8\xe7\xa4\xba"
    "\xe6\x95\xb0\xe5\xad\xa6\xe8\xbf\x90\xe7\xae\x97\xef\xbc\x8c\xe8\x80\x8c"
    "\xe5\x9b\xbe\xe5\xbd\xa2\xe8\xbe\xb9\xe7\xbc\x98\xe8\xa1\xa8\xe7\xa4\xba"
    "\xe5\x9c\xa8\xe5\xae\x83\xe4\xbb\xac\xe4\xb9\x8b\xe9\x97\xb4\xe6\xb5\x81"
    "\xe5\x8a\xa8\xe7\x9a\x84\xe5\xa4\x9a\xe7\xbb\xb4\xe6\x95\xb0\xe6\x8d\xae"
    "\xe9\x98\xb5\xe5\x88\x97\xef\xbc\x88\xe5\xbc\xa0\xe9\x87\x8f\xef\xbc\x89"
    "\xe3\x80\x82",
    "\xe8\xbf\x99\xe7\xa7\x8d\xe7\x81\xb5\xe6\xb4\xbb\xe7\x9a\x84\xe4\xbd\x93"
    "\xe7\xb3\xbb\xe7\xbb\x93\xe6\x9e\x84\xe4\xbd\xbf\xe6\x82\xa8\xe5\x8f\xaf"
    "\xe4\xbb\xa5\xe5\xb0\x86\xe8\xae\xa1\xe7\xae\x97\xe9\x83\xa8\xe7\xbd\xb2"
    "\xe5\x88\xb0\xe6\xa1\x8c\xe9\x9d\xa2\xef\xbc\x8c\xe6\x9c\x8d\xe5\x8a\xa1"
    "\xe5\x99\xa8\xe6\x88\x96\xe7\xa7\xbb\xe5\x8a\xa8\xe8\xae\xbe\xe5\xa4\x87"
    "\xe4\xb8\xad\xe7\x9a\x84\xe4\xb8\x80\xe4\xb8\xaa\xe6\x88\x96\xe5\xa4\x9a"
    "\xe4\xb8\xaa CPU\xe6\x88\x96GPU\xef\xbc\x8c\xe8\x80\x8c\xe6\x97\xa0\xe9"
    "\x9c\x80\xe9\x87\x8d\xe5\x86\x99\xe4\xbb\xa3\xe7\xa0\x81\xe3\x80\x82",
    "TensorFlow\xe8\xbf\x98\xe5\x8c\x85\xe6\x8b\xac[TensorBoard]\xef\xbc\x88"
    "https://www.tensorflow.org/guide/summaries_and_tensorboard\xef\xbc\x89\xef"
    "\xbc\x8c\xe8\xbf\x99\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe6\x95\xb0\xe6"
    "\x8d\xae\xe5\x8f\xaf\xe8\xa7\x86\xe5\x8c\x96\xe5\xb7\xa5\xe5\x85\xb7\xe5"
    "\x8c\x85\xe3\x80\x82",
    "TensorFlow\xe6\x9c\x80\xe5\x88\x9d\xe6\x98\xaf\xe7\x94\xb1\xe7\xa0\x94\xe7"
    "\xa9\xb6\xe4\xba\xba\xe5\x91\x98\xe5\x92\x8c\xe5\xb7\xa5\xe7\xa8\x8b\xe5"
    "\xb8\x88\xe5\x9c\xa8Google\xe6\x9c\xba\xe5\x99\xa8\xe6\x99\xba\xe8\x83\xbd"
    "\xe7\xa0\x94\xe7\xa9\xb6\xe7\xbb\x84\xe7\xbb\x87\xe7\x9a\x84Google Brain"
    "\xe5\x9b\xa2\xe9\x98\x9f\xe5\xbc\x80\xe5\x8f\x91\xe7\x9a\x84\xef\xbc\x8c"
    "\xe7\x9b\xae\xe7\x9a\x84\xe6\x98\xaf\xe8\xbf\x9b\xe8\xa1\x8c\xe6\x9c\xba"
    "\xe5\x99\xa8\xe5\xad\xa6\xe4\xb9\xa0\xe5\x92\x8c\xe6\xb7\xb1\xe5\xba\xa6"
    "\xe7\xa5\x9e\xe7\xbb\x8f\xe7\xbd\x91\xe7\xbb\x9c\xe7\xa0\x94\xe7\xa9\xb6"
    "\xe3\x80\x82",
    "\xe8\xaf\xa5\xe7\xb3\xbb\xe7\xbb\x9f\xe8\xb6\xb3\xe4\xbb\xa5\xe9\x80\x82"
    "\xe7\x94\xa8\xe4\xba\x8e\xe5\x90\x84\xe7\xa7\x8d\xe5\x85\xb6\xe4\xbb\x96"
    "\xe9\xa2\x86\xe5\x9f\x9f\xe4\xb9\x9f\xe6\x98\xaf\xe5\xa6\x82\xe6\xad\xa4"
    "\xe3\x80\x82",
    "TensorFlow\xe6\x8f\x90\xe4\xbe\x9b\xe7\xa8\xb3\xe5\xae\x9a\xe7\x9a\x84"
    "Python API\xe5\x92\x8c C API\xef\xbc\x8c\xe4\xbb\xa5\xe5\x8f\x8a\xe6\xb2"
    "\xa1\xe6\x9c\x89 API\xe5\x90\x91\xe5\x90\x8e\xe5\x85\xbc\xe5\xae\xb9\xe6"
    "\x80\xa7\xe4\xbf\x9d\xe8\xaf\x81\xef\xbc\x8c\xe5\xa6\x82 C ++\xef\xbc\x8c"
    "Go\xef\xbc\x8cJava\xef\xbc\x8cJavaScript\xe5\x92\x8cSwift\xe3\x80\x82",
};

const char* const kByteUnit = "BYTE";
const char* const kUTF8Unit = "UTF8_CHAR";

Tensor GetTestTensor(int batch) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc mht_0(mht_0_v, 284, "", "./tensorflow/core/kernels/substr_op_test.cc", "GetTestTensor");

  const int sz = TF_ARRAYSIZE(ascii_lines);
  Tensor t(DT_STRING, {batch});
  auto s = t.flat<tstring>();
  for (int i = 0; i < batch; ++i) {
    s(i) = ascii_lines[i % sz];
  }
  return t;
}

Tensor GetTestUTF8Tensor(int batch) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc mht_1(mht_1_v, 297, "", "./tensorflow/core/kernels/substr_op_test.cc", "GetTestUTF8Tensor");

  const int sz = TF_ARRAYSIZE(unicode_lines);
  Tensor t(DT_STRING, {batch});
  auto s = t.flat<tstring>();
  for (int i = 0; i < batch; ++i) {
    s(i) = unicode_lines[i % sz];
  }
  return t;
}

Graph* SetupSubstrGraph(const Tensor& input, const int32_t pos,
                        const int32_t len, const char* const unit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc mht_2(mht_2_v, 311, "", "./tensorflow/core/kernels/substr_op_test.cc", "SetupSubstrGraph");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor position(DT_INT32, TensorShape({}));
  position.flat<int32>().setConstant(pos);
  Tensor length(DT_INT32, TensorShape({}));
  length.flat<int32>().setConstant(len);

  TF_CHECK_OK(NodeBuilder("substr_op", "Substr")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, position))
                  .Input(test::graph::Constant(g, length))
                  .Attr("unit", unit)
                  .Finalize(g, nullptr /* node */));
  return g;
}

static void BM_SubstrByte(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc mht_3(mht_3_v, 330, "", "./tensorflow/core/kernels/substr_op_test.cc", "BM_SubstrByte");

  const int batch_size = state.range(0);

  Tensor input = GetTestTensor(batch_size);
  Graph* g = SetupSubstrGraph(input, 3, 30, kByteUnit);
  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetItemsProcessed(state.iterations());
}

static void BM_SubstrUTF8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_op_testDTcc mht_4(mht_4_v, 342, "", "./tensorflow/core/kernels/substr_op_test.cc", "BM_SubstrUTF8");

  const int batch_size = state.range(0);

  Tensor input = GetTestUTF8Tensor(batch_size);
  Graph* g = SetupSubstrGraph(input, 3, 30, kUTF8Unit);
  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_SubstrByte)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);
BENCHMARK(BM_SubstrUTF8)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

}  // end namespace tensorflow
