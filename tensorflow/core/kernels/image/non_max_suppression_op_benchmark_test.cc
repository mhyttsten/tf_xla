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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_benchmark_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_benchmark_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_benchmark_testDTcc() {
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* CombinedNonMaxSuppression(int batches, int box_num, int class_num,
                                        int q) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_benchmark_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/image/non_max_suppression_op_benchmark_test.cc", "CombinedNonMaxSuppression");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor boxes(DT_FLOAT, TensorShape({batches, box_num, q, 4}));
  boxes.flat<float>().setRandom();
  Tensor scores(DT_FLOAT, TensorShape({batches, box_num, class_num}));
  scores.flat<float>().setRandom();

  Tensor max_output_size_per_class(100);
  Tensor max_total_size(9000);
  Tensor iou_threshold(float(0.3));
  Tensor score_threshold(float(0.25));

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "CombinedNonMaxSuppression")
                  .Input(test::graph::Constant(g, boxes))
                  .Input(test::graph::Constant(g, scores))
                  .Input(test::graph::Constant(g, max_output_size_per_class))
                  .Input(test::graph::Constant(g, max_total_size))
                  .Input(test::graph::Constant(g, iou_threshold))
                  .Input(test::graph::Constant(g, score_threshold))
                  .Attr("pad_per_class", false)
                  .Attr("clip_boxes", true)
                  .Finalize(g, &ret));
  return g;
}

#define BM_CombinedNonMaxSuppressionDev(DEVICE, B, BN, CN, Q)         \
  static void BM_CombinedNMS_##DEVICE##_##B##_##BN##_##CN##_##Q(      \
      ::testing::benchmark::State& state) {                           \
    test::Benchmark(#DEVICE, CombinedNonMaxSuppression(B, BN, CN, Q), \
                    /*old_benchmark_api*/ false)                      \
        .Run(state);                                                  \
    state.SetItemsProcessed(state.iterations() * B);                  \
  }                                                                   \
  BENCHMARK(BM_CombinedNMS_##DEVICE##_##B##_##BN##_##CN##_##Q);

#define BM_Batch(BN, CN, Q)                            \
  BM_CombinedNonMaxSuppressionDev(cpu, 1, BN, CN, Q);  \
  BM_CombinedNonMaxSuppressionDev(cpu, 28, BN, CN, Q); \
  BM_CombinedNonMaxSuppressionDev(cpu, 32, BN, CN, Q); \
  BM_CombinedNonMaxSuppressionDev(cpu, 64, BN, CN, Q);

#define BN_Boxes_Number(CN, Q) \
  BM_Batch(500, CN, Q);        \
  BM_Batch(1000, CN, Q);       \
  BM_Batch(1917, CN, Q);       \
  BM_Batch(2500, CN, Q);

BN_Boxes_Number(25, 1);
BN_Boxes_Number(25, 25);
BN_Boxes_Number(90, 1);
BN_Boxes_Number(90, 90);
BN_Boxes_Number(200, 1);
BN_Boxes_Number(200, 200);

}  // namespace tensorflow
