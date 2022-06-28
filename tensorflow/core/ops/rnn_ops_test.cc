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
class MHTracer_DTPStensorflowPScorePSopsPSrnn_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSrnn_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSrnn_ops_testDTcc() {
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

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static string JoinedCopies(const string& s, int copies) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSrnn_ops_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/ops/rnn_ops_test.cc", "JoinedCopies");

  string res;
  for (int i = 0; i < copies; ++i) {
    strings::StrAppend(&res, i > 0 ? ";" : "", s);
  }
  return res;
}

TEST(RnnOpsTest, GRUBlockCell_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCell");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];?;?;?;?",
           "[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1]");
}

TEST(RnnOpsTest, GRUBlockCellGrad_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCellGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;?;[?];?;?;?;?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];[?,?];?;?;?;?;?;?;?",
           "in0;[d0_0,d1_1];[d0_0,d1_1];[d0_0,d2_1]");
}

TEST(RnnOpsTest, LSTMBlockCell_ShapeFn) {
  ShapeInferenceTestOp op("LSTMBlockCell");

  // Last 6 inputs don't affect shape inference.
  string input_suffix = strings::StrCat(";", JoinedCopies("?", 6));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?" + input_suffix);
  INFER_ERROR("must be rank 2", op, "?;[?]" + input_suffix);

  // Output
  INFER_OK(op, "?;?" + input_suffix, JoinedCopies("[?,?]", 7));
  INFER_OK(op, "[?,?];[?,?]" + input_suffix, JoinedCopies("[d0_0,d1_1]", 7));
}

TEST(RnnOpsTest, LSTMBlockCellGrad_ShapeFn) {
  ShapeInferenceTestOp op("LSTMBlockCellGrad");

  // Last 14 inputs don't affect shape inference.
  string input_suffix = strings::StrCat(";", JoinedCopies("?", 14));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?" + input_suffix);
  INFER_ERROR("must be rank 2", op, "?;[?]" + input_suffix);

  // Output
  INFER_OK(op, "?;?" + input_suffix, "[?,?];[?,?];[?];[?];[?]");
  INFER_OK(op, "[?,?];[?,?]" + input_suffix,
           "[d0_0,d1_1];[d0_0,?];[d1_1];[d1_1];[d1_1]");
  INFER_OK(op, "[1,2];[3,4]" + input_suffix,
           "[d0_0,d1_1];[d0_0,16];[d1_1];[d1_1];[d1_1]");
}

TEST(RnnOpsTest, BlockLSTM_ShapeFn) {
  ShapeInferenceTestOp op("BlockLSTM");

  TF_ASSERT_OK(NodeDefBuilder("test", "BlockLSTM")
                   .Input({"seq_len_max", 0, DT_INT64})
                   .Input({"x", 0, DT_FLOAT})
                   .Input({"cs_prev", 0, DT_FLOAT})
                   .Input({"h_prev", 0, DT_FLOAT})
                   .Input({"w", 0, DT_FLOAT})
                   .Input({"wci", 0, DT_FLOAT})
                   .Input({"wcf", 0, DT_FLOAT})
                   .Input({"wco", 0, DT_FLOAT})
                   .Input({"b", 0, DT_FLOAT})
                   .Finalize(&op.node_def));

  // Middle inputs don't affect shape inference.
  string infix = ";" + JoinedCopies("?", 6) + ";";

  // Rank checks.
  INFER_ERROR("must be rank 3", op, "?;[?]" + infix + "?");
  INFER_ERROR("must be rank 1", op, "?;?" + infix + "[?,?]");

  // Output
  INFER_OK(op, "?;?" + infix + "?", JoinedCopies("[?,?,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "?", JoinedCopies("[d1_0,d1_1,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "[?]", JoinedCopies("[d1_0,d1_1,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "[20]", JoinedCopies("[d1_0,d1_1,5]", 7));

  // cell_size must be divisible by 4.
  INFER_ERROR("must be evenly divisible", op, "?;?" + infix + "[11]");
}

TEST(RnnOpsTest, BlockLSTMGrad_ShapeFn) {
  ShapeInferenceTestOp op("BlockLSTMGrad");
  TF_ASSERT_OK(NodeDefBuilder("test", "BlockLSTMGrad")
                   .Input({"seq_len_max", 0, DT_INT64})
                   .Input({"x", 0, DT_FLOAT})
                   .Input({"cs_prev", 0, DT_FLOAT})
                   .Input({"h_prev", 0, DT_FLOAT})
                   .Input({"w", 0, DT_FLOAT})
                   .Input({"wci", 0, DT_FLOAT})
                   .Input({"wcf", 0, DT_FLOAT})
                   .Input({"wco", 0, DT_FLOAT})
                   .Input({"b", 0, DT_FLOAT})
                   .Input({"i", 0, DT_FLOAT})
                   .Input({"cs", 0, DT_FLOAT})
                   .Input({"f", 0, DT_FLOAT})
                   .Input({"o", 0, DT_FLOAT})
                   .Input({"ci", 0, DT_FLOAT})
                   .Input({"co", 0, DT_FLOAT})
                   .Input({"h", 0, DT_FLOAT})
                   .Input({"cs_grad", 0, DT_FLOAT})
                   .Input({"h_grad", 0, DT_FLOAT})
                   .Finalize(&op.node_def));

  // Last inputs don't affect shape inference.
  string suffix = ";" + JoinedCopies("?", 9);

  // Rank check for x
  INFER_ERROR("must be rank 3", op, "?;[?];?;?;?;?;?;?;?" + suffix);

  // Rank checks for cs_prev through b.
  INFER_ERROR("must be rank 2", op, "?;?;[1];?;?;?;?;?;?" + suffix);
  INFER_ERROR("must be rank 2", op, "?;?;?;[1];?;?;?;?;?" + suffix);
  INFER_ERROR("must be rank 2", op, "?;?;?;?;[1];?;?;?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[1,?];?;?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;[1,?];?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;?;[1,?];?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;?;?;[1,?]" + suffix);

  // Output with all input knowns makes known rank outputs.
  INFER_OK(
      op, JoinedCopies("?", 18),
      "[?,?,?];" + JoinedCopies("[?,?]", 3) + ";" + JoinedCopies("[?]", 4));

  // Output with copies input shapes to output.
  string input = strings::StrCat("?;[?,?,?];", JoinedCopies("[?,?]", 3), ";",
                                 JoinedCopies("[?]", 4), suffix);
  string expected = "in1";
  for (int i = 1; i < 8; ++i) {
    strings::StrAppend(&expected, ";in", (i + 1));
  }
  INFER_OK(op, input, expected);
}

}  // namespace tensorflow
