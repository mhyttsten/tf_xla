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
class MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc() {
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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CudnnRNNOpsTest, ParamsSize_ShapeFn) {
  ShapeInferenceTestOp op("CudnnRNNParamsSize");
  INFER_OK(op, "[];[];[]", "[1]");
  INFER_OK(op, "?;[];[]", "[1]");
  INFER_OK(op, "[];?;[]", "[1]");
  INFER_OK(op, "[];[];?", "[1]");
  INFER_OK(op, "[];?;?", "[1]");
  INFER_OK(op, "?;?;?", "[1]");

  INFER_ERROR("Shape must be rank 0 ", op, "[1,2];?;[]");
  INFER_ERROR("Shape must be rank 0 ", op, "?;[2];[]");
  INFER_ERROR("Shape must be rank 0 ", op, "?;?;[1]");
}

TEST(CudnnRNNOpsTest, ForwardLstm_ShapeFn) {
  int seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {seq_length, batch_size,
                                   num_units * dir_count};
  auto shape_to_str = [](const std::vector<int>& v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/ops/cudnn_rnn_ops_test.cc", "lambda");

    return strings::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_h_shape), ";", "[?]");
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in1;?";

  ShapeInferenceTestOp op("CudnnRNN");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNN")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?]");
  // Disabled because the kernel does not check shape of input_c.
  // INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV2Lstm_ShapeFn) {
  int seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {seq_length, batch_size,
                                   num_units * dir_count};
  auto shape_to_str = [](const std::vector<int>& v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc mht_1(mht_1_v, 261, "", "./tensorflow/core/ops/cudnn_rnn_ops_test.cc", "lambda");

    return strings::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_h_shape), ";", "[?]");
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in1;?;?";

  ShapeInferenceTestOp op("CudnnRNNV2");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV2")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?]");
  // Disabled because the kernel does not check shape of input_c.
  // INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV3Lstm_ShapeFn) {
  int max_seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {max_seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> input_c_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {max_seq_length, batch_size,
                                   num_units * dir_count};
  std::vector<int> seq_lengths_shape = {batch_size};
  auto shape_to_str = [](const std::vector<int>& v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc mht_2(mht_2_v, 304, "", "./tensorflow/core/ops/cudnn_rnn_ops_test.cc", "lambda");

    return strings::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_c_shape), ";", "[?]", ";",
      shape_to_str(seq_lengths_shape));
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in2;?;?";

  ShapeInferenceTestOp op("CudnnRNNV3");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV3")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Input({"sequence_lengths", 0, DT_INT32})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV3Gru) {
  int max_seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {max_seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> input_c_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {max_seq_length, batch_size,
                                   num_units * dir_count};
  std::vector<int> seq_lengths_shape = {batch_size};
  auto shape_to_str = [](const std::vector<int>& v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPScudnn_rnn_ops_testDTcc mht_3(mht_3_v, 349, "", "./tensorflow/core/ops/cudnn_rnn_ops_test.cc", "lambda");

    return strings::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_c_shape), ";", "[?]", ";",
      shape_to_str(seq_lengths_shape));
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;[];?;?";

  ShapeInferenceTestOp op("CudnnRNNV3");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV3")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Input({"sequence_lengths", 0, DT_INT32})
                   .Attr("rnn_mode", "gru")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[];[];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[];[?];[]");
}

}  // end namespace tensorflow
