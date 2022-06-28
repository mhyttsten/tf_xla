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
class MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc() {
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

#include <chrono>
#include <thread>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/util/determinism_test_util.h"

namespace tensorflow {
namespace {

class PrintingV2GraphTest : public OpsTestBase {
 protected:
  Status Init(const string& output_stream = "log(warning)") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/kernels/logging_ops_test.cc", "Init");

    TF_CHECK_OK(NodeDefBuilder("op", "PrintV2")
                    .Input(FakeInput(DT_STRING))
                    .Attr("output_stream", output_stream)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(PrintingV2GraphTest, StringSuccess) {
  TF_ASSERT_OK(Init());
  AddInputFromArray<tstring>(TensorShape({}), {"bar"});
  TF_ASSERT_OK(RunOpKernel());
}

TEST_F(PrintingV2GraphTest, InvalidOutputStream) {
  ASSERT_NE(::tensorflow::Status::OK(), (Init("invalid_output_stream")));
}

TEST_F(PrintingV2GraphTest, InvalidInputRank) {
  TF_ASSERT_OK(Init());
  AddInputFromArray<tstring>(TensorShape({2}), {"bar", "foo"});
  ASSERT_NE(::tensorflow::Status::OK(), RunOpKernel());
}

class PrintingGraphTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type1, DataType input_type2, string msg = "",
              int first_n = -1, int summarize = 3) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/logging_ops_test.cc", "Init");

    TF_CHECK_OK(NodeDefBuilder("op", "Print")
                    .Input(FakeInput(input_type1))
                    .Input(FakeInput(2, input_type2))
                    .Attr("message", msg)
                    .Attr("first_n", first_n)
                    .Attr("summarize", summarize)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(PrintingGraphTest, Int32Success_6) {
  TF_ASSERT_OK(Init(DT_INT32, DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, Int32Success_Summarize6) {
  TF_ASSERT_OK(Init(DT_INT32, DT_INT32, "", -1, 6));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, StringSuccess) {
  TF_ASSERT_OK(Init(DT_INT32, DT_STRING));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<tstring>(TensorShape({}), {"foo"});
  AddInputFromArray<tstring>(TensorShape({}), {"bar"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, MsgSuccess) {
  TF_ASSERT_OK(Init(DT_INT32, DT_STRING, "Message: "));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<tstring>(TensorShape({}), {"foo"});
  AddInputFromArray<tstring>(TensorShape({}), {"bar"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, FirstNSuccess) {
  TF_ASSERT_OK(Init(DT_INT32, DT_STRING, "", 3));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<tstring>(TensorShape({}), {"foo"});
  AddInputFromArray<tstring>(TensorShape({}), {"bar"});
  // run 4 times but we only print 3 as intended
  for (int i = 0; i < 4; i++) TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

class TimestampTest : public OpsTestBase {
 protected:
  Status Init() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_ops_testDTcc mht_2(mht_2_v, 310, "", "./tensorflow/core/kernels/logging_ops_test.cc", "Init");

    TF_CHECK_OK(NodeDefBuilder("op", "Timestamp").Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(TimestampTest, WaitAtLeast) {
  TF_ASSERT_OK(Init());
  TF_ASSERT_OK(RunOpKernel());
  double ts1 = *((*GetOutput(0)).flat<double>().data());

  // wait 1 second
  std::this_thread::sleep_for(std::chrono::seconds(1));

  TF_ASSERT_OK(RunOpKernel());
  double ts2 = *((*GetOutput(0)).flat<double>().data());

  EXPECT_LE(1.0, ts2 - ts1);
}

TEST_F(TimestampTest, DeterminismError) {
  test::DeterministicOpsScope det_scope;
  TF_ASSERT_OK(Init());
  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(
                  error::FAILED_PRECONDITION,
                  "Timestamp cannot be called when determinism is enabled"));
}

}  // end namespace
}  // end namespace tensorflow
