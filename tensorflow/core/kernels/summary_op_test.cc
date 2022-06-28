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
class MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc() {
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

#include <functional>
#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("expected_str: \"" + expected_str + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/summary_op_test.cc", "EXPECT_SummaryMatches");

  Summary expected;
  CHECK(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

// --------------------------------------------------------------------------
// SummaryHistoOp
// --------------------------------------------------------------------------
class SummaryHistoOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType dt) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/summary_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "HistogramSummary")
                     .Input(FakeInput())
                     .Input(FakeInput(dt))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryHistoOpTest, SimpleFloat) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromArray<float>(TensorShape({3, 2}),
                           {0.1f, -0.7f, 4.1f, 4., 5.f, 4.f});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7500  StdDev: 2.20\n"
      "Min: -0.7000  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, SimpleDouble) {
  MakeOp(DT_DOUBLE);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromArray<double>(TensorShape({3, 2}), {0.1, -0.7, 4.1, 4., 5., 4.});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7500  StdDev: 2.20\n"
      "Min: -0.7000  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, SimpleHalf) {
  MakeOp(DT_HALF);

  // Feed and run
  AddInputFromList<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromList<Eigen::half>(TensorShape({3, 2}),
                                {0.1, -0.7, 4.1, 4., 5., 4.});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7502  StdDev: 2.20\n"
      "Min: -0.7002  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, Error_WrongDimsTags) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({2, 1}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2}), {1.0f, -0.73f});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "tags must be scalar")) << s;
}

TEST_F(SummaryHistoOpTest, Error_TooManyTagValues) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({2}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2, 1}), {1.0f, -0.73f});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "tags must be scalar")) << s;
}

// --------------------------------------------------------------------------
// SummaryMergeOp
// --------------------------------------------------------------------------
class SummaryMergeOpTest : public OpsTestBase {
 protected:
  void MakeOp(int num_inputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_op_testDTcc mht_2(mht_2_v, 349, "", "./tensorflow/core/kernels/summary_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "MergeSummary")
                     .Input(FakeInput(num_inputs))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryMergeOpTest, Simple) {
  MakeOp(1);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tag2\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag3\" simple_value: 10000.0 }", &s2));
  Summary s3;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag4\" simple_value: 11.0 }", &s3));

  AddInputFromArray<tstring>(
      TensorShape({3}),
      {s1.SerializeAsString(), s2.SerializeAsString(), s3.SerializeAsString()});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  EXPECT_SummaryMatches(summary,
                        "value { tag: \"tag1\" simple_value: 1.0 } "
                        "value { tag: \"tag2\" simple_value: -0.73 } "
                        "value { tag: \"tag3\" simple_value: 10000.0 }"
                        "value { tag: \"tag4\" simple_value: 11.0 }");
}

TEST_F(SummaryMergeOpTest, Simple_MultipleInputs) {
  MakeOp(3);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tag2\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag3\" simple_value: 10000.0 }", &s2));
  Summary s3;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag4\" simple_value: 11.0 }", &s3));

  AddInputFromArray<tstring>(TensorShape({}), {s1.SerializeAsString()});
  AddInputFromArray<tstring>(TensorShape({}), {s2.SerializeAsString()});
  AddInputFromArray<tstring>(TensorShape({}), {s3.SerializeAsString()});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  EXPECT_SummaryMatches(summary,
                        "value { tag: \"tag1\" simple_value: 1.0 } "
                        "value { tag: \"tag2\" simple_value: -0.73 } "
                        "value { tag: \"tag3\" simple_value: 10000.0 }"
                        "value { tag: \"tag4\" simple_value: 11.0 }");
}

TEST_F(SummaryMergeOpTest, Error_MismatchedSize) {
  MakeOp(1);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tagduplicate\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tagduplicate\" simple_value: 1.0 } ", &s2));
  AddInputFromArray<tstring>(TensorShape({2}),
                             {s1.SerializeAsString(), s2.SerializeAsString()});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "Duplicate tag")) << s;
}

}  // namespace
}  // namespace tensorflow
