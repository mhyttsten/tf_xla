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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_stats.h"

#include <utility>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

string CheckAndRemoveDoc(const string& doc) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("doc: \"" + doc + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/profiler/internal/tfprof_show_test.cc", "CheckAndRemoveDoc");

  auto pos = doc.find("Profile:");
  CHECK(pos != doc.npos);
  return doc.substr(pos + 9);
}

class TFProfShowTest : public ::testing::Test {
 protected:
  TFProfShowTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/profiler/internal/tfprof_show_test.cc", "TFProfShowTest");

    string graph_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/graph.pbtxt");
    std::unique_ptr<tensorflow::GraphDef> graph_pb(new tensorflow::GraphDef());
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), graph_path, graph_pb.get(), false));

    std::unique_ptr<tensorflow::RunMetadata> run_meta_pb(
        new tensorflow::RunMetadata());
    string run_meta_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/run_meta");
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), run_meta_path, run_meta_pb.get(), true));

    std::unique_ptr<OpLogProto> op_log_pb(new OpLogProto());
    string op_log_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/tfprof_log");
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), op_log_path, op_log_pb.get()));

    string ckpt_path = io::JoinPath(testing::TensorFlowSrcRoot(),
                                    "core/profiler/internal/testdata/ckpt");
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader(
        new checkpoint::CheckpointReader(ckpt_path, status));
    CHECK(TF_GetCode(status) == TF_OK);
    TF_DeleteStatus(status);

    tf_stats_.reset(new TFStats(std::move(graph_pb), std::move(run_meta_pb),
                                std::move(op_log_pb), std::move(ckpt_reader)));
    tf_stats_->BuildAllViews();
  }

  string TestToFromProto(const string& cmd, const Options& opts,
                         bool show_multi_node = false) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("cmd: \"" + cmd + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_testDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/profiler/internal/tfprof_show_test.cc", "TestToFromProto");

    string profile_file = io::JoinPath(testing::TmpDir(), "profile");
    tf_stats_->WriteProfile(profile_file);
    TFStats new_stats(profile_file, nullptr);
    new_stats.BuildAllViews();
    if (show_multi_node) {
      new_stats.ShowMultiGraphNode(cmd, opts);
    } else {
      new_stats.ShowGraphNode(cmd, opts);
    }
    string dump_str;
    TF_CHECK_OK(ReadFileToString(Env::Default(),
                                 opts.output_options.at("outfile"), &dump_str));
    return dump_str;
  }

  std::unique_ptr<TFStats> tf_stats_;
};

TEST_F(TFProfShowTest, DumpScopeMode) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name",
      {"VariableV2"},  // accout_type_regexes
      {".*"}, {""}, {".*"}, {""}, false,
      {"params", "bytes", "peak_bytes", "residual_bytes", "output_bytes",
       "micros", "accelerator_micros", "cpu_micros", "float_ops"},
      "file", {{"outfile", dump_file}});
  tf_stats_->ShowGraphNode("scope", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "node name | # parameters | # float_ops | requested bytes | peak bytes | "
      "residual bytes | output bytes | total execution time | accelerator "
      "execution time | cpu execution time\n_TFProfRoot (--/451 params, --/0 "
      "flops, --/2.56KB, --/2.56KB, --/2.56KB, --/2.56KB, --/13us, --/0us, "
      "--/13us)\n  DW (3x3x3x6, 162/162 params, 0/0 flops, 1.28KB/1.28KB, "
      "1.28KB/1.28KB, 1.28KB/1.28KB, 1.28KB/1.28KB, 2us/2us, 0us/0us, "
      "2us/2us)\n  DW2 (2x2x6x12, 288/288 params, 0/0 flops, 1.28KB/1.28KB, "
      "1.28KB/1.28KB, 1.28KB/1.28KB, 1.28KB/1.28KB, 11us/11us, 0us/0us, "
      "11us/11us)\n  ScalarW (1, 1/1 params, 0/0 flops, 0B/0B, 0B/0B, 0B/0B, "
      "0B/0B, 0us/0us, 0us/0us, 0us/0us)\n",
      CheckAndRemoveDoc(dump_str));

  EXPECT_EQ(dump_str, TestToFromProto("scope", opts));
}

TEST_F(TFProfShowTest, DumpAcceleratorAndCPUMicros) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "cpu_micros",
               {".*"},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"accelerator_micros", "cpu_micros"}, "file",
               {{"outfile", dump_file}});
  tf_stats_->ShowGraphNode("scope", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "node name | accelerator execution time | cpu execution "
      "time\n_TFProfRoot (--/404us, --/4.54ms)\n  Conv2D (226us/226us, "
      "4.07ms/4.07ms)\n  Conv2D_1 (178us/178us, 419us/419us)\n  "
      "_retval_Conv2D_1_0_0 (0us/0us, 41us/41us)\n  DW2 (0us/0us, 11us/11us)\n "
      "   DW2/Assign (0us/0us, 0us/0us)\n    DW2/Initializer (0us/0us, "
      "0us/0us)\n      DW2/Initializer/random_normal (0us/0us, 0us/0us)\n      "
      "  DW2/Initializer/random_normal/RandomStandardNormal (0us/0us, "
      "0us/0us)\n        DW2/Initializer/random_normal/mean (0us/0us, "
      "0us/0us)\n        DW2/Initializer/random_normal/mul (0us/0us, "
      "0us/0us)\n        DW2/Initializer/random_normal/shape (0us/0us, "
      "0us/0us)\n        DW2/Initializer/random_normal/stddev (0us/0us, "
      "0us/0us)\n    DW2/read (0us/0us, 0us/0us)\n  DW (0us/0us, 2us/2us)\n    "
      "DW/Assign (0us/0us, 0us/0us)\n    DW/Initializer (0us/0us, 0us/0us)\n   "
      "   DW/Initializer/random_normal (0us/0us, 0us/0us)\n        "
      "DW/Initializer/random_normal/RandomStandardNormal (0us/0us, 0us/0us)\n  "
      "      DW/Initializer/random_normal/mean (0us/0us, 0us/0us)\n        "
      "DW/Initializer/random_normal/mul (0us/0us, 0us/0us)\n        "
      "DW/Initializer/random_normal/shape (0us/0us, 0us/0us)\n        "
      "DW/Initializer/random_normal/stddev (0us/0us, 0us/0us)\n    DW/read "
      "(0us/0us, 0us/0us)\n  zeros (0us/0us, 2us/2us)\n  ScalarW (0us/0us, "
      "0us/0us)\n    ScalarW/Assign (0us/0us, 0us/0us)\n    "
      "ScalarW/Initializer (0us/0us, 0us/0us)\n      "
      "ScalarW/Initializer/random_normal (0us/0us, 0us/0us)\n        "
      "ScalarW/Initializer/random_normal/RandomStandardNormal (0us/0us, "
      "0us/0us)\n        ScalarW/Initializer/random_normal/mean (0us/0us, "
      "0us/0us)\n        ScalarW/Initializer/random_normal/mul (0us/0us, "
      "0us/0us)\n        ScalarW/Initializer/random_normal/shape (0us/0us, "
      "0us/0us)\n        ScalarW/Initializer/random_normal/stddev (0us/0us, "
      "0us/0us)\n    ScalarW/read (0us/0us, 0us/0us)\n  init (0us/0us, "
      "0us/0us)\n",
      CheckAndRemoveDoc(dump_str));

  EXPECT_EQ(dump_str, TestToFromProto("scope", opts));
}

TEST_F(TFProfShowTest, DumpOpMode) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, "params",
      {".*"},  // accout_type_regexes
      {".*"}, {""}, {".*"}, {""}, false,
      {"params", "bytes", "micros", "float_ops", "occurrence", "input_shapes"},
      "file", {{"outfile", dump_file}});
  tf_stats_->ShowMultiGraphNode("op", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "nodename|requestedbytes|totalexecutiontime|acceleratorexecutiontime|"
      "cpuexecutiontime|#parameters|#float_ops|opoccurrence(run|defined)|"
      "inputshapes\nVariableV22.56KB(100.00%,8.40%),13us(100.00%,0.26%),0us("
      "100.00%,0.00%),13us(100.00%,0.29%),451params(100.00%,100.00%),0float_"
      "ops(100.00%,0.00%),2|3\n\ninput_type:\t(run*2|defined*3)\texec_time:"
      "13us\n\nAdd0B(0.00%,0.00%),0us(99.74%,0.00%),0us(100.00%,0.00%),0us(99."
      "71%,0.00%),0params(0.00%,0.00%),0float_ops(100.00%,0.00%),0|3\n\ninput_"
      "type:0:1,\t1:1\t(run*0|defined*1)\texec_time:0us\ninput_type:0:2x2x6x12,"
      "\t1:1\t(run*0|defined*1)\texec_time:0us\ninput_type:0:3x3x3x6,\t1:1\t("
      "run*0|defined*1)\texec_time:0us\n\nAssign0B(0.00%,0.00%),0us(99.74%,0."
      "00%),0us(100.00%,0.00%),0us(99.71%,0.00%),0params(0.00%,0.00%),0float_"
      "ops(100.00%,0.00%),0|3\n\ninput_type:0:1,\t1:1\t(run*0|defined*1)\texec_"
      "time:0us\ninput_type:0:2x2x6x12,\t1:2x2x6x12\t(run*0|defined*1)\texec_"
      "time:0us\ninput_type:0:3x3x3x6,\t1:3x3x3x6\t(run*0|defined*1)\texec_"
      "time:0us\n\nConst0B(0.00%,0.00%),2us(99.74%,0.04%),0us(100.00%,0.00%),"
      "2us(99.71%,0.04%),0params(0.00%,0.00%),0float_ops(100.00%,0.00%),1|"
      "10\n\ninput_type:\t(run*1|defined*10)\texec_time:2us\n\nConv2D27.90KB("
      "91.60%,91.60%),4.89ms(99.70%,98.87%),404us(100.00%,100.00%),4.49ms(99."
      "67%,98.77%),0params(0.00%,0.00%),10.44kfloat_ops(100.00%,100.00%),2|"
      "2\n\ninput_type:0:2x3x3x6,\t1:2x2x6x12\t(run*1|defined*1)\texec_time:"
      "597us\ninput_type:0:2x6x6x3,\t1:3x3x3x6\t(run*1|defined*1)\texec_time:4."
      "29ms\n\nIdentity0B(0.00%,0.00%),0us(0.83%,0.00%),0us(0.00%,0.00%),0us(0."
      "90%,0.00%),0params(0.00%,0.00%),0float_ops(0.00%,0.00%),0|3\n\ninput_"
      "type:0:1\t(run*0|defined*1)\texec_time:0us\ninput_type:0:2x2x6x12\t(run*"
      "0|defined*1)\texec_time:0us\ninput_type:0:3x3x3x6\t(run*0|defined*1)"
      "\texec_time:0us\n\n",
      StringReplace(CheckAndRemoveDoc(dump_str), " ", ""));

  EXPECT_EQ(dump_str, TestToFromProto("op", opts, true));
}
}  // namespace tfprof
}  // namespace tensorflow
