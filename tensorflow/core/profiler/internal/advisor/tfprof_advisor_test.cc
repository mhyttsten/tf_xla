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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPStfprof_advisor_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPStfprof_advisor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPStfprof_advisor_testDTcc() {
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

#include "tensorflow/core/profiler/internal/advisor/tfprof_advisor.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfprof {

class TFProfAdvisorTest : public ::testing::Test {
 protected:
  TFProfAdvisorTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPStfprof_advisor_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/profiler/internal/advisor/tfprof_advisor_test.cc", "TFProfAdvisorTest");

    stats_.reset(new TFStats(std::unique_ptr<GraphDef>(new GraphDef()), nullptr,
                             nullptr, nullptr));

    stats_->AddNodeForTest(
        0, CreateNode("n1", "Conv2D", {{"data_format", "NHWC"}}, 0, 10, 2));
    stats_->AddNodeForTest(0, CreateNode("n2", "Conv2D", {}, 0, 20, 2));
    stats_->BuildAllViews();
    advisor_.reset(new Advisor(stats_.get()));
  }

  std::unique_ptr<TFGraphNode> CreateNode(const string& name,
                                          const string& type,
                                          std::map<string, string> attrs,
                                          int64_t step, int64_t start_miros,
                                          int64_t end_rel_micros) {
    node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
    NodeDef* def = node_defs_.back().get();

    def->set_name(name);
    def->set_op(type);
    for (const auto& attr : attrs) {
      (*def->mutable_attr())[attr.first].set_s(attr.second);
    }
    std::unique_ptr<TFGraphNode> node(new TFGraphNode(def, -1, nullptr));

    NodeExecStats node_stat;
    node_stat.set_all_start_micros(start_miros);
    node_stat.set_op_end_rel_micros(end_rel_micros);
    node->AddStepStat(step, "/job:localhost/replica:0/task:0/device:GPU:0",
                      node_stat);
    node->AddStepStat(step,
                      "/job:localhost/replica:0/task:0/device:GPU:0:stream:all",
                      node_stat);
    node->AddStepStat(step,
                      "/job:localhost/replica:0/task:0/device:GPU:0:stream:0",
                      node_stat);
    return node;
  }

  std::unique_ptr<TFStats> stats_;
  std::unique_ptr<Advisor> advisor_;
  std::vector<std::unique_ptr<NodeDef>> node_defs_;
};

TEST_F(TFProfAdvisorTest, Basics) {
  AdvisorOptionsProto options = Advisor::DefaultOptions();
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_TRUE(advice.checkers().find(kCheckers[0]) != advice.checkers().end());
  EXPECT_TRUE(advice.checkers().find(kCheckers[1]) != advice.checkers().end());
  EXPECT_TRUE(advice.checkers().find(kCheckers[2]) != advice.checkers().end());
}

TEST_F(TFProfAdvisorTest, OperationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[1]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_EQ(advice.checkers().at(kCheckers[1]).reports_size(), 1);
  EXPECT_TRUE(
      absl::StrContains(advice.checkers().at(kCheckers[1]).reports(0), "NCHW"));
}

TEST_F(TFProfAdvisorTest, UtilizationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[0]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_EQ(advice.checkers().at(kCheckers[0]).reports_size(), 1);
  EXPECT_TRUE(absl::StrContains(advice.checkers().at(kCheckers[0]).reports(0),
                                "low utilization"));
}

TEST_F(TFProfAdvisorTest, ExpensiveOperationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[2]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_TRUE(absl::StrContains(advice.checkers().at(kCheckers[2]).reports(0),
                                "top 1 operation type: Conv2D"));
}

}  // namespace tfprof
}  // namespace tensorflow
