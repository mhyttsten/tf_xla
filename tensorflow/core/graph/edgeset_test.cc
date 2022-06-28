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
class MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc() {
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

#include "tensorflow/core/graph/edgeset.h"

#include <set>
#include <vector>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
class EdgeSetTest : public ::testing::Test {
 public:
  EdgeSetTest() : edges_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/graph/edgeset_test.cc", "EdgeSetTest");
}
  ~EdgeSetTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/graph/edgeset_test.cc", "~EdgeSetTest");
 delete[] edges_; }

  void MakeEdgeSet(int n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc mht_2(mht_2_v, 204, "", "./tensorflow/core/graph/edgeset_test.cc", "MakeEdgeSet");

    if (edges_) {
      delete[] edges_;
    }
    edges_ = new Edge[n];
    eset_.clear();
    model_.clear();
    for (int i = 0; i < n; i++) {
      eset_.insert(&edges_[i]);
      model_.insert(&edges_[i]);
    }
  }

  void CheckSame() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgeset_testDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/graph/edgeset_test.cc", "CheckSame");

    EXPECT_EQ(model_.size(), eset_.size());
    EXPECT_EQ(model_.empty(), eset_.empty());
    std::vector<const Edge*> modelv(model_.begin(), model_.end());
    std::vector<const Edge*> esetv(eset_.begin(), eset_.end());
    std::sort(modelv.begin(), modelv.end());
    std::sort(esetv.begin(), esetv.end());
    EXPECT_EQ(modelv.size(), esetv.size());
    for (size_t i = 0; i < modelv.size(); i++) {
      EXPECT_EQ(modelv[i], esetv[i]) << i;
    }
  }

  static constexpr int kInline = 64 / sizeof(const void*);
  Edge nonexistent_;
  Edge* edges_;
  EdgeSet eset_;
  std::set<const Edge*> model_;
};

namespace {

TEST_F(EdgeSetTest, Ops) {
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    CheckSame();
    EXPECT_EQ((n == 0), eset_.empty());
    EXPECT_EQ(n, eset_.size());

    eset_.clear();
    model_.clear();
    CheckSame();

    eset_.insert(&edges_[0]);
    model_.insert(&edges_[0]);
    CheckSame();
  }
}

// Try insert/erase of existing elements at different positions.
TEST_F(EdgeSetTest, Exists) {
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    for (int pos = 0; pos < n; pos++) {
      auto p = eset_.insert(&edges_[pos]);
      EXPECT_FALSE(p.second);
      EXPECT_EQ(&edges_[pos], *p.first);

      EXPECT_EQ(1, eset_.erase(&edges_[pos]));
      model_.erase(&edges_[pos]);
      CheckSame();
    }
  }
}

// Try insert/erase of non-existent element.
TEST_F(EdgeSetTest, DoesNotExist) {
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    EXPECT_EQ(0, eset_.erase(&nonexistent_));
    auto p = eset_.insert(&nonexistent_);
    EXPECT_TRUE(p.second);
    EXPECT_EQ(&nonexistent_, *p.first);
  }
}

}  // namespace
}  // namespace tensorflow
