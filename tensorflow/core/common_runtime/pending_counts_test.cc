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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_counts_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_counts_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_counts_testDTcc() {
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

#include "tensorflow/core/common_runtime/pending_counts.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

using std::unique_ptr;

namespace tensorflow {

TEST(PendingCounts, Simple) {
  const int C = 300;
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> h(C);
  for (int id = 0; id < C; id++) {
    h[id] = layout.CreateHandle(id, id);
  }

  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(h[id], id);
  }
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(h[id]), id);
    EXPECT_EQ(c.dead_count(h[id]), 0);
  }

  for (int id = 0; id < C; id++) {
    c.increment_dead_count(h[id]);
    // The dead count is no longer updated once pending is 0.
    EXPECT_EQ(c.dead_count(h[id]), (id == 0) ? 0 : 1);
  }

  EXPECT_EQ(c.decrement_pending(h[1], 1), 0);
  EXPECT_EQ(c.decrement_pending(h[3], 1), 2);
  EXPECT_EQ(c.decrement_pending(h[3], 1), 1);
  c.decrement_pending(h[5], 1);
  c.decrement_pending(h[5], 3);
  c.decrement_pending(h[170], 1);
  c.decrement_pending(h[170], 13);
  EXPECT_EQ(c.pending(h[1]), 0);
  EXPECT_EQ(c.pending(h[3]), 1);
  EXPECT_EQ(c.pending(h[5]), 1);
  EXPECT_EQ(c.pending(h[170]), 156);
}

TEST(PendingCounts, CopyConstructor) {
  const int C = 300;
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> h(C);
  for (int id = 0; id < C; id++) {
    h[id] = layout.CreateHandle(id, id);
  }
  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(h[id], id);
  }
  PendingCounts c2(c);
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(h[id]), c2.pending(h[id]));
    EXPECT_EQ(c.dead_count(h[id]), c2.dead_count(h[id]));
  }
}

TEST(PendingCounts, MarkLiveShowsUpAsCount) {
  PendingCounts::Layout layout;
  PendingCounts::Handle handles[2];
  handles[0] = layout.CreateHandle(5, 4);
  handles[1] = layout.CreateHandle(15, 4);
  for (int id = 0; id < 2; id++) {
    PendingCounts::Handle h = handles[id];
    // Test for both packed and large.
    int count = (id == 0) ? 5 : 15;

    PendingCounts c(layout);
    c.set_initial_count(h, count);
    EXPECT_EQ(c.pending(h), count);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), count - 1);
    // mark_live should be idempotent
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), count - 1);

    c.decrement_pending(h, count - 1);
    EXPECT_EQ(c.pending(h), 0);

    // mark_live should be idempotent
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
    c.mark_started(h);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
    c.mark_completed(h);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
  }
}

TEST(PendingCounts, StateIsCorrect) {
  const int C = 20;
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> handles(C);
  for (int id = 0; id < C; id++) {
    handles[id] = layout.CreateHandle(id, id);
  }
  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(handles[id], id);
  }

  for (int id = 0; id < C; id++) {
    PendingCounts::Handle h = handles[id];
    while (c.pending(h) > 0) {
      EXPECT_EQ(c.node_state(h), PendingCounts::PENDING_NOTREADY);
      c.decrement_pending(h, 1);
    }
    EXPECT_EQ(c.node_state(h), PendingCounts::PENDING_READY);
    c.mark_started(h);
    EXPECT_EQ(c.node_state(h), PendingCounts::STARTED);
    c.mark_completed(h);
    EXPECT_EQ(c.node_state(h), PendingCounts::COMPLETED);
  }
}

TEST(PendingCounts, AdjustForActivation) {
  PendingCounts::Layout layout;
  PendingCounts::Handle handles[2];
  handles[0] = layout.CreateHandle(5, 4);
  handles[1] = layout.CreateHandle(15, 4);
  for (int id = 0; id < 2; id++) {
    PendingCounts::Handle h = handles[id];
    // Test for both packed and large.
    int count = (id == 0) ? 5 : 15;

    PendingCounts c(layout);
    c.set_initial_count(h, count);
    EXPECT_EQ(c.pending(h), count);

    // Don't increment the dead count this time
    PendingCounts::AdjustResult result = c.adjust_for_activation(h, false);
    EXPECT_EQ(c.pending(h), count - 1);
    EXPECT_TRUE(result.any_pending);
    EXPECT_EQ(c.dead_count(h), 0);
    EXPECT_FALSE(result.any_dead);

    // Increment the dead count this time
    result = c.adjust_for_activation(h, true);
    EXPECT_EQ(c.pending(h), count - 2);
    EXPECT_TRUE(result.any_pending);
    EXPECT_EQ(c.dead_count(h), 1);
    EXPECT_TRUE(result.any_dead);
  }
}

TEST(PendingCounts, AdjustForActivationAtomic) {
  PendingCounts::Layout layout;
  PendingCounts::Handle handles[2];
  const int kInitialCounts[2] = {6, 16};
  handles[0] = layout.CreateHandle(kInitialCounts[0], 0);
  handles[1] = layout.CreateHandle(kInitialCounts[1], 0);
  PendingCounts c(layout);
  c.set_initial_count(handles[0], kInitialCounts[0]);
  c.set_initial_count(handles[1], kInitialCounts[1]);

  Env* env = Env::Default();
  std::atomic<bool> start{false};
  std::vector<unique_ptr<Thread>> threads;
  for (int t = 0; t < 2; t++) {
    threads.emplace_back(env->StartThread({}, "tester", [&]() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_counts_testDTcc mht_0(mht_0_v, 356, "", "./tensorflow/core/common_runtime/pending_counts_test.cc", "lambda");

      while (!start) {
      }
      for (int i = 0; i < kInitialCounts[0] / 2; i++) {
        c.adjust_for_activation_atomic(handles[0], false);
      }
      for (int i = 0; i < kInitialCounts[1] / 2; i++) {
        c.adjust_for_activation_atomic(handles[1], false);
      }
    }));
  }
  start = true;
  threads.clear();  // Joins the threads.

  EXPECT_EQ(c.pending(handles[0]), 0);
  EXPECT_EQ(c.pending(handles[1]), 0);
}

}  // namespace tensorflow
