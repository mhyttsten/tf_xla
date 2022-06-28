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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_scheduler_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_scheduler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_scheduler_testDTcc() {
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

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_scheduler_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/batching_util/batch_scheduler_test.cc", "FakeTask");
}

  ~FakeTask() override = default;

  size_t size() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_scheduler_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/kernels/batching_util/batch_scheduler_test.cc", "size");
 return size_; }

 private:
  const size_t size_;

  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

TEST(BatchTest, Basic) {
  Batch<FakeTask> batch;

  EXPECT_EQ(0, batch.num_tasks());
  EXPECT_TRUE(batch.empty());
  EXPECT_EQ(0, batch.size());
  EXPECT_FALSE(batch.IsClosed());

  auto task0 = new FakeTask(3);
  batch.AddTask(std::unique_ptr<FakeTask>(task0));

  EXPECT_EQ(1, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size(), batch.size());
  EXPECT_EQ(task0->size(), batch.task(0).size());
  EXPECT_FALSE(batch.IsClosed());

  auto task1 = new FakeTask(7);
  batch.AddTask(std::unique_ptr<FakeTask>(task1));

  EXPECT_EQ(2, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size() + task1->size(), batch.size());
  EXPECT_EQ(task1->size(), batch.task(1).size());
  EXPECT_EQ(task1->size(), batch.mutable_task(1)->size());
  EXPECT_FALSE(batch.IsClosed());

  batch.Close();
  EXPECT_TRUE(batch.IsClosed());

  EXPECT_EQ(2, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size() + task1->size(), batch.size());
  EXPECT_EQ(task0->size(), batch.task(0).size());
  EXPECT_EQ(task1->size(), batch.task(1).size());

  EXPECT_EQ(7, batch.RemoveTask()->size());
  EXPECT_EQ(3, batch.size());
  EXPECT_EQ(3, batch.RemoveTask()->size());
  EXPECT_EQ(0, batch.size());
  EXPECT_TRUE(batch.empty());
}

TEST(BatchTest, WaitUntilClosed) {
  Batch<FakeTask> batch;
  batch.AddTask(std::unique_ptr<FakeTask>(new FakeTask(3)));
  EXPECT_FALSE(batch.IsClosed());

  std::unique_ptr<Thread> close_thread(
      Env::Default()->StartThread(ThreadOptions(), "test", [&batch]() {
        Env::Default()->SleepForMicroseconds(100);
        batch.Close();
      }));
  batch.WaitUntilClosed();
  EXPECT_TRUE(batch.IsClosed());
}

TEST(BatchTest, DeletionBlocksUntilClosed) {
  Batch<FakeTask>* batch = new Batch<FakeTask>;
  batch->AddTask(std::unique_ptr<FakeTask>(new FakeTask(3)));
  EXPECT_FALSE(batch->IsClosed());

  Notification do_delete, deleted;
  std::unique_ptr<Thread> delete_thread(Env::Default()->StartThread(
      ThreadOptions(), "test", [&batch, &do_delete, &deleted]() {
        do_delete.WaitForNotification();
        delete batch;
        deleted.Notify();
      }));
  do_delete.Notify();
  Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
  EXPECT_FALSE(deleted.HasBeenNotified());
  batch->Close();
  deleted.WaitForNotification();
}

TEST(BatchTest, RemoveAllTasks) {
  Batch<FakeTask> batch;

  auto task0 = new FakeTask(3);
  batch.AddTask(std::unique_ptr<FakeTask>(task0));

  auto task1 = new FakeTask(7);
  batch.AddTask(std::unique_ptr<FakeTask>(task1));

  batch.Close();
  EXPECT_TRUE(batch.IsClosed());

  std::vector<std::unique_ptr<FakeTask>> tasks_in_batch =
      batch.RemoveAllTasks();
  EXPECT_EQ(2, tasks_in_batch.size());
  EXPECT_TRUE(batch.empty());

  EXPECT_EQ(task0, tasks_in_batch[0].get());
  EXPECT_EQ(task1, tasks_in_batch[1].get());

  // RemoveAllTasks returns empty vector from the second call and on, since
  // batch is closed.
  EXPECT_THAT(batch.RemoveAllTasks(), ::testing::IsEmpty());  // second call
  EXPECT_THAT(batch.RemoveAllTasks(), ::testing::IsEmpty());  // third call
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
