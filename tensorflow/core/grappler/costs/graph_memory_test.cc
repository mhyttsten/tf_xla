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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memory_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memory_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memory_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphMemoryTest : public ::testing::Test {
 protected:
  std::unordered_map<string, DeviceProperties> devices_;

 public:
  GraphMemoryTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memory_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/costs/graph_memory_test.cc", "GraphMemoryTest");

    devices_["/CPU:0"].set_type("CPU");
    devices_["/CPU:0"].set_num_cores(1);
    devices_["/CPU:0"].set_frequency(1);
    devices_["/CPU:0"].set_bandwidth(1);

    devices_["/GPU:0"].set_type("GPU");
    devices_["/GPU:0"].set_num_cores(1);
    devices_["/GPU:0"].set_frequency(1);
    devices_["/CPU:0"].set_bandwidth(1);
    (*devices_["/GPU:0"].mutable_environment())["architecture"] = "3";
  }
};

TEST_F(GraphMemoryTest, Basic) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"/CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);
  const GraphMemory::MemoryUsage& mem_usage =
      memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(120, mem_usage.used_memory);

  std::set<string> tensors;
  for (const auto& t : mem_usage.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  // When the execution of the 'Sign' node completes, TF can start executing
  // 'Sign_1' and release the memory used by 'x'. Since we can't be sure of
  // the order in which this takes place, in the worst case the 3 tensors are in
  // memory.
  std::set<string> expected;
  expected.insert("Sign:0");
  expected.insert("Sign_1:0");
  expected.insert("x:0");
  EXPECT_EQ(expected, tensors);
}

TEST_F(GraphMemoryTest, UnknownBatchSize) {
  TrivialTestGraphInputYielder fake_input(4, 1, -1, false, {"/CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);
  // Same maths as before, except that batch size is unknown and therefore
  // assumed to be one.
  const GraphMemory::MemoryUsage& mem_usage =
      memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(16, mem_usage.used_memory);

  std::set<string> tensors;
  for (const auto& t : mem_usage.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> expected;
  expected.insert("Const/Const:0");
  expected.insert("Sign:0");
  expected.insert("x:0");
  EXPECT_EQ(expected, tensors);
}

TEST_F(GraphMemoryTest, MultiDevice) {
  TrivialTestGraphInputYielder fake_input(4, 2, 1024 * 1024, false,
                                          {"/CPU:0", "/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);

  const GraphMemory::MemoryUsage& cpu_mem = memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(16777216, cpu_mem.used_memory);
  std::set<string> cpu_tensors;
  for (const auto& t : cpu_mem.live_tensors) {
    cpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> cpu_expected;
  cpu_expected.insert("Recv_Sign_1_0_on_/CPU_0:0");
  cpu_expected.insert("Sign:0");
  cpu_expected.insert("x:0");
  cpu_expected.insert("AddN:0");
  EXPECT_EQ(cpu_expected, cpu_tensors);

  const GraphMemory::MemoryUsage& gpu_mem = memory.GetPeakMemoryUsage("/GPU:0");
  EXPECT_EQ(16777216, gpu_mem.used_memory);
  std::set<string> gpu_tensors;
  for (const auto& t : gpu_mem.live_tensors) {
    gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> gpu_expected;
  gpu_expected.insert("Recv_AddN_0_on_/GPU_0:0");
  gpu_expected.insert("Sign_1:0");
  gpu_expected.insert("AddN_1:0");
  gpu_expected.insert("AddN_3:0");
  EXPECT_EQ(gpu_expected, gpu_tensors);
}

TEST_F(GraphMemoryTest, GpuSwapping) {
  TrivialTestGraphInputYielder fake_input(4, 2, 1024 * 1024, false, {"/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  {
    // Estimate the max memory usage for the graph.
    GraphMemory memory(item);
    Status s = memory.InferStatically(devices_);
    TF_CHECK_OK(s);

    const GraphMemory::MemoryUsage& gpu_mem =
        memory.GetPeakMemoryUsage("/GPU:0");
    EXPECT_EQ(20971520, gpu_mem.used_memory);
    std::set<string> gpu_tensors;
    for (const auto& t : gpu_mem.live_tensors) {
      gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
    }
    std::set<string> gpu_expected;
    gpu_expected.insert("Sign:0");
    gpu_expected.insert("Sign_1:0");
    gpu_expected.insert("AddN:0");
    gpu_expected.insert("AddN_1:0");
    gpu_expected.insert("AddN_2:0");
    EXPECT_EQ(gpu_expected, gpu_tensors);
  }

  {
    // Swap the first input to node AddN_1: its fanin (the square nodes) should
    // not appear in the max cut anymore.
    for (auto& node : *item.graph.mutable_node()) {
      if (node.name() == "AddN_1") {
        (*node.mutable_attr())["_swap_to_host"].mutable_list()->add_i(0);
      }
    }
    GraphMemory memory(item);
    Status s = memory.InferStatically(devices_);
    TF_CHECK_OK(s);
    const GraphMemory::MemoryUsage& new_gpu_mem =
        memory.GetPeakMemoryUsage("/GPU:0");
    EXPECT_EQ(20971520, new_gpu_mem.used_memory);
    std::set<string> new_gpu_tensors;
    for (const auto& t : new_gpu_mem.live_tensors) {
      new_gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
    }
    std::set<string> new_gpu_expected;
    new_gpu_expected.insert("AddN:0");
    new_gpu_expected.insert("AddN_1:0");
    new_gpu_expected.insert("AddN_2:0");
    new_gpu_expected.insert("AddN_3:0");
    new_gpu_expected.insert("AddN_4:0");
    EXPECT_EQ(new_gpu_expected, new_gpu_tensors);
  }
}

TEST_F(GraphMemoryTest, CtrlDependencies) {
  // Build a simple graph with a control dependency.
  Scope s = Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a").WithDevice("/CPU:0"), 10.0f, {3});
  Output v =
      ops::Variable(s.WithOpName("v").WithDevice("/CPU:0"), {3}, DT_FLOAT);
  Output assign =
      ops::Assign(s.WithOpName("assign").WithDevice("/CPU:0"), v, a);
  ops::NoOp init(
      s.WithOpName("init").WithDevice("/CPU:0").WithControlDependencies(
          assign));

  GrapplerItem item;
  item.fetch.push_back("init");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphMemory memory(item);
  Status status = memory.InferStatically(devices_);
  TF_CHECK_OK(status);

  const GraphMemory::MemoryUsage& mem = memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(36, mem.used_memory);
  std::set<string> tensors;
  for (const auto& t : mem.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> expected;
  expected.insert("a:0");
  expected.insert("v:0");
  expected.insert("assign:0");
  EXPECT_EQ(expected, tensors);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
