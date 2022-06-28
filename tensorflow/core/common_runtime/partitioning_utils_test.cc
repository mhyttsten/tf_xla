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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc() {
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

#include "tensorflow/core/common_runtime/partitioning_utils.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class PartitioningUtilsTest : public ::testing::Test {
 public:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "SetUp");

    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 2});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &devices));
    device0_ = devices[0].get();
    device1_ = devices[1].get();
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));

    for (auto d : device_mgr_->ListDevices()) {
      device_set_.AddDevice(d);
    }
  }

  void SwapGraph(Graph* graph, bool assign_device = false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "SwapGraph");

    Scope s = Scope::NewRootScope();
    if (assign_device) {
      s = s.WithDevice(device0_->name());
    }
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto id_x = ops::Identity(s.WithOpName("id_x"), x);
    auto id_y = ops::Identity(s.WithOpName("id_y"), y);
    auto dx_retval = ops::_Retval(s.WithOpName("retval1"), id_y, 0);
    auto dy_retval = ops::_Retval(s.WithOpName("retval2"), id_x, 1);
    TF_ASSERT_OK(s.ToGraph(graph));

    if (assign_device) {
      FunctionLibraryDefinition flib_def(OpRegistry::Global());
      Placer placer(graph, "", &flib_def, &device_set_, device0_);
      TF_ASSERT_OK(placer.Run());
    }
  }

  void TwoDeviceSwapGraph(Graph* graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "TwoDeviceSwapGraph");

    Scope s = Scope::NewRootScope();
    Scope s1 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:0");
    Scope s2 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:1");
    auto x = ops::_Arg(s1.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s2.WithOpName("y"), DT_FLOAT, 1);
    auto id_x = ops::Identity(s1.WithOpName("id_x"), x);
    auto id_y = ops::Identity(s2.WithOpName("id_y"), y);
    auto dx_retval = ops::_Retval(s2.WithOpName("retval1"), id_y, 0);
    auto dy_retval = ops::_Retval(s1.WithOpName("retval2"), id_x, 1);
    TF_ASSERT_OK(s.ToGraph(graph));
    FunctionLibraryDefinition flib_def(OpRegistry::Global());
    Placer placer(graph, "", &flib_def, &device_set_, device0_);
    TF_ASSERT_OK(placer.Run());
  }

  // Fills subgraph with an identify function arg->identity->ret
  // where each node has type `dtype` and arg/ret nodes have
  // indices `arg_index` and `ret_index`.
  void SubGraph(Graph* subgraph, DataType dtype,
                gtl::ArraySlice<int> arg_indices,
                gtl::ArraySlice<int> ret_indices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "SubGraph");

    Scope s = Scope::NewRootScope();
    Scope s1 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:0");
    CHECK_EQ(arg_indices.size(), ret_indices.size());
    for (size_t i = 0; i < arg_indices.size(); ++i) {
      auto x = ops::_Arg(s1.WithOpName("x"), dtype, arg_indices[i]);
      auto id_x = ops::Identity(s1.WithOpName("id_x"), x);
      auto dx_retval =
          ops::_Retval(s1.WithOpName("retval1"), id_x, ret_indices[i]);
    }
    TF_ASSERT_OK(s.ToGraph(subgraph));
    FunctionLibraryDefinition flib_def(OpRegistry::Global());
    Placer placer(subgraph, "", &flib_def, &device_set_, device0_);
    TF_ASSERT_OK(placer.Run());
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device0_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  Device* device1_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  DeviceSet device_set_;
};

TEST_F(PartitioningUtilsTest, GraphWithoutAssignedDevicesFails) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  SwapGraph(graph.get());

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(errors::IsInvalidArgument(status)) << status.ToString();
}

TEST_F(PartitioningUtilsTest, OneDevice) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  SwapGraph(graph.get(), true);
  int num_nodes = graph->num_op_nodes();

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_EQ(1, subgraphs.size());
  const auto& pair = *subgraphs.begin();
  ASSERT_EQ("/job:a/replica:0/task:0/device:CPU:0", pair.first);
  ASSERT_EQ(num_nodes, pair.second->num_op_nodes());
}

TEST_F(PartitioningUtilsTest, TwoDevices) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  TwoDeviceSwapGraph(graph.get());

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_EQ(2, subgraphs.size());

  const auto& part1 = subgraphs["/job:a/replica:0/task:0/device:CPU:0"];
  ASSERT_EQ(3, part1->num_op_nodes());
  const auto& part2 = subgraphs["/job:a/replica:0/task:0/device:CPU:1"];
  ASSERT_EQ(3, part2->num_op_nodes());
}

TEST_F(PartitioningUtilsTest, InsertTransferOpsWithOneDevice) {
  // A graph with three nodes that are on the same device.
  // x(_Arg, device0) -> id_x(Identity, device0) -> ret_x(_Retval, device0)
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  Scope scope = Scope::NewRootScope().WithDevice(device0_->name());

  auto x = ops::_Arg(scope.WithOpName("x"), DT_FLOAT, 0);
  auto id_x = ops::Identity(scope.WithOpName("id_x"), x);
  auto ret_x = ops::_Retval(scope.WithOpName("ret_x"), id_x, 0);
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global());
  Placer placer(graph.get(), "", &flib_def, &device_set_, device0_);
  TF_ASSERT_OK(placer.Run());

  // No Send/Recv node initially.
  EXPECT_EQ(graph->num_op_nodes(), 3);
  int send_count = 0, recv_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsSend())
      ++send_count;
    else if (op->IsRecv())
      ++recv_count;
  }
  ASSERT_EQ(send_count, 0);
  ASSERT_EQ(recv_count, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> new_graph,
                          InsertTransferOps(device_set_, std::move(graph)));

  // No Send/Recv node is added, as all nodes are on the same device.
  EXPECT_EQ(new_graph->num_op_nodes(), 3);
  send_count = recv_count = 0;
  for (const auto* op : new_graph->op_nodes()) {
    if (op->IsSend())
      ++send_count;
    else if (op->IsRecv())
      ++recv_count;
  }
  EXPECT_EQ(send_count, 0);
  EXPECT_EQ(recv_count, 0);
}

TEST_F(PartitioningUtilsTest, InsertTransferOpsWithTwoDevices) {
  // A graph with three nodes that are on two devices.
  // x(_Arg, device0) -> id_x(Identity, device1) -> ret_x(_Retval, device0)
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  Scope scope = Scope::NewRootScope();
  Scope scope1 = scope.WithDevice(device0_->name());
  Scope scope2 = scope.WithDevice(device1_->name());

  auto x = ops::_Arg(scope1.WithOpName("x"), DT_FLOAT, 0);
  auto id_x = ops::Identity(scope2.WithOpName("id_x"), x);
  auto ret_x = ops::_Retval(scope1.WithOpName("ret_x"), id_x, 0);
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global());
  Placer placer(graph.get(), "", &flib_def, &device_set_, device0_);
  TF_ASSERT_OK(placer.Run());

  // No Send/Recv node initially.
  EXPECT_EQ(graph->num_op_nodes(), 3);
  int send_count = 0, recv_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsSend())
      ++send_count;
    else if (op->IsRecv())
      ++recv_count;
  }
  ASSERT_EQ(send_count, 0);
  ASSERT_EQ(recv_count, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> new_graph,
                          InsertTransferOps(device_set_, std::move(graph)));

  // Two pairs of Send/Recv nodes are inserted.
  EXPECT_EQ(new_graph->num_op_nodes(), 7);
  send_count = recv_count = 0;
  for (const auto* op : new_graph->op_nodes()) {
    if (op->IsSend())
      ++send_count;
    else if (op->IsRecv())
      ++recv_count;
  }
  EXPECT_EQ(send_count, 2);
  EXPECT_EQ(recv_count, 2);
}

void CheckRetIndices(const std::vector<int>& expected,
                     const std::vector<int>& actual) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_4(mht_4_v, 434, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "CheckRetIndices");

  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], actual[i]) << " at index " << i;
  }
}

void CheckArgIndices(const std::vector<FunctionArgIndex>& expected,
                     const std::vector<FunctionArgIndex>& actual) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_5(mht_5_v, 445, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "CheckArgIndices");

  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i].index, actual[i].index) << " at index " << i;
    ASSERT_EQ(expected[i].sub_index, actual[i].sub_index) << " at index " << i;
  }
}

void CheckAlloc(const std::vector<bool>& expected,
                const std::vector<AllocatorAttributes>& actual) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_6(mht_6_v, 457, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "CheckAlloc");

  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], actual[i].on_host()) << " at index " << i;
  }
}

void CheckIndex(const Node& node, int expected_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utils_testDTcc mht_7(mht_7_v, 467, "", "./tensorflow/core/common_runtime/partitioning_utils_test.cc", "CheckIndex");

  const AttrValue* attr_value;
  TF_ASSERT_OK(node.attrs().Find("index", &attr_value));
  int index = static_cast<int>(attr_value->i());
  ASSERT_EQ(expected_index, index);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRets) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_FLOAT, {3}, {5});

  std::vector<FunctionArgIndex> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs, /*ints_on_device=*/false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckArgIndices({{3, -1}}, arg_indices);
  CheckRetIndices({5}, ret_indices);
  CheckAlloc({false}, arg_alloc_attrs);
  CheckAlloc({false}, ret_alloc_attrs);

  std::unordered_map<string, Node*> nodes = graph->BuildNodeNameIndex();
  ASSERT_EQ(1, nodes.count("x"));
  CheckIndex(*nodes["x"], 0);
  ASSERT_EQ(1, nodes.count("retval1"));
  CheckIndex(*nodes["retval1"], 0);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRetsIntsNotOnDevice) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_INT32, {3}, {5});

  std::vector<FunctionArgIndex> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs, /*ints_on_device=*/false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckAlloc({true}, arg_alloc_attrs);
  CheckAlloc({true}, ret_alloc_attrs);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRetsIntsOnDevice) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_INT32, {3}, {5});

  std::vector<FunctionArgIndex> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs, /*ints_on_device=*/true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckAlloc({false}, arg_alloc_attrs);
  CheckAlloc({false}, ret_alloc_attrs);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRets_Order) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_FLOAT, {9, 7, 5, 3, 1}, {2, 4, 6, 8, 10});

  const std::map<int, int> sub_indices = {
      {7, 2}, {3, 1}, {1, 0}, {5, 2}, {9, 0}};
  const AttrValue* attr_value;
  for (Node* n : graph->op_nodes()) {
    if (n->IsArg()) {
      TF_ASSERT_OK(n->attrs().Find("index", &attr_value));
      n->AddAttr("sub_index",
                 sub_indices.at(static_cast<int>(attr_value->i())));
    }
  }

  std::vector<FunctionArgIndex> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs, /*ints_on_device=*/false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckArgIndices({{1, 0}, {3, 1}, {5, 2}, {7, 2}, {9, 0}}, arg_indices);
  CheckRetIndices({2, 4, 6, 8, 10}, ret_indices);
  CheckAlloc({false, false, false, false, false}, arg_alloc_attrs);
  CheckAlloc({false, false, false, false, false}, ret_alloc_attrs);
}

}  // anonymous namespace
}  // namespace tensorflow
