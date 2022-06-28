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
class MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

using ::testing::_;
using testing::matchers::AssignedDevice;
using testing::matchers::Attr;
using testing::matchers::Const;
using testing::matchers::CtrlDeps;
using testing::matchers::Inputs;
using testing::matchers::Name;
using testing::matchers::NodeWith;
using testing::matchers::Op;
using testing::matchers::Out;

// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "FakeDevice");
}

  Status Sync() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "Sync");
 return errors::Unimplemented("FakeDevice::Sync()"); }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "GetAllocator");
 return nullptr; }

  static std::unique_ptr<Device> Make(const string& name, const string& type) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(DeviceType(type).type());
    return absl::make_unique<FakeDevice>(device_attributes);
  }
};

const char* kHostName = "/job:worker/replica:0/task:0/device:CPU:0";
const char* kDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";

Status IncreaseDynamismForAutoJit(const Scope& s,
                                  std::unique_ptr<Graph>* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "IncreaseDynamismForAutoJit");

  std::vector<std::unique_ptr<Device>> devices;
  devices.push_back(FakeDevice::Make(kDeviceName, DEVICE_GPU));
  devices.push_back(FakeDevice::Make(kHostName, DEVICE_CPU));

  std::unique_ptr<DeviceSet> device_set(new DeviceSet());
  for (auto& device : devices) {
    device_set->AddDevice(device.get());
  }

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.device_set = device_set.get();
  options.session_options = &session_options;

  // Scope::ToGraph seems to drop assigned devices, probably because it goes
  // through a GraphDef.  So explicitly maintain the device assignment.
  std::unordered_map<string, string> assigned_device_names;
  for (Node* n : s.graph()->nodes()) {
    assigned_device_names[n->name()] = n->assigned_device_name();
  }
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  for (Node* n : graph->nodes()) {
    n->set_assigned_device_name(assigned_device_names[n->name()]);
  }

  IncreaseDynamismForAutoJitPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return Status::OK();
}

TEST(SliceToDynamicSliceRewriteTest, Basic) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  const int64_t zero_64 = 0;
  const int32_t zero_32 = 0;
  const int64_t one_64 = 1;

  auto m_input = Out(NodeWith(Op("Placeholder"), Name("input")));
  auto m_begin_s64 = Out(NodeWith(
      Op("Cast"), Inputs(Out(NodeWith(Op("Placeholder"), Name("begin"))))));
  auto m_input_shape = Out(NodeWith(Op("Shape"), Inputs(m_input)));
  auto m_slice_size_0 = Out(NodeWith(
      Op("Sub"), AssignedDevice(kHostName),
      Inputs(
          Out(NodeWith(Op("Slice"), AssignedDevice(kHostName),
                       Inputs(m_input_shape, Const(zero_64), Const(one_64)))),
          Out(NodeWith(Op("Slice"), AssignedDevice(kHostName),
                       Inputs(m_begin_s64, Const(zero_64), Const(one_64)))))));
  auto m_dynamic_slice_size =
      Out(NodeWith(Op("ConcatV2"), AssignedDevice(kHostName),
                   Inputs(m_slice_size_0, Const(static_cast<int64_t>(500)),
                          Const(zero_32))));

  std::vector<string> compile_time_constant_inputs;
  compile_time_constant_inputs.push_back("size");
  auto m_dynamic_slice = NodeWith(
      Op("Slice"), AssignedDevice(kDeviceName),
      Attr(kXlaCompileTimeConstantInputsAttr, compile_time_constant_inputs),
      Inputs(m_input, m_begin_s64, m_dynamic_slice_size));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice, m_dynamic_slice);
}

TEST(SliceToDynamicSliceRewriteTest, SliceFromVector) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  EXPECT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(result->nodes(), Not(Contains(NodeWith(Op("ConcatV2")))));
}

TEST(SliceToDynamicSliceRewriteTest, ControlDependencePreserved) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output control_pred = ops::Placeholder(root.WithOpName("control"), DT_BOOL);
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);
  root.graph()->AddControlEdge(control_pred.node(), slice.node());

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice,
              NodeWith(Op("Slice"),
                       CtrlDeps(NodeWith(Op("Placeholder"), Name("control")))));
}

int64_t ToInt64(int v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_4(mht_4_v, 372, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "ToInt64");
 return static_cast<int64_t>(v); }

TEST(SliceToDynamicSliceRewriteTest, Int64Indices) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size =
      ops::Const(root.WithOpName("size"), {ToInt64(-1), ToInt64(500)});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(), Not(Contains(NodeWith(Op("Cast")))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteInvalidSlice) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);

  // The shape refiner throws an error if we use a bogus constant value for
  // size.  So we first use a Placeholder to placate the shape refiner, and
  // later replace it with a bogus constant.
  Output size_placeholder =
      ops::Placeholder(root.WithOpName("size_placeholder"), DT_INT32);
  Output slice =
      ops::Slice(root.WithOpName("slice"), input, begin, size_placeholder);

  Output size = ops::Const(root.WithOpName("size"), {-8, 500});
  TF_ASSERT_OK(root.graph()->UpdateEdge(/*new_src=*/size.node(),
                                        /*new_src_index=*/0,
                                        /*dst=*/slice.node(), /*dst_index=*/2));

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteUnclusteredSlice) {
  Scope root =
      Scope::NewRootScope().ExitOnError().WithAssignedDevice(kDeviceName);

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteSliceWithNonConstSize) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size = ops::Placeholder(root.WithOpName("size"), DT_INT64);
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, ScalarSlice) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size = ops::Const<int64_t>(root.WithOpName("size"), {});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice,
              NodeWith(Op("Slice"), Attr(kXlaCompileTimeConstantInputsAttr),
                       Inputs(_, _, Out(NodeWith(Name(size.node()->name()))))));
}

TEST(SliceToDynamicSliceRewriteTest, IndicesNotVector) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  auto ToInt64 = [](int v) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_pass_testDTcc mht_5(mht_5_v, 489, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass_test.cc", "lambda");
 return static_cast<int64_t>(v); };

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);

  // The C++ node bindings immediately error out when we try construct a bogus
  // slice so we first use a placeholder to construct the Slice and then replace
  // the input.
  Output size_placeholder = ops::Placeholder(root.WithOpName("size"), DT_INT64);
  Output slice =
      ops::Slice(root.WithOpName("slice"), input, begin, size_placeholder);

  Output size =
      ops::Const(root.WithOpName("size"), {{ToInt64(-1)}, {ToInt64(500)}});
  TF_ASSERT_OK(root.graph()->UpdateEdge(size.node(), 0, slice.node(), 2));

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, SliceWithSliceInput) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size_a = ops::Const(root.WithOpName("size_a"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size_a);

  Output size_b = ops::Const(root.WithOpName("size_a"), {-1, 200});
  Output slice_with_slice_input = ops::Slice(
      root.WithOpName("slice_with_slice_input"), slice, begin, size_b);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(),
      "slice_with_slice_input/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_EQ(static_shaped_slice->output_type(0), DT_FLOAT)
      << "Expected DT_FLOAT, was "
      << DataType_Name(static_shaped_slice->output_type(0));
  EXPECT_THAT(
      static_shaped_slice,
      NodeWith(
          Op("Slice"),
          Inputs(Out(NodeWith(
                     Op("Slice"),
                     Name("slice/static_shaped_slice/static_shaped_slice"))),
                 _, _)));
}

TEST(SliceToDynamicSliceRewriteTest, SliceWithSliceBegin) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input_float =
      ops::Placeholder(root.WithOpName("input_float"), DT_FLOAT);
  Output input_i64 = ops::Placeholder(root.WithOpName("input_i64"), DT_INT64);

  Output begin_begin =
      ops::Placeholder(root.WithOpName("begin_begin"), DT_INT32);
  Output begin_size = ops::Const(root.WithOpName("begin_size"), {-1});
  Output begin =
      ops::Slice(root.WithOpName("begin"), input_i64, begin_begin, begin_size);

  Output size =
      ops::Const(root.WithOpName("size"), {ToInt64(-1), ToInt64(200)});
  Output slice_with_slice_begin = ops::Slice(
      root.WithOpName("slice_with_slice_begin"), input_float, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(),
      "slice_with_slice_begin/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_EQ(static_shaped_slice->output_type(0), DT_FLOAT)
      << "Expected DT_FLOAT, was "
      << DataType_Name(static_shaped_slice->output_type(0));
  EXPECT_THAT(
      static_shaped_slice,
      NodeWith(
          Op("Slice"),
          Inputs(_,
                 Out(NodeWith(
                     Op("Slice"),
                     Name("begin/static_shaped_slice/static_shaped_slice"))),
                 _)));
}

// New constants being created need to have control dependencies copied to
// ensure correct control flow analysis in TF V2.
TEST(SliceToDynamicSliceRewriteTest, WithControlDepsToConstant) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  // Add an additional dependency that should still exist in with the new size
  // variables.
  Output dependency = ops::Placeholder(root.WithOpName("dependency"), DT_BOOL);
  root.graph()->AddControlEdge(dependency.node(), size.node());

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  // Check that the new constants have control dependencies.
  Node* const_0 = testing::FindNodeByName(result.get(),
                                          "slice/static_shaped_slice/const_0");
  EXPECT_NE(const_0, nullptr);
  EXPECT_THAT(const_0,
              NodeWith(Op("Const"), CtrlDeps(NodeWith(Op("Placeholder"),
                                                      Name("dependency")))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteSliceWithConstBegin) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Const(root.WithOpName("begin"), {10, 10});
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* slice_node = testing::FindNodeByName(result.get(), "slice");
  EXPECT_THAT(slice_node,
              NodeWith(Op("Slice"), Inputs(Out(NodeWith(Op("Placeholder"))),
                                           Out(NodeWith(Op("Const"))),
                                           Out(NodeWith(Op("Const"))))));
}

}  // namespace
}  // namespace tensorflow
