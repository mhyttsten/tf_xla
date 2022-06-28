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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
using ::absl::StrCat;
using ::testing::ElementsAre;

struct TestParam {
  bool static_engine;
};

class TRTEngineOpTestBase : public OpsTestBase {
 public:
  void AddSimpleTrtOp(DataType dtype, int max_cached_engines_count = 1,
                      PartialTensorShape shape = PartialTensorShape({-1, -1}),
                      bool use_implicit_batch = true,
                      bool allow_build_at_runtime = true,
                      bool static_engine = false) {
    // Create the GPU device.
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

    // Create simple TF graph.
    Scope s = Scope::NewRootScope();
    auto feed = ops::_Arg(s.WithOpName("TensorRTInputPH_0"), dtype, 0);
    auto add = ops::Add(s.WithOpName("add"), feed, feed);
    ops::_Retval(s.WithOpName("TensorRTOutputPH_0"), add, 0);

    // Serialize the graph. TRTEngineOp will convert it using dynamic mode.
    GraphDef graph_def;
    TF_ASSERT_OK(s.ToGraphDef(&graph_def));
    Graph* graph = s.graph();
    TF_ASSERT_OK(convert::RegisterGraphToFunctionLibrary(graph_def, graph,
                                                         std::string(kOpName)));
    TF_ASSERT_OK(flib_def_->AddLibrary(graph->flib_def()));

    string segment_string;
    if (static_engine) {
      convert::TRTOptimizationPass::ConversionParams params;
      convert::EngineInfo info;
      info.segment_graph_def.CopyFrom(graph_def);
      info.precision_mode = TrtPrecisionMode::FP32;
      info.max_workspace_size_bytes = 1 << 20;
      info.engine_name = "TRTEngineOP_000_000";
      params.use_implicit_batch = use_implicit_batch;
      params.trt_logger_name = "DefaultLogger";

      TrtShapeOptimizationProfile profile;
      TensorShape my_shape;
      // We set profile 0 to be incompatible with the input used in the test.
      // This way we ensure that profile selection is tested.
      TF_CHECK_OK(
          TensorShapeUtils::MakeShape(std::vector<int32>{4, 2}, &my_shape));
      profile.AddShape({my_shape, {}});
      TF_CHECK_OK(
          TensorShapeUtils::MakeShape(std::vector<int32>{1, 2}, &my_shape));
      profile.AddShape({my_shape, {}});

      profile.InitProfiles({shape}, ProfileStrategy::kOptimal);
      std::vector<PartialTensorShape> shape_vec{shape, {}};
      TF_CHECK_OK(convert::CreateStaticEngine(
          params, info, 1, shape_vec, &profile, &segment_string, nullptr));
    }

    // Create the op.
    // In implicit batch mode, the input shapes that we specify here are not
    // used for engine creation, we use the concrete shapes during inference
    // time for creating the engine.
    // In explicit batch mode, the input shapes attribute is used to define
    // the network for the TensorRT engine.
    OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
    NameAttrList function;
    function.set_name(StrCat(std::string(kOpName), "_native_segment"));
    // We disable allow_soft_placement when executing the native segment of the
    // TRTEngineOp for the following reasons:
    //    OpsTestBase only allow one device in the device manager.
    //    We need to define the GPU device to test TRTEngineOp.
    //    When allow_soft_placement is true, the TensorFlow runtime produces an
    //      error if a CPU device is not defined
    //      (see ProcessFunctionLibraryRuntime::InstantiateMultiDevice).
    TF_ASSERT_OK(NodeDefBuilder(std::string(kOpName), "TRTEngineOp")
                     .Input(FakeInput(1, dtype))
                     .Attr("input_shapes", {shape})
                     .Attr("output_shapes", {shape})
                     .Attr("static_engine", static_engine)
                     .Attr("segment_func", function)
                     .Attr("serialized_segment", segment_string)
                     .Attr("calibration_data", "")
                     .Attr("max_cached_engines_count", max_cached_engines_count)
                     .Attr("workspace_size_bytes", 1 << 20)
                     .Attr("precision_mode", "FP32")
                     .Attr("use_calibration", false)
                     .Attr("profile_strategy", "optimal")
                     .Attr("_use_implicit_batch", use_implicit_batch)
                     .Attr("_allow_build_at_runtime", allow_build_at_runtime)
                     .Attr("_allow_soft_placement", false)
                     .Attr("OutT", {dtype})
                     .Finalize(OpsTestBase::node_def()));
    TF_ASSERT_OK(InitOpWithFunctionLibrary());
  }

  static const absl::string_view kOpName;

  template <typename T>
  void AddSimpleInput(const TensorShape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc mht_0(mht_0_v, 327, "", "./tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op_test.cc", "AddSimpleInput");

    std::vector<T> input(shape.num_elements());
    std::iota(input.begin(), input.end(), T(0));
    OpsTestBase::AddInputFromArray<T>(shape, input);
  }

  void ResetInputs() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc mht_1(mht_1_v, 336, "", "./tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op_test.cc", "ResetInputs");

    inputs_.clear();
    for (auto& temp : tensors_) {
      delete temp;
    }
    tensors_.clear();
  }

 private:
  Status InitOpWithFunctionLibrary() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc mht_2(mht_2_v, 348, "", "./tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op_test.cc", "InitOpWithFunctionLibrary");

    OpKernel* kernel = nullptr;
    auto flr = pflr_->GetFLR(device_->name());
    std::shared_ptr<const NodeProperties> props;
    Status status = NodeProperties::CreateFromNodeDef(
        node_def_, flr->GetFunctionLibraryDefinition(), &props);
    if (status.ok()) {
      status.Update(CreateOpKernel(device_type_, device_, allocator(), flr,
                                   props, TF_GRAPH_DEF_VERSION, &kernel));
    }
    kernel_ = std::unique_ptr<OpKernel>(kernel);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }
};

class TRTEngineOpTestWithParam
    : public TRTEngineOpTestBase,
      public ::testing::WithParamInterface<TestParam> {
 public:
  TRTEngineOpTestWithParam() : param_(GetParam()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSkernelsPStrt_engine_op_testDTcc mht_3(mht_3_v, 371, "", "./tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op_test.cc", "TRTEngineOpTestWithParam");
}

 protected:
  TestParam param_;
};

const absl::string_view TRTEngineOpTestBase::kOpName = "myop";

constexpr std::array<TestParam, 2> TestParameters{TestParam{false},
                                                  TestParam{true}};

INSTANTIATE_TEST_CASE_P(TRTEngineOpTestInstantiation, TRTEngineOpTestWithParam,
                        ::testing::ValuesIn(TestParameters));

TEST_F(TRTEngineOpTestBase, DynamicEngines) {
  // Test dynamic engine creation during inference time
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/4);

  // Execute the op with batch size > 1.
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({2, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(device_->resource_manager()->Lookup(
      std::string(kTfTrtContainerName), std::string(kOpName), &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // It should contain only one engine.
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));

  // Execute the op with batch size 1. It should reuse existing engine to
  // execute.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(1, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));

  // Execute the op with a larger batch size.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({3, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(2, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({3, 2})}));

  // Execute the op with an input that has different non-batch dimension.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({10, 10}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  // Execute it again with an input that has the same non-batch dimension but
  // smallest batch size. It should find the correct engine to use.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 10}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(3, cache->size());  // Should only create 3 engines in total.
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({3, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({10, 10})}));
}

TEST_F(TRTEngineOpTestBase, AllowBuildAtRuntime) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      PartialTensorShape({-1, -1}),
                                      /*use_implicit_batch=*/true,
                                      /*allow_build_at_runtime=*/false);

  // Execute the op
  TensorShape input_shape({2, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(device_->resource_manager()->Lookup(
      std::string(kTfTrtContainerName), std::string(kOpName), &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // It should contain a placeholder with an empty cuda_engine (to mark that
  // engine creation was not successful for the given input shape).
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  ASSERT_EQ(1, cache->count({input_shape}));
  EngineContext* ectx = cache->at({input_shape}).get();
  EXPECT_EQ(ectx->cuda_engine, nullptr);
}

TEST_P(TRTEngineOpTestWithParam, ExplicitBatch) {
  // Test inference in explicit batch mode with static input shapes. Static
  // shapes in this context means that the TensorRT knows all the input shapes
  // during engine creation time.
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      /*shape=*/PartialTensorShape({1, 2}),
                                      /*use_implicit_batch=*/false,
                                      /*allow_build_at_runtime=*/true,
                                      /*static_engine=*/param_.static_engine);

  TensorShape input_shape({1, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(device_->resource_manager()->Lookup(
      std::string(kTfTrtContainerName), std::string(kOpName), &cache_resource));
  core::ScopedUnref sc(cache_resource);

  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  ASSERT_EQ(1, cache->count({input_shape}));
  EngineContext* ectx = cache->at({input_shape}).get();
  EXPECT_NE(ectx->cuda_engine, nullptr);
}

TEST_P(TRTEngineOpTestWithParam, DynamicShapes) {
  // Test inference in explicit batch mode with dynamic input shapes. Dynamic
  // shapes in this context means that some input shapes for TensorRT are
  // unknown during engine creation time. When we create the network, the
  // unknow shapes are repsesented as -1. Before we run inference, these shapes
  // have to be specified by calling setBindingDimensions.
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      /*shape=*/PartialTensorShape({-1, -1}),
                                      /*use_implicit_batch=*/false,
                                      /*allow_build_at_runtime=*/true,
                                      param_.static_engine);

  TensorShape input_shape({1, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);

  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(device_->resource_manager()->Lookup(
      std::string(kTfTrtContainerName), std::string(kOpName), &cache_resource));
  core::ScopedUnref sc(cache_resource);

  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  ASSERT_EQ(1, cache->count({input_shape}));
  EngineContext* ectx = cache->at({input_shape}).get();
  EXPECT_NE(ectx->cuda_engine, nullptr);

  // Execute the op with an incompatible shape.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 37}));
  // Test that the op runs. This should fall back to native segment.
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  // We should still have a single engine that is not compatible with the input.
  EXPECT_EQ(1, cache->size());
  EXPECT_EQ(0, cache->count({TensorShape({1, 37})}));
}

template <typename T>
class TRTEngineOpTest : public TRTEngineOpTestBase {};

using TypeList = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(TRTEngineOpTest, TypeList);

TYPED_TEST(TRTEngineOpTest, Basic) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DataTypeToEnum<TypeParam>::v());

  // Execute the op.
  OpsTestBase::AddInputFromArray<TypeParam>(TensorShape({1, 2}),
                                            {TypeParam(0.0f), TypeParam(1.0f)});
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Verify the result.
  Tensor* output = OpsTestBase::GetOutput(0);
  EXPECT_THAT(
      absl::Span<const TypeParam>(output->template flat<TypeParam>().data(),
                                  output->NumElements()),
      ElementsAre(TypeParam(0.0f), TypeParam(2.0f)));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
