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
class MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc() {
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

#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ScopedAllocatorOpTest : public OpsTestBase {
 protected:
  void MakeOp(const TensorShape& shape,
              const gtl::ArraySlice<TensorShape> shapes, DataType dtype,
              const string& name, int32_t id, int32_t expected_call_count) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "MakeOp");

    TF_EXPECT_OK(NodeDefBuilder("scoped_allocator_op", "_ScopedAllocator")
                     .Attr("T", dtype)
                     .Attr("shape", shape)
                     .Attr("shapes", shapes)
                     .Attr("sa_name", name)
                     .Attr("id", id)
                     .Attr("expected_call_count", expected_call_count)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    TF_ASSERT_OK(RunOpKernel());

    // Allocate and Deallocate the tensors so that memory is not leaked
    AllocatorAttributes attr;
    Allocator* allocator;
    for (size_t i = 0; i < shapes.size(); i++) {
      attr.scope_id = id + i + 1;
      allocator = device_->GetScopedAllocator(attr, context_->step_id());
      Tensor temp(allocator, dtype, shapes[i]);
    }
  }
};

TEST_F(ScopedAllocatorOpTest, Simple) {
  MakeOp(TensorShape({8}), {TensorShape({8})}, DT_FLOAT, "test", 120, 1);
  MakeOp(TensorShape({1024}), {TensorShape({32, 32})}, DT_DOUBLE, "test1", 130,
         1);
  MakeOp(TensorShape({204}),
         {TensorShape({64}), TensorShape({3, 3}), TensorShape({5, 5, 5})},
         DT_HALF, "test2", 140, 3);
  MakeOp(TensorShape({1024}), {TensorShape({512}), TensorShape({64, 8})},
         DT_UINT32, "test3", 150, 2);
}

// PrepOp is common to ConcatOp tests and SplitOpTests.
// It allocates a backing tensor that is large enough to hold all slices defined
// by fields, creates ScopedAllocatorInstances for each field, allocates the
// tensors, and assigns them as inputs to the op.
// We won't use the AddInput* suite of functions from ops_testutil.h because
// they allocate new tensors for each input.  We need to mimic what a
// ScopedAllocator would do.
void PrepOp(DataType dtype, int32_t id,
            const std::vector<TensorShape>& fields_shapes,
            std::vector<ScopedAllocator::Field>* fields,
            Tensor** backing_tensor, Allocator* allocator,
            ScopedAllocatorMgr* sam, const string& op_name,
            std::vector<Tensor>* tensors,
            gtl::InlinedVector<TensorValue, 4>* inputs,
            const DataTypeVector& input_types) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_1(mht_1_v, 265, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "PrepOp");

  ScopedAllocatorMgr::PopulateFields(id, fields_shapes, dtype, fields);
  // We don't simply allocate a tensor with shape as backing_tensor_shape,
  // because we need to account for padding in the fields.  We actually need a
  // tensor of size at least (fields[-1].offset + fields[-1].bytes_allocated).
  size_t num_bytes = fields->back().offset + fields->back().bytes_allocated;
  int32_t num_elements = num_bytes / DataTypeSize(dtype);
  CHECK_EQ(num_bytes % DataTypeSize(dtype), 0);

  *backing_tensor = new Tensor(allocator, dtype, {num_elements});
  int64_t step_id = 10;
  Status s = sam->AddScopedAllocator(**backing_tensor, step_id, id,
                                     "sa_" + op_name + "_test", *fields,
                                     fields_shapes.size());
  TF_ASSERT_OK(s);

  ScopedAllocatorContainer* sac = sam->GetContainer(step_id);
  std::vector<ScopedAllocatorInstance*> sa_instances(fields_shapes.size(),
                                                     nullptr);
  for (size_t i = 0; i < fields_shapes.size(); i++) {
    sa_instances[i] = sac->GetInstance(id + i + 1);
    tensors->push_back(Tensor(sa_instances[i], dtype, fields_shapes[i]));
  }
  // Now add the tensor as an input to ScopedAllocator<op_name>Op.
  // Order matters here, so first add the backing tensor, then the slices.
  inputs->reserve(1 + tensors->size());
  CHECK_GT(input_types.size(), inputs->size());
  CHECK_EQ(input_types[inputs->size()], dtype);
  inputs->push_back({nullptr, *backing_tensor});
  for (size_t i = 0; i < tensors->size(); i++) {
    CHECK_EQ(input_types[inputs->size()], dtype);
    inputs->push_back({nullptr, &((*tensors)[i])});
  }
}

class ScopedAllocatorConcatOpTest : public OpsTestBase {
 protected:
  void BuildNodeDef(const TensorShape& shape, DataType dtype,
                    const string& name, int32_t id, int32_t num_tensors) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_2(mht_2_v, 307, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "BuildNodeDef");

    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_concat_op", "_ScopedAllocatorConcat")
            .Attr("shape", shape)
            .Attr("T", dtype)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Input(FakeInput(dtype))               // backing tensor
            .Input(FakeInput(num_tensors, dtype))  // list of tensors
            .Finalize(node_def()));
    shape_ = shape;
    reshape_ = false;
  }

  void BuildNodeDefWithReshape(const TensorShape& shape, DataType dtype,
                               bool reshape, const string& name, int32_t id,
                               int32_t num_tensors) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_3(mht_3_v, 328, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "BuildNodeDefWithReshape");

    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_concat_op", "_ScopedAllocatorConcat")
            .Attr("shape", shape)
            .Attr("T", dtype)
            .Attr("reshape", reshape)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Input(FakeInput(dtype))               // backing tensor
            .Input(FakeInput(num_tensors, dtype))  // list of tensors
            .Finalize(node_def()));
    shape_ = shape;
    reshape_ = reshape;
  }

  void MakeOp(const TensorShape& shape, DataType dtype, bool reshape,
              const string& name, int32_t id, int32_t num_tensors) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_4(mht_4_v, 349, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "MakeOp");

    BuildNodeDefWithReshape(shape, dtype, reshape, name, id, num_tensors);
    TF_EXPECT_OK(InitOp());
  }

  void ExecOp(DataType dtype, int32_t id,
              const std::vector<TensorShape>& fields_shapes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_5(mht_5_v, 358, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "ExecOp");

    Tensor* backing_tensor = nullptr;
    std::vector<Tensor> tensors;
    std::vector<ScopedAllocator::Field> fields;
    PrepOp(dtype, id, fields_shapes, &fields, &backing_tensor, allocator(),
           device_->GetScopedAllocatorMgr(), "concat", &tensors, &inputs_,
           input_types_);

    TF_ASSERT_OK(RunOpKernel());

    // Check input and output are same tensor.
    const Tensor& input = context_->input(0);
    OpOutputList output_list;
    Status s = context_->output_list("output", &output_list);
    TF_ASSERT_OK(s);
    const Tensor& output = *(output_list[0]);
    CHECK_EQ(DMAHelper::base(&input), DMAHelper::base(&output));
    CHECK_EQ(input.dtype(), output.dtype());
    CHECK_EQ(input.NumElements(), output.NumElements());
    if (reshape_) {
      CHECK_EQ(shape_, output.shape());
    } else {
      TensorShape expected_shape({input.NumElements()});
      CHECK_EQ(expected_shape, output.shape());
    }

    // Free the backing tensor which was allocated in PrepOp.
    delete backing_tensor;
  }

 private:
  TensorShape shape_;
  bool reshape_;
};

TEST_F(ScopedAllocatorConcatOpTest, Success1) {
  MakeOp({32}, DT_FLOAT, false, "test", 120, 2);
  ExecOp(DT_FLOAT, 120, {{16}, {16}});
}

TEST_F(ScopedAllocatorConcatOpTest, Success2) {
  MakeOp({2, 2, 2}, DT_DOUBLE, false, "test", 120, 2);
  ExecOp(DT_DOUBLE, 120, {{2, 2}, {2, 2}});
}

TEST_F(ScopedAllocatorConcatOpTest, Success3) {
  MakeOp({3, 3, 3}, DT_HALF, false, "test", 120, 3);
  ExecOp(DT_HALF, 120, {{3, 3}, {3, 3}, {3, 3}});
}

TEST_F(ScopedAllocatorConcatOpTest, Reshape) {
  MakeOp({2, 2, 4}, DT_DOUBLE, true, "test", 120, 2);

  // The elements of the third parameter to ExecOp must be multiples of
  // Allocator::kAllocatorAlignment in size.  If they are not, the backing
  // tensor allocated by PrepOp will have too many elements and reshaping
  // will fail.
  ExecOp(DT_DOUBLE, 120, {{2, 4}, {2, 4}});
}

TEST_F(ScopedAllocatorConcatOpTest, NoReshapeAttr) {
  BuildNodeDef({3, 4, 4}, DT_HALF, "test", 120, 3);
  TF_EXPECT_OK(InitOp());
  ExecOp(DT_HALF, 120, {{4, 4}, {4, 4}, {4, 4}});
}

TEST_F(ScopedAllocatorConcatOpTest, FailDtypeCheck) {
  MakeOp({8}, DT_FLOAT, false, "test", 120, 2);
  EXPECT_DEATH(ExecOp(DT_DOUBLE, 120, {{4}, {4}}), "");
}

TEST_F(ScopedAllocatorConcatOpTest, FailNumElementsCheck) {
  MakeOp({32}, DT_FLOAT, false, "test", 120, 2);
  AddInputFromArray<float>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<float>({4}, {0, 1, 2, 3});
  AddInputFromArray<float>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

// This test should fail because the backing tensor and the input tensors are
// unrelated, i.e. the inputs are not slices of the backing tensor.
TEST_F(ScopedAllocatorConcatOpTest, FailBounds) {
  MakeOp({8}, DT_DOUBLE, false, "test", 120, 2);
  AddInputFromArray<double>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<double>({4}, {0, 1, 2, 3});
  AddInputFromArray<double>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

class ScopedAllocatorSplitOpTest : public OpsTestBase {
 protected:
  void BuildNodeDef(const TensorShape& in_shape, DataType dtype,
                    const string& name, int32_t id, int32_t num_tensors,
                    const std::vector<TensorShape>& out_shapes) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_6(mht_6_v, 457, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "BuildNodeDef");

    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_split_op", "_ScopedAllocatorSplit")
            .Attr("T", dtype)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Attr("shapes", out_shapes)
            .Input(FakeInput(dtype))  // backing tensor and input
            .Input(
                FakeInput(num_tensors, dtype))  // list of subtensors to forward
            .Finalize(node_def()));
  }

  void MakeOp(const TensorShape& in_shape, DataType dtype, const string& name,
              int32_t id, int32_t num_tensors,
              const std::vector<TensorShape>& out_shapes) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_7(mht_7_v, 477, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "MakeOp");

    BuildNodeDef(in_shape, dtype, name, id, num_tensors, out_shapes);
    TF_EXPECT_OK(InitOp());
  }

  // Similar to ConcatOpTest, we add inputs that are allocated from
  // ScopedAllocator so that the memory lines up nicely.
  void ExecOp(DataType dtype, int32_t id,
              const std::vector<TensorShape>& fields_shapes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_ops_testDTcc mht_8(mht_8_v, 488, "", "./tensorflow/core/kernels/scoped_allocator_ops_test.cc", "ExecOp");

    Tensor* backing_tensor = nullptr;
    std::vector<Tensor> tensors;
    std::vector<ScopedAllocator::Field> fields;
    PrepOp(dtype, id, fields_shapes, &fields, &backing_tensor, allocator(),
           device_->GetScopedAllocatorMgr(), "split", &tensors, &inputs_,
           input_types_);

    TF_ASSERT_OK(RunOpKernel());

    // Check that outputs are slices of backing tensor.
    const Tensor& input = context_->input(0);
    const void* lower_limit = DMAHelper::base(&input);
    const char* lower_limit_c =
        static_cast<const char*>(lower_limit);  // for pointer arithmetic
    OpOutputList output_list;
    Status s = context_->output_list("output", &output_list);
    TF_ASSERT_OK(s);
    for (int i = 0; i < output_list.size(); i++) {
      const Tensor& output = *(output_list[i]);
      const void* expected_base =
          static_cast<const void*>(lower_limit_c + fields[i].offset);
      CHECK_EQ(output.dtype(), input.dtype());
      CHECK_EQ(expected_base, DMAHelper::base(&output));
      CHECK_EQ(output.NumElements(), fields_shapes[i].num_elements());
    }

    // Free the backing tensor which was allocated in PrepOp.
    delete backing_tensor;
  }
};

TEST_F(ScopedAllocatorSplitOpTest, Success1) {
  MakeOp({32}, DT_FLOAT, "test", 120, 2, {{16}, {16}});
  ExecOp(DT_FLOAT, 120, {{16}, {16}});
}

TEST_F(ScopedAllocatorSplitOpTest, Success2) {
  MakeOp({2, 2, 2}, DT_DOUBLE, "test", 120, 2, {{2, 2}, {2, 2}});
  ExecOp(DT_DOUBLE, 120, {{2, 2}, {2, 2}});
}

TEST_F(ScopedAllocatorSplitOpTest, Success3) {
  MakeOp({3, 3, 3}, DT_HALF, "test", 120, 3, {{3, 3}, {3, 3}, {3, 3}});
  ExecOp(DT_HALF, 120, {{3, 3}, {3, 3}, {3, 3}});
}

TEST_F(ScopedAllocatorSplitOpTest, FailNLessThan2) {
  BuildNodeDef({4, 4}, DT_FLOAT, "test", 120, 1, {{4, 4}});
  Status s = InitOp();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

TEST_F(ScopedAllocatorSplitOpTest, FailDtypeCheck) {
  MakeOp({8}, DT_FLOAT, "test", 120, 2, {{4}, {4}});
  EXPECT_DEATH(ExecOp(DT_HALF, 120, {{4}, {4}}), "");
}

TEST_F(ScopedAllocatorSplitOpTest, FailBounds) {
  MakeOp({8}, DT_DOUBLE, "test", 120, 2, {{4}, {4}});
  AddInputFromArray<double>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<double>({4}, {0, 1, 2, 3});
  AddInputFromArray<double>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

}  // end namespace tensorflow
