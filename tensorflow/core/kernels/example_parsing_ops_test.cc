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
class MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc() {
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

#include <unordered_map>

#include "absl/base/call_once.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef std::map<std::tuple<int, int, int>, Tensor> ExampleTensorMap;

// Fillers to fill the underlying repeated array in protobuf.
class BytesFiller {
 public:
  BytesFiller() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "BytesFiller");
}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_bytes_list()->add_value("abcd1234abcd1234abcd1234abcd1234!");
    }
  }
  Tensor make_dense_default(int feature_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "make_dense_default");

    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_STRING;
};

class Int64Filler {
 public:
  Int64Filler() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "Int64Filler");
}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_int64_list()->add_value(1729);
    }
  }
  Tensor make_dense_default(int feature_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "make_dense_default");

    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_INT64;
};

class FloatFiller {
 public:
  FloatFiller() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "FloatFiller");
}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_float_list()->add_value(1.729);
    }
  }
  Tensor make_dense_default(int feature_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_5(mht_5_v, 261, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "make_dense_default");

    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_FLOAT;
};

template <typename T>
struct ExampleStore {
 private:
  static ExampleTensorMap serialized_example;
  static absl::once_flag flags_init;

 public:
  static ExampleTensorMap& GetSerializedExample() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "GetSerializedExample");

    absl::call_once(flags_init, [] {
      AddExample(&serialized_example, 10, 1, 1);
      AddExample(&serialized_example, 100, 1, 1);
      AddExample(&serialized_example, 1000, 1, 1);
      AddExample(&serialized_example, 10, 128, 1);
      AddExample(&serialized_example, 100, 128, 1);
      AddExample(&serialized_example, 1000, 128, 1);
      AddExample(&serialized_example, 10, 512, 1);
      AddExample(&serialized_example, 100, 512, 1);
      AddExample(&serialized_example, 1000, 512, 1);
      AddExample(&serialized_example, 1, 1, 10);
      AddExample(&serialized_example, 1, 1, 100);
      AddExample(&serialized_example, 1, 1, 1000);
      AddExample(&serialized_example, 1, 1, 10000);
      AddExample(&serialized_example, 1, 1, 100000);
      AddExample(&serialized_example, 1, 1, 1000000);
      AddExample(&serialized_example, 10, 1, 100000);
      AddExample(&serialized_example, 100, 1, 10000);
      AddExample(&serialized_example, 1000, 1, 1000);
    });
    return serialized_example;
  }
  typedef T Filler;
  static void AddExample(ExampleTensorMap* examples, int num_keys,
                         int batch_size, int feature_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "AddExample");

    Example example;
    Filler fill;
    Tensor record_string(DT_STRING, TensorShape({batch_size}));
    auto string_t = record_string.vec<tstring>();
    example.Clear();
    for (int b = 0; b < batch_size; ++b) {
      for (int k = 0; k < num_keys; ++k) {
        string k_str = strings::Printf("feature_%d", k);
        Feature f;
        fill(&f, feature_size);
        Features* features = example.mutable_features();
        (*features->mutable_feature())[k_str] = f;
      }
      CHECK(SerializeToTString(example, &string_t(b)));
    }
    (*examples)[std::make_tuple(batch_size, num_keys, feature_size)] =
        record_string;
  }
};
template <typename T>
ExampleTensorMap ExampleStore<T>::serialized_example;
template <typename T>
absl::once_flag ExampleStore<T>::flags_init;

template struct ExampleStore<BytesFiller>;
template struct ExampleStore<Int64Filler>;
template struct ExampleStore<FloatFiller>;

enum BenchmarkType { kDense, kSparse, kVarLenDense, kRagged };

template <typename S, BenchmarkType b_type>
struct BenchmarkOptions {
  int benchmark_type = b_type;
  typedef S Store;
  typename S::Filler filler;
};

template <typename Options>
static Graph* ParseExample(int batch_size, int num_keys, int feature_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_8(mht_8_v, 347, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "ParseExample");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized = Options::Store::GetSerializedExample()[std::make_tuple(
      batch_size, num_keys, feature_size)];
  Tensor names(DT_STRING, TensorShape({batch_size}));

  std::vector<NodeBuilder::NodeOut> sparse_keys;
  std::vector<NodeBuilder::NodeOut> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<PartialTensorShape> dense_shapes;
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    Tensor key(DT_STRING, TensorShape());
    key.scalar<tstring>()() = strings::Printf("feature_%d", i);
    switch (opt.benchmark_type) {
      case kDense:
        dense_keys.emplace_back(test::graph::Constant(g, key));
        dense_defaults.emplace_back(test::graph::Constant(
            g, opt.filler.make_dense_default(feature_size)));
        dense_shapes.push_back(PartialTensorShape({feature_size}));
        break;
      case kVarLenDense:
        dense_keys.emplace_back(test::graph::Constant(g, key));
        dense_defaults.emplace_back(
            test::graph::Constant(g, opt.filler.make_dense_default(1)));
        dense_shapes.push_back(PartialTensorShape({-1}));
        break;
      case kSparse:
        sparse_keys.emplace_back(test::graph::Constant(g, key));
        sparse_types.push_back(opt.filler.dtype);
        break;
    }
  }

  Node* ret;
  TF_EXPECT_OK(NodeBuilder(g->NewName("n"), "ParseExample")
                   .Input(test::graph::Constant(g, serialized))
                   .Input(test::graph::Constant(g, names))
                   .Input(sparse_keys)
                   .Input(dense_keys)
                   .Input(dense_defaults)
                   .Attr("sparse_types", sparse_types)
                   .Attr("dense_shapes", dense_shapes)
                   .Finalize(g, &ret));

  FixupSourceAndSinkEdges(g);
  return g;
}

template <typename Options>
static Graph* ParseExampleV2(int batch_size, int num_keys, int feature_size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_9(mht_9_v, 401, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "ParseExampleV2");

  bool scalar_input = (batch_size == 0);
  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized_batch =
      Options::Store::GetSerializedExample()[std::make_tuple(
          scalar_input ? 1 : batch_size, num_keys, feature_size)];
  Tensor serialized_example(DT_STRING, TensorShape());
  Tensor names(DT_STRING,
               scalar_input ? TensorShape({}) : TensorShape({batch_size}));
  Tensor* serialized;

  if (scalar_input) {
    serialized_example.scalar<tstring>()() = serialized_batch.vec<tstring>()(0);
    serialized = &serialized_example;
  } else {
    serialized = &serialized_batch;
  }

  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<DataType> ragged_value_types;
  std::vector<DataType> ragged_split_types;
  std::vector<PartialTensorShape> dense_shapes;
  Tensor keys_t(DT_STRING, {static_cast<int32>(num_keys)});
  auto keys_flat = keys_t.flat<tstring>();
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    keys_flat(i) = strings::Printf("feature_%d", i);
    switch (opt.benchmark_type) {
      case kDense:
        dense_defaults.emplace_back(test::graph::Constant(
            g, opt.filler.make_dense_default(feature_size)));
        dense_shapes.push_back(PartialTensorShape({feature_size}));
        break;
      case kVarLenDense:
        dense_defaults.emplace_back(
            test::graph::Constant(g, opt.filler.make_dense_default(1)));
        dense_shapes.push_back(PartialTensorShape({-1}));
        break;
      case kSparse:
        sparse_types.push_back(opt.filler.dtype);
        break;
      case kRagged:
        ragged_value_types.push_back(opt.filler.dtype);
        ragged_split_types.push_back(DT_INT32);
        break;
    }
  }

  Tensor empty_keys(DT_STRING, {0});
  auto bm_type = opt.benchmark_type;
  auto& sparse_keys = (bm_type == kSparse) ? keys_t : empty_keys;
  auto& dense_keys =
      (bm_type == kDense || bm_type == kVarLenDense) ? keys_t : empty_keys;
  auto& ragged_keys = (bm_type == kRagged) ? keys_t : empty_keys;
  int num_sparse = opt.benchmark_type == kSparse ? num_keys : 0;

  Node* ret;
  TF_EXPECT_OK(NodeBuilder(g->NewName("n"), "ParseExampleV2")
                   .Input(test::graph::Constant(g, *serialized))
                   .Input(test::graph::Constant(g, names))
                   .Input(test::graph::Constant(g, sparse_keys))
                   .Input(test::graph::Constant(g, dense_keys))
                   .Input(test::graph::Constant(g, ragged_keys))
                   .Input(dense_defaults)
                   .Attr("num_sparse", num_sparse)
                   .Attr("sparse_types", sparse_types)
                   .Attr("ragged_value_types", ragged_value_types)
                   .Attr("ragged_split_types", ragged_split_types)
                   .Attr("dense_shapes", dense_shapes)
                   .Finalize(g, &ret));

  FixupSourceAndSinkEdges(g);
  return g;
}

template <typename Options>
static Graph* ParseSingleExample(int num_keys, int feature_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSexample_parsing_ops_testDTcc mht_10(mht_10_v, 481, "", "./tensorflow/core/kernels/example_parsing_ops_test.cc", "ParseSingleExample");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized_batch_1 =
      Options::Store::GetSerializedExample()[std::make_tuple(1, num_keys,
                                                             feature_size)];
  Tensor serialized(DT_STRING, TensorShape());
  serialized.scalar<tstring>()() = serialized_batch_1.vec<tstring>()(0);

  std::vector<string> sparse_keys;
  std::vector<string> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<PartialTensorShape> dense_shapes;
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    string key = strings::Printf("feature_%d", i);
    switch (opt.benchmark_type) {
      case kDense:
        dense_keys.push_back(key),
            dense_defaults.emplace_back(test::graph::Constant(
                g, opt.filler.make_dense_default(feature_size)));
        dense_shapes.push_back(PartialTensorShape({feature_size}));
        break;
      case kVarLenDense:
        dense_keys.push_back(key),
            dense_defaults.emplace_back(
                test::graph::Constant(g, opt.filler.make_dense_default(1)));
        dense_shapes.push_back(PartialTensorShape({-1}));
        break;
      case kSparse:
        sparse_keys.push_back(key), sparse_types.push_back(opt.filler.dtype);
        break;
    }
  }

  Node* ret;
  TF_EXPECT_OK(NodeBuilder(g->NewName("n"), "ParseSingleExample")
                   .Input(test::graph::Constant(g, serialized))
                   .Input(dense_defaults)
                   .Attr<int64_t>("num_sparse", sparse_keys.size())
                   .Attr("sparse_keys", sparse_keys)
                   .Attr("sparse_types", sparse_types)
                   .Attr("dense_keys", dense_keys)
                   .Attr("dense_shapes", dense_shapes)
                   .Finalize(g, &ret));

  FixupSourceAndSinkEdges(g);
  return g;
}

// Benchmark settings (Sparse, Dense) X (Bytes, Int64, Float)
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kSparse> SparseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kDense> DenseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kVarLenDense>
    VarLenDenseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kRagged> RaggedString;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kSparse> SparseInt64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kDense> DenseInt64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kVarLenDense>
    VarLenDenseInt64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kRagged> RaggedInt64;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kSparse> SparseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kDense> DenseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kVarLenDense>
    VarLenDenseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kRagged> RaggedFloat;

// B == batch_size, K == num_keys. F == feature_size.
// K must be one of 10, 100, 1000
#define BM_ParseExample(TYPE, B, K, F)                                    \
  static void BM_ParseExample##_##TYPE##_##B##_##K##_##F(                 \
      ::testing::benchmark::State& state) {                               \
    int64_t items_per_iter = static_cast<int64_t>(B) * K * F;             \
    test::Benchmark("cpu", ParseExample<TYPE>(B, K, F), nullptr, nullptr, \
                    nullptr, "SINGLE_THREADED_EXECUTOR", false)           \
        .Run(state);                                                      \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *    \
                            items_per_iter);                              \
  }                                                                       \
  BENCHMARK(BM_ParseExample##_##TYPE##_##B##_##K##_##F)->UseRealTime();

#define BM_AllParseExample(Type)       \
  BM_ParseExample(Type, 1, 10, 1);     \
  BM_ParseExample(Type, 128, 10, 1);   \
  BM_ParseExample(Type, 512, 10, 1);   \
  BM_ParseExample(Type, 1, 100, 1);    \
  BM_ParseExample(Type, 128, 100, 1);  \
  BM_ParseExample(Type, 512, 100, 1);  \
  BM_ParseExample(Type, 1, 1000, 1);   \
  BM_ParseExample(Type, 128, 1000, 1); \
  BM_ParseExample(Type, 512, 1000, 1); \
  BM_ParseExample(Type, 1, 1, 1000000);

BM_AllParseExample(SparseString);
BM_AllParseExample(DenseString);
BM_AllParseExample(VarLenDenseString);
BM_AllParseExample(SparseInt64);
BM_AllParseExample(DenseInt64);
BM_AllParseExample(VarLenDenseInt64);
BM_AllParseExample(SparseFloat);
BM_AllParseExample(DenseFloat);
BM_AllParseExample(VarLenDenseFloat);

// B == batch_size, K == num_keys. F == feature_size.
// K must be one of 10, 100, 1000
// B=0 indicates that a scalar input should be used (instead of a vector).
#define BM_ParseExampleV2(TYPE, B, K, F)                                    \
  static void BM_ParseExampleV2##_##TYPE##_##B##_##K##_##F(                 \
      ::testing::benchmark::State& state) {                                 \
    int64_t items_per_iter = static_cast<int64_t>(std::max(B, 1)) * K * F;  \
    test::Benchmark("cpu", ParseExampleV2<TYPE>(B, K, F), nullptr, nullptr, \
                    nullptr, "SINGLE_THREADED_EXECUTOR",                    \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *      \
                            items_per_iter);                                \
  }                                                                         \
  BENCHMARK(BM_ParseExampleV2##_##TYPE##_##B##_##K##_##F)->UseRealTime();

#define BM_AllParseExampleV2(Type)        \
  /* Vector Inputs */                     \
  BM_ParseExampleV2(Type, 1, 10, 1);      \
  BM_ParseExampleV2(Type, 128, 10, 1);    \
  BM_ParseExampleV2(Type, 512, 10, 1);    \
  BM_ParseExampleV2(Type, 1, 100, 1);     \
  BM_ParseExampleV2(Type, 128, 100, 1);   \
  BM_ParseExampleV2(Type, 512, 100, 1);   \
  BM_ParseExampleV2(Type, 1, 1000, 1);    \
  BM_ParseExampleV2(Type, 128, 1000, 1);  \
  BM_ParseExampleV2(Type, 512, 1000, 1);  \
  BM_ParseExampleV2(Type, 1, 1, 1000000); \
  /* Scalar Inputs */                     \
  BM_ParseExampleV2(Type, 0, 10, 1);      \
  BM_ParseExampleV2(Type, 0, 100, 1);     \
  BM_ParseExampleV2(Type, 0, 1000, 1);    \
  BM_ParseExampleV2(Type, 0, 1, 10);      \
  BM_ParseExampleV2(Type, 0, 1, 100);     \
  BM_ParseExampleV2(Type, 0, 1, 1000);    \
  BM_ParseExampleV2(Type, 0, 1, 10000);   \
  BM_ParseExampleV2(Type, 0, 1, 100000);  \
  BM_ParseExampleV2(Type, 0, 1, 1000000); \
  BM_ParseExampleV2(Type, 0, 10, 100000); \
  BM_ParseExampleV2(Type, 0, 100, 10000); \
  BM_ParseExampleV2(Type, 0, 1000, 1000);

BM_AllParseExampleV2(SparseString);
BM_AllParseExampleV2(DenseString);
BM_AllParseExampleV2(VarLenDenseString);
BM_AllParseExampleV2(RaggedString);
BM_AllParseExampleV2(SparseInt64);
BM_AllParseExampleV2(DenseInt64);
BM_AllParseExampleV2(VarLenDenseInt64);
BM_AllParseExampleV2(RaggedInt64);
BM_AllParseExampleV2(SparseFloat);
BM_AllParseExampleV2(DenseFloat);
BM_AllParseExampleV2(VarLenDenseFloat);
BM_AllParseExampleV2(RaggedFloat);

// K == num_keys. F == feature_size.
// K must be one of 10, 100, 1000
#define BM_ParseSingleExample(TYPE, K, F)                                    \
  void BM_ParseSingleExample##_##TYPE##_1_##K##_##F(                         \
      ::testing::benchmark::State& state) {                                  \
    int64_t items_per_iter = K * F;                                          \
    test::Benchmark("cpu", ParseSingleExample<TYPE>(K, F), nullptr, nullptr, \
                    nullptr, "SINGLE_THREADED_EXECUTOR",                     \
                    /*old_benchmark_api=*/false)                             \
        .Run(state);                                                         \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *       \
                            items_per_iter);                                 \
  }                                                                          \
  BENCHMARK(BM_ParseSingleExample##_##TYPE##_1_##K##_##F)->UseRealTime();

#define BM_AllParseSingleExample(Type)     \
  BM_ParseSingleExample(Type, 10, 1);      \
  BM_ParseSingleExample(Type, 100, 1);     \
  BM_ParseSingleExample(Type, 1000, 1);    \
  BM_ParseSingleExample(Type, 1, 10);      \
  BM_ParseSingleExample(Type, 1, 100);     \
  BM_ParseSingleExample(Type, 1, 1000);    \
  BM_ParseSingleExample(Type, 1, 10000);   \
  BM_ParseSingleExample(Type, 1, 100000);  \
  BM_ParseSingleExample(Type, 1, 1000000); \
  BM_ParseSingleExample(Type, 10, 100000); \
  BM_ParseSingleExample(Type, 100, 10000); \
  BM_ParseSingleExample(Type, 1000, 1000);

BM_AllParseSingleExample(SparseString);
BM_AllParseSingleExample(DenseString);
BM_AllParseSingleExample(VarLenDenseString);
BM_AllParseSingleExample(SparseInt64);
BM_AllParseSingleExample(DenseInt64);
BM_AllParseSingleExample(VarLenDenseInt64);
BM_AllParseSingleExample(SparseFloat);
BM_AllParseSingleExample(DenseFloat);
BM_AllParseSingleExample(VarLenDenseFloat);

}  // end namespace tensorflow
