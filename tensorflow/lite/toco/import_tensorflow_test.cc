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
class MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc() {
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
#include "tensorflow/lite/toco/import_tensorflow.h"
#include "tensorflow/lite/toco/toco_port.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/testing/util.h"

namespace toco {

using tensorflow::AttrValue;
using tensorflow::DT_BOOL;
using tensorflow::DT_COMPLEX64;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_INVALID;
using tensorflow::DT_QUINT8;
using tensorflow::DT_STRING;
using tensorflow::DT_UINT16;
using tensorflow::DT_UINT32;
using tensorflow::NodeDef;
using tensorflow::Status;
using ::testing::ElementsAre;

namespace internal {
using ConverterType = tensorflow::Status (*)(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    const ModelFlags& model_flags, Model* model);
using ConverterMapType = std::unordered_map<std::string, ConverterType>;

ConverterMapType GetTensorFlowNodeConverterMap();
ConverterMapType GetTensorFlowNodeConverterMapForFlex();
Status ImportTensorFlowNode(const NodeDef&, const TensorFlowImportFlags&,
                            const ModelFlags& model_flags, Model*,
                            const ConverterMapType&);
}  // namespace internal

namespace {

Status ImportNode(const NodeDef& node, Model* model) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_0(mht_0_v, 230, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "ImportNode");

  const auto converter = internal::GetTensorFlowNodeConverterMap();
  return internal::ImportTensorFlowNode(node, TensorFlowImportFlags(),
                                        ModelFlags(), model, converter);
}

Status ImportFlexNode(const NodeDef& node, Model* model) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "ImportFlexNode");

  // Empty converter => all nodes are flex nodes.
  const auto converter = internal::ConverterMapType();
  return internal::ImportTensorFlowNode(node, TensorFlowImportFlags(),
                                        ModelFlags(), model, converter);
}

Status ImportNode(const NodeDef& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "ImportNode");

  Model model;
  return ImportNode(node, &model);
}

NodeDef BuildNode(
    const std::string& op,
    const std::vector<std::initializer_list<int>>& output_shapes) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "BuildNode");

  NodeDef node;
  node.set_op(op);
  node.set_name("Node1");
  node.add_input();
  node.set_input(0, "Node0");

  AttrValue::ListValue* shapes =
      (*node.mutable_attr())["_output_shapes"].mutable_list();
  for (const auto& output_shape : output_shapes) {
    tensorflow::TensorShapeProto* shape = shapes->add_shape();
    for (int64_t output_shape_dim : output_shape) {
      auto shape_dim = shape->add_dim();
      shape_dim->set_size(output_shape_dim);
    }
  }

  return node;
}

namespace {
void BuildConstNode(std::initializer_list<int64_t> shape,
                    tensorflow::DataType dtype, int64_t num_elements,
                    NodeDef* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_4(mht_4_v, 286, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "BuildConstNode");

  node->set_op("Const");
  node->set_name("Node1");

  // An attribute describing the type of this const node.
  AttrValue dtype_attr;
  SetAttrValue(dtype, &dtype_attr);
  (*node->mutable_attr())["dtype"] = dtype_attr;

  // An attribute describing the content of this const node.
  tensorflow::TensorProto t;
  t.set_dtype(dtype);
  auto* s = t.mutable_tensor_shape();
  for (auto d : shape) {
    s->add_dim()->set_size(d);
  }

  switch (dtype) {
    case DT_FLOAT:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_float_val(i / 10000.0 + 1);
      }
      break;
    case DT_INT32:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_int_val(i % std::numeric_limits<int>::max() + 1);
      }
      break;
    case DT_UINT32:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_int_val(i % std::numeric_limits<uint32_t>::max() + 1);
      }
      break;
    case DT_QUINT8:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_int_val(i % std::numeric_limits<uint8_t>::max() + 1);
      }
      break;
    case DT_INT64:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_int64_val(i + 1);
      }
      break;
    case DT_UINT16:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_int_val(i % std::numeric_limits<uint16_t>::max() + 1);
      }
      break;
    case DT_STRING:
      break;
    case DT_BOOL:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_bool_val((i % 2) == 0);
      }
      break;
    case DT_COMPLEX64:
      for (int64_t i = 0; i < num_elements; ++i) {
        t.add_scomplex_val(i / 10000.0 + 1);
        t.add_scomplex_val(-i / 10000.0 - 1);
      }
      break;
    default:
      break;
  }

  AttrValue value_attr;
  SetAttrValue(t, &value_attr);
  (*node->mutable_attr())["value"] = value_attr;
}
}  //  namespace

TEST(FlexImportTest, ConditionalConst) {
  Model model;
  auto build_and_import_node =
      [&model](const std::string& name, std::initializer_list<int64_t> shape,
               tensorflow::DataType dtype, int64_t num_elements) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_5(mht_5_v, 365, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "lambda");

        NodeDef node;
        BuildConstNode(shape, dtype, num_elements, &node);
        node.set_name(name);

        const auto converter = internal::GetTensorFlowNodeConverterMapForFlex();
        return internal::ImportTensorFlowNode(node, TensorFlowImportFlags(),
                                              ModelFlags(), &model, converter);
      };

  EXPECT_TRUE(build_and_import_node("Known", {1, 2, 3}, DT_INT32, 6).ok());
  EXPECT_TRUE(build_and_import_node("BadType", {1, 2, 3}, DT_INVALID, 6).ok());
  EXPECT_TRUE(build_and_import_node("Unknown", {1, -2, 3}, DT_INT32, 6).ok());

  // We expect the "Known" node to be converted into an array, while the
  // "Unknown" and "BadType" nodes are kept as operators.
  EXPECT_EQ(model.operators.size(), 2);
  EXPECT_TRUE(model.HasArray("Known"));
  EXPECT_FALSE(model.HasArray("Unknown"));
  EXPECT_FALSE(model.HasArray("BadType"));
}

TEST(FlexImportTest, SoftmaxWithBeta) {
  NodeDef node;
  node.set_op("Softmax");
  node.set_name("softmax");
  node.add_input();
  node.set_input(0, "logits");

  AttrValue dtype_attr;
  SetAttrValue(0.5, &dtype_attr);
  (*node.mutable_attr())["_softmax_beta"] = dtype_attr;
  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kSoftmax);
  const SoftmaxOperator* op =
      static_cast<const SoftmaxOperator*>(model.operators[0].get());
  EXPECT_EQ(op->beta, 0.5);
}

TEST(FlexImportTest, SoftmaxWithoutBeta) {
  NodeDef node;
  node.set_op("Softmax");
  node.set_name("softmax");
  node.add_input();
  node.set_input(0, "logits");

  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kSoftmax);
  const SoftmaxOperator* op =
      static_cast<const SoftmaxOperator*>(model.operators[0].get());
  EXPECT_EQ(op->beta, 1.0);
}

class ShapeImportTest : public ::testing::TestWithParam<tensorflow::DataType> {
};

TEST_P(ShapeImportTest, ShapeElementIsNegative) {
  NodeDef node;
  BuildConstNode({1, -2, 10}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(
      status.error_message(),
      "Tensor shape should not include negative values\n\t (while processing "
      "node 'Node1')");
}

TEST_P(ShapeImportTest, ShapeElementIsZero) {
  NodeDef node;
  // Const nodes with zero-sized, non-scalar shapes are still not importable.
  BuildConstNode({1, 0, 10}, GetParam(), 0, &node);

  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  const auto& array = model.GetArray("Node1");
  EXPECT_THAT(array.shape().dims(), ::testing::ElementsAre());
}

// Note how this is subtly different thant ShapeElementIsZero above, where toco
// removes all shape information after import.
TEST_P(ShapeImportTest, ShapeIsOneDimZero) {
  NodeDef node;
  BuildConstNode({0}, GetParam(), 0, &node);

  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  const auto& array = model.GetArray("Node1");
  // We would like to have [0] shapes actually import correctly, but
  // for some reason that slows everything down.
  EXPECT_THAT(array.shape().dims(), ::testing::ElementsAre());
}

TEST_P(ShapeImportTest, ShapeElementTooLarge) {
  NodeDef node;
  BuildConstNode({3000000000}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(status.error_message(),
            "Shape element overflows\n\t (while processing node 'Node1')");
}

TEST_P(ShapeImportTest, ShapeTooLarge) {
  NodeDef node;
  BuildConstNode({1000000, 2000000, 2000000, 2000000}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(status.error_message(),
            "Tensor shape is too large\n\t (while processing node 'Node1')");
}

TEST_P(ShapeImportTest, ValidShapeButZeroElements) {
  NodeDef node;
  BuildConstNode({1, 2, 2, 2}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_THAT(status.error_message(),
              ::testing::MatchesRegex(
                  "Neither input_content .0. nor .*_val .0. have the right "
                  "dimensions .8. for this .* tensor\n\t .while processing "
                  "node 'Node1'."));
}

std::vector<tensorflow::DataType> TestTypes() {
  return {DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL, DT_QUINT8, DT_COMPLEX64};
}

INSTANTIATE_TEST_SUITE_P(ShapeImportTest, ShapeImportTest,
                         ::testing::ValuesIn(TestTypes()));

class ContentImportTest : public ::testing::Test {
 public:
  template <ArrayDataType T>
  std::vector<DataType<T>> ImportAndGetData(const NodeDef& node) {
    Model model;
    auto status = ImportNode(node, &model);
    CHECK(status.ok()) << status.error_message();
    const auto& array = model.GetArray("Node1");
    return array.GetBuffer<T>().data;
  }
  void RemoveTrailingElements(NodeDef* node, int num) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_6(mht_6_v, 511, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "RemoveTrailingElements");

    tensorflow::TensorProto* p =
        node->mutable_attr()->at("value").mutable_tensor();
    for (int i = 0; i < num; ++i) {
      if (p->int_val_size() > 0) p->mutable_int_val()->RemoveLast();
      if (p->int64_val_size() > 0) p->mutable_int64_val()->RemoveLast();
      if (p->float_val_size() > 0) p->mutable_float_val()->RemoveLast();
      if (p->bool_val_size() > 0) p->mutable_bool_val()->RemoveLast();
      if (p->scomplex_val_size() > 0) p->mutable_scomplex_val()->RemoveLast();
      if (p->scomplex_val_size() > 0) p->mutable_scomplex_val()->RemoveLast();
    }
  }
};

TEST_F(ContentImportTest, Int32) {
  constexpr ArrayDataType kType = ArrayDataType::kInt32;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_INT32, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 5));
  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST_F(ContentImportTest, Int64) {
  constexpr ArrayDataType kType = ArrayDataType::kInt64;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_INT64, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 5));
  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST_F(ContentImportTest, Quint8) {
  constexpr ArrayDataType kType = ArrayDataType::kUint8;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_QUINT8, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 5));
  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST_F(ContentImportTest, Bool) {
  constexpr ArrayDataType kType = ArrayDataType::kBool;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_BOOL, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 0, 1, 0, 1, 0));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 0, 1, 0, 1, 1));
  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST_F(ContentImportTest, Float) {
  constexpr ArrayDataType kType = ArrayDataType::kFloat;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_FLOAT, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node),
              ElementsAre(1.0000, 1.0001, 1.0002, 1.0003, 1.0004, 1.0005));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(ImportAndGetData<kType>(node),
              ElementsAre(1.0000, 1.0001, 1.0002, 1.0003, 1.0004, 1.0004));
  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(ImportAndGetData<kType>(node),
              ElementsAre(1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000));
}

TEST_F(ContentImportTest, Complex64) {
  constexpr ArrayDataType kType = ArrayDataType::kComplex64;

  NodeDef node;
  BuildConstNode({1, 2, 3}, DT_COMPLEX64, 6, &node);

  using cplx = std::complex<float>;
  EXPECT_THAT(
      ImportAndGetData<kType>(node),
      ElementsAre(std::complex<float>(1.0000, -1.0000), cplx(1.0001, -1.0001),
                  cplx(1.0002, -1.0002), cplx(1.0003, -1.0003),
                  cplx(1.0004, -1.0004), cplx(1.0005, -1.0005)));
  RemoveTrailingElements(&node, 1);
  EXPECT_THAT(
      ImportAndGetData<kType>(node),
      ElementsAre(std::complex<float>(1.0000, -1.0000), cplx(1.0001, -1.0001),
                  cplx(1.0002, -1.0002), cplx(1.0003, -1.0003),
                  cplx(1.0004, -1.0004), cplx(1.0004, -1.0004)));

  RemoveTrailingElements(&node, 4);
  EXPECT_THAT(
      ImportAndGetData<kType>(node),
      ElementsAre(std::complex<float>(1.0000, -1.0000), cplx(1.0000, -1.0000),
                  cplx(1.0000, -1.0000), cplx(1.0000, -1.0000),
                  cplx(1.0000, -1.0000), cplx(1.0000, -1.0000)));
}

std::vector<std::pair<tensorflow::DataType, ArrayDataType>> UnaryTestTypes() {
  return {{DT_FLOAT, ArrayDataType::kFloat},
          {DT_INT32, ArrayDataType::kInt32},
          {DT_INT64, ArrayDataType::kInt64}};
}

class TensorContentTest : public ::testing::Test {
 public:
  template <ArrayDataType T>
  std::vector<DataType<T>> ImportAndGetData(const NodeDef& node) {
    Model model;
    auto status = ImportNode(node, &model);
    CHECK(status.ok()) << status.error_message();
    const auto& nodearray = model.GetArray("Node1");
    return nodearray.GetBuffer<T>().data;
  }
  template <class T>
  void NodeWithTensorContent(std::initializer_list<int64_t> shape,
                             tensorflow::DataType dtype, int64_t num_elements,
                             NodeDef* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_7(mht_7_v, 642, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "NodeWithTensorContent");

    node->set_op("Const");
    node->set_name("Node1");

    // An attribute describing the type of this const node.
    AttrValue dtype_attr;
    SetAttrValue(dtype, &dtype_attr);
    (*node->mutable_attr())["dtype"] = dtype_attr;

    auto allocated_content = absl::make_unique<T[]>(num_elements);

    // An attribute describing the content of this const node.
    tensorflow::TensorProto t;
    t.set_dtype(dtype);
    auto* s = t.mutable_tensor_shape();
    for (const auto& d : shape) {
      s->add_dim()->set_size(d);
    }

    switch (dtype) {
      case DT_FLOAT:
        for (int64_t i = 0; i < num_elements; ++i) {
          allocated_content[i] = i / 10000.0 + 1;
        }
        break;
      case DT_INT32:
        for (int64_t i = 0; i < num_elements; ++i) {
          allocated_content[i] = i % std::numeric_limits<int>::max() + 1;
        }
        break;
      case DT_QUINT8:
        for (int64_t i = 0; i < num_elements; ++i) {
          allocated_content[i] = i % std::numeric_limits<uint8_t>::max() + 1;
        }
        break;
      case DT_INT64:
        for (int64_t i = 0; i < num_elements; ++i) {
          allocated_content[i] = i + 1;
        }
        break;
      case DT_STRING:
        break;
      case DT_BOOL:
        for (int64_t i = 0; i < num_elements; ++i) {
          allocated_content[i] = ((i % 2) == 0);
        }
        break;
      default:
        break;
    }
    t.set_tensor_content(
        std::string(reinterpret_cast<const char*>(allocated_content.get()),
                    num_elements * sizeof(T)));

    AttrValue value_attr;
    SetAttrValue(t, &value_attr);
    (*node->mutable_attr())["value"] = value_attr;

    allocated_content.reset();
  }
};

TEST_F(TensorContentTest, Int64) {
  constexpr ArrayDataType kType = ArrayDataType::kInt64;

  NodeDef node;
  NodeWithTensorContent<int64_t>({1, 2, 3}, DT_INT64, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST_F(TensorContentTest, Int32) {
  constexpr ArrayDataType kType = ArrayDataType::kInt32;

  NodeDef node;
  NodeWithTensorContent<int>({1, 2, 3}, DT_INT32, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST_F(TensorContentTest, Float) {
  constexpr ArrayDataType kType = ArrayDataType::kFloat;

  NodeDef node;
  NodeWithTensorContent<float>({1, 2, 3}, DT_FLOAT, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node),
              ElementsAre(1.0000, 1.0001, 1.0002, 1.0003, 1.0004, 1.0005));
}

TEST_F(TensorContentTest, Quint8) {
  constexpr ArrayDataType kType = ArrayDataType::kUint8;

  NodeDef node;
  NodeWithTensorContent<uint8_t>({1, 2, 3}, DT_QUINT8, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST_F(TensorContentTest, Bool) {
  constexpr ArrayDataType kType = ArrayDataType::kBool;

  NodeDef node;
  NodeWithTensorContent<bool>({1, 2, 3}, DT_BOOL, 6, &node);

  EXPECT_THAT(ImportAndGetData<kType>(node), ElementsAre(1, 0, 1, 0, 1, 0));
}

class TypeImportTest : public ::testing::TestWithParam<
                           std::pair<tensorflow::DataType, ArrayDataType>> {
 protected:
  TypeImportTest() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_8(mht_8_v, 756, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "TypeImportTest");
}

  void BuildUnaryNode(const std::string& op_name, tensorflow::DataType dtype,
                      NodeDef* node) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_9(mht_9_v, 763, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "BuildUnaryNode");

    node->set_op(op_name);
    node->set_name("Node1");

    node->add_input();
    node->set_input(0, "Node0");

    AttrValue dtype_attr;
    SetAttrValue(dtype, &dtype_attr);
    (*node->mutable_attr())["T"] = dtype_attr;
  }
};

TEST_P(TypeImportTest, BasicTypeInference) {
  NodeDef node;
  BuildUnaryNode("Atan", GetParam().first, &node);

  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());
  ASSERT_THAT(op->output_data_types, ::testing::ElementsAre(GetParam().second));
}
INSTANTIATE_TEST_SUITE_P(BasicTypeInference, TypeImportTest,
                         ::testing::ValuesIn(UnaryTestTypes()));

TEST(ImportTest, TypeInferenceWithFixedOutputType) {
  // Create an op that has a fixed output type (bool).
  Model model;
  EXPECT_TRUE(ImportNode(BuildNode("IsFinite", {{1, 2}, {2, 3}}), &model).ok());
  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());

  // The static output type should be indicated in the imported op.
  ASSERT_THAT(op->output_data_types,
              ::testing::ElementsAre(ArrayDataType::kBool));
}

TEST(ImportTest, FailedTypeInference) {
  // Create a unary op with no Type ("T") annotation.
  NodeDef node;
  node.set_op("Atan");
  node.set_name("Node1");
  node.add_input();
  node.set_input(0, "Node0");

  Model model;
  EXPECT_TRUE(ImportNode(node, &model).ok());

  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());
  ASSERT_TRUE(op->output_data_types.empty());
}

TEST(ImportTest, UnsupportedOpWithOutputShapes) {
  // Create an unsupported op with output shapes.
  Model model;
  EXPECT_TRUE(ImportNode(BuildNode("Atan", {{1, 2}, {2, 3}}), &model).ok());
  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());

  // The output shapes should be imported.
  ASSERT_EQ(op->output_shapes.size(), 2);
  ASSERT_THAT(op->output_shapes[0].dims(), ::testing::ElementsAre(1, 2));
  ASSERT_THAT(op->output_shapes[1].dims(), ::testing::ElementsAre(2, 3));
}

TEST(ImportTest, UnsupportedOpWithWildcardOutputShapes) {
  // Create an unsupported op with wildcard output shapes.
  Model model;
  EXPECT_TRUE(ImportNode(BuildNode("Atan", {{-1, 2}}), &model).ok());
  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());

  // Wildcard shapes aren't yet supported.
  ASSERT_TRUE(op->output_shapes.empty());
}

TEST(ImportTest, UnsupportedOpWithMultipleOutputs) {
  // This test needs an existing TensorFlow op to run correctly, because it
  // read the OpDef from the global registry. The complex output setup of
  // ParseExample allows us to test all nuances here, but we will need to add
  // attributes to match the specification in the OpDef.
  NodeDef node = BuildNode("ParseExample", {});

  // Nsparse defines how many sparse indices and shapes there are. Here we set
  // Nsparse to 2, meaning there will be 2 INT64 tensors for 'sparse_indices'
  // and 2 INT64 tensors for 'sparse_shapes. The type of those tensors is
  // defined in the OpDef.
  {
    AttrValue value_attr;
    SetAttrValue(2, &value_attr);
    (*node.mutable_attr())["Nsparse"] = value_attr;
  }

  // The there will be a number of 'sparse_values' tensors, defined by the
  // attribute 'sparse_types', which is a list of types.
  {
    AttrValue value_attr;
    std::vector<tensorflow::DataType> types;
    types.push_back(tensorflow::DT_FLOAT);
    types.push_back(tensorflow::DT_STRING);
    SetAttrValue(types, &value_attr);
    (*node.mutable_attr())["sparse_types"] = value_attr;
  }

  // And finally there will be 'dense_values' tensors, which are controlled by
  // the 'Tdense' attribute.
  {
    AttrValue value_attr;
    std::vector<tensorflow::DataType> types;
    types.push_back(tensorflow::DT_STRING);
    types.push_back(tensorflow::DT_FLOAT);
    types.push_back(tensorflow::DT_INT64);
    SetAttrValue(types, &value_attr);
    (*node.mutable_attr())["Tdense"] = value_attr;
  }

  Model model;
  EXPECT_TRUE(ImportFlexNode(node, &model).ok());

  ASSERT_THAT(model.operators.size(), ::testing::Ge(1));
  ASSERT_EQ(model.operators[0]->type, OperatorType::kUnsupported);
  const TensorFlowUnsupportedOperator* op =
      static_cast<const TensorFlowUnsupportedOperator*>(
          model.operators[0].get());

  ASSERT_EQ(op->outputs.size(), 9);
  ASSERT_EQ(op->output_data_types.size(), 9);

  // The 'sparse_indices' output tensors.
  ASSERT_EQ(op->outputs[0], "Node1");
  ASSERT_EQ(op->outputs[1], "Node1:1");
  ASSERT_EQ(op->output_data_types[0], ArrayDataType::kInt64);
  ASSERT_EQ(op->output_data_types[1], ArrayDataType::kInt64);

  // The 'sparse_values' output tensors.
  ASSERT_EQ(op->outputs[2], "Node1:2");
  ASSERT_EQ(op->outputs[3], "Node1:3");
  ASSERT_EQ(op->output_data_types[2], ArrayDataType::kFloat);
  ASSERT_EQ(op->output_data_types[3], ArrayDataType::kString);

  // The 'sparse_shapes' output tensors.
  ASSERT_EQ(op->outputs[4], "Node1:4");
  ASSERT_EQ(op->outputs[5], "Node1:5");
  ASSERT_EQ(op->output_data_types[4], ArrayDataType::kInt64);
  ASSERT_EQ(op->output_data_types[5], ArrayDataType::kInt64);

  // The 'dense_shapes' output tensors.
  ASSERT_EQ(op->outputs[6], "Node1:6");
  ASSERT_EQ(op->outputs[7], "Node1:7");
  ASSERT_EQ(op->outputs[8], "Node1:8");
  ASSERT_EQ(op->output_data_types[6], ArrayDataType::kString);
  ASSERT_EQ(op->output_data_types[7], ArrayDataType::kFloat);
  ASSERT_EQ(op->output_data_types[8], ArrayDataType::kInt64);
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPSimport_tensorflow_testDTcc mht_10(mht_10_v, 942, "", "./tensorflow/lite/toco/import_tensorflow_test.cc", "main");

  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
