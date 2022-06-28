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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc() {
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

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

Array<int64_t> MakeArray(absl::Span<const int64_t> dimensions,
                         absl::Span<const int64_t> contents) {
  Array<int64_t> a(dimensions);
  std::copy(contents.begin(), contents.end(), a.begin());
  return a;
}

OpMetadata GetMetadata(const std::string& op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/hlo_sharding_test.cc", "GetMetadata");

  OpMetadata metadata;
  metadata.set_op_name(op_name);
  return metadata;
}

std::vector<OpMetadata> SingleMetadata() { return {GetMetadata("a")}; }

std::vector<OpMetadata> ListMetadata() {
  return {GetMetadata("b"), GetMetadata("c")};
}

class HloShardingTest : public HloTestBase {};

TEST_F(HloShardingTest, Replicate) {
  HloSharding sharding = HloSharding::Replicate();
  EXPECT_TRUE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_TRUE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(65535));

  HloSharding other = HloSharding::Replicate();
  EXPECT_EQ(other, sharding);

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/2));
  EXPECT_FALSE(sharding.HasUniqueDevice());
}

TEST_F(HloShardingTest, DevicePlacement) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_FALSE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(5));
  EXPECT_EQ(5, sharding.GetUniqueDevice());

  HloSharding other = HloSharding::Replicate();
  EXPECT_NE(other, sharding);

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/6));
  EXPECT_IS_NOT_OK(
      sharding.Validate(ShapeUtil::MakeShape(U32, {4}), /*num_devices=*/5));

  ShapeTree<HloSharding> shape_tree =
      sharding.GetAsShapeTree(ShapeUtil::MakeShape(U32, {4}));
  EXPECT_EQ(shape_tree.element({}), sharding);
  EXPECT_TRUE(shape_tree.IsLeaf({}));
}

TEST_F(HloShardingTest, ProtoRoundTrip) {
  OpSharding proto;
  proto.set_type(OpSharding::TUPLE);
  auto* tiled = proto.add_tuple_shardings();
  tiled->set_type(OpSharding::OTHER);
  tiled->add_tile_assignment_devices(0);
  tiled->add_tile_assignment_devices(1);
  tiled->add_tile_assignment_dimensions(1);
  tiled->add_tile_assignment_dimensions(2);
  *tiled->add_metadata() = GetMetadata("a");
  *tiled->add_metadata() = GetMetadata("b");
  auto* replicated = proto.add_tuple_shardings();
  replicated->set_type(OpSharding::REPLICATED);
  *replicated->add_metadata() = GetMetadata("c");
  auto* manual = proto.add_tuple_shardings();
  manual->set_type(OpSharding::MANUAL);
  HloSharding sharding = HloSharding::FromProto(proto).ConsumeValueOrDie();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(proto, sharding.ToProto()));
}

TEST_F(HloShardingTest, Tile) {
  {
    // Test should fail because of a duplicate tile assignment.
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 0, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                       /*num_devices=*/4));
  }

  {
    // Test should fail because of more devices used than `num_device`.
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 1, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4, 6}),
                                       /*num_devices=*/2));
  }

  {
    // Test should pass.
    Shape shape = ShapeUtil::MakeShape(U32, {4, 5});
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {3, 5}),
                                   /*num_devices=*/5));

    EXPECT_EQ(0, sharding.DeviceForTileIndex({0, 0}));
    EXPECT_EQ(3, sharding.DeviceForTileIndex({0, 1}));
    EXPECT_EQ(2, sharding.DeviceForTileIndex({1, 0}));
    EXPECT_EQ(1, sharding.DeviceForTileIndex({1, 1}));

    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 0),
              (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 3),
              (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 2),
              (std::vector<int64_t>{2, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 1),
              (std::vector<int64_t>{2, 3}));

    EXPECT_FALSE(sharding.HasUniqueDevice());
  }
}

// Tests that empty tuple is supported.
TEST_F(HloShardingTest, EmptySingleTuple) {
  HloSharding sharding = HloSharding::SingleTuple(ShapeUtil::MakeTupleShape({}),
                                                  HloSharding::AssignDevice(0));
  EXPECT_TRUE(sharding.ExtractSingleSharding());
}

TEST_F(HloShardingTest, NestedTuple) {
  // nested_tuple_shape = (f32[], (f32[3]), f32[4, 6])
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {}),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3})}),
      ShapeUtil::MakeShape(F32, {4, 6}),
  });

  HloSharding tiled_sharding = HloSharding::Tile(Array<int64_t>({{0, 1}}));
  OpSharding proto;
  proto.set_type(OpSharding::TUPLE);
  *proto.add_tuple_shardings() = HloSharding::Replicate().ToProto();
  *proto.add_tuple_shardings() = HloSharding::AssignDevice(0).ToProto();
  *proto.add_tuple_shardings() = tiled_sharding.ToProto();
  HloSharding tuple_sharding =
      HloSharding::FromProto(proto).ConsumeValueOrDie();

  ShapeTree<HloSharding> shape_tree =
      tuple_sharding.GetAsShapeTree(nested_tuple_shape);
  EXPECT_EQ(shape_tree.element({0}), HloSharding::Replicate());
  EXPECT_EQ(shape_tree.element({1, 0}), HloSharding::AssignDevice(0));
  EXPECT_EQ(shape_tree.element({2}), tiled_sharding);

  EXPECT_IS_OK(tuple_sharding.Validate(nested_tuple_shape, /*num_devices=*/5));
  // Test should fail because tuple element count does not match.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeTupleShape({}),
                                           /*num_devices=*/5));
  // Test should fail because the input type is not a tuple.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeShape(F32, {}),
                                           /*num_devices=*/5));
}

TEST_F(HloShardingTest, NormalizeTrivialSubgroupToManual) {
  HloSharding sharding =
      HloSharding::Subgroup(MakeArray({1, 2, 1}, {0, 1}),
                            {OpSharding::MANUAL, OpSharding::REPLICATED});
  EXPECT_TRUE(sharding.IsManual());
}

TEST_F(HloShardingTest, Hash) {
  auto hash_compare_equal = [](const HloSharding& a, const HloSharding& b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc mht_1(mht_1_v, 374, "", "./tensorflow/compiler/xla/service/hlo_sharding_test.cc", "lambda");

    if (absl::HashOf(a) != absl::HashOf(b)) {
      return false;
    }
    return a == b;
  };

  {
    HloSharding sharding1 = HloSharding::Replicate();
    HloSharding sharding2 = HloSharding::Replicate();
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::AssignDevice(1);
    HloSharding sharding2 = HloSharding::AssignDevice(1);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::AssignDevice(1);
    HloSharding sharding2 = HloSharding::AssignDevice(2);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  HloSharding default_sharding = HloSharding::Replicate();
  {
    ShapeTree<HloSharding> shape_tree(ShapeUtil::MakeTupleShape({}),
                                      default_sharding);
    HloSharding sharding1 = HloSharding::Replicate();
    HloSharding sharding2 = HloSharding::Tuple(shape_tree);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree(ShapeUtil::MakeTupleShape({}),
                                      default_sharding);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree1(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree1.mutable_element({0}) = HloSharding::Replicate();
    ShapeTree<HloSharding> shape_tree2(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree2.mutable_element({0}) = HloSharding::AssignDevice(0);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree1);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree2);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree1(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree1.mutable_element({0}) = HloSharding::AssignDevice(0);
    ShapeTree<HloSharding> shape_tree2(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree2.mutable_element({0}) = HloSharding::AssignDevice(0);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree1);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree2);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }
}

using ShardingWithMetadataParamType =
    std::tuple<std::vector<OpMetadata>, std::string>;

TEST_F(HloShardingTest, ToStringReplicatedTest) {
  HloSharding sharding = HloSharding::Replicate();
  EXPECT_EQ(sharding.ToString(), "{replicated}");
}

class HloReplicateShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloReplicateShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::Replicate(std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false), "{replicated}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloReplicateShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{replicated}"),
        std::make_tuple(SingleMetadata(),
                        "{replicated metadata={op_name=\"a\"}}"),
        std::make_tuple(
            ListMetadata(),
            "{replicated metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringAssignDeviceTest) {
  HloSharding sharding = HloSharding::AssignDevice(7);
  EXPECT_EQ(sharding.ToString(), "{maximal device=7}");
}

class HloAssignDeviceShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloAssignDeviceShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::AssignDevice(7, std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{maximal device=7}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloAssignDeviceShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{maximal device=7}"),
        std::make_tuple(SingleMetadata(),
                        "{maximal device=7 metadata={op_name=\"a\"}}"),
        std::make_tuple(
            ListMetadata(),
            "{maximal device=7 metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringTiledTest) {
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64_t>({{{2, 3}}, {{5, 7}}}));
  EXPECT_EQ(sharding.ToString(), "{devices=[2,1,2]2,3,5,7}");
}

class HloTiledShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloTiledShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::Tile(
      Array3D<int64_t>({{{2, 3}}, {{5, 7}}}), std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{devices=[2,1,2]2,3,5,7}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloTiledShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{devices=[2,1,2]2,3,5,7}"),
        std::make_tuple(SingleMetadata(),
                        "{devices=[2,1,2]2,3,5,7 metadata={op_name=\"a\"}}"),
        std::make_tuple(ListMetadata(),
                        "{devices=[2,1,2]2,3,5,7 metadata={{op_name=\"b\"}, "
                        "{op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringTupleTest) {
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(U32, {7, 25}),
                                 ShapeUtil::MakeShape(S32, {9, 11})}),
      {HloSharding::Replicate(), HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
       HloSharding::AssignDevice(3)});
  EXPECT_EQ(sharding.ToString(),
            "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
}

TEST_F(HloShardingTest, ToStringTupleWithMetadataTest) {
  auto metadata = SingleMetadata();
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(U32, {7, 25}),
                                 ShapeUtil::MakeShape(S32, {9, 11})}),
      {HloSharding::Replicate({GetMetadata("d")}),
       HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
       HloSharding::AssignDevice(3, {GetMetadata("e")})});
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            "{{replicated metadata={op_name=\"d\"}}, {devices=[1,2]3,5}, "
            "{maximal device=3 metadata={op_name=\"e\"}}}");
}

TEST_F(HloShardingTest, OstreamTest) {
  HloSharding sharding =
      HloSharding::Tile(Array4D<int64_t>({{{{0, 1}, {2, 3}}}}));
  std::ostringstream oss;
  oss << sharding;
  EXPECT_EQ(oss.str(), "{devices=[1,1,2,2]0,1,2,3}");
}

class HloParseShardingWithMetadataTest
    : public ::testing::TestWithParam<std::vector<OpMetadata>> {};

TEST_P(HloParseShardingWithMetadataTest, ParseHloString) {
  auto check = [](const HloSharding& sharding) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_testDTcc mht_2(mht_2_v, 575, "", "./tensorflow/compiler/xla/service/hlo_sharding_test.cc", "lambda");

    TF_ASSERT_OK_AND_ASSIGN(
        auto parsed_sharding,
        ParseSharding(sharding.ToString(/*include_metadata=*/true)));
    EXPECT_EQ(sharding, parsed_sharding);
  };
  check(HloSharding::Replicate(GetParam()));
  check(HloSharding::AssignDevice(2, GetParam()));
  check(HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}}), GetParam()));
  // Empty tuple. One sharding is required for empty tuples, as we need to be
  // able to assign sharding to them, even though they have no leaves.
  check(HloSharding::Tuple(ShapeUtil::MakeTupleShape({}),
                           {HloSharding::Replicate(GetParam())}));
  {
    // Non-nested tuple.
    auto tuple_shape =
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 1, 5, 7}),
                                   ShapeUtil::MakeShape(F32, {3, 5, 7}),
                                   ShapeUtil::MakeShape(F32, {3, 7})});
    check(HloSharding::Tuple(
        tuple_shape,
        {HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}})),
         HloSharding::Replicate(GetParam()), HloSharding::AssignDevice(1)}));
  }
  {
    // Nested tuple.
    auto tuple_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(F32, {3, 1, 5, 7}),
         ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5, 7}),
                                    ShapeUtil::MakeShape(F32, {3, 7})})});
    std::vector<HloSharding> leaf_shardings = {
        HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}})),
        HloSharding::Replicate(), HloSharding::AssignDevice(1, GetParam())};
    ShapeTree<HloSharding> sharding_tree(tuple_shape, HloSharding::Replicate());
    // Assign leaf_shardings to sharding_tree leaves.
    auto it = leaf_shardings.begin();
    for (auto& index_to_sharding : sharding_tree.leaves()) {
      index_to_sharding.second = *it++;
    }
    check(HloSharding::Tuple(sharding_tree));
  }
}

INSTANTIATE_TEST_SUITE_P(ParseHloString, HloParseShardingWithMetadataTest,
                         ::testing::Values(std::vector<OpMetadata>(),
                                           SingleMetadata(), ListMetadata()));

TEST_F(HloShardingTest, WithMetadataNoOverwrite) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_new_metadata =
        sharding.WithMetadata(SingleMetadata(), /*overwrite=*/false);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.metadata().front(), SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/false);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding.metadata().front(), sharding_new_metadata.metadata().front()));
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, SingleMetadata())});
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/false);
    EXPECT_TRUE(sharding_new_metadata.metadata().empty());
    ASSERT_TRUE(sharding_new_metadata.IsTuple());
    ASSERT_EQ(sharding_new_metadata.tuple_elements().size(), 3);

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[0].metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.tuple_elements()[0].metadata().front(),
        SingleMetadata().front()));

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[1].metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(protobuf_util::ProtobufEquals(
          sharding_new_metadata.tuple_elements()[1].metadata()[i],
          ListMetadata()[i]));
    }

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[2].metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.tuple_elements()[2].metadata().front(),
        SingleMetadata().front()));
  }
}

TEST_F(HloShardingTest, WithMetadataOverwrite) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_new_metadata =
        sharding.WithMetadata(SingleMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.metadata().front(), SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(protobuf_util::ProtobufEquals(
          sharding_new_metadata.metadata()[i], ListMetadata()[i]));
    }
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, SingleMetadata())});
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/true);
    EXPECT_TRUE(sharding_new_metadata.metadata().empty());
    ASSERT_TRUE(sharding_new_metadata.IsTuple());
    ASSERT_EQ(sharding_new_metadata.tuple_elements().size(), 3);

    for (const auto& sub_sharding : sharding_new_metadata.tuple_elements()) {
      ASSERT_EQ(sub_sharding.metadata().size(), 2);
      for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(protobuf_util::ProtobufEquals(sub_sharding.metadata()[i],
                                                  ListMetadata()[i]));
      }
    }
  }
}

TEST_F(HloShardingTest, WithoutMetadata) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, ListMetadata())});
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
    ASSERT_TRUE(sharding_no_metadata.IsTuple());
    EXPECT_EQ(sharding_no_metadata.tuple_elements().size(), 3);
    for (const auto& sub_sharding : sharding_no_metadata.tuple_elements()) {
      EXPECT_TRUE(sub_sharding.metadata().empty());
    }
  }
}

}  // namespace
}  // namespace xla
