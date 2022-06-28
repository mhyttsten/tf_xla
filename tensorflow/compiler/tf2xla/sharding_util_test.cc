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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc() {
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
#include "tensorflow/compiler/tf2xla/sharding_util.h"

#include <functional>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CoreUtilTest, ParseShardingFromDevice) {
  Graph graph(OpRegistry::Global());

  auto core_from_sharding =
      [](absl::optional<xla::OpSharding> sharding) -> int64 {
    if (sharding.has_value() &&
        sharding.value().type() == xla::OpSharding::MAXIMAL) {
      return sharding.value().tile_assignment_devices(0);
    } else {
      return -1;
    }
  };

  auto parse_status = ParseShardingFromDevice("", 1);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));
  parse_status = ParseShardingFromDevice("", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:-1", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:55", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(55, core_from_sharding(parse_status.ValueOrDie()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:100", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/cpu:0", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));
}

class ShardingWithMetadataTest
    : public ::testing::TestWithParam<xla::OpSharding> {};

TEST_P(ShardingWithMetadataTest, GetShardingFromNode) {
  NodeDef node_def;
  {
    node_def.set_op("_Arg");
    node_def.set_name("arg");
    AttrValue xla_sharding;
    xla_sharding.set_s("");
    AttrValue index;
    index.set_i(0);
    AttrValue type;
    type.set_type(DataType::DT_FLOAT);
    node_def.mutable_attr()->insert(
        {{"_XlaSharding", xla_sharding}, {"index", index}, {"T", type}});
  }

  auto check_metadata = [](const xla::OpSharding& sharding) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc mht_0(mht_0_v, 246, "", "./tensorflow/compiler/tf2xla/sharding_util_test.cc", "lambda");

    ASSERT_EQ(sharding.metadata_size(), 1);
    const auto& metadata = sharding.metadata(0);
    EXPECT_EQ(metadata.op_type(), "_Arg");
    EXPECT_EQ(metadata.op_name(), "arg");
  };

  auto test_sharding_metadata =
      [&check_metadata](
          const std::function<StatusOr<absl::optional<xla::OpSharding>>()>&
              fn) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/tf2xla/sharding_util_test.cc", "lambda");

        auto status_or_sharding = fn();
        TF_ASSERT_OK(status_or_sharding.status());
        ASSERT_TRUE(status_or_sharding.ValueOrDie().has_value());
        auto& sharding = status_or_sharding.ValueOrDie();
        ASSERT_TRUE(sharding.has_value());
        if (sharding->type() == xla::OpSharding::TUPLE) {
          EXPECT_TRUE(sharding->metadata().empty());
          for (const auto& sharding_element : sharding->tuple_shardings()) {
            check_metadata(sharding_element);
          }
        } else {
          check_metadata(sharding.value());
        }
      };

  {
    test_sharding_metadata([&node_def]() {
      return GetShardingFromNodeDef(node_def, /*add_metadata=*/true);
    });
  }

  {
    test_sharding_metadata([&node_def]() {
      return ParseShardingFromDevice(node_def, /*num_cores_per_replica=*/1,
                                     /*add_metadata=*/true);
    });
  }

  {
    Graph graph(OpRegistry::Global());
    Status status;
    Node* node = graph.AddNode(node_def, &status);
    TF_ASSERT_OK(status);

    test_sharding_metadata([node]() {
      return ParseShardingFromDevice(*node, /*num_cores_per_replica=*/1,
                                     /*add_metadata=*/true);
    });
  }
}

xla::OpSharding CreateTupleSharding() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_util_testDTcc mht_2(mht_2_v, 304, "", "./tensorflow/compiler/tf2xla/sharding_util_test.cc", "CreateTupleSharding");

  xla::OpSharding sharding;
  sharding.set_type(xla::OpSharding::TUPLE);
  sharding.add_tuple_shardings()->set_type(xla::OpSharding::REPLICATED);
  sharding.add_tuple_shardings()->set_type(xla::OpSharding::REPLICATED);
  return sharding;
}

INSTANTIATE_TEST_SUITE_P(GetShardingFromNode, ShardingWithMetadataTest,
                         ::testing::Values(xla::sharding_builder::Replicate(),
                                           CreateTupleSharding()));

}  // namespace tensorflow
