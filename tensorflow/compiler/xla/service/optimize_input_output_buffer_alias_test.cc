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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc() {
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

#include "tensorflow/compiler/xla/service/optimize_input_output_buffer_alias.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// Tests that UserBufferAlias properly maps input and output buffer indices of
// various shapes for aliasing.
class OptimizeInputOutputBufferAliasTest : public HloTestBase {
 protected:
  OptimizeInputOutputBufferAliasTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/optimize_input_output_buffer_alias_test.cc", "OptimizeInputOutputBufferAliasTest");

    r1f32_ = ShapeUtil::MakeShape(F32, {4});
    r2f32_ = ShapeUtil::MakeShape(F32, {4, 5});
    r3f32_ = ShapeUtil::MakeShape(F32, {4, 5, 6});
    r4f32_ = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});

    optimize_pass_ = absl::make_unique<OptimizeInputOutputBufferAlias>();
  }

  // Returns the number of output indices that aliases with the input.
  int64_t AliasCount() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/optimize_input_output_buffer_alias_test.cc", "AliasCount");

    int64_t count = 0;

    config_.ForEachAlias(
        [&](const ShapeIndex&, const HloInputOutputAliasConfig::Alias&) {
          count++;
        });
    return count;
  }

  bool BuildAliasConfig(const Shape& input_shape, const Shape& output_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoptimize_input_output_buffer_alias_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/optimize_input_output_buffer_alias_test.cc", "BuildAliasConfig");

    config_ = HloInputOutputAliasConfig(output_shape);
    auto changed = optimize_pass_->Build(input_shape, output_shape, &config_);
    TF_CHECK_OK(changed.status());

    return changed.ValueOrDie();
  }

  std::unique_ptr<OptimizeInputOutputBufferAlias> optimize_pass_;

  HloInputOutputAliasConfig config_;

  Shape r1f32_;
  Shape r2f32_;
  Shape r3f32_;
  Shape r4f32_;
};

// All shapes are different, so no aliasing is available.
TEST_F(OptimizeInputOutputBufferAliasTest, AllDifferentBufferSizes) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_});
  Shape output = ShapeUtil::MakeTupleShape({r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// Input and output shapes are equal, so buffers can alias at the same index.
TEST_F(OptimizeInputOutputBufferAliasTest, OrderedNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);
  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex{1});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{2});
  EXPECT_EQ(config_.GetAliasedOutput(0, {3}), ShapeIndex{3});
}

// Only a subset of the tuple element shapes match between the input and the
// output.
TEST_F(OptimizeInputOutputBufferAliasTest, PartialReuseNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r1f32_, r2f32_, r2f32_});
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 2);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{1});
}

// The output shape is reverse of the input shape, but we can still reuse all
// the buffers.
TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape({r4f32_, r3f32_, r2f32_, r1f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{3});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex{2});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{1});
  EXPECT_EQ(config_.GetAliasedOutput(0, {3}), ShapeIndex{0});
}

TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({r1f32_}), r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape(
      {r1f32_, ShapeUtil::MakeTupleShape({r3f32_, r2f32_}), r2f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 3);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0, 0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex({1, 1}));
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex({1, 0}));
}

}  // namespace xla
