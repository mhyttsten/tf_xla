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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc() {
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

#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class SymbolicShapesTest : public ::testing::Test {
 protected:
  TensorShapeProto MakeUnknown() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/grappler/utils/symbolic_shapes_test.cc", "MakeUnknown");

    TensorShapeProto shape;
    shape.set_unknown_rank(true);
    return shape;
  }

  TensorShapeProto MakeShape(std::vector<int> dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/grappler/utils/symbolic_shapes_test.cc", "MakeShape");

    TensorShapeProto shape;
    for (int dim_size : dims) {
      TensorShapeProto::Dim dim;
      dim.set_size(dim_size);
      *shape.add_dim() = dim;
    }
    return shape;
  }
};

bool operator<(const TensorShapeProto& lhs, const TensorShapeProto& rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapes_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/grappler/utils/symbolic_shapes_test.cc", "operator<");

  return CompareSymbolicallyShapedTensorSizes(lhs, rhs);
}

TEST_F(SymbolicShapesTest, ShapeIsSymbolicallyDefined) {
  EXPECT_FALSE(ShapeIsSymbolicallyDefined(MakeUnknown()));
  EXPECT_FALSE(ShapeIsSymbolicallyDefined(MakeShape({-1, 2})));

  EXPECT_TRUE(ShapeIsSymbolicallyDefined(MakeShape({1, 2})));
  EXPECT_TRUE(ShapeIsSymbolicallyDefined(MakeShape({-2, 2})));
}

TEST_F(SymbolicShapesTest, ShapesSymbolicallyEqual) {
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeUnknown(), MakeUnknown()));
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeShape({-1, 2}), MakeShape({-1, 2})));
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeShape({-2, 2}), MakeShape({-3, 2})));

  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({1, 2}), MakeShape({1, 2})));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, 2}), MakeShape({-2, 2})));
}

TEST_F(SymbolicShapesTest, ShapesBroadcastable) {
  EXPECT_FALSE(ShapesBroadcastable(MakeUnknown(), MakeUnknown()));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2}), MakeShape({1, -3})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-1, 2}), MakeShape({-1, 2})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2, 2}), MakeShape({-3, 2})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2, 4}), MakeShape({-2, 8})));

  EXPECT_TRUE(ShapesBroadcastable(MakeShape({1, 2}), MakeShape({1, 2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 2}), MakeShape({-2, 2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 32}), MakeShape({-2, 1})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 1}), MakeShape({1, -2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 1}), MakeShape({1, -3})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-3}), MakeShape({-2, -3})));

  TensorShapeProto output_shape;
  EXPECT_TRUE(
      ShapeAfterBroadcast(MakeShape({1, 2}), MakeShape({1, 2}), &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({1, 2}), output_shape));
  EXPECT_TRUE(ShapeAfterBroadcast(MakeShape({-2, 2}), MakeShape({-2, 2}),
                                  &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, 2}), output_shape));
  EXPECT_TRUE(ShapeAfterBroadcast(MakeShape({-2, 32}), MakeShape({-2, 1}),
                                  &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, 32}), output_shape));
  EXPECT_TRUE(ShapeAfterBroadcast(MakeShape({-2, 1}), MakeShape({1, -2}),
                                  &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, -2}), output_shape));
  EXPECT_TRUE(ShapeAfterBroadcast(MakeShape({-2, 1}), MakeShape({1, -3}),
                                  &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, -3}), output_shape));
  EXPECT_TRUE(
      ShapeAfterBroadcast(MakeShape({-3}), MakeShape({-2, -3}), &output_shape));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, -3}), output_shape));
}

TEST_F(SymbolicShapesTest, CompareSymbolicallyShapedTensorSizes) {
  EXPECT_TRUE(MakeShape({1, 1, 32}) < MakeShape({32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({2048}));
  EXPECT_TRUE(MakeShape({1, -2, 32}) < MakeShape({-2, 32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({-2, 32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({-1, 32, 32}));
  EXPECT_TRUE(MakeShape({1, -2, 32}) < MakeShape({-2, -2, 32}));

  EXPECT_FALSE(MakeShape({1, -2, 32}) < MakeShape({-3, 32, 32}));
  EXPECT_FALSE(MakeShape({1, -1, 32}) < MakeShape({1, -1, 32}));
  EXPECT_FALSE(MakeShape({1, -1, 32}) < MakeShape({-1, -1, 32}));
  EXPECT_FALSE(MakeShape({-1, -1, 32}) < MakeShape({1, -1, 32}));
}

TEST_F(SymbolicShapesTest, RankAndNumCoeff) {
  EXPECT_EQ(2, Rank(MakeShape({32, 32})));
  EXPECT_EQ(32 * 32, NumCoefficients(MakeShape({32, 32})));
  EXPECT_EQ(2, Rank(MakeShape({-2, 32})));
  EXPECT_EQ(-1, NumCoefficients(MakeShape({-2, 32})));
  TensorShapeProto shape;
  shape.set_unknown_rank(true);
  EXPECT_EQ(-1, Rank(shape));
  EXPECT_EQ(-1, NumCoefficients(shape));
}

TEST_F(SymbolicShapesTest, SizeRatio) {
  EXPECT_EQ(16, ComputeSizeRatio(MakeShape({32, 32}), MakeShape({32, 2})));
  EXPECT_EQ(16, ComputeSizeRatio(MakeShape({-2, 32}), MakeShape({-2, 2})));
  EXPECT_EQ(16,
            ComputeSizeRatio(MakeShape({-2, -2, 32}), MakeShape({-2, 2, -2})));
  EXPECT_EQ(-1,
            ComputeSizeRatio(MakeShape({-2, -2, 32}), MakeShape({-2, 2, 2})));
  EXPECT_EQ(-1,
            ComputeSizeRatio(MakeShape({-2, 2, 32}), MakeShape({-2, 2, -2})));
  EXPECT_EQ(-1, ComputeSizeRatio(MakeShape({-2, -2}), MakeShape({-2, 2})));
  EXPECT_EQ(-1, ComputeSizeRatio(MakeShape({-2, 32}), MakeShape({-2, -2})));
  EXPECT_EQ(1, ComputeSizeRatio(MakeShape({-2, -3}), MakeShape({-3, -2})));
  EXPECT_EQ(-1, ComputeSizeRatio(MakeShape({-1, 32}), MakeShape({-2, 2})));
  EXPECT_EQ(-1, ComputeSizeRatio(MakeShape({-1, 32}), MakeShape({-2, 0})));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
