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
class MHTracer_DTPStensorflowPScorePSopsPSlookup_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSlookup_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSlookup_ops_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(LookupOpsTest, LookupTableFindV2_ShapeFn) {
  ShapeInferenceTestOp op("LookupTableFindV2");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[?];?;?");
  TF_ASSERT_OK(NodeDefBuilder("test", "LookupTableFindV2")
                   .Input({"table_handle", 0, DT_RESOURCE})
                   .Input({"keys", 0, DT_INT64})
                   .Input({"default_value", 0, DT_FLOAT})
                   .Attr("Tin", DT_INT64)
                   .Attr("Tout", DT_FLOAT)
                   .Finalize(&op.node_def));
  std::vector<std::vector<ShapeInferenceTestOp::ShapeAndType>> types;
  auto set_types = [&op, &types](DataType key_type, DataType value_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_ops_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/ops/lookup_ops_test.cc", "lambda");

    types.emplace_back();
    auto& table = types.back();
    table.emplace_back("[3]", key_type);
    table.emplace_back("[4]", value_type);
    op.input_resource_handle_shapes_and_types = {&table, nullptr, nullptr};
  };
  // If there's no input handle shapes and types, output shape is unknown.
  INFER_OK(op, "[];[?,3];[4]", "?");
  // Set input handle with mismatched key type.
  set_types(DT_INT32, DT_FLOAT);
  INFER_ERROR("read value with wrong dtype", op, "[];[?,3];[4]");
  // Set input handle with mismatched value type.
  set_types(DT_INT64, DT_INT64);
  INFER_ERROR("read value with wrong dtype", op, "[];[?,3];[4]");
  // Set input handle with matched types.
  set_types(DT_INT64, DT_FLOAT);
  INFER_OK(op, "[];[?,3];[4]", "[d1_0,4]");
  INFER_OK(op, "[];[1,3];[4]", "[d1_0,4]");
  INFER_OK(op, "[];[1,?];[4]", "[d1_0,4]");
}

TEST(LookupOpsTest, LookupTableExportV2_ShapeFn) {
  ShapeInferenceTestOp op("LookupTableExportV2");
  TF_ASSERT_OK(NodeDefBuilder("test", "LookupTableExportV2")
                   .Input({"table_handle", 0, DT_RESOURCE})
                   .Attr("Tkeys", DT_INT64)
                   .Attr("Tvalues", DT_FLOAT)
                   .Finalize(&op.node_def));
  std::vector<std::vector<ShapeInferenceTestOp::ShapeAndType>> types;
  auto set_types = [&op, &types](DataType key_type, DataType value_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_ops_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/ops/lookup_ops_test.cc", "lambda");

    types.emplace_back();
    auto& table = types.back();
    table.emplace_back("[3]", key_type);
    table.emplace_back("[4]", value_type);
    op.input_resource_handle_shapes_and_types = {&table};
  };
  // Set input handle with mismatched key type.
  set_types(DT_INT32, DT_FLOAT);
  INFER_ERROR("read value with wrong dtype", op, "[]");
  // Set input handle with mismatched value type.
  set_types(DT_INT64, DT_INT64);
  INFER_ERROR("read value with wrong dtype", op, "[]");
  // Set input handle with matched types.
  set_types(DT_INT64, DT_FLOAT);
  INFER_OK(op, "[]", "?;?");
}

// TODO(b/169969017): add shape fn tests for rest of the ops.

}  // namespace
}  // namespace tensorflow
