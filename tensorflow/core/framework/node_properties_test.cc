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
class MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc() {
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
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_properties.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

OpDef ToOpDef(const OpDefBuilder& builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/framework/node_properties_test.cc", "ToOpDef");

  OpRegistrationData op_reg_data;
  EXPECT_TRUE(builder.Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

class MockOpRegistry : public OpRegistryInterface {
 public:
  MockOpRegistry()
      : op_reg_(ToOpDef(OpDefBuilder("Foo")
                            .Input("f: float")
                            .Input("i: int32")
                            .Output("of: double"))) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/framework/node_properties_test.cc", "MockOpRegistry");
}
  ~MockOpRegistry() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/framework/node_properties_test.cc", "~MockOpRegistry");
}

  // Returns an error status and sets *op_reg_data to nullptr if no OpDef is
  // registered under that name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/framework/node_properties_test.cc", "LookUp");

    if (op_type_name == "Foo") {
      *op_reg_data = &op_reg_;
      return Status::OK();
    } else {
      *op_reg_data = nullptr;
      return errors::InvalidArgument("Op type named ", op_type_name,
                                     " not found");
    }
  }

  const OpDef* get_op_def_addr() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/framework/node_properties_test.cc", "get_op_def_addr");
 return &op_reg_.op_def; }

 private:
  const OpRegistrationData op_reg_;
};

void ValidateNodeProperties(const NodeProperties& props, const OpDef* op_def,
                            const NodeDef& node_def,
                            const DataTypeVector& input_types,
                            const DataTypeVector& output_types) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_properties_testDTcc mht_5(mht_5_v, 249, "", "./tensorflow/core/framework/node_properties_test.cc", "ValidateNodeProperties");

  EXPECT_EQ(props.op_def, op_def);
  EXPECT_EQ(props.node_def.name(), node_def.name());
  ASSERT_EQ(props.input_types.size(), input_types.size());
  for (int i = 0; i < input_types.size(); ++i) {
    EXPECT_EQ(props.input_types[i], input_types[i]);
    EXPECT_EQ(props.input_types_slice[i], input_types[i]);
  }
  ASSERT_EQ(props.output_types.size(), output_types.size());
  for (int i = 0; i < output_types.size(); ++i) {
    EXPECT_EQ(props.output_types[i], output_types[i]);
    EXPECT_EQ(props.output_types_slice[i], output_types[i]);
  }
}

}  // namespace

TEST(NodeProperties, Contructors) {
  OpDef op_def;
  NodeDef node_def;
  node_def.set_name("foo");
  DataTypeVector input_types{DT_FLOAT, DT_INT32};
  DataTypeVector output_types{DT_DOUBLE};
  DataTypeSlice input_types_slice(input_types);
  DataTypeSlice output_types_slice(output_types);

  // Construct from slices.
  NodeProperties props_from_slices(&op_def, node_def, input_types_slice,
                                   output_types_slice);
  ValidateNodeProperties(props_from_slices, &op_def, node_def, input_types,
                         output_types);

  // Construct from vectors.
  NodeProperties props_from_vectors(&op_def, node_def, input_types,
                                    output_types);
  ValidateNodeProperties(props_from_vectors, &op_def, node_def, input_types,
                         output_types);
}

TEST(NodeProperties, CreateFromNodeDef) {
  MockOpRegistry op_registry;
  NodeDef node_def;
  node_def.set_name("bar");
  node_def.set_op("Foo");
  node_def.add_input("f_in");
  node_def.add_input("i_in");

  std::shared_ptr<const NodeProperties> props;
  EXPECT_TRUE(
      NodeProperties::CreateFromNodeDef(node_def, &op_registry, &props).ok());

  DataTypeVector input_types{DT_FLOAT, DT_INT32};
  DataTypeVector output_types{DT_DOUBLE};
  ValidateNodeProperties(*props, op_registry.get_op_def_addr(), node_def,
                         input_types, output_types);

  // The OpDef lookup should fail for this one:
  node_def.set_op("Baz");
  std::shared_ptr<const NodeProperties> props_bad;
  EXPECT_FALSE(
      NodeProperties::CreateFromNodeDef(node_def, &op_registry, &props_bad)
          .ok());
  EXPECT_EQ(props_bad, nullptr);
}
}  // namespace tensorflow
