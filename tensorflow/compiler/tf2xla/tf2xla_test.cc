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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc() {
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

#include "tensorflow/compiler/tf2xla/tf2xla.h"

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

AttrValue TypeAttrValue(DataType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/tf2xla_test.cc", "TypeAttrValue");

  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
}

GraphDef SumGraph() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/tf2xla/tf2xla_test.cc", "SumGraph");

  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* sum = graph_def.add_node();
  sum->set_name("sum");
  sum->set_op("Add");
  sum->add_input("x");
  sum->add_input("y");
  (*sum->mutable_attr())["T"] = TypeAttrValue(DT_INT32);
  return graph_def;
}

tf2xla::Config SumConfig() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/tf2xla/tf2xla_test.cc", "SumConfig");

  tf2xla::Config config;
  config.add_feed()->mutable_id()->set_node_name("x");
  config.add_feed()->mutable_id()->set_node_name("y");
  config.add_fetch()->mutable_id()->set_node_name("sum");
  return config;
}

TEST(ConvertGraphDefToXla, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, client, &computation));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32>(32);
  auto x_global_or = client->TransferToServer(x_literal);
  auto y_global_or = client->TransferToServer(y_literal);
  TF_EXPECT_OK(x_global_or.status());
  TF_EXPECT_OK(y_global_or.status());
  std::unique_ptr<xla::GlobalData> x_global =
      std::move(x_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> y_global =
      std::move(y_global_or.ValueOrDie());

  // Execute and check result.
  auto result_or =
      client->ExecuteAndTransfer(computation, {x_global.get(), y_global.get()});
  TF_EXPECT_OK(result_or.status());
  xla::Literal result = std::move(result_or.ValueOrDie());
  EXPECT_EQ("(\ns32[] 42\n)", result.ToString());

  config.mutable_feed(0)->mutable_id()->set_output_index(
      123); /* invalid output_index */
  EXPECT_TRUE(errors::IsInvalidArgument(
      ConvertGraphDefToXla(graph_def, config, client, &computation)));
}

TEST(ConvertGraphDefToXla, SumWithUnusedArgument) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();
  NodeDef* unused = graph_def.add_node();
  unused->set_name("unused");
  unused->set_op("Placeholder");
  (*unused->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  config.add_feed()->mutable_id()->set_node_name("unused");

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, client, &computation));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32>(32);
  auto x_global_or = client->TransferToServer(x_literal);
  auto y_global_or = client->TransferToServer(y_literal);
  auto unused_global_or = client->TransferToServer(y_literal);
  TF_EXPECT_OK(x_global_or.status());
  TF_EXPECT_OK(y_global_or.status());
  TF_EXPECT_OK(unused_global_or.status());
  std::unique_ptr<xla::GlobalData> x_global =
      std::move(x_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> y_global =
      std::move(y_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> unused_global =
      std::move(unused_global_or.ValueOrDie());

  // Execute and check result.
  auto result_or = client->ExecuteAndTransfer(
      computation, {x_global.get(), y_global.get(), unused_global.get()});
  TF_EXPECT_OK(result_or.status());
  xla::Literal result = std::move(result_or.ValueOrDie());
  EXPECT_EQ("(\ns32[] 42\n)", result.ToString());
}

}  // namespace
}  // namespace tensorflow
