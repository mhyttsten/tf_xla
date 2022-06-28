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
class MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/grappler/grappler.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

void optimize_func(void* optimizer, const TF_Buffer* graph_buf,
                   const TF_GrapplerItem* item, TF_Buffer* optimized_graph_buf,
                   TF_Status* tf_status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/c/experimental/grappler/grappler_test.cc", "optimize_func");
}

void PopulateDefaultParam(TP_OptimizerRegistrationParams* params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/c/experimental/grappler/grappler_test.cc", "PopulateDefaultParam");

  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  params->optimizer->create_func = nullptr;
  params->optimizer->optimize_func = optimize_func;
  params->optimizer->destroy_func = nullptr;
}

TEST(Grappler, SuccessfulRegistration) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Success";
    params->optimizer_configs->remapping = TF_TriState_Off;
  };

  TF_ASSERT_OK(InitGraphPlugin(plugin_init));
  ASSERT_EQ(PluginGraphOptimizerRegistry::CreateOptimizers(
                std::set<string>{"Success"})
                .size(),
            1);
  ConfigList config = PluginGraphOptimizerRegistry::GetPluginConfigs(
      true, std::set<string>{"Success"});
  ASSERT_EQ(config.toggle_config["remapping"], RewriterConfig::OFF);
}

TEST(Grappler, MultiplePluginRegistration) {
  auto plugin_init_0 = [](TP_OptimizerRegistrationParams* const params,
                          TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Device0";
  };
  auto plugin_init_1 = [](TP_OptimizerRegistrationParams* const params,
                          TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Device1";
  };

  TF_ASSERT_OK(InitGraphPlugin(plugin_init_0));
  TF_ASSERT_OK(InitGraphPlugin(plugin_init_1));
  ASSERT_EQ(PluginGraphOptimizerRegistry::CreateOptimizers(
                std::set<string>{"Device0", "Device1"})
                .size(),
            2);
}

TEST(Grappler, DeviceTypeNotSet) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = nullptr;
  };

  tensorflow::Status status = InitGraphPlugin(plugin_init);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(
      status.error_message(),
      "'device_type' field in TP_OptimizerRegistrationParams must be set.");
}

TEST(Grappler, OptimizeFuncNotSet) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "FuncNotSet";
    params->optimizer->optimize_func = nullptr;
  };

  tensorflow::Status status = InitGraphPlugin(plugin_init);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.error_message(),
            "'optimize_func' field in TP_Optimizer must be set.");
}

TEST(TF_GrapplerItem, NodesToPreserve) {
  GrapplerItem item;
  item.fetch = std::vector<string>{"Conv", "BiasAdd"};
  std::unordered_set<string> nodes_preserved = item.NodesToPreserve();
  TF_GrapplerItem* c_item = reinterpret_cast<TF_GrapplerItem*>(&item);

  int list_total_size = 0;
  for (const string& s : nodes_preserved) {
    list_total_size += s.size();
  }

  size_t storage_size = 0;
  int num_values = 0;
  TF_Status* status = TF_NewStatus();
  TF_GetNodesToPreserveListSize(c_item, &num_values, &storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(nodes_preserved.size(), num_values);
  EXPECT_EQ(list_total_size, storage_size);

  std::unique_ptr<char*[]> values(new char*[nodes_preserved.size()]);
  std::unique_ptr<size_t[]> lens(new size_t[nodes_preserved.size()]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetNodesToPreserveList(c_item, values.get(), lens.get(),
                            nodes_preserved.size(), storage.get(), storage_size,
                            status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (size_t i = 0; i < nodes_preserved.size(); ++i) {
    EXPECT_EQ(nodes_preserved.find(string(static_cast<const char*>(values[i]),
                                          lens[i])) != nodes_preserved.end(),
              true);
  }
  TF_DeleteStatus(status);
}

TEST(TF_GrapplerItem, FetchNodes) {
  GrapplerItem item;
  item.fetch = std::vector<string>{"Conv", "BiasAdd"};
  TF_GrapplerItem* c_item = reinterpret_cast<TF_GrapplerItem*>(&item);

  int list_total_size = 0;
  for (const string& s : item.fetch) {
    list_total_size += s.size();
  }

  size_t storage_size = 0;
  int num_values = 0;
  TF_Status* status = TF_NewStatus();
  TF_GetFetchNodesListSize(c_item, &num_values, &storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(item.fetch.size(), num_values);
  EXPECT_EQ(list_total_size, storage_size);

  std::unique_ptr<char*[]> values(new char*[item.fetch.size()]);
  std::unique_ptr<size_t[]> lens(new size_t[item.fetch.size()]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetFetchNodesList(c_item, values.get(), lens.get(), item.fetch.size(),
                       storage.get(), storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (size_t i = 0; i < item.fetch.size(); ++i) {
    EXPECT_EQ(item.fetch[i].size(), lens[i]) << i;
    EXPECT_EQ(item.fetch[i],
              string(static_cast<const char*>(values[i]), lens[i]))
        << i;
  }
  TF_DeleteStatus(status);
}

TEST(TF_GraphProperties, InputProperties) {
  std::unique_ptr<SingleMachine> cluster(new SingleMachine(5 * 60, 3, 0));
  TF_ASSERT_OK(cluster->Provision());

  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_Status* status = TF_NewStatus();
  TF_GraphProperties* graph_properties =
      TF_NewGraphProperties(reinterpret_cast<TF_GrapplerItem*>(&item));
  TF_InferStatically(graph_properties, true, false, false, false, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "AddN") {
      int num_values = 0;
      TF_GetInputPropertiesListSize(graph_properties, node.name().c_str(),
                                    &num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      EXPECT_EQ(num_values, 1);

      std::vector<TF_Buffer*> in_props_buf(num_values, TF_NewBuffer());

      TF_GetInputPropertiesList(graph_properties, node.name().c_str(),
                                in_props_buf.data(), num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      tensorflow::OpInfo::TensorProperties in_props;
      Status s = tensorflow::BufferToMessage(in_props_buf[0], &in_props);
      TF_ASSERT_OK(s);

      EXPECT_EQ(DT_FLOAT, in_props.dtype());
      EXPECT_FALSE(in_props.shape().unknown_rank());
      EXPECT_EQ(2, in_props.shape().dim_size());
      EXPECT_EQ(10, in_props.shape().dim(0).size());
      EXPECT_EQ(1, in_props.shape().dim(1).size());

      for (int i = 0; i < in_props_buf.size(); i++)
        TF_DeleteBuffer(in_props_buf[i]);
    }
  }
  TF_DeleteGraphProperties(graph_properties);
  TF_DeleteStatus(status);
  TF_ASSERT_OK(cluster->Shutdown());
}

TEST(TF_GraphProperties, OutputProperties) {
  std::unique_ptr<SingleMachine> cluster(new SingleMachine(5 * 60, 3, 0));
  TF_ASSERT_OK(cluster->Provision());

  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_Status* status = TF_NewStatus();
  TF_GraphProperties* graph_properties =
      TF_NewGraphProperties(reinterpret_cast<TF_GrapplerItem*>(&item));
  TF_InferStatically(graph_properties, true, false, false, false, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "AddN") {
      int num_values = 0;
      TF_GetOutputPropertiesListSize(graph_properties, node.name().c_str(),
                                     &num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      EXPECT_EQ(num_values, 1);

      std::vector<TF_Buffer*> out_props_buf(num_values, TF_NewBuffer());

      TF_GetOutputPropertiesList(graph_properties, node.name().c_str(),
                                 out_props_buf.data(), num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      tensorflow::OpInfo::TensorProperties out_props;
      Status s = tensorflow::BufferToMessage(out_props_buf[0], &out_props);
      TF_ASSERT_OK(s);

      EXPECT_EQ(DT_FLOAT, out_props.dtype());
      EXPECT_FALSE(out_props.shape().unknown_rank());
      EXPECT_EQ(2, out_props.shape().dim_size());
      EXPECT_EQ(10, out_props.shape().dim(0).size());
      EXPECT_EQ(1, out_props.shape().dim(1).size());

      for (int i = 0; i < out_props_buf.size(); i++)
        TF_DeleteBuffer(out_props_buf[i]);
    }
  }
  TF_DeleteStatus(status);
  TF_DeleteGraphProperties(graph_properties);
  TF_ASSERT_OK(cluster->Shutdown());
}

TEST(TF_FunctionLibraryDefinition, LookUpOpDef) {
  TF_Buffer* g_buf = TF_NewBuffer();
  TF_Buffer* op_buf = TF_NewBuffer();
  TF_Status* status = TF_NewStatus();
  GraphDef g_def;
  Status s = MessageToBuffer(g_def, g_buf);
  TF_ASSERT_OK(s);
  TF_FunctionLibraryDefinition* func =
      TF_NewFunctionLibraryDefinition(g_buf, status);

  TF_LookUpOpDef(func, "Add", op_buf, status);
  string actual_string(reinterpret_cast<const char*>(op_buf->data),
                       op_buf->length);
  ASSERT_EQ(TF_OK, TF_GetCode(status));

  const OpDef* expected_op_def;
  TF_ASSERT_OK(OpRegistry::Global()->LookUpOpDef("Add", &expected_op_def));
  string expected_serialized;
  expected_op_def->SerializeToString(&expected_serialized);
  EXPECT_EQ(expected_serialized, actual_string);
  TF_DeleteBuffer(g_buf);
  TF_DeleteBuffer(op_buf);
  TF_DeleteStatus(status);
  TF_DeleteFunctionLibraryDefinition(func);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
