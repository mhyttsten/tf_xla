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
class MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc {
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
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/example/example_parser_configuration.h"

#include <vector>

#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

Status FindNodeIndexByName(const tensorflow::GraphDef& graph,
                           const string& node_name, int* node_idx) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/example/example_parser_configuration.cc", "FindNodeIndexByName");

  for (int i = 0; i < graph.node_size(); ++i) {
    const auto& node = graph.node(i);
    if (node.name() == node_name) {
      *node_idx = i;
      return Status::OK();
    }
  }
  return errors::InvalidArgument(node_name, " not found in GraphDef");
}

Status ExtractExampleParserConfiguration(
    const tensorflow::GraphDef& graph, const string& node_name,
    tensorflow::Session* session,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/example/example_parser_configuration.cc", "ExtractExampleParserConfiguration");

  int node_idx;
  TF_RETURN_IF_ERROR(FindNodeIndexByName(graph, node_name, &node_idx));

  const auto& node = graph.node(node_idx);
  if (node.op() != "ParseExample") {
    return errors::InvalidArgument(node_name, " node is not a ParseExample op");
  }

  auto& attr_map = node.attr();
  auto num_sparse = attr_map.at("Nsparse").i();
  auto num_dense = attr_map.at("Ndense").i();
  fixed_len_features->resize(num_dense);
  var_len_features->resize(num_sparse);

  auto tdense = attr_map.at("Tdense");
  auto dense_shapes = attr_map.at("dense_shapes");
  auto sparse_types = attr_map.at("sparse_types");

  // Consistency check attributes.
  if (tdense.list().type_size() != num_dense) {
    return errors::InvalidArgument("Node attr Tdense has ",
                                   tdense.list().type_size(),
                                   " elements != Ndense attr: ", num_dense);
  }

  if (dense_shapes.list().shape_size() != num_dense) {
    return errors::InvalidArgument("Node attr dense_shapes has ",
                                   dense_shapes.list().shape_size(),
                                   " elements != Ndense attr: ", num_dense);
  }

  if (sparse_types.list().type_size() != num_sparse) {
    return errors::InvalidArgument("Node attr sparse_types has ",
                                   sparse_types.list().type_size(),
                                   " elements != NSparse attr: ", num_sparse);
  }

  for (int i = 0; i < tdense.list().type_size(); ++i) {
    (*fixed_len_features)[i].dtype = tdense.list().type(i);
    // Convert TensorShapeProto to TensorShape.
    (*fixed_len_features)[i].shape = TensorShape(dense_shapes.list().shape(i));
  }

  for (int i = 0; i < sparse_types.list().type_size(); ++i) {
    (*var_len_features)[i].dtype = sparse_types.list().type(i);
  }

  // We must fetch the configuration input tensors to the ParseExample op.
  // Skipping index = 0, which is the serialized proto input.
  std::vector<string> fetch_names(node.input_size() - 1);
  for (int i = 1; i < node.input_size(); ++i) {
    fetch_names[i - 1] = node.input(i);
  }

  std::vector<Tensor> op_input_tensors;

  TF_RETURN_IF_ERROR(session->Run({},               // no_inputs,
                                  fetch_names, {},  // no target_node_names,
                                  &op_input_tensors));

  // The input tensors are laid out sequentially in a flat manner.
  // Here are the various start offsets.
  int sparse_keys_start = 1;
  int dense_keys_start = sparse_keys_start + num_sparse;
  int dense_defaults_start = dense_keys_start + num_dense;

  for (int i = 0; i < num_sparse; ++i) {
    int input_idx = sparse_keys_start + i;
    (*var_len_features)[i].key =
        op_input_tensors[input_idx].scalar<tstring>()();
  }

  for (int i = 0; i < num_dense; ++i) {
    FixedLenFeature& config = (*fixed_len_features)[i];
    int dense_keys_offset = dense_keys_start + i;
    config.key = op_input_tensors[dense_keys_offset].scalar<tstring>()();

    int defaults_offset = dense_defaults_start + i;
    config.default_value = op_input_tensors[defaults_offset];
  }

  // The output tensors are laid out sequentially in a flat manner.
  // Here are the various start offsets.
  int sparse_indices_output_start = 0;
  int sparse_values_output_start = sparse_indices_output_start + num_sparse;
  int sparse_shapes_output_start = sparse_values_output_start + num_sparse;
  int dense_values_output_start = sparse_shapes_output_start + num_sparse;

  string node_output_prefix = strings::StrCat(node_name, ":");

  for (int i = 0; i < num_sparse; ++i) {
    VarLenFeature& config = (*var_len_features)[i];

    int indices_offset = sparse_indices_output_start + i;
    config.indices_output_tensor_name =
        strings::StrCat(node_output_prefix, indices_offset);

    int values_offset = sparse_values_output_start + i;
    config.values_output_tensor_name =
        strings::StrCat(node_output_prefix, values_offset);

    int shapes_offset = sparse_shapes_output_start + i;
    config.shapes_output_tensor_name =
        strings::StrCat(node_output_prefix, shapes_offset);
  }

  for (int i = 0; i < num_dense; ++i) {
    int output_idx = dense_values_output_start + i;
    (*fixed_len_features)[i].values_output_tensor_name =
        strings::StrCat(node_output_prefix, output_idx);
  }
  return Status::OK();
}

Status ExampleParserConfigurationProtoToFeatureVectors(
    const ExampleParserConfiguration& config_proto,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configurationDTcc mht_2(mht_2_v, 343, "", "./tensorflow/core/example/example_parser_configuration.cc", "ExampleParserConfigurationProtoToFeatureVectors");

  const auto& feature_map = config_proto.feature_map();
  for (auto it = feature_map.cbegin(); it != feature_map.cend(); ++it) {
    string key = it->first;
    const auto& config = it->second;
    if (config.has_fixed_len_feature()) {
      const auto& fixed_config = config.fixed_len_feature();
      FixedLenFeature f;
      f.key = key;
      f.dtype = fixed_config.dtype();
      f.shape = TensorShape(fixed_config.shape());
      Tensor default_value(f.dtype, f.shape);
      if (!default_value.FromProto(fixed_config.default_value())) {
        return errors::InvalidArgument(
            "Invalid default_value in config proto ",
            fixed_config.default_value().DebugString());
      }
      f.default_value = default_value;
      f.values_output_tensor_name = fixed_config.values_output_tensor_name();
      fixed_len_features->push_back(f);
    } else {
      const auto& var_len_config = config.var_len_feature();
      VarLenFeature v;
      v.key = key;
      v.dtype = var_len_config.dtype();
      v.values_output_tensor_name = var_len_config.values_output_tensor_name();
      v.indices_output_tensor_name =
          var_len_config.indices_output_tensor_name();
      v.shapes_output_tensor_name = var_len_config.shapes_output_tensor_name();
      var_len_features->push_back(v);
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
