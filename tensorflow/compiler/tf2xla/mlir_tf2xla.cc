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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc() {
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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace tensorflow {

namespace {

// A fake device to simulate the presence of a CPU.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/tf2xla/mlir_tf2xla.cc", "FakeDevice");
}

  Status Sync() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/tf2xla/mlir_tf2xla.cc", "Sync");
 return errors::Unimplemented("FakeDevice::Sync()"); }
};

// Translates the graph input information from tf2xla:::Config to
// GraphImportConfig.
Status ConvertInputInfo(
    const tf2xla::Config& config,
    const std::unordered_map<std::string, std::string>& feed_name_remap,
    GraphImportConfig* specs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/tf2xla/mlir_tf2xla.cc", "ConvertInputInfo");

  std::vector<std::string> array_names;
  std::vector<std::string> data_types;
  std::vector<llvm::Optional<std::vector<int>>> shapes;
  for (const tf2xla::Feed& feed : config.feed()) {
    std::string place_holder_name =
        feed_name_remap.at(TensorIdToString(feed.id()));
    array_names.push_back(place_holder_name);
    data_types.push_back(
        feed.type() == DT_INVALID ? "" : DataType_Name(feed.type()));
    if (feed.shape().unknown_rank()) {
      shapes.push_back(llvm::None);
      continue;
    }
    std::vector<int> dims;
    dims.reserve(feed.shape().dim_size());
    absl::c_for_each(feed.shape().dim(), [&](const TensorShapeProto::Dim d) {
      dims.push_back(d.size());
    });
    shapes.push_back(dims);
  }

  return ParseInputArrayInfo(array_names, data_types, shapes, &specs->inputs);
}

// Translates the graph output information from tf2xla:::Config to
// GraphImportConfig.
Status ConvertOutputInfo(const tf2xla::Config& config,
                         GraphImportConfig* specs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/tf2xla/mlir_tf2xla.cc", "ConvertOutputInfo");

  std::vector<std::string> array_names;
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    array_names.push_back(fetch.id().node_name());
  }

  return ParseOutputArrayInfo(array_names, &specs->outputs);
}

}  // namespace

Status ConvertGraphDefToXlaViaMlir(
    GraphDef graph_def, const tf2xla::Config& config,
    xla::XlaComputation* computation, absl::string_view debug_info_filename,
    absl::string_view debug_info_path_begin_marker) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("debug_info_filename: \"" + std::string(debug_info_filename.data(), debug_info_filename.size()) + "\"");
   mht_4_v.push_back("debug_info_path_begin_marker: \"" + std::string(debug_info_path_begin_marker.data(), debug_info_path_begin_marker.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_tf2xlaDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/tf2xla/mlir_tf2xla.cc", "ConvertGraphDefToXlaViaMlir");

  // AddPlaceholdersForFeeds prepares for PruneGraphDefInto and serves two
  // purposes: (1) It creates a placeholder node for each feed, so that
  // PruneGraphDefInfo can prune away the node containing the feed. (2) It
  // is also a workaround for b/149029125. It replaces a feed representation
  // with a placeholder node that contains a single output.
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graph_def.library());
  std::unique_ptr<Graph> graph(new Graph(flib_def));
  std::unordered_map<string, string> feed_name_remap;
  TF_RETURN_IF_ERROR(AddPlaceholdersForFeeds(config, graph->op_registry(),
                                             &feed_name_remap, &graph_def));

  // TODO(b/149024678): remove this workaround after the ticket is fixed.
  //   Prune the GraphDef because MLIR importer doesn't allow unknown ops in
  //   graph nodes even the nodes are not needed for computing the outputs.
  GraphDef pruned_graph_def;
  TF_RETURN_IF_ERROR(PruneGraphDefInto(config, graph_def, &pruned_graph_def));

  GraphImportConfig specs;
  specs.prune_unused_nodes = false;
  specs.convert_legacy_fed_inputs = false;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  TF_RETURN_IF_ERROR(ConvertInputInfo(config, feed_name_remap, &specs));
  TF_RETURN_IF_ERROR(ConvertOutputInfo(config, &specs));

  GraphDebugInfo debug_info;
  if (!debug_info_filename.empty()) {
    TF_RETURN_IF_ERROR(LoadProtoFromFile(debug_info_filename, &debug_info));

    if (!debug_info_path_begin_marker.empty()) {
      for (size_t i = 0, e = debug_info.files_size(); i < e; ++i) {
        std::string* file_name = debug_info.mutable_files(i);
        size_t location =
            file_name->rfind(std::string(debug_info_path_begin_marker));
        if (location != std::string::npos) {
          *file_name = file_name->substr(location +
                                         debug_info_path_begin_marker.length());
        }
      }
    }
  }

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      ConvertGraphdefToMlir(pruned_graph_def, debug_info, specs, &context));

  // Construct a CPU device and add the device to the operations.
  DeviceSet device_set;
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:CPU:0");
  attr.set_device_type(DeviceType("CPU").type());
  FakeDevice device(attr);
  device_set.AddDevice(&device);
  AddDevicesToOp(*module, &device_set);

  TF_RETURN_IF_ERROR(mlir::TF::RunBridgeWithStandardPipeline(
      *module, /*enable_logging=*/VLOG_IS_ON(1), /*enable_inliner=*/true));

  // Convert the MLIR module to XLA computation. If the input graph can't be
  // lowered down to a single graph node with a single island by the previous
  // step, this step will return an error.
  return ConvertMLIRToXlaComputation(
      *module, /*device_type=*/"XLA_CPU_JIT", computation,
      /*use_tuple_args=*/false, /*prefer_tf2xla=*/false,
      /*return_tuple=*/true);
}

}  // namespace tensorflow
