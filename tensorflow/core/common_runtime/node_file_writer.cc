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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc() {
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

#include "tensorflow/core/common_runtime/node_file_writer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace {

// Avoiding writing to disk very commonly executed ops that are known to be
// deterministic. This reduces the filesize.
const absl::flat_hash_set<std::string>* const kOpsToSkipWriting =
    new absl::flat_hash_set<std::string>{"Add",
                                         "AddV2",
                                         "BroadcastTo",
                                         "Cast",
                                         "ConcatV2",
                                         "Const",
                                         "_EagerConst",
                                         "Enter",
                                         "Exit",
                                         "Fill",
                                         "_HostSend",
                                         "Identity",
                                         "Less",
                                         "MatrixDiagV3",
                                         "Merge",
                                         "Mul",
                                         "NextIteration",
                                         "Pack",
                                         "RandomStandardNormal",
                                         "RandomUniform",
                                         "Range",
                                         "RealDiv",
                                         "Reshape",
                                         "_Send",
                                         "Shape",
                                         "StridedSlice",
                                         "Sub",
                                         "Switch",
                                         "Transpose",
                                         "_XlaCompile"};

// If a host int32 input has at most this many elements, the tensor value will
// be written to the file.
const int kMaxInt32Elems = 10;

}  // namespace

namespace tensorflow {

/*static*/ StatusOr<NodeFileWriter*>
tensorflow::NodeFileWriter::GetNodeFileWriterIfEnabled(
    const std::string& device_name, Env* env) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc mht_0(mht_0_v, 241, "", "./tensorflow/core/common_runtime/node_file_writer.cc", "tensorflow::NodeFileWriter::GetNodeFileWriterIfEnabled");

  // First get the directory from TF_NODE_FILE_WRITER_DIRECTORY.
  static const std::string* const node_dir = [] {
    std::string node_dir;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_NODE_FILE_WRITER_DIRECTORY", "", &node_dir));
    if (node_dir == "test_undeclared_outputs_dir") {
      bool env_set = io::GetTestUndeclaredOutputsDir(&node_dir);
      if (!env_set || node_dir.empty()) {
        LOG(WARNING)
            << "TF_NODE_FILE_WRITER_DIRECTORY was set to "
               "'test_undeclared_outputs_dir', but the environmental "
               "variable TEST_UNDECLARED_OUTPUTS_DIR does not exist or "
               "is empty. NodeDef collection will be skipped.";
      } else {
        node_dir = io::JoinPath(node_dir, "node_defs");
      }
    }
    return new std::string{node_dir};
  }();
  if (node_dir->empty()) {
    return nullptr;
  }

  static mutex mu(LINKER_INITIALIZED);
  // Cache a NodeFileWriter* for each device name, so that different Sessions
  // each share the same NodeFileWriters. Sharing NodeFileWriters reduces the
  // total size of the outputted files, since it means if multiple Sessions run
  // the same op, the op is only written recorded to disk once.
  static auto* device_name_to_writer =
      new absl::flat_hash_map<std::string, NodeFileWriter*>{};
  mutex_lock l(mu);
  auto it = device_name_to_writer->find(device_name);
  if (it == device_name_to_writer->end()) {
    Status s = env->CreateDir(*node_dir);
    if (!s.ok() && s.code() != error::ALREADY_EXISTS) {
      return s;
    }

    // Put the device name in the filename for debugging purposes. Also append
    // random number in case multiple processes write out nodes concurrently.
    std::string filename = strings::StrCat(
        "node_defs", absl::StrReplaceAll(device_name, {{"/", "_"}, {":", "_"}}),
        "_", random::New64());

    auto* writer = new NodeFileWriter{io::JoinPath(*node_dir, filename)};
    s = writer->Init(env);
    if (!s.ok()) {
      delete writer;
      return s;
    }
    it = device_name_to_writer->insert({device_name, writer}).first;
  }
  return it->second;
}

Status NodeFileWriter::RecordNodeExecution(OpKernel* op_kernel,
                                           OpKernelContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc mht_1(mht_1_v, 301, "", "./tensorflow/core/common_runtime/node_file_writer.cc", "NodeFileWriter::RecordNodeExecution");

  if (kOpsToSkipWriting->count(op_kernel->type_string())) {
    return Status::OK();
  }
  NodeDef def;
  def.set_name("NodeFileWriter");
  def.set_op(op_kernel->def().op());
  *def.mutable_attr() = op_kernel->def().attr();
  // The input shapes/dtypes are stored in the 'attr' section of the NodeDef
  AttrValue& input_shapes = (*def.mutable_attr())["_input_shapes"];
  AttrValue& input_dtypes = (*def.mutable_attr())["_input_dtypes"];
  for (int i = 0; i < context->num_inputs(); i++) {
    if (!context->has_input(i) || context->input_is_ref(i)) {
      // Calling context->input(i) requires the input to exist and not be a ref,
      // so return immediately if that is not the case.
      return Status::OK();
    }
    TensorShapeProto* shape_proto = input_shapes.mutable_list()->add_shape();
    const Tensor& input = context->input(i);
    input.shape().AsProto(shape_proto);
    input_dtypes.mutable_list()->add_type(context->input_dtype(i));
    // Store small int32 host inputs, as they often represent shapes.
    if (input.NumElements() <= kMaxInt32Elems && input.dtype() == DT_INT32 &&
        context->input_memory_type(i) == HOST_MEMORY) {
      AttrValue& input_tensor =
          (*def.mutable_attr())[strings::StrCat("_input_tensor_", i)];
      input.AsProtoField(input_tensor.mutable_tensor());
    } else if (!DataTypeIsFloating(input.dtype())) {
      // Skip ops with non-floating-point inputs, since these are not useful
      // when testing determinism.
      return Status::OK();
    }
  }
  return MaybeWriteNodeDefToFile(def);
}

Status NodeFileWriter::MaybeWriteNodeDefToFile(const NodeDef& def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSnode_file_writerDTcc mht_2(mht_2_v, 340, "", "./tensorflow/core/common_runtime/node_file_writer.cc", "NodeFileWriter::MaybeWriteNodeDefToFile");

  std::string def_str = def.SerializeAsString();
  uint64 size = def_str.size();
  std::string size_as_str;
  // The file consists of a series of records, each consisting of a 64-bit
  // little endian integer representing the size of the serialized NodeDef,
  // followed by the serialized NodeDef.
  for (unsigned int i = 0; i < 8; i++) {
    size_as_str.push_back((size >> (i * 8)) & 0xff);
  }

  EqualGraphDefOptions options;
  options.ignore_internal_attrs = false;
  uint64 hash = NodeDefHash(def, options);

  mutex_lock l{mu_};
  if (written_hashes_.count(hash) == 0) {
    TF_RETURN_IF_ERROR(node_def_file_->Append(size_as_str));
    TF_RETURN_IF_ERROR(node_def_file_->Append(def_str));
    written_hashes_.insert(hash);
    // Flush after each write, since NodeFileWriters are never destructed so the
    // file is never closed.
    TF_RETURN_IF_ERROR(node_def_file_->Flush());
  }
  return Status::OK();
}

}  // namespace tensorflow
