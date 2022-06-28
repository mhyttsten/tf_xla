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
class MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc() {
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

#include "tensorflow/core/data/serialization_utils.h"

#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph_def_builder.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";
constexpr char kComponent[] = "component";
constexpr char kNumComponents[] = "num_components";
constexpr char kNumElements[] = "num_elements";
constexpr char kIsDataset[] = ".is_dataset";
constexpr char kOutputNode[] = ".output_node";

// We assume that all keys are of the form <iterator_prefix>:<name>. We extract
// the iterator name by getting rid of everything post the final colon.
Status GetIteratorName(StringPiece key, string* name) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/data/serialization_utils.cc", "GetIteratorName");

  if (!str_util::StartsWith(key, data::kFullNameRandomHex)) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  std::vector<string> split_keys = str_util::Split(key, data::kPipe);
  if (split_keys.size() != 2) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  string real_key = split_keys[1];
  const int pos = real_key.rfind(kColon);
  *name = real_key.substr(0, pos);
  return Status::OK();
}

Status FromGraphDef(FunctionLibraryRuntime* flr, const GraphDef& graph_def,
                    const std::vector<std::pair<string, Tensor>>& input_list,
                    const string& output_node, Tensor* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("output_node: \"" + output_node + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/data/serialization_utils.cc", "FromGraphDef");

  FunctionLibraryRuntime* cloned_flr = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  TF_RETURN_IF_ERROR(flr->Clone(&lib_def, &pflr, &cloned_flr, true));
  TF_RETURN_IF_ERROR(AddToFunctionLibrary(lib_def.get(), graph_def.library()));
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(cloned_flr->device());
  TF_RETURN_IF_ERROR(graph_runner.Run(&graph, cloned_flr, input_list,
                                      {output_node}, &outputs));
  *result = outputs[0];
  return Status::OK();
}

// FindStatefulOps searches `graph_def` for all of its stateful ops storing
// their names in `stateful_op_names`.
Status FindStatefulOps(const GraphDef& graph_def,
                       std::vector<string>* stateful_op_names) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/data/serialization_utils.cc", "FindStatefulOps");

  FunctionLibraryDefinition lib_def(OpRegistry::Global(), graph_def.library());

  // Iterate over all nodes in the graph.
  for (const auto& node : graph_def.node()) {
    // Each Dataset graph has a _Retval op in the end which is marked stateful
    if (node.op() == FunctionLibraryDefinition::kRetOp) continue;
    if (!IsNodeStateful(lib_def, node).ok()) {
      stateful_op_names->push_back(node.op());
    }
  }

  // Iterate over all functions.
  for (const auto& fdef : graph_def.library().function()) {
    if (!fdef.signature().is_stateful()) continue;
    for (const auto& node : fdef.node_def()) {
      if (!IsNodeStateful(lib_def, node).ok()) {
        stateful_op_names->push_back(
            absl::StrCat(node.op(), " in function: ", fdef.signature().name()));
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status ReadElementsFromCheckpoint(IteratorContext* ctx,
                                  IteratorStateReader* reader,
                                  StringPiece key_prefix,
                                  std::vector<std::vector<Tensor>>* elements) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_3(mht_3_v, 287, "", "./tensorflow/core/data/serialization_utils.cc", "ReadElementsFromCheckpoint");

  int64_t num_elements;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_prefix, kNumElements, &num_elements));
  DCHECK(elements->empty());
  elements->reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    std::string element_prefix = absl::StrCat(key_prefix, "::", i);
    int64_t num_components;
    TF_RETURN_IF_ERROR(
        reader->ReadScalar(element_prefix, kNumComponents, &num_components));
    elements->emplace_back();
    std::vector<Tensor>& element = elements->at(i);
    element.reserve(num_components);
    for (int j = 0; j < num_components; ++j) {
      element.emplace_back();
      TF_RETURN_IF_ERROR(reader->ReadTensor(
          ctx->flr(), element_prefix, absl::StrCat(kComponent, "[", j, "]"),
          &element.back()));
    }
  }
  return Status::OK();
}

Status WriteElementsToCheckpoint(
    IteratorStateWriter* writer, StringPiece key_prefix,
    const std::vector<std::vector<Tensor>>& elements) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_4(mht_4_v, 316, "", "./tensorflow/core/data/serialization_utils.cc", "WriteElementsToCheckpoint");

  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_prefix, kNumElements, elements.size()));
  for (int i = 0; i < elements.size(); ++i) {
    const std::vector<Tensor>& element = elements[i];
    std::string element_prefix = absl::StrCat(key_prefix, "::", i);
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(element_prefix, kNumComponents, element.size()));
    for (int j = 0; j < elements[i].size(); ++j) {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          element_prefix, absl::StrCat(kComponent, "[", j, "]"), element[j]));
    }
  }
  return Status::OK();
}

VariantTensorDataReader::VariantTensorDataReader(
    const std::vector<const tensorflow::VariantTensorData*>& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_5(mht_5_v, 336, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::VariantTensorDataReader");

  for (const auto& d : data) {
    string metadata;
    d->get_metadata(&metadata);
    auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
    const string name = keys[0];
    data_[name] = d;
    map_[name] = std::map<string, size_t>();
    for (size_t i = 1; i < keys.size(); ++i) {
      map_[name][keys[i]] = i - 1;
    }
  }
}

Status VariantTensorDataReader::ReadScalar(StringPiece key,
                                           int64_t* val) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_6(mht_6_v, 354, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadScalar");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalar(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           int64_t* val) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_7(mht_7_v, 364, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadScalar");

  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece key,
                                           tstring* val) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_8(mht_8_v, 372, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadScalar");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalar(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           tstring* val) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_9(mht_9_v, 382, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadScalar");

  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece key, Tensor* val) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_10(mht_10_v, 389, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadTensor");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensor(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                           StringPiece key, Tensor* val) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_11(mht_11_v, 399, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadTensor");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensorInternal(flr, name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece name, StringPiece key,
                                           Tensor* val) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_12(mht_12_v, 409, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadTensor");

  return ReadTensor(/*flr=*/nullptr, name, key, val);
}

Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                           StringPiece name, StringPiece key,
                                           Tensor* val) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_13(mht_13_v, 418, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadTensor");

  return ReadTensorInternal(flr, name, key, val);
}

bool VariantTensorDataReader::Contains(StringPiece key) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_14(mht_14_v, 425, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::Contains");

  string name;
  if (!GetIteratorName(key, &name).ok()) {
    return false;
  }
  return Contains(name, key);
}

bool VariantTensorDataReader::Contains(StringPiece n, StringPiece key) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_15(mht_15_v, 436, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::Contains");

  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return false;
  }
  const auto& bucket = it->second;
  return bucket.find(string(key)) != bucket.end();
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece n,
                                                   StringPiece key,
                                                   T* val) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_16(mht_16_v, 452, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadScalarInternal");

  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return errors::NotFound(name);
  }
  const auto& bucket = it->second;
  auto key_it = bucket.find(string(key));
  if (key_it == bucket.end()) {
    return errors::NotFound(key);
  }
  *val = data_.at(name)->tensors(key_it->second).scalar<T>()();
  return Status::OK();
}

Status VariantTensorDataReader::ReadTensorInternal(FunctionLibraryRuntime* flr,
                                                   StringPiece n,
                                                   StringPiece key,
                                                   Tensor* val) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_17(mht_17_v, 473, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadTensorInternal");

  if (Contains(n, strings::StrCat(key, kIsDataset))) {
    return ReadDatasetInternal(flr, n, key, val);
  }
  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return errors::NotFound(name);
  }
  const auto& bucket = it->second;
  auto key_it = bucket.find(string(key));
  if (key_it == bucket.end()) {
    return errors::NotFound(key);
  }
  *val = data_.at(name)->tensors(key_it->second);
  return Status::OK();
}

Status VariantTensorDataReader::ReadDatasetInternal(FunctionLibraryRuntime* flr,
                                                    StringPiece n,
                                                    StringPiece key,
                                                    Tensor* val) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_18(mht_18_v, 497, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataReader::ReadDatasetInternal");

  if (flr == nullptr) {
    return errors::Internal(
        "Function library runtime is needed to restore a dataset.");
  }
  tstring output_node, serialized_graph_def;
  TF_RETURN_IF_ERROR(
      ReadScalar(n, strings::StrCat(key, kOutputNode), &output_node));
  TF_RETURN_IF_ERROR(
      ReadScalar(n, strings::StrCat(key), &serialized_graph_def));
  GraphDef graph_def;
  graph_def.ParseFromString(serialized_graph_def);
  TF_RETURN_IF_ERROR(FromGraphDef(flr, graph_def, {}, output_node, val));
  return Status::OK();
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const int64_t val) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_19(mht_19_v, 517, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteScalar");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalar(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const int64_t val) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_20(mht_20_v, 527, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteScalar");

  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const tstring& val) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("val: \"" + (std::string)val + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_21(mht_21_v, 536, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteScalar");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalar(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const tstring& val) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("val: \"" + (std::string)val + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_22(mht_22_v, 547, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteScalar");

  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece key,
                                            const Tensor& val) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_23(mht_23_v, 555, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteTensor");

  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteTensor(name, key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece name, StringPiece key,
                                            const Tensor& val) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_24(mht_24_v, 565, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteTensor");

  return WriteTensorInternal(name, key, val);
}

void VariantTensorDataWriter::MaybeFlush() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_25(mht_25_v, 572, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::MaybeFlush");

  if (is_flushed_) return;
  for (auto& keys : keys_) {
    const string name = keys.first;
    string metadata = name;
    for (size_t i = 0; i < keys_[name].size(); ++i) {
      strings::StrAppend(&metadata, kDelimiter, keys_[name][i]);
    }
    data_[name]->set_metadata(metadata);
  }
  is_flushed_ = true;
}

void VariantTensorDataWriter::Reset() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_26(mht_26_v, 588, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::Reset");

  is_flushed_ = false;
  data_.clear();
  keys_.clear();
}

void VariantTensorDataWriter::ReleaseData(
    std::vector<std::unique_ptr<VariantTensorData>>* variants) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_27(mht_27_v, 598, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::ReleaseData");

  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(std::move(it.second));
  }
  Reset();
}

void VariantTensorDataWriter::GetData(
    std::vector<const VariantTensorData*>* variants) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_28(mht_28_v, 610, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::GetData");

  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(it.second.get());
  }
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece name,
                                                    StringPiece key,
                                                    const T& val) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_29(mht_29_v, 623, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteScalarInternal");

  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(name, key, val_t);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece n,
                                                    StringPiece key,
                                                    const Tensor& val) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_30(mht_30_v, 638, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteTensorInternal");

  DatasetBase* dataset;
  if (GetDatasetFromVariantTensor(val, &dataset).ok()) {
    return WriteDatasetInternal(n, key, dataset);
  }
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteTensor after GetData or ReleaseData is called");
  }
  DCHECK_EQ(key.find(kDelimiter), string::npos);
  string name(n);
  if (keys_.count(name) == 0) {
    keys_[name] = std::vector<string>();
  }
  keys_[name].push_back(string(key));
  if (data_.count(name) == 0) {
    data_[name] = absl::make_unique<VariantTensorData>();
    data_[name]->set_type_name("tensorflow::Iterator");
  }
  *(data_[name]->add_tensors()) = val;
  return Status::OK();
}

Status VariantTensorDataWriter::WriteDatasetInternal(
    StringPiece n, StringPiece key, const DatasetBase* dataset) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_31(mht_31_v, 665, "", "./tensorflow/core/data/serialization_utils.cc", "VariantTensorDataWriter::WriteDatasetInternal");

  GraphDef graph_def;
  SerializationContext ctx((SerializationContext::Params()));
  TF_RETURN_IF_ERROR(AsGraphDef(dataset, std::move(ctx), &graph_def));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
      break;
    }
  }
  string result;
  graph_def.SerializeToString(&result);
  TF_RETURN_IF_ERROR(WriteScalar(n, strings::StrCat(key, kIsDataset), ""));
  TF_RETURN_IF_ERROR(
      WriteScalar(n, strings::StrCat(key, kOutputNode), output_node));
  TF_RETURN_IF_ERROR(WriteScalar(n, key, result));
  return Status::OK();
}

Status AsGraphDefForRewrite(OpKernelContext* ctx, const DatasetBase* input,
                            std::vector<std::pair<string, Tensor>>* input_list,
                            GraphDef* result, string* dataset_node) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_32(mht_32_v, 690, "", "./tensorflow/core/data/serialization_utils.cc", "AsGraphDefForRewrite");

  SerializationContext::Params params(ctx);
  params.input_list = input_list;
  params.external_state_policy =
      SerializationContext::ExternalStatePolicy::kIgnore;
  params.is_graph_rewrite = true;
  SerializationContext serialization_ctx(params);
  TF_RETURN_IF_ERROR(AsGraphDef(input, std::move(serialization_ctx), result));

  // Symbolic `_Retval` node indicates which node corresponds to the dataset.
  for (const auto& node : result->node()) {
    if (node.op() == "_Retval") {
      *dataset_node = node.input(0);
    }
  }
  return Status::OK();
}

Status AsGraphDef(const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSdataPSserialization_utilsDTcc mht_33(mht_33_v, 713, "", "./tensorflow/core/data/serialization_utils.cc", "AsGraphDef");

  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kFail) {
    TF_RETURN_IF_ERROR(dataset->CheckExternalState());
  }
  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kWarn) {
    std::vector<string> stateful_op_names;
    TF_RETURN_IF_ERROR(FindStatefulOps(*graph_def, &stateful_op_names));
    if (!stateful_op_names.empty()) {
      LOG(WARNING) << "We found the following stateful ops in the dataset "
                      "construction graph whose state would not be "
                      "serialized and might "
                      "cause subtle bugs: "
                   << absl::StrJoin(stateful_op_names, ", ");
    }
  }
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* output_node = nullptr;
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &output_node));
  // Insert a purely symbolic _Retval node to indicate to consumers which node
  // represents `dataset`.
  ops::UnaryOp("_Retval", output_node,
               b.opts()
                   .WithName("dataset")
                   .WithAttr("T", DT_VARIANT)
                   .WithAttr("index", 0));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
