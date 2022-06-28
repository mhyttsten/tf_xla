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
class MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/memory_types.h"

#include <utility>

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// Returns the largest endpoint of anything in the name_map.
int GetTotal(const NameRangeMap& name_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/framework/memory_types.cc", "GetTotal");

  int total = 0;
  for (const auto& item : name_map) {
    total = std::max(total, item.second.second);
  }
  return total;
}

// Fills memory_types for either input or output, setting everything
// to DEVICE_MEMORY except those args in host_memory_args.  Removes
// elements of host_memory_args that were used.
void MemoryTypesHelper(const NameRangeMap& name_map,
                       std::vector<string>* host_memory_args,
                       MemoryTypeVector* memory_types) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/framework/memory_types.cc", "MemoryTypesHelper");

  // Update args that have been marked as in "HOST_MEMORY".
  size_t keep = 0;
  for (size_t i = 0; i < host_memory_args->size(); ++i) {
    auto iter = name_map.find((*host_memory_args)[i]);
    if (iter != name_map.end()) {
      for (int j = iter->second.first; j < iter->second.second; ++j) {
        (*memory_types)[j] = HOST_MEMORY;
      }
    } else {
      // (*host_memory_args)[i] not found, save it for the next pass.
      if (i > keep) (*host_memory_args)[keep] = (*host_memory_args)[i];
      ++keep;
    }
  }
  host_memory_args->resize(keep);
}

bool IsFunctionCallOp(const string& op_type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/framework/memory_types.cc", "IsFunctionCallOp");

  return op_type == "SymbolicGradient" || op_type == "PartitionedCall" ||
         op_type == "StatefulPartitionedCall" || op_type == "While" ||
         op_type == "StatelessWhile";
}

}  // namespace

MemoryType MTypeFromDType(const DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/framework/memory_types.cc", "MTypeFromDType");

  return (dtype == DT_INT32 || DataTypeAlwaysOnHost(dtype)) ? HOST_MEMORY
                                                            : DEVICE_MEMORY;
}

MemoryType MTypeFromDTypeIntsOnDevice(const DataType dtype) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/framework/memory_types.cc", "MTypeFromDTypeIntsOnDevice");

  return DataTypeAlwaysOnHost(dtype) ? HOST_MEMORY : DEVICE_MEMORY;
}

Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          const DeviceType& device_type, const NodeDef& ndef,
                          MemoryTypeVector* inp_mtypes,
                          MemoryTypeVector* out_mtypes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/framework/memory_types.cc", "MemoryTypesForNode");

  // Look up the Op registered for this op name.
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(ndef.op(), &op_def));

  // Look up the Kernel registered for this node def.
  const KernelDef* kdef = nullptr;
  Status status =
      FindKernelDef(device_type, ndef, &kdef, nullptr /* kernel_class_name */);

  DataTypeVector inp_dtypes;
  DataTypeVector out_dtypes;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &inp_dtypes, &out_dtypes));

  inp_mtypes->clear();
  out_mtypes->clear();

  bool has_xla_compile = [&] {
    const auto& it = ndef.attr().find(kXlaMustCompileAttr);
    return it != ndef.attr().end() && it->second.b();
  }();

  bool has_kernel_def = status.ok() && !IsFunctionCallOp(ndef.op());
  auto host_memory_required = [&](const DataType& dt) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_typesDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/framework/memory_types.cc", "lambda");

    bool int32_on_device =
        has_kernel_def || device_type.type_string() == "TPU" || has_xla_compile;
    return DataTypeAlwaysOnHost(dt) || (dt == DT_INT32 && !int32_on_device);
  };

  if (has_kernel_def) {
    // Gets the input/output names and their corresponding endpoint ranges.
    NameRangeMap inp_names;
    NameRangeMap out_names;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(ndef, *op_def, &inp_names, &out_names));

    // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
    inp_mtypes->resize(GetTotal(inp_names), DEVICE_MEMORY);
    out_mtypes->resize(GetTotal(out_names), DEVICE_MEMORY);

    // Fills in host memory types based on the kernel def.
    const auto& from_proto = kdef->host_memory_arg();
    std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());
    MemoryTypesHelper(inp_names, &host_memory_args, inp_mtypes);
    MemoryTypesHelper(out_names, &host_memory_args, out_mtypes);
    if (!host_memory_args.empty()) {
      return errors::InvalidArgument(
          "HostMemory args '", absl::StrJoin(host_memory_args, "', '"),
          "' not found in OpDef: ", SummarizeOpDef(*op_def));
    }
  } else {
    // Set all the datatype to DEVICE_MEMORY by default, later on change it to
    // HOST_MEMORY where it is required by the datatype.
    inp_mtypes->resize(inp_dtypes.size(), DEVICE_MEMORY);
    out_mtypes->resize(out_dtypes.size(), DEVICE_MEMORY);
  }
  CHECK_LE(inp_mtypes->size(), inp_dtypes.size());
  CHECK_LE(out_mtypes->size(), out_dtypes.size());

  // Mark e.g. all resource and string types as host memory.
  for (int i = 0; i < inp_mtypes->size(); ++i) {
    if (host_memory_required(inp_dtypes[i])) {
      (*inp_mtypes)[i] = HOST_MEMORY;
    }
  }
  for (int i = 0; i < out_mtypes->size(); ++i) {
    if (host_memory_required(out_dtypes[i])) {
      (*out_mtypes)[i] = HOST_MEMORY;
    }
  }

  std::vector<int32> hostmem_attr;
  if (TryGetNodeAttr(ndef, "_input_hostmem", &hostmem_attr)) {
    for (int32_t i : hostmem_attr) {
      if (0 <= i && i < inp_mtypes->size()) {
        (*inp_mtypes)[i] = HOST_MEMORY;
      }
    }
  }
  hostmem_attr.clear();
  if (TryGetNodeAttr(ndef, "_output_hostmem", &hostmem_attr)) {
    for (int32_t i : hostmem_attr) {
      if (0 <= i && i < out_mtypes->size()) {
        (*out_mtypes)[i] = HOST_MEMORY;
      }
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
