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
class MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc() {
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

#include "tensorflow/core/util/proto/descriptors.h"

#include "absl/strings/match.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/descriptor_pool_registry.h"

namespace tensorflow {
namespace {

Status CreatePoolFromSet(const protobuf::FileDescriptorSet& set,
                         std::unique_ptr<protobuf::DescriptorPool>* out_pool) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/util/proto/descriptors.cc", "CreatePoolFromSet");

  *out_pool = absl::make_unique<protobuf::DescriptorPool>();
  for (const auto& file : set.file()) {
    if ((*out_pool)->BuildFile(file) == nullptr) {
      return errors::InvalidArgument("Failed to load FileDescriptorProto: ",
                                     file.DebugString());
    }
  }
  return Status::OK();
}

// Build a `DescriptorPool` from the named file or URI. The file or URI
// must be available to the current TensorFlow environment.
//
// The file must contain a serialized `FileDescriptorSet`. See
// `GetDescriptorPool()` for more information.
Status GetDescriptorPoolFromFile(
    tensorflow::Env* env, const string& filename,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/util/proto/descriptors.cc", "GetDescriptorPoolFromFile");

  Status st = env->FileExists(filename);
  if (!st.ok()) {
    return st;
  }
  // Read and parse the FileDescriptorSet.
  protobuf::FileDescriptorSet descs;
  std::unique_ptr<ReadOnlyMemoryRegion> buf;
  st = env->NewReadOnlyMemoryRegionFromFile(filename, &buf);
  if (!st.ok()) {
    return st;
  }
  if (!descs.ParseFromArray(buf->data(), buf->length())) {
    return errors::InvalidArgument(
        "descriptor_source contains invalid FileDescriptorSet: ", filename);
  }
  return CreatePoolFromSet(descs, owned_desc_pool);
}

Status GetDescriptorPoolFromBinary(
    const string& source,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("source: \"" + source + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/util/proto/descriptors.cc", "GetDescriptorPoolFromBinary");

  if (!absl::StartsWith(source, "bytes://")) {
    return errors::InvalidArgument(absl::StrCat(
        "Source does not represent serialized file descriptor set proto. ",
        "This may be due to a missing dependency on the file containing ",
        "REGISTER_DESCRIPTOR_POOL(\"", source, "\", ...);"));
  }
  // Parse the FileDescriptorSet.
  protobuf::FileDescriptorSet proto;
  if (!proto.ParseFromString(string(absl::StripPrefix(source, "bytes://")))) {
    return errors::InvalidArgument(absl::StrCat(
        "Source does not represent serialized file descriptor set proto. ",
        "This may be due to a missing dependency on the file containing ",
        "REGISTER_DESCRIPTOR_POOL(\"", source, "\", ...);"));
  }
  return CreatePoolFromSet(proto, owned_desc_pool);
}

}  // namespace

Status GetDescriptorPool(
    Env* env, string const& descriptor_source,
    protobuf::DescriptorPool const** desc_pool,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdescriptorsDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/util/proto/descriptors.cc", "GetDescriptorPool");

  // Attempt to lookup the pool in the registry.
  auto pool_fn = DescriptorPoolRegistry::Global()->Get(descriptor_source);
  if (pool_fn != nullptr) {
    return (*pool_fn)(desc_pool, owned_desc_pool);
  }

  // If there is no pool function registered for the given source, let the
  // runtime find the file or URL.
  Status status =
      GetDescriptorPoolFromFile(env, descriptor_source, owned_desc_pool);
  if (status.ok()) {
    *desc_pool = owned_desc_pool->get();
    return Status::OK();
  }

  status = GetDescriptorPoolFromBinary(descriptor_source, owned_desc_pool);
  *desc_pool = owned_desc_pool->get();
  return status;
}

}  // namespace tensorflow
