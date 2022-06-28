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
class MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc() {
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

#include "tensorflow/core/kernels/initializable_lookup_table.h"

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

Status InitializableLookupTable::Find(OpKernelContext* ctx, const Tensor& keys,
                                      Tensor* values,
                                      const Tensor& default_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/initializable_lookup_table.cc", "InitializableLookupTable::Find");

  if (!is_initialized()) {
    return errors::FailedPrecondition("Table not initialized.");
  }
  // Do not let the use migrate before the check;  table is used without
  // a lock by the readers.
  std::atomic_thread_fence(std::memory_order_acquire);
  return DoFind(keys, values, default_value);
}

Status InitializableLookupTable::ImportValues(OpKernelContext* ctx,
                                              const Tensor& keys,
                                              const Tensor& values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/kernels/initializable_lookup_table.cc", "InitializableLookupTable::ImportValues");

  lookup::KeyValueTensorIterator iter(&keys, &values);
  auto serializer = absl::make_unique<InitializerSerializer>(
      [keys, values](GraphDefBuilder* builder, Node* table, Node** out) {
        Node* keys_node =
            ops::SourceOp("Const", builder->opts()
                                       .WithAttr("dtype", keys.dtype())
                                       .WithAttr("value", keys));
        Node* values_node =
            ops::SourceOp("Const", builder->opts()
                                       .WithAttr("dtype", values.dtype())
                                       .WithAttr("value", values));
        Node* import_table =
            ops::TernaryOp("LookupTableImportV2", table, keys_node, values_node,
                           builder->opts()
                               .WithAttr("Tin", keys.dtype())
                               .WithAttr("Tout", values.dtype()));
        *out = ops::UnaryOp("Identity", table,
                            builder->opts().WithControlInput(import_table));
        return Status::OK();
      });

  return Initialize(iter, std::move(serializer));
}

Status InitializableLookupTable::Initialize(InitTableIterator& iter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/kernels/initializable_lookup_table.cc", "InitializableLookupTable::Initialize");

  return Initialize(iter, /*serializer=*/nullptr);
}

Status InitializableLookupTable::Initialize(
    InitTableIterator& iter,
    std::unique_ptr<InitializerSerializer> serializer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/kernels/initializable_lookup_table.cc", "InitializableLookupTable::Initialize");

  if (!iter.Valid()) {
    return iter.status();
  }
  TF_RETURN_IF_ERROR(
      CheckKeyAndValueTensorsForInsert(iter.keys(), iter.values()));

  mutex_lock l(mu_);
  if (is_initialized()) {
    bool result;
    TF_RETURN_IF_ERROR(AreEntriesSame(iter, &result));
    // If the table is already initialized, we make sure that the entries in the
    // table are the same that we want to initialize the table with.
    if (!result) {
      return errors::FailedPrecondition(
          "Table was already initialized with "
          "different data.");
    } else {
      return Status::OK();
    }
  }
  TF_RETURN_IF_ERROR(DoLazyPrepare([&iter]() { return iter.total_size(); }));
  while (iter.Valid()) {
    TF_RETURN_IF_ERROR(DoInsert(iter.keys(), iter.values()));
    iter.Next();
  }
  if (!errors::IsOutOfRange(iter.status())) {
    return iter.status();
  }

  initializer_serializer_ = std::move(serializer);
  is_initialized_.store(true, std::memory_order_release);
  return Status::OK();
}

Status InitializableLookupTable::AreEntriesSame(const InitTableIterator& iter,
                                                bool* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTcc mht_4(mht_4_v, 286, "", "./tensorflow/core/kernels/initializable_lookup_table.cc", "InitializableLookupTable::AreEntriesSame");

  *result = static_cast<size_t>(iter.total_size()) == size();
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
