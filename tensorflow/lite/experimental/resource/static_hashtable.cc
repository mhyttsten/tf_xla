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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc() {
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

#include "tensorflow/lite/experimental/resource/static_hashtable.h"

#include <memory>
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"

namespace tflite {
namespace resource {
namespace internal {

template <typename KeyType, typename ValueType>
TfLiteStatus StaticHashtable<KeyType, ValueType>::Lookup(
    TfLiteContext* context, const TfLiteTensor* keys, TfLiteTensor* values,
    const TfLiteTensor* default_value) {
  if (!is_initialized_) {
    context->ReportError(context,
                         "hashtable need to be initialized before using");
    return kTfLiteError;
  }
  const int size =
      MatchingFlatSize(GetTensorShape(keys), GetTensorShape(values));

  auto key_tensor_reader = TensorReader<KeyType>(keys);
  auto value_tensor_writer = TensorWriter<ValueType>(values);
  auto default_value_tensor_reader = TensorReader<ValueType>(default_value);
  ValueType first_default_value = default_value_tensor_reader.GetData(0);

  for (int i = 0; i < size; ++i) {
    auto result = map_.find(key_tensor_reader.GetData(i));
    if (result != map_.end()) {
      value_tensor_writer.SetData(i, result->second);
    } else {
      value_tensor_writer.SetData(i, first_default_value);
    }
  }

  // This is for a string tensor case in order to write buffer back to the
  // actual tensor destination. Otherwise, it does nothing since the scalar data
  // will be written into the tensor storage directly.
  value_tensor_writer.Commit();

  return kTfLiteOk;
}

template <typename KeyType, typename ValueType>
TfLiteStatus StaticHashtable<KeyType, ValueType>::Import(
    TfLiteContext* context, const TfLiteTensor* keys,
    const TfLiteTensor* values) {
  // Import nodes can be invoked twice because the converter will not extract
  // the initializer graph separately from the original graph. The invocations
  // after the first call will be ignored.
  if (is_initialized_) {
    return kTfLiteOk;
  }

  const int size =
      MatchingFlatSize(GetTensorShape(keys), GetTensorShape(values));

  auto key_tensor_reader = TensorReader<KeyType>(keys);
  auto value_tensor_writer = TensorReader<ValueType>(values);
  for (int i = 0; i < size; ++i) {
    map_.insert({key_tensor_reader.GetData(i), value_tensor_writer.GetData(i)});
  }

  is_initialized_ = true;
  return kTfLiteOk;
}

LookupInterface* CreateStaticHashtable(TfLiteType key_type,
                                       TfLiteType value_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc mht_0(mht_0_v, 253, "", "./tensorflow/lite/experimental/resource/static_hashtable.cc", "CreateStaticHashtable");

  if (key_type == kTfLiteInt64 && value_type == kTfLiteString) {
    return new StaticHashtable<std::int64_t, std::string>(key_type, value_type);
  } else if (key_type == kTfLiteString && value_type == kTfLiteInt64) {
    return new StaticHashtable<std::string, std::int64_t>(key_type, value_type);
  }
  return nullptr;
}

}  // namespace internal

void CreateHashtableResourceIfNotAvailable(ResourceMap* resources,
                                           int resource_id,
                                           TfLiteType key_dtype,
                                           TfLiteType value_dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc mht_1(mht_1_v, 270, "", "./tensorflow/lite/experimental/resource/static_hashtable.cc", "CreateHashtableResourceIfNotAvailable");

  if (resources->count(resource_id) != 0) {
    return;
  }
  auto* hashtable = internal::CreateStaticHashtable(key_dtype, value_dtype);
  resources->emplace(resource_id, std::unique_ptr<LookupInterface>(hashtable));
}

LookupInterface* GetHashtableResource(ResourceMap* resources, int resource_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSstatic_hashtableDTcc mht_2(mht_2_v, 281, "", "./tensorflow/lite/experimental/resource/static_hashtable.cc", "GetHashtableResource");

  auto it = resources->find(resource_id);
  if (it != resources->end()) {
    return static_cast<LookupInterface*>(it->second.get());
  }
  return nullptr;
}

}  // namespace resource
}  // namespace tflite
