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
class MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc() {
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

#include "tensorflow/core/example/feature_util.h"

namespace tensorflow {

namespace internal {
Feature& ExampleFeature(const std::string& name, Example* example) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/example/feature_util.cc", "ExampleFeature");

  return *GetFeature(name, example);
}

}  // namespace internal

template <>
bool HasFeature<>(const std::string& key, const Features& features) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/example/feature_util.cc", "HasFeature<>");

  return (features.feature().find(key) != features.feature().end());
}

template <>
bool HasFeature<protobuf_int64>(const std::string& key,
                                const Features& features) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/example/feature_util.cc", "HasFeature<protobuf_int64>");

  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kInt64List);
}

template <>
bool HasFeature<float>(const std::string& key, const Features& features) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/example/feature_util.cc", "HasFeature<float>");

  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kFloatList);
}

template <>
bool HasFeature<std::string>(const std::string& key, const Features& features) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_4(mht_4_v, 234, "", "./tensorflow/core/example/feature_util.cc", "HasFeature<std::string>");

  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kBytesList);
}

template <>
bool HasFeature<tstring>(const std::string& key, const Features& features) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_5(mht_5_v, 245, "", "./tensorflow/core/example/feature_util.cc", "HasFeature<tstring>");

  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kBytesList);
}

bool HasFeatureList(const std::string& key,
                    const SequenceExample& sequence_example) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_6(mht_6_v, 256, "", "./tensorflow/core/example/feature_util.cc", "HasFeatureList");

  auto& feature_list = sequence_example.feature_lists().feature_list();
  return (feature_list.find(key) != feature_list.end());
}

template <>
const protobuf::RepeatedField<protobuf_int64>& GetFeatureValues<protobuf_int64>(
    const Feature& feature) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<protobuf_int64>");

  return feature.int64_list().value();
}

template <>
protobuf::RepeatedField<protobuf_int64>* GetFeatureValues<protobuf_int64>(
    Feature* feature) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_8(mht_8_v, 275, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<protobuf_int64>");

  return feature->mutable_int64_list()->mutable_value();
}

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_9(mht_9_v, 284, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<float>");

  return feature.float_list().value();
}

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature* feature) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_10(mht_10_v, 292, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<float>");

  return feature->mutable_float_list()->mutable_value();
}

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<tstring>(
    const Feature& feature) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_11(mht_11_v, 301, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<tstring>");

  return feature.bytes_list().value();
}

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<std::string>(
    const Feature& feature) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_12(mht_12_v, 310, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<std::string>");

  return feature.bytes_list().value();
}

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<tstring>(
    Feature* feature) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_13(mht_13_v, 319, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<tstring>");

  return feature->mutable_bytes_list()->mutable_value();
}

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_14(mht_14_v, 328, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureValues<std::string>");

  return feature->mutable_bytes_list()->mutable_value();
}

const protobuf::RepeatedPtrField<Feature>& GetFeatureList(
    const std::string& key, const SequenceExample& sequence_example) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_15(mht_15_v, 337, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureList");

  return sequence_example.feature_lists().feature_list().at(key).feature();
}

protobuf::RepeatedPtrField<Feature>* GetFeatureList(
    const std::string& feature_list_key, SequenceExample* sequence_example) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("feature_list_key: \"" + feature_list_key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_16(mht_16_v, 346, "", "./tensorflow/core/example/feature_util.cc", "GetFeatureList");

  return (*sequence_example->mutable_feature_lists()
               ->mutable_feature_list())[feature_list_key]
      .mutable_feature();
}

template <>
void ClearFeatureValues<protobuf_int64>(Feature* feature) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_17(mht_17_v, 356, "", "./tensorflow/core/example/feature_util.cc", "ClearFeatureValues<protobuf_int64>");

  feature->mutable_int64_list()->Clear();
}

template <>
void ClearFeatureValues<float>(Feature* feature) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_18(mht_18_v, 364, "", "./tensorflow/core/example/feature_util.cc", "ClearFeatureValues<float>");

  feature->mutable_float_list()->Clear();
}

template <>
void ClearFeatureValues<std::string>(Feature* feature) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_19(mht_19_v, 372, "", "./tensorflow/core/example/feature_util.cc", "ClearFeatureValues<std::string>");

  feature->mutable_bytes_list()->Clear();
}

template <>
void ClearFeatureValues<tstring>(Feature* feature) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_20(mht_20_v, 380, "", "./tensorflow/core/example/feature_util.cc", "ClearFeatureValues<tstring>");

  feature->mutable_bytes_list()->Clear();
}

template <>
Features* GetFeatures<Features>(Features* proto) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_21(mht_21_v, 388, "", "./tensorflow/core/example/feature_util.cc", "GetFeatures<Features>");

  return proto;
}

template <>
Features* GetFeatures<Example>(Example* proto) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_22(mht_22_v, 396, "", "./tensorflow/core/example/feature_util.cc", "GetFeatures<Example>");

  return proto->mutable_features();
}

template <>
const Features& GetFeatures<Features>(const Features& proto) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_23(mht_23_v, 404, "", "./tensorflow/core/example/feature_util.cc", "GetFeatures<Features>");

  return proto;
}

template <>
const Features& GetFeatures<Example>(const Example& proto) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTcc mht_24(mht_24_v, 412, "", "./tensorflow/core/example/feature_util.cc", "GetFeatures<Example>");

  return proto.features();
}

template <>
const protobuf::RepeatedField<protobuf_int64>& GetFeatureValues<protobuf_int64>(
    const Feature& feature);

template <>
protobuf::RepeatedField<protobuf_int64>* GetFeatureValues<protobuf_int64>(
    Feature* feature);

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature);

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature* feature);

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<std::string>(
    const Feature& feature);

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<tstring>(
    const Feature& feature);

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature);

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<tstring>(
    Feature* feature);

}  // namespace tensorflow
