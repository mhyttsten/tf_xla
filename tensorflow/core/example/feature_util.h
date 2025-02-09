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

// A set of lightweight wrappers which simplify access to Feature protos.
//
// TensorFlow Example proto uses associative maps on top of oneof fields.
// SequenceExample proto uses associative map of FeatureList.
// So accessing feature values is not very convenient.
//
// For example, to read a first value of integer feature "tag":
//   int id = example.features().feature().at("tag").int64_list().value(0);
//
// to add a value:
//   auto features = example->mutable_features();
//   (*features->mutable_feature())["tag"].mutable_int64_list()->add_value(id);
//
// For float features you have to use float_list, for string - bytes_list.
//
// To do the same with this library:
//   int id = GetFeatureValues<int64_t>("tag", example).Get(0);
//   GetFeatureValues<int64_t>("tag", &example)->Add(id);
//
// Modification of bytes features is slightly different:
//   auto tag = GetFeatureValues<string>("tag", &example);
//   *tag->Add() = "lorem ipsum";
//
// To copy multiple values into a feature:
//   AppendFeatureValues({1,2,3}, "tag", &example);
//
// GetFeatureValues gives you access to underlying data - RepeatedField object
// (RepeatedPtrField for byte list). So refer to its documentation of
// RepeatedField for full list of supported methods.
//
// NOTE: Due to the nature of oneof proto fields setting a feature of one type
// automatically clears all values stored as another type with the same feature
// key.
//
// This library also has tools to work with SequenceExample protos.
//
// To get a value from SequenceExample.context:
//   int id = GetFeatureValues<protobuf_int64>("tag", se.context()).Get(0);
// To add a value to the context:
//   GetFeatureValues<protobuf_int64>("tag", se.mutable_context())->Add(42);
//
// To add values to feature_lists:
//   AppendFeatureValues({4.0},
//                       GetFeatureList("images", &se)->Add());
//   AppendFeatureValues({5.0, 3.0},
//                       GetFeatureList("images", &se)->Add());
// This will create a feature list keyed as "images" with two features:
//   feature_lists {
//     feature_list {
//       key: "images"
//       value {
//         feature { float_list { value: [4.0] } }
//         feature { float_list { value: [5.0, 3.0] } }
//       }
//     }
//   }
//
// Functions exposed by this library:
//   HasFeature<[FeatureType]>(key, proto) -> bool
//     Returns true if a feature with the specified key, and optionally
//     FeatureType, belongs to the Features or Example proto.
//   HasFeatureList(key, sequence_example) -> bool
//     Returns true if SequenceExample has a feature_list with the key.
//
//   GetFeatureValues<FeatureType>(key, proto) -> RepeatedField<FeatureType>
//     Returns values for the specified key and the FeatureType.
//     Supported types for the proto: Example, Features.
//   GetFeatureList(key, sequence_example) -> RepeatedPtrField<Feature>
//     Returns Feature protos associated with a key.
//
//   AppendFeatureValues(begin, end, feature)
//   AppendFeatureValues(container or initializer_list, feature)
//     Copies values into a Feature.
//   AppendFeatureValues(begin, end, key, proto)
//   AppendFeatureValues(container or initializer_list, key, proto)
//     Copies values into Features and Example protos with the specified key.
//
//   ClearFeatureValues<FeatureType>(feature)
//     Clears the feature's repeated field of the given type.
//
//   SetFeatureValues(begin, end, feature)
//   SetFeatureValues(container or initializer_list, feature)
//     Clears a Feature, then copies values into it.
//   SetFeatureValues(begin, end, key, proto)
//   SetFeatureValues(container or initializer_list, key, proto)
//     Clears Features or Example protos with the specified key,
//     then copies values into them.
//
// Auxiliary functions, it is unlikely you'll need to use them directly:
//   GetFeatures(proto) -> Features
//     A convenience function to get Features proto.
//     Supported types for the proto: Example, Features.
//   GetFeature(key, proto) -> Feature
//     Returns a Feature proto for the specified key.
//     Supported types for the proto: Example, Features.
//   GetFeatureValues<FeatureType>(feature) -> RepeatedField<FeatureType>
//     Returns values of the feature for the FeatureType.

#ifndef TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_
#define TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh() {
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


#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

#include "absl/base/macros.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace internal {

// TODO(gorban): Update all clients in a followup CL.
// Returns a reference to a feature corresponding to the name.
// Note: it will create a new Feature if it is missing in the example.
ABSL_DEPRECATED("Use GetFeature instead.")
Feature& ExampleFeature(const std::string& name, Example* example);

// Specializations of RepeatedFieldTrait define a type of RepeatedField
// corresponding to a selected feature type.
template <typename FeatureType>
struct RepeatedFieldTrait;

template <>
struct RepeatedFieldTrait<protobuf_int64> {
  using Type = protobuf::RepeatedField<protobuf_int64>;
};

template <>
struct RepeatedFieldTrait<float> {
  using Type = protobuf::RepeatedField<float>;
};

template <>
struct RepeatedFieldTrait<tstring> {
  using Type = protobuf::RepeatedPtrField<std::string>;
};

template <>
struct RepeatedFieldTrait<std::string> {
  using Type = protobuf::RepeatedPtrField<std::string>;
};

// Specializations of FeatureTrait define a type of feature corresponding to a
// selected value type.
template <typename ValueType, class Enable = void>
struct FeatureTrait;

template <typename ValueType>
struct FeatureTrait<ValueType, typename std::enable_if<
                                   std::is_integral<ValueType>::value>::type> {
  using Type = protobuf_int64;
};

template <typename ValueType>
struct FeatureTrait<
    ValueType,
    typename std::enable_if<std::is_floating_point<ValueType>::value>::type> {
  using Type = float;
};

template <typename T>
struct is_string
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_string<std::string> : std::true_type {};

template <>
struct is_string<::tensorflow::StringPiece> : std::true_type {};

template <>
struct is_string<tstring> : std::true_type {};

template <typename ValueType>
struct FeatureTrait<
    ValueType, typename std::enable_if<is_string<ValueType>::value>::type> {
  using Type = std::string;
};

}  //  namespace internal

// Returns true if sequence_example has a feature_list with the specified key.
bool HasFeatureList(const std::string& key,
                    const SequenceExample& sequence_example);

template <typename T>
struct TypeHasFeatures : std::false_type {};

template <>
struct TypeHasFeatures<Example> : std::true_type {};

template <>
struct TypeHasFeatures<Features> : std::true_type {};

// A family of template functions to return mutable Features proto from a
// container proto. Supported ProtoTypes: Example, Features.
template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value, Features*>::type
GetFeatures(ProtoType* proto);

template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value,
                        const Features&>::type
GetFeatures(const ProtoType& proto);

// Base declaration of a family of template functions to return a read only
// repeated field of feature values.
template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const Feature& feature);

// Returns a read only repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes: Example, Features.
template <typename FeatureType, typename ProtoType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const std::string& key, const ProtoType& proto) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_0(mht_0_v, 411, "", "./tensorflow/core/example/feature_util.h", "GetFeatureValues");

  return GetFeatureValues<FeatureType>(GetFeatures(proto).feature().at(key));
}

// Returns a mutable repeated field of a feature values.
template <typename FeatureType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    Feature* feature);

// Returns a mutable repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes: Example, Features.
template <typename FeatureType, typename ProtoType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    const std::string& key, ProtoType* proto) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_1(mht_1_v, 428, "", "./tensorflow/core/example/feature_util.h", "GetFeatureValues");

  ::tensorflow::Feature& feature =
      (*GetFeatures(proto)->mutable_feature())[key];
  return GetFeatureValues<FeatureType>(&feature);
}

// Returns a read-only Feature proto for the specified key, throws
// std::out_of_range if the key is not found. Supported types for the proto:
// Example, Features.
template <typename ProtoType>
const Feature& GetFeature(const std::string& key, const ProtoType& proto) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_2(mht_2_v, 442, "", "./tensorflow/core/example/feature_util.h", "GetFeature");

  return GetFeatures(proto).feature().at(key);
}

// Returns a mutable Feature proto for the specified key, creates a new if
// necessary. Supported types for the proto: Example, Features.
template <typename ProtoType>
Feature* GetFeature(const std::string& key, ProtoType* proto) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_3(mht_3_v, 453, "", "./tensorflow/core/example/feature_util.h", "GetFeature");

  return &(*GetFeatures(proto)->mutable_feature())[key];
}

// Returns a repeated field with features corresponding to a feature_list key.
const protobuf::RepeatedPtrField<Feature>& GetFeatureList(
    const std::string& key, const SequenceExample& sequence_example);

// Returns a mutable repeated field with features corresponding to a
// feature_list key. It will create a new FeatureList if necessary.
protobuf::RepeatedPtrField<Feature>* GetFeatureList(
    const std::string& feature_list_key, SequenceExample* sequence_example);

template <typename IteratorType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         Feature* feature) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_4(mht_4_v, 471, "", "./tensorflow/core/example/feature_util.h", "AppendFeatureValues");

  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  std::copy(first, last,
            protobuf::RepeatedFieldBackInserter(
                GetFeatureValues<FeatureType>(feature)));
}

template <typename ValueType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         Feature* feature) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_5(mht_5_v, 484, "", "./tensorflow/core/example/feature_util.h", "AppendFeatureValues");

  using FeatureType = typename internal::FeatureTrait<ValueType>::Type;
  auto* values = GetFeatureValues<FeatureType>(feature);
  values->Reserve(container.size());
  std::move(container.begin(), container.end(),
            protobuf::RepeatedFieldBackInserter(values));
}

namespace internal {

// HasSize<T>::value is true_type if T has a size() member.
template <typename T, typename = void>
struct HasSize : std::false_type {};

template <typename T>
struct HasSize<T, absl::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// Reserves the container's size, if a container.size() method exists.
template <typename ContainerType, typename RepeatedFieldType>
auto ReserveIfSizeAvailable(const ContainerType& container,
                            RepeatedFieldType& values) ->
    typename std::enable_if_t<HasSize<ContainerType>::value, void> {
  values.Reserve(container.size());
}

template <typename ContainerType, typename RepeatedFieldType>
auto ReserveIfSizeAvailable(const ContainerType& container,
                            RepeatedFieldType& values) ->
    typename std::enable_if_t<!HasSize<ContainerType>::value, void> {}

}  // namespace internal

template <typename ContainerType>
void AppendFeatureValues(const ContainerType& container, Feature* feature) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_6(mht_6_v, 521, "", "./tensorflow/core/example/feature_util.h", "AppendFeatureValues");

  using IteratorType = typename ContainerType::const_iterator;
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  auto* values = GetFeatureValues<FeatureType>(feature);
  internal::ReserveIfSizeAvailable(container, *values);
  std::copy(container.begin(), container.end(),
            protobuf::RepeatedFieldBackInserter(values));
}

// Copies elements from the range, defined by [first, last) into the feature
// obtainable from the (proto, key) combination.
template <typename IteratorType, typename ProtoType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         const std::string& key, ProtoType* proto) {
  AppendFeatureValues(first, last, GetFeature(key, GetFeatures(proto)));
}

// Copies all elements from the container into a feature.
template <typename ContainerType, typename ProtoType>
void AppendFeatureValues(const ContainerType& container, const std::string& key,
                         ProtoType* proto) {
  AppendFeatureValues<ContainerType>(container,
                                     GetFeature(key, GetFeatures(proto)));
}

// Copies all elements from the initializer list into a Feature contained by
// Features or Example proto.
template <typename ValueType, typename ProtoType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         const std::string& key, ProtoType* proto) {
  AppendFeatureValues<ValueType>(container,
                                 GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string).
template <typename... FeatureType>
void ClearFeatureValues(Feature* feature);

// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType>
void SetFeatureValues(IteratorType first, IteratorType last, Feature* feature) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_7(mht_7_v, 567, "", "./tensorflow/core/example/feature_util.h", "SetFeatureValues");

  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(first, last, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      Feature* feature) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_8(mht_8_v, 581, "", "./tensorflow/core/example/feature_util.h", "SetFeatureValues");

  using FeatureType = typename internal::FeatureTrait<ValueType>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(container, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType>
void SetFeatureValues(const ContainerType& container, Feature* feature) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSexamplePSfeature_utilDTh mht_9(mht_9_v, 593, "", "./tensorflow/core/example/feature_util.h", "SetFeatureValues");

  using IteratorType = typename ContainerType::const_iterator;
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(container, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType, typename ProtoType>
void SetFeatureValues(IteratorType first, IteratorType last,
                      const std::string& key, ProtoType* proto) {
  SetFeatureValues(first, last, GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType, typename ProtoType>
void SetFeatureValues(const ContainerType& container, const std::string& key,
                      ProtoType* proto) {
  SetFeatureValues<ContainerType>(container,
                                  GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType, typename ProtoType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      const std::string& key, ProtoType* proto) {
  SetFeatureValues<ValueType>(container, GetFeature(key, GetFeatures(proto)));
}

// Returns true if a feature with the specified key belongs to the Features.
// The template parameter pack accepts zero or one template argument - which
// is FeatureType. If the FeatureType not specified (zero template arguments)
// the function will not check the feature type. Otherwise it will return false
// if the feature has a wrong type.
template <typename... FeatureType>
bool HasFeature(const std::string& key, const Features& features);

// Returns true if a feature with the specified key belongs to the Example.
// Doesn't check feature type if used without FeatureType, otherwise the
// specialized versions return false if the feature has a wrong type.
template <typename... FeatureType>
bool HasFeature(const std::string& key, const Example& example) {
  return HasFeature<FeatureType...>(key, GetFeatures(example));
}

// TODO(gorban): update all clients in a followup CL.
template <typename... FeatureType>
ABSL_DEPRECATED("Use HasFeature instead.")
bool ExampleHasFeature(const std::string& key, const Example& example) {
  return HasFeature<FeatureType...>(key, example);
}

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_
