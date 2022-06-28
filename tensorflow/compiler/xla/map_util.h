/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh() {
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


#include <functional>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// FindOrDie returns a const reference to the value associated with
// the given key if it exists. Crashes otherwise.
//
// This is intended as a replacement for operator[] as an rvalue (for reading)
// when the key is guaranteed to exist.
template <class Collection>
const typename Collection::value_type::second_type& FindOrDie(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/map_util.h", "FindOrDie");

  typename Collection::const_iterator it = collection.find(key);
  CHECK(it != collection.end()) << "Map key not found: " << key;
  return it->second;
}

// Same as above, but returns a non-const reference.
template <class Collection>
typename Collection::value_type::second_type& FindOrDie(
    Collection& collection,  // NOLINT
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/map_util.h", "FindOrDie");

  typename Collection::iterator it = collection.find(key);
  CHECK(it != collection.end()) << "Map key not found: " << key;
  return it->second;
}

// Like FindOrDie but returns an error instead of dying if `key` is not in
// `container`.
template <class Collection>
StatusOr<
    std::reference_wrapper<const typename Collection::value_type::second_type>>
MaybeFind(const Collection& collection,
          const typename Collection::value_type::first_type& key) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    std::ostringstream os;
    os << key;
    return NotFound("key not found: %s", os.str());
  }
  return {it->second};
}

// Returns a const reference to the value associated with the given key if it
// exists, otherwise returns a const reference to the provided default value.
//
// WARNING: If a temporary object is passed as the default "value,"
// this function will return a reference to that temporary object,
// which will be destroyed at the end of the statement. A common
// example: if you have a map with string values, and you pass a char*
// as the default "value," either use the returned value immediately
// or store it in a string (not string&).
template <class Collection>
const typename Collection::value_type::second_type& FindOrDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/map_util.h", "FindOrDefault");

  auto it = collection.find(key);
  if (it != collection.end()) return it->second;
  return value;
}

// Inserts the key-value pair into the collection. Dies if key was already
// present.
template <class Collection, class Key, class Value>
void InsertOrDie(Collection* const collection, Key&& key, Value&& value) {
  auto p = collection->insert(
      std::make_pair(std::forward<Key>(key), std::forward<Value>(value)));
  CHECK(p.second) << "duplicate key: " << key;
}

// Returns true if and only if the given collection contains the given key.
template <class Collection, class Key>
bool ContainsKey(const Collection& collection, const Key& key) {
  return collection.find(key) != collection.end();
}

// Returns a function that returns whether the map contains the given key.
template <class Key, class Value>
auto IsKeyIn(const absl::flat_hash_map<Key, Value>& map) {
  return [&](const Key& key) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_3(mht_3_v, 285, "", "./tensorflow/compiler/xla/map_util.h", "lambda");
 return map.contains(key); };
}

// Returns a function that returns whether the set contains the given value.
template <class T>
auto IsValueIn(const absl::flat_hash_set<T>& set) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/map_util.h", "IsValueIn");

  return [&](const T& value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmap_utilDTh mht_5(mht_5_v, 297, "", "./tensorflow/compiler/xla/map_util.h", "lambda");
 return set.contains(value); };
}

// Inserts `value` into `set`. Dies if it was already present.
template <class Set, class Value>
void InsertOrDie(Set* const set, Value&& value) {
  CHECK(set->insert(std::forward<Value>(value)).second)
      << "duplicate value: " << value;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_
