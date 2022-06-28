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

// This file provides utility functions for use with STL map-like data
// structures, such as std::map and hash_map. Some functions will also work with
// sets, such as ContainsKey().

#ifndef TENSORFLOW_LIB_GTL_MAP_UTIL_H_
#define TENSORFLOW_LIB_GTL_MAP_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh() {
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


#include <stddef.h>

#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/lib/gtl/subtle/map_traits.h"

namespace tensorflow {
namespace gtl {

// Returns a pointer to the const value associated with the given key if it
// exists, or NULL otherwise.
template <class Collection>
const typename Collection::value_type::second_type* FindOrNull(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/lib/gtl/map_util.h", "FindOrNull");

  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return nullptr;
  }
  return &it->second;
}

// Same as above but returns a pointer to the non-const value.
template <class Collection>
typename Collection::value_type::second_type* FindOrNull(
    Collection& collection,  // NOLINT
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/lib/gtl/map_util.h", "FindOrNull");

  typename Collection::iterator it = collection.find(key);
  if (it == collection.end()) {
    return nullptr;
  }
  return &it->second;
}

// Returns the pointer value associated with the given key. If none is found,
// NULL is returned. The function is designed to be used with a map of keys to
// pointers.
//
// This function does not distinguish between a missing key and a key mapped
// to a NULL value.
template <class Collection>
typename Collection::value_type::second_type FindPtrOrNull(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_2(mht_2_v, 244, "", "./tensorflow/core/lib/gtl/map_util.h", "FindPtrOrNull");

  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return typename Collection::value_type::second_type();
  }
  return it->second;
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
const typename Collection::value_type::second_type& FindWithDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_3(mht_3_v, 268, "", "./tensorflow/core/lib/gtl/map_util.h", "FindWithDefault");

  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return value;
  }
  return it->second;
}

// Inserts the given key-value pair into the collection. Returns true if and
// only if the key from the given pair didn't previously exist. Otherwise, the
// value in the map is replaced with the value from the given pair.
template <class Collection>
bool InsertOrUpdate(Collection* const collection,
                    const typename Collection::value_type& vt) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_4(mht_4_v, 284, "", "./tensorflow/core/lib/gtl/map_util.h", "InsertOrUpdate");

  std::pair<typename Collection::iterator, bool> ret = collection->insert(vt);
  if (!ret.second) {
    // update
    ret.first->second = vt.second;
    return false;
  }
  return true;
}

// Same as above, except that the key and value are passed separately.
template <class Collection>
bool InsertOrUpdate(Collection* const collection,
                    const typename Collection::value_type::first_type& key,
                    const typename Collection::value_type::second_type& value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_5(mht_5_v, 301, "", "./tensorflow/core/lib/gtl/map_util.h", "InsertOrUpdate");

  return InsertOrUpdate(collection,
                        typename Collection::value_type(key, value));
}

// Inserts the given key and value into the given collection if and only if the
// given key did NOT already exist in the collection. If the key previously
// existed in the collection, the value is not changed. Returns true if the
// key-value pair was inserted; returns false if the key was already present.
template <class Collection>
bool InsertIfNotPresent(Collection* const collection,
                        const typename Collection::value_type& vt) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_6(mht_6_v, 315, "", "./tensorflow/core/lib/gtl/map_util.h", "InsertIfNotPresent");

  return collection->insert(vt).second;
}

// Same as above except the key and value are passed separately.
template <class Collection>
bool InsertIfNotPresent(
    Collection* const collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_7(mht_7_v, 327, "", "./tensorflow/core/lib/gtl/map_util.h", "InsertIfNotPresent");

  return InsertIfNotPresent(collection,
                            typename Collection::value_type(key, value));
}

// Looks up a given key and value pair in a collection and inserts the key-value
// pair if it's not already present. Returns a reference to the value associated
// with the key.
template <class Collection>
typename Collection::value_type::second_type& LookupOrInsert(
    Collection* const collection, const typename Collection::value_type& vt) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_8(mht_8_v, 340, "", "./tensorflow/core/lib/gtl/map_util.h", "LookupOrInsert");

  return collection->insert(vt).first->second;
}

// Same as above except the key-value are passed separately.
template <class Collection>
typename Collection::value_type::second_type& LookupOrInsert(
    Collection* const collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_9(mht_9_v, 352, "", "./tensorflow/core/lib/gtl/map_util.h", "LookupOrInsert");

  return LookupOrInsert(collection,
                        typename Collection::value_type(key, value));
}

// Saves the reverse mapping into reverse. Returns true if values could all be
// inserted.
template <typename M, typename ReverseM>
bool ReverseMap(const M& m, ReverseM* reverse) {
  bool all_unique = true;
  for (const auto& kv : m) {
    if (!InsertOrUpdate(reverse, kv.second, kv.first)) {
      all_unique = false;
    }
  }
  return all_unique;
}

// Like ReverseMap above, but returns its output m. Return type has to
// be specified explicitly. Example:
// M::M(...) : m_(...), r_(ReverseMap<decltype(r_)>(m_)) {}
template <typename ReverseM, typename M>
ReverseM ReverseMap(const M& m) {
  typename std::remove_const<ReverseM>::type reverse;
  ReverseMap(m, &reverse);
  return reverse;
}

// Erases the m item identified by the given key, and returns the value
// associated with that key. It is assumed that the value (i.e., the
// mapped_type) is a pointer. Returns null if the key was not found in the
// m.
//
// Examples:
//   std::map<string, MyType*> my_map;
//
// One line cleanup:
//     delete EraseKeyReturnValuePtr(&my_map, "abc");
//
// Use returned value:
//     std::unique_ptr<MyType> value_ptr(
//         EraseKeyReturnValuePtr(&my_map, "abc"));
//     if (value_ptr.get())
//       value_ptr->DoSomething();
//
template <typename Collection>
typename Collection::value_type::second_type EraseKeyReturnValuePtr(
    Collection* collection,
    const typename Collection::value_type::first_type& key) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmap_utilDTh mht_10(mht_10_v, 403, "", "./tensorflow/core/lib/gtl/map_util.h", "EraseKeyReturnValuePtr");

  auto it = collection->find(key);
  if (it == collection->end()) return nullptr;
  auto v = gtl::subtle::GetMapped(*it);
  collection->erase(it);
  return v;
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_MAP_UTIL_H_
