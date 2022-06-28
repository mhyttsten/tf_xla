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

#ifndef TENSORFLOW_GRAPH_EDGESET_H_
#define TENSORFLOW_GRAPH_EDGESET_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh() {
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

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {

class Edge;

// An unordered set of edges.  Uses very little memory for small sets.
// Unlike gtl::FlatSet, EdgeSet does NOT allow mutations during
// iteration.
class EdgeSet {
 public:
  EdgeSet();
  ~EdgeSet();

  typedef const Edge* key_type;
  typedef const Edge* value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  class const_iterator;
  typedef const_iterator iterator;

  bool empty() const;
  size_type size() const;
  void clear();
  std::pair<iterator, bool> insert(value_type value);
  size_type erase(key_type key);
  void reserve(size_type new_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_0(mht_0_v, 219, "", "./tensorflow/core/graph/edgeset.h", "reserve");

    if (new_size > kInline) {
      auto s = new gtl::FlatSet<const Edge*>(new_size);
      s->insert(reinterpret_cast<const Edge**>(std::begin(ptrs_)),
                reinterpret_cast<const Edge**>(&ptrs_[0] + size()));
      ptrs_[0] = this;
      ptrs_[1] = s;
    }
  }

  // Caller is not allowed to mutate the EdgeSet while iterating.
  const_iterator begin() const;
  const_iterator end() const;

 private:
  // Up to kInline elements are stored directly in ptrs_ (nullptr means none).
  // If ptrs_[0] == this then ptrs_[1] points to a set<const Edge*>.
  // kInline must be >= 2, and is chosen such that ptrs_ fills a 64 byte
  // cacheline.
  static constexpr int kInline = 64 / sizeof(const void*);
  const void* ptrs_[kInline];

  gtl::FlatSet<const Edge*>* get_set() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_1(mht_1_v, 244, "", "./tensorflow/core/graph/edgeset.h", "get_set");

    if (ptrs_[0] == this) {
      return static_cast<gtl::FlatSet<const Edge*>*>(
          const_cast<void*>(ptrs_[1]));
    } else {
      return nullptr;
    }
  }

// To detect mutations while iterating.
#ifdef NDEBUG
  void RegisterMutation() {}
#else
  uint32 mutations_ = 0;
  void RegisterMutation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_2(mht_2_v, 261, "", "./tensorflow/core/graph/edgeset.h", "RegisterMutation");
 mutations_++; }
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(EdgeSet);
};

class EdgeSet::const_iterator {
 public:
  typedef typename EdgeSet::value_type value_type;
  typedef const typename EdgeSet::value_type& reference;
  typedef const typename EdgeSet::value_type* pointer;
  typedef typename EdgeSet::difference_type difference_type;
  typedef std::forward_iterator_tag iterator_category;

  const_iterator() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_3(mht_3_v, 278, "", "./tensorflow/core/graph/edgeset.h", "const_iterator");
}

  const_iterator& operator++();
  const_iterator operator++(int /*unused*/);
  const value_type* operator->() const;
  value_type operator*() const;
  bool operator==(const const_iterator& other) const;
  bool operator!=(const const_iterator& other) const {
    return !(*this == other);
  }

 private:
  friend class EdgeSet;

  void const* const* array_iter_ = nullptr;
  typename gtl::FlatSet<const Edge*>::const_iterator tree_iter_;

#ifdef NDEBUG
  inline void Init(const EdgeSet* e) {}
  inline void CheckNoMutations() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_4(mht_4_v, 300, "", "./tensorflow/core/graph/edgeset.h", "CheckNoMutations");
}
#else
  inline void Init(const EdgeSet* e) {
    owner_ = e;
    init_mutations_ = e->mutations_;
  }
  inline void CheckNoMutations() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_5(mht_5_v, 309, "", "./tensorflow/core/graph/edgeset.h", "CheckNoMutations");

    CHECK_EQ(init_mutations_, owner_->mutations_);
  }
  const EdgeSet* owner_ = nullptr;
  uint32 init_mutations_ = 0;
#endif
};

inline EdgeSet::EdgeSet() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_6(mht_6_v, 320, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::EdgeSet");

  for (int i = 0; i < kInline; i++) {
    ptrs_[i] = nullptr;
  }
}

inline EdgeSet::~EdgeSet() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_7(mht_7_v, 329, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::~EdgeSet");
 delete get_set(); }

inline bool EdgeSet::empty() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_8(mht_8_v, 334, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::empty");
 return size() == 0; }

inline EdgeSet::size_type EdgeSet::size() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_9(mht_9_v, 339, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::size");

  auto s = get_set();
  if (s) {
    return s->size();
  } else {
    size_t result = 0;
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i]) result++;
    }
    return result;
  }
}

inline void EdgeSet::clear() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_10(mht_10_v, 355, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::clear");

  RegisterMutation();
  delete get_set();
  for (int i = 0; i < kInline; i++) {
    ptrs_[i] = nullptr;
  }
}

inline EdgeSet::const_iterator EdgeSet::begin() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_11(mht_11_v, 366, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::begin");

  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (s) {
    ci.tree_iter_ = s->begin();
  } else {
    ci.array_iter_ = &ptrs_[0];
  }
  return ci;
}

inline EdgeSet::const_iterator EdgeSet::end() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_12(mht_12_v, 381, "", "./tensorflow/core/graph/edgeset.h", "EdgeSet::end");

  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (s) {
    ci.tree_iter_ = s->end();
  } else {
    ci.array_iter_ = &ptrs_[size()];
  }
  return ci;
}

inline EdgeSet::const_iterator& EdgeSet::const_iterator::operator++() {
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    ++array_iter_;
  } else {
    ++tree_iter_;
  }
  return *this;
}

inline EdgeSet::const_iterator EdgeSet::const_iterator::operator++(
    int /*unused*/) {
  CheckNoMutations();
  const_iterator tmp = *this;
  operator++();
  return tmp;
}

// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.
inline const EdgeSet::const_iterator::value_type* EdgeSet::const_iterator::
operator->() const {
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return reinterpret_cast<const value_type*>(array_iter_);
  } else {
    return tree_iter_.operator->();
  }
}

// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.
inline EdgeSet::const_iterator::value_type EdgeSet::const_iterator::operator*()
    const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSedgesetDTh mht_13(mht_13_v, 429, "", "./tensorflow/core/graph/edgeset.h", "*");

  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return static_cast<value_type>(*array_iter_);
  } else {
    return *tree_iter_;
  }
}

inline bool EdgeSet::const_iterator::operator==(
    const const_iterator& other) const {
  DCHECK((array_iter_ == nullptr) == (other.array_iter_ == nullptr))
      << "Iterators being compared must be from same set that has not "
      << "been modified since the iterator was constructed";
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return array_iter_ == other.array_iter_;
  } else {
    return other.array_iter_ == nullptr && tree_iter_ == other.tree_iter_;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_EDGESET_H_
