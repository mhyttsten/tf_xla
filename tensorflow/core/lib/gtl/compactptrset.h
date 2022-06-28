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

#ifndef TENSORFLOW_CORE_LIB_GTL_COMPACTPTRSET_H_
#define TENSORFLOW_CORE_LIB_GTL_COMPACTPTRSET_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh() {
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


#include <type_traits>
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace gtl {

// CompactPointerSet<T> is like a std::unordered_set<T> but is optimized
// for small sets (<= 1 element).  T must be a pointer type.
template <typename T>
class CompactPointerSet {
 private:
  using BigRep = FlatSet<T>;

 public:
  using value_type = T;

  CompactPointerSet() : rep_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/lib/gtl/compactptrset.h", "CompactPointerSet");
}

  ~CompactPointerSet() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_1(mht_1_v, 209, "", "./tensorflow/core/lib/gtl/compactptrset.h", "~CompactPointerSet");

    static_assert(
        std::is_pointer<T>::value,
        "CompactPointerSet<T> can only be used with T's that are pointers");
    if (isbig()) delete big();
  }

  CompactPointerSet(const CompactPointerSet& other) : rep_(0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_2(mht_2_v, 219, "", "./tensorflow/core/lib/gtl/compactptrset.h", "CompactPointerSet");
 *this = other; }

  CompactPointerSet& operator=(const CompactPointerSet& other) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_3(mht_3_v, 224, "", "./tensorflow/core/lib/gtl/compactptrset.h", "=");

    if (this == &other) return *this;
    if (other.isbig()) {
      // big => any
      if (!isbig()) MakeBig();
      *big() = *other.big();
    } else if (isbig()) {
      // !big => big
      big()->clear();
      if (other.rep_ != 0) {
        big()->insert(reinterpret_cast<T>(other.rep_));
      }
    } else {
      // !big => !big
      rep_ = other.rep_;
    }
    return *this;
  }

  class iterator {
   public:
    typedef ssize_t difference_type;
    typedef T value_type;
    typedef const T* pointer;
    typedef const T& reference;
    typedef ::std::forward_iterator_tag iterator_category;

    explicit iterator(uintptr_t rep)
        : bigrep_(false), single_(reinterpret_cast<T>(rep)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_4(mht_4_v, 255, "", "./tensorflow/core/lib/gtl/compactptrset.h", "iterator");
}
    explicit iterator(typename BigRep::iterator iter)
        : bigrep_(true), single_(nullptr), iter_(iter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_5(mht_5_v, 260, "", "./tensorflow/core/lib/gtl/compactptrset.h", "iterator");
}

    iterator& operator++() {
      if (bigrep_) {
        ++iter_;
      } else {
        DCHECK(single_ != nullptr);
        single_ = nullptr;
      }
      return *this;
    }
    // maybe post-increment?

    bool operator==(const iterator& other) const {
      if (bigrep_) {
        return iter_ == other.iter_;
      } else {
        return single_ == other.single_;
      }
    }
    bool operator!=(const iterator& other) const { return !(*this == other); }

    const T& operator*() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_6(mht_6_v, 285, "", "./tensorflow/core/lib/gtl/compactptrset.h", "*");

      if (bigrep_) {
        return *iter_;
      } else {
        DCHECK(single_ != nullptr);
        return single_;
      }
    }

   private:
    friend class CompactPointerSet;
    bool bigrep_;
    T single_;
    typename BigRep::iterator iter_;
  };
  using const_iterator = iterator;

  bool empty() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/lib/gtl/compactptrset.h", "empty");
 return isbig() ? big()->empty() : (rep_ == 0); }
  size_t size() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_8(mht_8_v, 309, "", "./tensorflow/core/lib/gtl/compactptrset.h", "size");
 return isbig() ? big()->size() : (rep_ == 0 ? 0 : 1); }

  void clear() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_9(mht_9_v, 314, "", "./tensorflow/core/lib/gtl/compactptrset.h", "clear");

    if (isbig()) {
      delete big();
    }
    rep_ = 0;
  }

  std::pair<iterator, bool> insert(T elem) {
    if (!isbig()) {
      if (rep_ == 0) {
        uintptr_t v = reinterpret_cast<uintptr_t>(elem);
        if (v == 0 || ((v & 0x3) != 0)) {
          // Cannot use small representation for nullptr.  Fall through.
        } else {
          rep_ = v;
          return {iterator(v), true};
        }
      }
      MakeBig();
    }
    auto p = big()->insert(elem);
    return {iterator(p.first), p.second};
  }

  template <typename InputIter>
  void insert(InputIter begin, InputIter end) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_10(mht_10_v, 342, "", "./tensorflow/core/lib/gtl/compactptrset.h", "insert");

    for (; begin != end; ++begin) {
      insert(*begin);
    }
  }

  const_iterator begin() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_11(mht_11_v, 351, "", "./tensorflow/core/lib/gtl/compactptrset.h", "begin");

    return isbig() ? iterator(big()->begin()) : iterator(rep_);
  }
  const_iterator end() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_12(mht_12_v, 357, "", "./tensorflow/core/lib/gtl/compactptrset.h", "end");

    return isbig() ? iterator(big()->end()) : iterator(0);
  }

  iterator find(T elem) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_13(mht_13_v, 364, "", "./tensorflow/core/lib/gtl/compactptrset.h", "find");

    if (rep_ == reinterpret_cast<uintptr_t>(elem)) {
      return iterator(rep_);
    } else if (!isbig()) {
      return iterator(0);
    } else {
      return iterator(big()->find(elem));
    }
  }

  size_t count(T elem) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_14(mht_14_v, 377, "", "./tensorflow/core/lib/gtl/compactptrset.h", "count");
 return find(elem) != end() ? 1 : 0; }

  size_t erase(T elem) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_15(mht_15_v, 382, "", "./tensorflow/core/lib/gtl/compactptrset.h", "erase");

    if (!isbig()) {
      if (rep_ == reinterpret_cast<uintptr_t>(elem)) {
        rep_ = 0;
        return 1;
      } else {
        return 0;
      }
    } else {
      return big()->erase(elem);
    }
  }

 private:
  // Size         rep_
  // -------------------------------------------------------------------------
  // 0            0
  // 1            The pointer itself (bottom bits == 00)
  // large        Pointer to a BigRep (bottom bits == 01)
  uintptr_t rep_;

  bool isbig() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_16(mht_16_v, 406, "", "./tensorflow/core/lib/gtl/compactptrset.h", "isbig");
 return (rep_ & 0x3) == 1; }
  BigRep* big() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_17(mht_17_v, 410, "", "./tensorflow/core/lib/gtl/compactptrset.h", "big");

    DCHECK(isbig());
    return reinterpret_cast<BigRep*>(rep_ - 1);
  }

  void MakeBig() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScompactptrsetDTh mht_18(mht_18_v, 418, "", "./tensorflow/core/lib/gtl/compactptrset.h", "MakeBig");

    DCHECK(!isbig());
    BigRep* big = new BigRep;
    if (rep_ != 0) {
      big->insert(reinterpret_cast<T>(rep_));
    }
    rep_ = reinterpret_cast<uintptr_t>(big) + 0x1;
  }
};

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_COMPACTPTRSET_H_
