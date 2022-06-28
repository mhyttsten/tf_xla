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

#ifndef TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_
#define TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh() {
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
#include <functional>
#include <initializer_list>
#include <iterator>
#include <utility>
#include "tensorflow/core/lib/gtl/flatrep.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {

// FlatMap<K,V,...> provides a map from K to V.
//
// The map is implemented using an open-addressed hash table.  A
// single array holds entire map contents and collisions are resolved
// by probing at a sequence of locations in the array.
template <typename Key, typename Val, class Hash = hash<Key>,
          class Eq = std::equal_to<Key>>
class FlatMap {
 private:
  // Forward declare some internal types needed in public section.
  struct Bucket;

  // We cannot use std::pair<> since internal representation stores
  // keys and values in separate arrays, so we make a custom struct
  // that holds references to the internal key, value elements.
  //
  // We define the struct as private ValueType, and typedef it as public
  // value_type, to work around a gcc bug when compiling the iterators.
  struct ValueType {
    typedef Key first_type;
    typedef Val second_type;

    const Key& first;
    Val& second;
    ValueType(const Key& k, Val& v) : first(k), second(v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_0(mht_0_v, 225, "", "./tensorflow/core/lib/gtl/flatmap.h", "ValueType");
}
  };

 public:
  typedef Key key_type;
  typedef Val mapped_type;
  typedef Hash hasher;
  typedef Eq key_equal;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef ValueType value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  FlatMap() : FlatMap(1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_1(mht_1_v, 244, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");
}

  explicit FlatMap(size_t N, const Hash& hf = Hash(), const Eq& eq = Eq())
      : rep_(N, hf, eq) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_2(mht_2_v, 250, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");
}

  FlatMap(const FlatMap& src) : rep_(src.rep_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_3(mht_3_v, 255, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");
}

  // Move constructor leaves src in a valid but unspecified state (same as
  // std::unordered_map).
  FlatMap(FlatMap&& src) : rep_(std::move(src.rep_)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_4(mht_4_v, 262, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");
}

  template <typename InputIter>
  FlatMap(InputIter first, InputIter last, size_t N = 1,
          const Hash& hf = Hash(), const Eq& eq = Eq())
      : FlatMap(N, hf, eq) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_5(mht_5_v, 270, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");

    insert(first, last);
  }

  FlatMap(std::initializer_list<std::pair<const Key, Val>> init, size_t N = 1,
          const Hash& hf = Hash(), const Eq& eq = Eq())
      : FlatMap(init.begin(), init.end(), N, hf, eq) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_6(mht_6_v, 279, "", "./tensorflow/core/lib/gtl/flatmap.h", "FlatMap");
}

  FlatMap& operator=(const FlatMap& src) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_7(mht_7_v, 284, "", "./tensorflow/core/lib/gtl/flatmap.h", "=");

    rep_.CopyFrom(src.rep_);
    return *this;
  }

  // Move-assignment operator leaves src in a valid but unspecified state (same
  // as std::unordered_map).
  FlatMap& operator=(FlatMap&& src) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_8(mht_8_v, 294, "", "./tensorflow/core/lib/gtl/flatmap.h", "=");

    rep_.MoveFrom(std::move(src.rep_));
    return *this;
  }

  ~FlatMap() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_9(mht_9_v, 302, "", "./tensorflow/core/lib/gtl/flatmap.h", "~FlatMap");
}

  void swap(FlatMap& x) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_10(mht_10_v, 307, "", "./tensorflow/core/lib/gtl/flatmap.h", "swap");
 rep_.swap(x.rep_); }
  void clear_no_resize() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_11(mht_11_v, 311, "", "./tensorflow/core/lib/gtl/flatmap.h", "clear_no_resize");
 rep_.clear_no_resize(); }
  void clear() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_12(mht_12_v, 315, "", "./tensorflow/core/lib/gtl/flatmap.h", "clear");
 rep_.clear(); }
  void reserve(size_t N) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_13(mht_13_v, 319, "", "./tensorflow/core/lib/gtl/flatmap.h", "reserve");
 rep_.Resize(std::max(N, size())); }
  void rehash(size_t N) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_14(mht_14_v, 323, "", "./tensorflow/core/lib/gtl/flatmap.h", "rehash");
 rep_.Resize(std::max(N, size())); }
  void resize(size_t N) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_15(mht_15_v, 327, "", "./tensorflow/core/lib/gtl/flatmap.h", "resize");
 rep_.Resize(std::max(N, size())); }
  size_t size() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_16(mht_16_v, 331, "", "./tensorflow/core/lib/gtl/flatmap.h", "size");
 return rep_.size(); }
  bool empty() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_17(mht_17_v, 335, "", "./tensorflow/core/lib/gtl/flatmap.h", "empty");
 return size() == 0; }
  size_t bucket_count() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_18(mht_18_v, 339, "", "./tensorflow/core/lib/gtl/flatmap.h", "bucket_count");
 return rep_.bucket_count(); }
  hasher hash_function() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_19(mht_19_v, 343, "", "./tensorflow/core/lib/gtl/flatmap.h", "hash_function");
 return rep_.hash_function(); }
  key_equal key_eq() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_20(mht_20_v, 347, "", "./tensorflow/core/lib/gtl/flatmap.h", "key_eq");
 return rep_.key_eq(); }

  class iterator {
   public:
    typedef typename FlatMap::difference_type difference_type;
    typedef typename FlatMap::value_type value_type;
    typedef typename FlatMap::pointer pointer;
    typedef typename FlatMap::reference reference;
    typedef ::std::forward_iterator_tag iterator_category;

    iterator() : b_(nullptr), end_(nullptr), i_(0) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_21(mht_21_v, 360, "", "./tensorflow/core/lib/gtl/flatmap.h", "iterator");
}

    // Make iterator pointing at first element at or after b.
    iterator(Bucket* b, Bucket* end) : b_(b), end_(end), i_(0) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_22(mht_22_v, 366, "", "./tensorflow/core/lib/gtl/flatmap.h", "iterator");
 SkipUnused(); }

    // Make iterator pointing exactly at ith element in b, which must exist.
    iterator(Bucket* b, Bucket* end, uint32 i) : b_(b), end_(end), i_(i) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_23(mht_23_v, 372, "", "./tensorflow/core/lib/gtl/flatmap.h", "iterator");

      FillValue();
    }

    reference operator*() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_24(mht_24_v, 379, "", "./tensorflow/core/lib/gtl/flatmap.h", "*");
 return *val(); }
    pointer operator->() { return val(); }
    bool operator==(const iterator& x) const {
      return b_ == x.b_ && i_ == x.i_;
    }
    bool operator!=(const iterator& x) const { return !(*this == x); }
    iterator& operator++() {
      DCHECK(b_ != end_);
      i_++;
      SkipUnused();
      return *this;
    }
    iterator operator++(int /*indicates postfix*/) {
      iterator tmp(*this);
      ++*this;
      return tmp;
    }

   private:
    friend class FlatMap;
    Bucket* b_;
    Bucket* end_;
    char space_ alignas(value_type)[sizeof(value_type)];
    uint32 i_;

    pointer val() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_25(mht_25_v, 407, "", "./tensorflow/core/lib/gtl/flatmap.h", "val");
 return reinterpret_cast<pointer>(space_); }
    void FillValue() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_26(mht_26_v, 411, "", "./tensorflow/core/lib/gtl/flatmap.h", "FillValue");
 new (space_) value_type(b_->key(i_), b_->val(i_)); }
    void SkipUnused() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_27(mht_27_v, 415, "", "./tensorflow/core/lib/gtl/flatmap.h", "SkipUnused");

      while (b_ < end_) {
        if (i_ >= Rep::kWidth) {
          i_ = 0;
          b_++;
        } else if (b_->marker[i_] < 2) {
          i_++;
        } else {
          FillValue();
          break;
        }
      }
    }
  };

  class const_iterator {
   private:
    mutable iterator rep_;  // Share state and logic with non-const iterator.
   public:
    typedef typename FlatMap::difference_type difference_type;
    typedef typename FlatMap::value_type value_type;
    typedef typename FlatMap::const_pointer pointer;
    typedef typename FlatMap::const_reference reference;
    typedef ::std::forward_iterator_tag iterator_category;

    const_iterator() : rep_() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_28(mht_28_v, 443, "", "./tensorflow/core/lib/gtl/flatmap.h", "const_iterator");
}
    const_iterator(Bucket* start, Bucket* end) : rep_(start, end) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_29(mht_29_v, 447, "", "./tensorflow/core/lib/gtl/flatmap.h", "const_iterator");
}
    const_iterator(Bucket* b, Bucket* end, uint32 i) : rep_(b, end, i) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_30(mht_30_v, 451, "", "./tensorflow/core/lib/gtl/flatmap.h", "const_iterator");
}

    reference operator*() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_31(mht_31_v, 456, "", "./tensorflow/core/lib/gtl/flatmap.h", "*");
 return *rep_.val(); }
    pointer operator->() const { return rep_.val(); }
    bool operator==(const const_iterator& x) const { return rep_ == x.rep_; }
    bool operator!=(const const_iterator& x) const { return rep_ != x.rep_; }
    const_iterator& operator++() {
      ++rep_;
      return *this;
    }
    const_iterator operator++(int /*indicates postfix*/) {
      const_iterator tmp(*this);
      ++*this;
      return tmp;
    }
  };

  iterator begin() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_32(mht_32_v, 474, "", "./tensorflow/core/lib/gtl/flatmap.h", "begin");
 return iterator(rep_.start(), rep_.limit()); }
  iterator end() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_33(mht_33_v, 478, "", "./tensorflow/core/lib/gtl/flatmap.h", "end");
 return iterator(rep_.limit(), rep_.limit()); }
  const_iterator begin() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_34(mht_34_v, 482, "", "./tensorflow/core/lib/gtl/flatmap.h", "begin");

    return const_iterator(rep_.start(), rep_.limit());
  }
  const_iterator end() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_35(mht_35_v, 488, "", "./tensorflow/core/lib/gtl/flatmap.h", "end");

    return const_iterator(rep_.limit(), rep_.limit());
  }

  size_t count(const Key& k) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_36(mht_36_v, 495, "", "./tensorflow/core/lib/gtl/flatmap.h", "count");
 return rep_.Find(k).found ? 1 : 0; }
  iterator find(const Key& k) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_37(mht_37_v, 499, "", "./tensorflow/core/lib/gtl/flatmap.h", "find");

    auto r = rep_.Find(k);
    return r.found ? iterator(r.b, rep_.limit(), r.index) : end();
  }
  const_iterator find(const Key& k) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_38(mht_38_v, 506, "", "./tensorflow/core/lib/gtl/flatmap.h", "find");

    auto r = rep_.Find(k);
    return r.found ? const_iterator(r.b, rep_.limit(), r.index) : end();
  }

  Val& at(const Key& k) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_39(mht_39_v, 514, "", "./tensorflow/core/lib/gtl/flatmap.h", "at");

    auto r = rep_.Find(k);
    DCHECK(r.found);
    return r.b->val(r.index);
  }
  const Val& at(const Key& k) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_40(mht_40_v, 522, "", "./tensorflow/core/lib/gtl/flatmap.h", "at");

    auto r = rep_.Find(k);
    DCHECK(r.found);
    return r.b->val(r.index);
  }

  template <typename P>
  std::pair<iterator, bool> insert(const P& p) {
    return Insert(p.first, p.second);
  }
  std::pair<iterator, bool> insert(const std::pair<const Key, Val>& p) {
    return Insert(p.first, p.second);
  }
  template <typename InputIter>
  void insert(InputIter first, InputIter last) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_41(mht_41_v, 539, "", "./tensorflow/core/lib/gtl/flatmap.h", "insert");

    for (; first != last; ++first) {
      insert(*first);
    }
  }

  Val& operator[](const Key& k) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_42(mht_42_v, 548, "", "./tensorflow/core/lib/gtl/flatmap.h", "lambda");
 return IndexOp(k); }
  Val& operator[](Key&& k) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_43(mht_43_v, 552, "", "./tensorflow/core/lib/gtl/flatmap.h", "lambda");
 return IndexOp(std::forward<Key>(k)); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return InsertPair(std::make_pair(std::forward<Args>(args)...));
  }

  size_t erase(const Key& k) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_44(mht_44_v, 562, "", "./tensorflow/core/lib/gtl/flatmap.h", "erase");

    auto r = rep_.Find(k);
    if (!r.found) return 0;
    rep_.Erase(r.b, r.index);
    return 1;
  }
  iterator erase(iterator pos) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_45(mht_45_v, 571, "", "./tensorflow/core/lib/gtl/flatmap.h", "erase");

    rep_.Erase(pos.b_, pos.i_);
    ++pos;
    return pos;
  }
  iterator erase(iterator pos, iterator last) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_46(mht_46_v, 579, "", "./tensorflow/core/lib/gtl/flatmap.h", "erase");

    for (; pos != last; ++pos) {
      rep_.Erase(pos.b_, pos.i_);
    }
    return pos;
  }

  std::pair<iterator, iterator> equal_range(const Key& k) {
    auto pos = find(k);
    if (pos == end()) {
      return std::make_pair(pos, pos);
    } else {
      auto next = pos;
      ++next;
      return std::make_pair(pos, next);
    }
  }
  std::pair<const_iterator, const_iterator> equal_range(const Key& k) const {
    auto pos = find(k);
    if (pos == end()) {
      return std::make_pair(pos, pos);
    } else {
      auto next = pos;
      ++next;
      return std::make_pair(pos, next);
    }
  }

  bool operator==(const FlatMap& x) const {
    if (size() != x.size()) return false;
    for (auto& p : x) {
      auto i = find(p.first);
      if (i == end()) return false;
      if (i->second != p.second) return false;
    }
    return true;
  }
  bool operator!=(const FlatMap& x) const { return !(*this == x); }

  // If key exists in the table, prefetch the associated value.  This
  // is a hint, and may have no effect.
  void prefetch_value(const Key& key) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_47(mht_47_v, 623, "", "./tensorflow/core/lib/gtl/flatmap.h", "prefetch_value");
 rep_.Prefetch(key); }

 private:
  using Rep = internal::FlatRep<Key, Bucket, Hash, Eq>;

  // Bucket stores kWidth <marker, key, value> triples.
  // The data is organized as three parallel arrays to reduce padding.
  struct Bucket {
    uint8 marker[Rep::kWidth];

    // Wrap keys and values in union to control construction and destruction.
    union Storage {
      struct {
        Key key[Rep::kWidth];
        Val val[Rep::kWidth];
      };
      Storage() {}
      ~Storage() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_48(mht_48_v, 643, "", "./tensorflow/core/lib/gtl/flatmap.h", "~Storage");
}
    } storage;

    Key& key(uint32 i) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_49(mht_49_v, 649, "", "./tensorflow/core/lib/gtl/flatmap.h", "key");

      DCHECK_GE(marker[i], 2);
      return storage.key[i];
    }
    Val& val(uint32 i) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_50(mht_50_v, 656, "", "./tensorflow/core/lib/gtl/flatmap.h", "val");

      DCHECK_GE(marker[i], 2);
      return storage.val[i];
    }
    template <typename V>
    void InitVal(uint32 i, V&& v) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_51(mht_51_v, 664, "", "./tensorflow/core/lib/gtl/flatmap.h", "InitVal");

      new (&storage.val[i]) Val(std::forward<V>(v));
    }
    void Destroy(uint32 i) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_52(mht_52_v, 670, "", "./tensorflow/core/lib/gtl/flatmap.h", "Destroy");

      storage.key[i].Key::~Key();
      storage.val[i].Val::~Val();
    }
    void MoveFrom(uint32 i, Bucket* src, uint32 src_index) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_53(mht_53_v, 677, "", "./tensorflow/core/lib/gtl/flatmap.h", "MoveFrom");

      new (&storage.key[i]) Key(std::move(src->storage.key[src_index]));
      new (&storage.val[i]) Val(std::move(src->storage.val[src_index]));
    }
    void CopyFrom(uint32 i, Bucket* src, uint32 src_index) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_54(mht_54_v, 684, "", "./tensorflow/core/lib/gtl/flatmap.h", "CopyFrom");

      new (&storage.key[i]) Key(src->storage.key[src_index]);
      new (&storage.val[i]) Val(src->storage.val[src_index]);
    }
  };

  template <typename Pair>
  std::pair<iterator, bool> InsertPair(Pair&& p) {
    return Insert(std::forward<decltype(p.first)>(p.first),
                  std::forward<decltype(p.second)>(p.second));
  }

  template <typename K, typename V>
  std::pair<iterator, bool> Insert(K&& k, V&& v) {
    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<K>(k));
    const bool inserted = !r.found;
    if (inserted) {
      r.b->InitVal(r.index, std::forward<V>(v));
    }
    return {iterator(r.b, rep_.limit(), r.index), inserted};
  }

  template <typename K>
  Val& IndexOp(K&& k) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmapDTh mht_55(mht_55_v, 711, "", "./tensorflow/core/lib/gtl/flatmap.h", "IndexOp");

    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<K>(k));
    Val* vptr = &r.b->val(r.index);
    if (!r.found) {
      new (vptr) Val();  // Initialize value in new slot.
    }
    return *vptr;
  }

  Rep rep_;
};

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_
