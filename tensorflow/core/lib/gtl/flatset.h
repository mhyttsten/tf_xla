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

#ifndef TENSORFLOW_CORE_LIB_GTL_FLATSET_H_
#define TENSORFLOW_CORE_LIB_GTL_FLATSET_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh() {
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

// FlatSet<K,...> provides a set of K.
//
// The map is implemented using an open-addressed hash table.  A
// single array holds entire map contents and collisions are resolved
// by probing at a sequence of locations in the array.
template <typename Key, class Hash = hash<Key>, class Eq = std::equal_to<Key>>
class FlatSet {
 private:
  // Forward declare some internal types needed in public section.
  struct Bucket;

 public:
  typedef Key key_type;
  typedef Key value_type;
  typedef Hash hasher;
  typedef Eq key_equal;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  FlatSet() : FlatSet(1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");
}

  explicit FlatSet(size_t N, const Hash& hf = Hash(), const Eq& eq = Eq())
      : rep_(N, hf, eq) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");
}

  FlatSet(const FlatSet& src) : rep_(src.rep_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_2(mht_2_v, 235, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");
}

  // Move constructor leaves src in a valid but unspecified state (same as
  // std::unordered_set).
  FlatSet(FlatSet&& src) : rep_(std::move(src.rep_)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_3(mht_3_v, 242, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");
}

  template <typename InputIter>
  FlatSet(InputIter first, InputIter last, size_t N = 1,
          const Hash& hf = Hash(), const Eq& eq = Eq())
      : FlatSet(N, hf, eq) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_4(mht_4_v, 250, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");

    insert(first, last);
  }

  FlatSet(std::initializer_list<value_type> init, size_t N = 1,
          const Hash& hf = Hash(), const Eq& eq = Eq())
      : FlatSet(init.begin(), init.end(), N, hf, eq) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_5(mht_5_v, 259, "", "./tensorflow/core/lib/gtl/flatset.h", "FlatSet");
}

  FlatSet& operator=(const FlatSet& src) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_6(mht_6_v, 264, "", "./tensorflow/core/lib/gtl/flatset.h", "=");

    rep_.CopyFrom(src.rep_);
    return *this;
  }

  // Move-assignment operator leaves src in a valid but unspecified state (same
  // as std::unordered_set).
  FlatSet& operator=(FlatSet&& src) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_7(mht_7_v, 274, "", "./tensorflow/core/lib/gtl/flatset.h", "=");

    rep_.MoveFrom(std::move(src.rep_));
    return *this;
  }

  ~FlatSet() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_8(mht_8_v, 282, "", "./tensorflow/core/lib/gtl/flatset.h", "~FlatSet");
}

  void swap(FlatSet& x) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_9(mht_9_v, 287, "", "./tensorflow/core/lib/gtl/flatset.h", "swap");
 rep_.swap(x.rep_); }
  void clear_no_resize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_10(mht_10_v, 291, "", "./tensorflow/core/lib/gtl/flatset.h", "clear_no_resize");
 rep_.clear_no_resize(); }
  void clear() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_11(mht_11_v, 295, "", "./tensorflow/core/lib/gtl/flatset.h", "clear");
 rep_.clear(); }
  void reserve(size_t N) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_12(mht_12_v, 299, "", "./tensorflow/core/lib/gtl/flatset.h", "reserve");
 rep_.Resize(std::max(N, size())); }
  void rehash(size_t N) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_13(mht_13_v, 303, "", "./tensorflow/core/lib/gtl/flatset.h", "rehash");
 rep_.Resize(std::max(N, size())); }
  void resize(size_t N) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_14(mht_14_v, 307, "", "./tensorflow/core/lib/gtl/flatset.h", "resize");
 rep_.Resize(std::max(N, size())); }
  size_t size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_15(mht_15_v, 311, "", "./tensorflow/core/lib/gtl/flatset.h", "size");
 return rep_.size(); }
  bool empty() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_16(mht_16_v, 315, "", "./tensorflow/core/lib/gtl/flatset.h", "empty");
 return size() == 0; }
  size_t bucket_count() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_17(mht_17_v, 319, "", "./tensorflow/core/lib/gtl/flatset.h", "bucket_count");
 return rep_.bucket_count(); }
  hasher hash_function() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_18(mht_18_v, 323, "", "./tensorflow/core/lib/gtl/flatset.h", "hash_function");
 return rep_.hash_function(); }
  key_equal key_eq() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_19(mht_19_v, 327, "", "./tensorflow/core/lib/gtl/flatset.h", "key_eq");
 return rep_.key_eq(); }

  class const_iterator {
   public:
    typedef typename FlatSet::difference_type difference_type;
    typedef typename FlatSet::value_type value_type;
    typedef typename FlatSet::const_pointer pointer;
    typedef typename FlatSet::const_reference reference;
    typedef ::std::forward_iterator_tag iterator_category;

    const_iterator() : b_(nullptr), end_(nullptr), i_(0) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_20(mht_20_v, 340, "", "./tensorflow/core/lib/gtl/flatset.h", "const_iterator");
}

    // Make iterator pointing at first element at or after b.
    const_iterator(Bucket* b, Bucket* end) : b_(b), end_(end), i_(0) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_21(mht_21_v, 346, "", "./tensorflow/core/lib/gtl/flatset.h", "const_iterator");

      SkipUnused();
    }

    // Make iterator pointing exactly at ith element in b, which must exist.
    const_iterator(Bucket* b, Bucket* end, uint32 i)
        : b_(b), end_(end), i_(i) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_22(mht_22_v, 355, "", "./tensorflow/core/lib/gtl/flatset.h", "const_iterator");
}

    reference operator*() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_23(mht_23_v, 360, "", "./tensorflow/core/lib/gtl/flatset.h", "*");
 return key(); }
    pointer operator->() const { return &key(); }
    bool operator==(const const_iterator& x) const {
      return b_ == x.b_ && i_ == x.i_;
    }
    bool operator!=(const const_iterator& x) const { return !(*this == x); }
    const_iterator& operator++() {
      DCHECK(b_ != end_);
      i_++;
      SkipUnused();
      return *this;
    }
    const_iterator operator++(int /*indicates postfix*/) {
      const_iterator tmp(*this);
      ++*this;
      return tmp;
    }

   private:
    friend class FlatSet;
    Bucket* b_;
    Bucket* end_;
    uint32 i_;

    reference key() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_24(mht_24_v, 387, "", "./tensorflow/core/lib/gtl/flatset.h", "key");
 return b_->key(i_); }
    void SkipUnused() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_25(mht_25_v, 391, "", "./tensorflow/core/lib/gtl/flatset.h", "SkipUnused");

      while (b_ < end_) {
        if (i_ >= Rep::kWidth) {
          i_ = 0;
          b_++;
        } else if (b_->marker[i_] < 2) {
          i_++;
        } else {
          break;
        }
      }
    }
  };

  typedef const_iterator iterator;

  iterator begin() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_26(mht_26_v, 410, "", "./tensorflow/core/lib/gtl/flatset.h", "begin");
 return iterator(rep_.start(), rep_.limit()); }
  iterator end() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_27(mht_27_v, 414, "", "./tensorflow/core/lib/gtl/flatset.h", "end");
 return iterator(rep_.limit(), rep_.limit()); }
  const_iterator begin() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_28(mht_28_v, 418, "", "./tensorflow/core/lib/gtl/flatset.h", "begin");

    return const_iterator(rep_.start(), rep_.limit());
  }
  const_iterator end() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_29(mht_29_v, 424, "", "./tensorflow/core/lib/gtl/flatset.h", "end");

    return const_iterator(rep_.limit(), rep_.limit());
  }

  size_t count(const Key& k) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_30(mht_30_v, 431, "", "./tensorflow/core/lib/gtl/flatset.h", "count");
 return rep_.Find(k).found ? 1 : 0; }
  iterator find(const Key& k) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_31(mht_31_v, 435, "", "./tensorflow/core/lib/gtl/flatset.h", "find");

    auto r = rep_.Find(k);
    return r.found ? iterator(r.b, rep_.limit(), r.index) : end();
  }
  const_iterator find(const Key& k) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_32(mht_32_v, 442, "", "./tensorflow/core/lib/gtl/flatset.h", "find");

    auto r = rep_.Find(k);
    return r.found ? const_iterator(r.b, rep_.limit(), r.index) : end();
  }

  std::pair<iterator, bool> insert(const Key& k) { return Insert(k); }
  std::pair<iterator, bool> insert(Key&& k) { return Insert(std::move(k)); }
  template <typename InputIter>
  void insert(InputIter first, InputIter last) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_33(mht_33_v, 453, "", "./tensorflow/core/lib/gtl/flatset.h", "insert");

    for (; first != last; ++first) {
      insert(*first);
    }
  }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<Args>(args)...);
    const bool inserted = !r.found;
    return {iterator(r.b, rep_.limit(), r.index), inserted};
  }

  size_t erase(const Key& k) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_34(mht_34_v, 470, "", "./tensorflow/core/lib/gtl/flatset.h", "erase");

    auto r = rep_.Find(k);
    if (!r.found) return 0;
    rep_.Erase(r.b, r.index);
    return 1;
  }
  iterator erase(iterator pos) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_35(mht_35_v, 479, "", "./tensorflow/core/lib/gtl/flatset.h", "erase");

    rep_.Erase(pos.b_, pos.i_);
    ++pos;
    return pos;
  }
  iterator erase(iterator pos, iterator last) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_36(mht_36_v, 487, "", "./tensorflow/core/lib/gtl/flatset.h", "erase");

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

  bool operator==(const FlatSet& x) const {
    if (size() != x.size()) return false;
    for (const auto& elem : x) {
      auto i = find(elem);
      if (i == end()) return false;
    }
    return true;
  }
  bool operator!=(const FlatSet& x) const { return !(*this == x); }

  // If key exists in the table, prefetch it.  This is a hint, and may
  // have no effect.
  void prefetch_value(const Key& key) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_37(mht_37_v, 530, "", "./tensorflow/core/lib/gtl/flatset.h", "prefetch_value");
 rep_.Prefetch(key); }

 private:
  using Rep = internal::FlatRep<Key, Bucket, Hash, Eq>;

  // Bucket stores kWidth <marker, key, value> triples.
  // The data is organized as three parallel arrays to reduce padding.
  struct Bucket {
    uint8 marker[Rep::kWidth];

    // Wrap keys in union to control construction and destruction.
    union Storage {
      Key key[Rep::kWidth];
      Storage() {}
      ~Storage() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_38(mht_38_v, 547, "", "./tensorflow/core/lib/gtl/flatset.h", "~Storage");
}
    } storage;

    Key& key(uint32 i) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_39(mht_39_v, 553, "", "./tensorflow/core/lib/gtl/flatset.h", "key");

      DCHECK_GE(marker[i], 2);
      return storage.key[i];
    }
    void Destroy(uint32 i) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_40(mht_40_v, 560, "", "./tensorflow/core/lib/gtl/flatset.h", "Destroy");
 storage.key[i].Key::~Key(); }
    void MoveFrom(uint32 i, Bucket* src, uint32 src_index) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_41(mht_41_v, 564, "", "./tensorflow/core/lib/gtl/flatset.h", "MoveFrom");

      new (&storage.key[i]) Key(std::move(src->storage.key[src_index]));
    }
    void CopyFrom(uint32 i, Bucket* src, uint32 src_index) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatsetDTh mht_42(mht_42_v, 570, "", "./tensorflow/core/lib/gtl/flatset.h", "CopyFrom");

      new (&storage.key[i]) Key(src->storage.key[src_index]);
    }
  };

  template <typename K>
  std::pair<iterator, bool> Insert(K&& k) {
    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<K>(k));
    const bool inserted = !r.found;
    return {iterator(r.b, rep_.limit(), r.index), inserted};
  }

  Rep rep_;
};

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_FLATSET_H_
