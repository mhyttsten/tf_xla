/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// This simple class finds the top n elements of an incrementally provided set
// of elements which you push one at a time.  If the number of elements exceeds
// n, the lowest elements are incrementally dropped.  At the end you get
// a vector of the top elements sorted in descending order (through Extract() or
// ExtractNondestructive()), or a vector of the top elements but not sorted
// (through ExtractUnsorted() or ExtractUnsortedNondestructive()).
//
// The value n is specified in the constructor.  If there are p elements pushed
// altogether:
//   The total storage requirements are O(min(n, p)) elements
//   The running time is O(p * log(min(n, p))) comparisons
// If n is a constant, the total storage required is a constant and the running
// time is linear in p.
//
// NOTE(zhifengc): There is a way to do this in O(min(n, p)) storage and O(p)
// runtime. The basic idea is to repeatedly fill up a buffer of 2 * n elements,
// discarding the lowest n elements whenever the buffer is full using a linear-
// time median algorithm. This may have better performance when the input
// sequence is partially sorted.
//
// NOTE(zhifengc): This class should be redesigned to avoid reallocating a
// vector for each Extract.

// Copied from tensorflow/core/lib/gtl/top_n.h
// TODO(b/111524997): Remove this file.
#ifndef TENSORFLOW_LITE_KERNELS_CTC_TOP_N_H_
#define TENSORFLOW_LITE_KERNELS_CTC_TOP_N_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh() {
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
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace gtl {

// Cmp is an stl binary predicate.  Note that Cmp is the "greater" predicate,
// not the more commonly used "less" predicate.
//
// If you use a "less" predicate here, the TopN will pick out the bottom N
// elements out of the ones passed to it, and it will return them sorted in
// ascending order.
//
// TopN is rule-of-zero copyable and movable if its members are.
template <class T, class Cmp = std::greater<T> >
class TopN {
 public:
  // The TopN is in one of the three states:
  //
  //  o UNORDERED: this is the state an instance is originally in,
  //    where the elements are completely orderless.
  //
  //  o BOTTOM_KNOWN: in this state, we keep the invariant that there
  //    is at least one element in it, and the lowest element is at
  //    position 0. The elements in other positions remain
  //    unsorted. This state is reached if the state was originally
  //    UNORDERED and a peek_bottom() function call is invoked.
  //
  //  o HEAP_SORTED: in this state, the array is kept as a heap and
  //    there are exactly (limit_+1) elements in the array. This
  //    state is reached when at least (limit_+1) elements are
  //    pushed in.
  //
  //  The state transition graph is at follows:
  //
  //             peek_bottom()                (limit_+1) elements
  //  UNORDERED --------------> BOTTOM_KNOWN --------------------> HEAP_SORTED
  //      |                                                           ^
  //      |                      (limit_+1) elements                  |
  //      +-----------------------------------------------------------+

  enum State { UNORDERED, BOTTOM_KNOWN, HEAP_SORTED };
  using UnsortedIterator = typename std::vector<T>::const_iterator;

  // 'limit' is the maximum number of top results to return.
  explicit TopN(size_t limit) : TopN(limit, Cmp()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_0(mht_0_v, 263, "", "./tensorflow/lite/kernels/ctc/top_n.h", "TopN");
}
  TopN(size_t limit, const Cmp &cmp) : limit_(limit), cmp_(cmp) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_1(mht_1_v, 267, "", "./tensorflow/lite/kernels/ctc/top_n.h", "TopN");
}

  size_t limit() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_2(mht_2_v, 272, "", "./tensorflow/lite/kernels/ctc/top_n.h", "limit");
 return limit_; }

  // Number of elements currently held by this TopN object.  This
  // will be no greater than 'limit' passed to the constructor.
  size_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_3(mht_3_v, 279, "", "./tensorflow/lite/kernels/ctc/top_n.h", "size");
 return std::min(elements_.size(), limit_); }

  bool empty() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_4(mht_4_v, 284, "", "./tensorflow/lite/kernels/ctc/top_n.h", "empty");
 return size() == 0; }

  // If you know how many elements you will push at the time you create the
  // TopN object, you can call reserve to preallocate the memory that TopN
  // will need to process all 'n' pushes.  Calling this method is optional.
  void reserve(size_t n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_5(mht_5_v, 292, "", "./tensorflow/lite/kernels/ctc/top_n.h", "reserve");
 elements_.reserve(std::min(n, limit_ + 1)); }

  // Push 'v'.  If the maximum number of elements was exceeded, drop the
  // lowest element and return it in 'dropped' (if given). If the maximum is not
  // exceeded, 'dropped' will remain unchanged. 'dropped' may be omitted or
  // nullptr, in which case it is not filled in.
  // Requires: T is CopyAssignable, Swappable
  void push(const T &v) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_6(mht_6_v, 302, "", "./tensorflow/lite/kernels/ctc/top_n.h", "push");
 push(v, nullptr); }
  void push(const T &v, T *dropped) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_7(mht_7_v, 306, "", "./tensorflow/lite/kernels/ctc/top_n.h", "push");
 PushInternal(v, dropped); }

  // Move overloads of push.
  // Requires: T is MoveAssignable, Swappable
  void push(T &&v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_8(mht_8_v, 313, "", "./tensorflow/lite/kernels/ctc/top_n.h", "push");
  // NOLINT(build/c++11)
    push(std::move(v), nullptr);
  }
  void push(T &&v, T *dropped) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_9(mht_9_v, 319, "", "./tensorflow/lite/kernels/ctc/top_n.h", "push");
  // NOLINT(build/c++11)
    PushInternal(std::move(v), dropped);
  }

  // Peeks the bottom result without calling Extract()
  const T &peek_bottom();

  // Extract the elements as a vector sorted in descending order.  The caller
  // assumes ownership of the vector and must delete it when done.  This is a
  // destructive operation.  The only method that can be called immediately
  // after Extract() is Reset().
  std::vector<T> *Extract();

  // Similar to Extract(), but makes no guarantees the elements are in sorted
  // order.  As with Extract(), the caller assumes ownership of the vector and
  // must delete it when done.  This is a destructive operation.  The only
  // method that can be called immediately after ExtractUnsorted() is Reset().
  std::vector<T> *ExtractUnsorted();

  // A non-destructive version of Extract(). Copy the elements in a new vector
  // sorted in descending order and return it.  The caller assumes ownership of
  // the new vector and must delete it when done.  After calling
  // ExtractNondestructive(), the caller can continue to push() new elements.
  std::vector<T> *ExtractNondestructive() const;

  // A non-destructive version of Extract(). Copy the elements to a given
  // vector sorted in descending order. After calling
  // ExtractNondestructive(), the caller can continue to push() new elements.
  // Note:
  //  1. The given argument must to be allocated.
  //  2. Any data contained in the vector prior to the call will be deleted
  //     from it. After the call the vector will contain only the elements
  //     from the data structure.
  void ExtractNondestructive(std::vector<T> *output) const;

  // A non-destructive version of ExtractUnsorted(). Copy the elements in a new
  // vector and return it, with no guarantees the elements are in sorted order.
  // The caller assumes ownership of the new vector and must delete it when
  // done.  After calling ExtractUnsortedNondestructive(), the caller can
  // continue to push() new elements.
  std::vector<T> *ExtractUnsortedNondestructive() const;

  // A non-destructive version of ExtractUnsorted(). Copy the elements into
  // a given vector, with no guarantees the elements are in sorted order.
  // After calling ExtractUnsortedNondestructive(), the caller can continue
  // to push() new elements.
  // Note:
  //  1. The given argument must to be allocated.
  //  2. Any data contained in the vector prior to the call will be deleted
  //     from it. After the call the vector will contain only the elements
  //     from the data structure.
  void ExtractUnsortedNondestructive(std::vector<T> *output) const;

  // Return an iterator to the beginning (end) of the container,
  // with no guarantees about the order of iteration. These iterators are
  // invalidated by mutation of the data structure.
  UnsortedIterator unsorted_begin() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_10(mht_10_v, 378, "", "./tensorflow/lite/kernels/ctc/top_n.h", "unsorted_begin");
 return elements_.begin(); }
  UnsortedIterator unsorted_end() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_11(mht_11_v, 382, "", "./tensorflow/lite/kernels/ctc/top_n.h", "unsorted_end");
 return elements_.begin() + size(); }

  // Accessor for comparator template argument.
  Cmp *comparator() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPStop_nDTh mht_12(mht_12_v, 388, "", "./tensorflow/lite/kernels/ctc/top_n.h", "comparator");
 return &cmp_; }

  // This removes all elements.  If Extract() or ExtractUnsorted() have been
  // called, this will put it back in an empty but useable state.
  void Reset();

 private:
  template <typename U>
  void PushInternal(U &&v, T *dropped);  // NOLINT(build/c++11)

  // elements_ can be in one of two states:
  //   elements_.size() <= limit_:  elements_ is an unsorted vector of elements
  //      pushed so far.
  //   elements_.size() > limit_:  The last element of elements_ is unused;
  //      the other elements of elements_ are an stl heap whose size is exactly
  //      limit_.  In this case elements_.size() is exactly one greater than
  //      limit_, but don't use "elements_.size() == limit_ + 1" to check for
  //      that because you'll get a false positive if limit_ == size_t(-1).
  std::vector<T> elements_;
  size_t limit_;  // Maximum number of elements to find
  Cmp cmp_;       // Greater-than comparison function
  State state_ = UNORDERED;
};

// ----------------------------------------------------------------------
// Implementations of non-inline functions

template <class T, class Cmp>
template <typename U>
void TopN<T, Cmp>::PushInternal(U &&v, T *dropped) {  // NOLINT(build/c++11)
  if (limit_ == 0) {
    if (dropped) *dropped = std::forward<U>(v);  // NOLINT(build/c++11)
    return;
  }
  if (state_ != HEAP_SORTED) {
    elements_.push_back(std::forward<U>(v));  // NOLINT(build/c++11)
    if (state_ == UNORDERED || cmp_(elements_.back(), elements_.front())) {
      // Easy case: we just pushed the new element back
    } else {
      // To maintain the BOTTOM_KNOWN state, we need to make sure that
      // the element at position 0 is always the smallest. So we put
      // the new element at position 0 and push the original bottom
      // element in the back.
      // Warning: this code is subtle.
      using std::swap;
      swap(elements_.front(), elements_.back());
    }
    if (elements_.size() == limit_ + 1) {
      // Transition from unsorted vector to a heap.
      std::make_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
      state_ = HEAP_SORTED;
    }
  } else {
    // Only insert the new element if it is greater than the least element.
    if (cmp_(v, elements_.front())) {
      elements_.back() = std::forward<U>(v);  // NOLINT(build/c++11)
      std::push_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
    } else {
      if (dropped) *dropped = std::forward<U>(v);  // NOLINT(build/c++11)
    }
  }
}

template <class T, class Cmp>
const T &TopN<T, Cmp>::peek_bottom() {
  TFLITE_DCHECK(!empty());
  if (state_ == UNORDERED) {
    // We need to do a linear scan to find out the bottom element
    int min_candidate = 0;
    for (size_t i = 1; i < elements_.size(); ++i) {
      if (cmp_(elements_[min_candidate], elements_[i])) {
        min_candidate = i;
      }
    }
    // By swapping the element at position 0 and the minimal
    // element, we transition to the BOTTOM_KNOWN state
    if (min_candidate != 0) {
      using std::swap;
      swap(elements_[0], elements_[min_candidate]);
    }
    state_ = BOTTOM_KNOWN;
  }
  return elements_.front();
}

template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::Extract() {
  auto out = new std::vector<T>;
  out->swap(elements_);
  if (state_ != HEAP_SORTED) {
    std::sort(out->begin(), out->end(), cmp_);
  } else {
    out->pop_back();
    std::sort_heap(out->begin(), out->end(), cmp_);
  }
  return out;
}

template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractUnsorted() {
  auto out = new std::vector<T>;
  out->swap(elements_);
  if (state_ == HEAP_SORTED) {
    // Remove the limit_+1'th element.
    out->pop_back();
  }
  return out;
}

template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractNondestructive() const {
  auto out = new std::vector<T>;
  ExtractNondestructive(out);
  return out;
}

template <class T, class Cmp>
void TopN<T, Cmp>::ExtractNondestructive(std::vector<T> *output) const {
  TFLITE_DCHECK(output);
  *output = elements_;
  if (state_ != HEAP_SORTED) {
    std::sort(output->begin(), output->end(), cmp_);
  } else {
    output->pop_back();
    std::sort_heap(output->begin(), output->end(), cmp_);
  }
}

template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractUnsortedNondestructive() const {
  auto elements = new std::vector<T>;
  ExtractUnsortedNondestructive(elements);
  return elements;
}

template <class T, class Cmp>
void TopN<T, Cmp>::ExtractUnsortedNondestructive(std::vector<T> *output) const {
  TFLITE_DCHECK(output);
  *output = elements_;
  if (state_ == HEAP_SORTED) {
    // Remove the limit_+1'th element.
    output->pop_back();
  }
}

template <class T, class Cmp>
void TopN<T, Cmp>::Reset() {
  elements_.clear();
  state_ = UNORDERED;
}

}  // namespace gtl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CTC_TOP_N_H_
