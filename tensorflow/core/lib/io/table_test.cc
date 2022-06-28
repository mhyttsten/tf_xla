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
class MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc() {
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

#include "tensorflow/core/lib/io/table.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/block.h"
#include "tensorflow/core/lib/io/block_builder.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace table {

namespace {
typedef std::pair<StringPiece, StringPiece> StringPiecePair;
}

namespace test {
static StringPiece RandomString(random::SimplePhilox* rnd, int len,
                                string* dst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/lib/io/table_test.cc", "RandomString");

  dst->resize(len);
  for (int i = 0; i < len; i++) {
    (*dst)[i] = static_cast<char>(' ' + rnd->Uniform(95));  // ' ' .. '~'
  }
  return StringPiece(*dst);
}
static string RandomKey(random::SimplePhilox* rnd, int len) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/lib/io/table_test.cc", "RandomKey");

  // Make sure to generate a wide variety of characters so we
  // test the boundary conditions for short-key optimizations.
  static const char kTestChars[] = {'\0', '\1', 'a',    'b',    'c',
                                    'd',  'e',  '\xfd', '\xfe', '\xff'};
  string result;
  for (int i = 0; i < len; i++) {
    result += kTestChars[rnd->Uniform(sizeof(kTestChars))];
  }
  return result;
}
static StringPiece CompressibleString(random::SimplePhilox* rnd,
                                      double compressed_fraction, size_t len,
                                      string* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/lib/io/table_test.cc", "CompressibleString");

  int raw = static_cast<int>(len * compressed_fraction);
  if (raw < 1) raw = 1;
  string raw_data;
  RandomString(rnd, raw, &raw_data);

  // Duplicate the random data until we have filled "len" bytes
  dst->clear();
  while (dst->size() < len) {
    dst->append(raw_data);
  }
  dst->resize(len);
  return StringPiece(*dst);
}
}  // namespace test

static void Increment(string* key) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/lib/io/table_test.cc", "Increment");
 key->push_back('\0'); }

// An STL comparator that compares two StringPieces
namespace {
struct STLLessThan {
  STLLessThan() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/lib/io/table_test.cc", "STLLessThan");
}
  bool operator()(const string& a, const string& b) const {
    return StringPiece(a).compare(StringPiece(b)) < 0;
  }
};
}  // namespace

class StringSink : public WritableFile {
 public:
  ~StringSink() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/lib/io/table_test.cc", "~StringSink");
}

  const string& contents() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/lib/io/table_test.cc", "contents");
 return contents_; }

  Status Close() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_7(mht_7_v, 288, "", "./tensorflow/core/lib/io/table_test.cc", "Close");
 return Status::OK(); }
  Status Flush() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_8(mht_8_v, 292, "", "./tensorflow/core/lib/io/table_test.cc", "Flush");
 return Status::OK(); }
  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_9(mht_9_v, 296, "", "./tensorflow/core/lib/io/table_test.cc", "Name");

    return errors::Unimplemented("StringSink does not support Name()");
  }
  Status Sync() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_10(mht_10_v, 302, "", "./tensorflow/core/lib/io/table_test.cc", "Sync");
 return Status::OK(); }
  Status Tell(int64_t* pos) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_11(mht_11_v, 306, "", "./tensorflow/core/lib/io/table_test.cc", "Tell");

    *pos = contents_.size();
    return Status::OK();
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_12(mht_12_v, 314, "", "./tensorflow/core/lib/io/table_test.cc", "Append");

    contents_.append(data.data(), data.size());
    return Status::OK();
  }

 private:
  string contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(const StringPiece& contents)
      : contents_(contents.data(), contents.size()), bytes_read_(0) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_13(mht_13_v, 329, "", "./tensorflow/core/lib/io/table_test.cc", "StringSource");
}

  ~StringSource() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_14(mht_14_v, 334, "", "./tensorflow/core/lib/io/table_test.cc", "~StringSource");
}

  uint64 Size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_15(mht_15_v, 339, "", "./tensorflow/core/lib/io/table_test.cc", "Size");
 return contents_.size(); }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_16(mht_16_v, 344, "", "./tensorflow/core/lib/io/table_test.cc", "Name");

    return errors::Unimplemented("StringSource does not support Name()");
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_17(mht_17_v, 353, "", "./tensorflow/core/lib/io/table_test.cc", "Read");

    if (offset > contents_.size()) {
      return errors::InvalidArgument("invalid Read offset");
    }
    if (offset + n > contents_.size()) {
      n = contents_.size() - offset;
    }
    memcpy(scratch, &contents_[offset], n);
    *result = StringPiece(scratch, n);
    bytes_read_ += n;
    return Status::OK();
  }

  uint64 BytesRead() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_18(mht_18_v, 369, "", "./tensorflow/core/lib/io/table_test.cc", "BytesRead");
 return bytes_read_; }

 private:
  string contents_;
  mutable uint64 bytes_read_;
};

typedef std::map<string, string, STLLessThan> KVMap;

// Helper class for tests to unify the interface between
// BlockBuilder/TableBuilder and Block/Table.
class Constructor {
 public:
  explicit Constructor() : data_(STLLessThan()) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_19(mht_19_v, 385, "", "./tensorflow/core/lib/io/table_test.cc", "Constructor");
}
  virtual ~Constructor() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_20(mht_20_v, 389, "", "./tensorflow/core/lib/io/table_test.cc", "~Constructor");
}

  void Add(const string& key, const StringPiece& value) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_21(mht_21_v, 395, "", "./tensorflow/core/lib/io/table_test.cc", "Add");

    data_[key] = string(value);
  }

  // Finish constructing the data structure with all the keys that have
  // been added so far.  Returns the keys in sorted order in "*keys"
  // and stores the key/value pairs in "*kvmap"
  void Finish(const Options& options, std::vector<string>* keys, KVMap* kvmap) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_22(mht_22_v, 405, "", "./tensorflow/core/lib/io/table_test.cc", "Finish");

    *kvmap = data_;
    keys->clear();
    for (KVMap::const_iterator it = data_.begin(); it != data_.end(); ++it) {
      keys->push_back(it->first);
    }
    data_.clear();
    Status s = FinishImpl(options, *kvmap);
    ASSERT_TRUE(s.ok()) << s.ToString();
  }

  // Construct the data structure from the data in "data"
  virtual Status FinishImpl(const Options& options, const KVMap& data) = 0;

  virtual Iterator* NewIterator() const = 0;

  virtual const KVMap& data() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_23(mht_23_v, 424, "", "./tensorflow/core/lib/io/table_test.cc", "data");
 return data_; }

 private:
  KVMap data_;
};

class BlockConstructor : public Constructor {
 public:
  BlockConstructor() : block_(nullptr) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_24(mht_24_v, 435, "", "./tensorflow/core/lib/io/table_test.cc", "BlockConstructor");
}
  ~BlockConstructor() override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_25(mht_25_v, 439, "", "./tensorflow/core/lib/io/table_test.cc", "~BlockConstructor");
 delete block_; }
  Status FinishImpl(const Options& options, const KVMap& data) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_26(mht_26_v, 443, "", "./tensorflow/core/lib/io/table_test.cc", "FinishImpl");

    delete block_;
    block_ = nullptr;
    BlockBuilder builder(&options);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
    }
    // Open the block
    data_ = string(builder.Finish());
    BlockContents contents;
    contents.data = data_;
    contents.cacheable = false;
    contents.heap_allocated = false;
    block_ = new Block(contents);
    return Status::OK();
  }
  Iterator* NewIterator() const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_27(mht_27_v, 463, "", "./tensorflow/core/lib/io/table_test.cc", "NewIterator");
 return block_->NewIterator(); }

 private:
  string data_;
  Block* block_;
};

class TableConstructor : public Constructor {
 public:
  TableConstructor() : source_(nullptr), table_(nullptr) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_28(mht_28_v, 475, "", "./tensorflow/core/lib/io/table_test.cc", "TableConstructor");
}
  ~TableConstructor() override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_29(mht_29_v, 479, "", "./tensorflow/core/lib/io/table_test.cc", "~TableConstructor");
 Reset(); }
  Status FinishImpl(const Options& options, const KVMap& data) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_30(mht_30_v, 483, "", "./tensorflow/core/lib/io/table_test.cc", "FinishImpl");

    Reset();
    StringSink sink;
    TableBuilder builder(options, &sink);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
      TF_CHECK_OK(builder.status());
    }
    Status s = builder.Finish();
    TF_CHECK_OK(s) << s.ToString();

    CHECK_EQ(sink.contents().size(), builder.FileSize());

    // Open the table
    source_ = new StringSource(sink.contents());
    Options table_options;
    return Table::Open(table_options, source_, sink.contents().size(), &table_);
  }

  Iterator* NewIterator() const override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_31(mht_31_v, 506, "", "./tensorflow/core/lib/io/table_test.cc", "NewIterator");
 return table_->NewIterator(); }

  uint64 ApproximateOffsetOf(const StringPiece& key) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_32(mht_32_v, 511, "", "./tensorflow/core/lib/io/table_test.cc", "ApproximateOffsetOf");

    return table_->ApproximateOffsetOf(key);
  }

  uint64 BytesRead() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_33(mht_33_v, 518, "", "./tensorflow/core/lib/io/table_test.cc", "BytesRead");
 return source_->BytesRead(); }

 private:
  void Reset() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_34(mht_34_v, 524, "", "./tensorflow/core/lib/io/table_test.cc", "Reset");

    delete table_;
    delete source_;
    table_ = nullptr;
    source_ = nullptr;
  }

  StringSource* source_;
  Table* table_;
};

enum TestType { TABLE_TEST, BLOCK_TEST };

struct TestArgs {
  TestType type;
  int restart_interval;
};

static const TestArgs kTestArgList[] = {
    {TABLE_TEST, 16}, {TABLE_TEST, 1}, {TABLE_TEST, 1024},
    {BLOCK_TEST, 16}, {BLOCK_TEST, 1}, {BLOCK_TEST, 1024},
};
static const int kNumTestArgs = sizeof(kTestArgList) / sizeof(kTestArgList[0]);

class Harness : public ::testing::Test {
 public:
  Harness() : constructor_(nullptr) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_35(mht_35_v, 553, "", "./tensorflow/core/lib/io/table_test.cc", "Harness");
}

  void Init(const TestArgs& args) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_36(mht_36_v, 558, "", "./tensorflow/core/lib/io/table_test.cc", "Init");

    delete constructor_;
    constructor_ = nullptr;
    options_ = Options();

    options_.block_restart_interval = args.restart_interval;
    // Use shorter block size for tests to exercise block boundary
    // conditions more.
    options_.block_size = 256;
    switch (args.type) {
      case TABLE_TEST:
        constructor_ = new TableConstructor();
        break;
      case BLOCK_TEST:
        constructor_ = new BlockConstructor();
        break;
    }
  }

  ~Harness() override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_37(mht_37_v, 580, "", "./tensorflow/core/lib/io/table_test.cc", "~Harness");
 delete constructor_; }

  void Add(const string& key, const string& value) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("key: \"" + key + "\"");
   mht_38_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_38(mht_38_v, 587, "", "./tensorflow/core/lib/io/table_test.cc", "Add");

    constructor_->Add(key, value);
  }

  void Test(random::SimplePhilox* rnd, int num_random_access_iters = 200) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_39(mht_39_v, 594, "", "./tensorflow/core/lib/io/table_test.cc", "Test");

    std::vector<string> keys;
    KVMap data;
    constructor_->Finish(options_, &keys, &data);

    TestForwardScan(keys, data);
    TestRandomAccess(rnd, keys, data, num_random_access_iters);
  }

  void TestForwardScan(const std::vector<string>& keys, const KVMap& data) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_40(mht_40_v, 606, "", "./tensorflow/core/lib/io/table_test.cc", "TestForwardScan");

    Iterator* iter = constructor_->NewIterator();
    ASSERT_TRUE(!iter->Valid());
    iter->SeekToFirst();
    for (KVMap::const_iterator model_iter = data.begin();
         model_iter != data.end(); ++model_iter) {
      ASSERT_EQ(ToStringPiecePair(data, model_iter), ToStringPiecePair(iter));
      iter->Next();
    }
    ASSERT_TRUE(!iter->Valid());
    delete iter;
  }

  void TestRandomAccess(random::SimplePhilox* rnd,
                        const std::vector<string>& keys, const KVMap& data,
                        int num_random_access_iters) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_41(mht_41_v, 624, "", "./tensorflow/core/lib/io/table_test.cc", "TestRandomAccess");

    static const bool kVerbose = false;
    Iterator* iter = constructor_->NewIterator();
    ASSERT_TRUE(!iter->Valid());
    KVMap::const_iterator model_iter = data.begin();
    if (kVerbose) fprintf(stderr, "---\n");
    for (int i = 0; i < num_random_access_iters; i++) {
      const int toss = rnd->Uniform(3);
      switch (toss) {
        case 0: {
          if (iter->Valid()) {
            if (kVerbose) fprintf(stderr, "Next\n");
            iter->Next();
            ++model_iter;
            ASSERT_EQ(ToStringPiecePair(data, model_iter),
                      ToStringPiecePair(iter));
          }
          break;
        }

        case 1: {
          if (kVerbose) fprintf(stderr, "SeekToFirst\n");
          iter->SeekToFirst();
          model_iter = data.begin();
          ASSERT_EQ(ToStringPiecePair(data, model_iter),
                    ToStringPiecePair(iter));
          break;
        }

        case 2: {
          string key = PickRandomKey(rnd, keys);
          model_iter = data.lower_bound(key);
          if (kVerbose)
            fprintf(stderr, "Seek '%s'\n", absl::CEscape(key).c_str());
          iter->Seek(StringPiece(key));
          ASSERT_EQ(ToStringPiecePair(data, model_iter),
                    ToStringPiecePair(iter));
          break;
        }
      }
    }
    delete iter;
  }

  StringPiecePair ToStringPiecePair(const KVMap& data,
                                    const KVMap::const_iterator& it) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_42(mht_42_v, 672, "", "./tensorflow/core/lib/io/table_test.cc", "ToStringPiecePair");

    if (it == data.end()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->first, it->second);
    }
  }

  StringPiecePair ToStringPiecePair(const KVMap& data,
                                    const KVMap::const_reverse_iterator& it) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_43(mht_43_v, 684, "", "./tensorflow/core/lib/io/table_test.cc", "ToStringPiecePair");

    if (it == data.rend()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->first, it->second);
    }
  }

  StringPiecePair ToStringPiecePair(const Iterator* it) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_44(mht_44_v, 695, "", "./tensorflow/core/lib/io/table_test.cc", "ToStringPiecePair");

    if (!it->Valid()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->key(), it->value());
    }
  }

  string PickRandomKey(random::SimplePhilox* rnd,
                       const std::vector<string>& keys) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_45(mht_45_v, 707, "", "./tensorflow/core/lib/io/table_test.cc", "PickRandomKey");

    if (keys.empty()) {
      return "foo";
    } else {
      const int index = rnd->Uniform(keys.size());
      string result = keys[index];
      switch (rnd->Uniform(3)) {
        case 0:
          // Return an existing key
          break;
        case 1: {
          // Attempt to return something smaller than an existing key
          if (!result.empty() && result[result.size() - 1] > '\0') {
            result[result.size() - 1]--;
          }
          break;
        }
        case 2: {
          // Return something larger than an existing key
          Increment(&result);
          break;
        }
      }
      return result;
    }
  }

 private:
  Options options_;
  Constructor* constructor_;
};

// Test empty table/block.
TEST_F(Harness, Empty) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 1, 17);
    random::SimplePhilox rnd(&philox);
    Test(&rnd);
  }
}

// Special test for a block with no restart entries.  The C++ leveldb
// code never generates such blocks, but the Java version of leveldb
// seems to.
TEST_F(Harness, ZeroRestartPointsInBlock) {
  char data[sizeof(uint32)];
  memset(data, 0, sizeof(data));
  BlockContents contents;
  contents.data = StringPiece(data, sizeof(data));
  contents.cacheable = false;
  contents.heap_allocated = false;
  Block block(contents);
  Iterator* iter = block.NewIterator();
  iter->SeekToFirst();
  ASSERT_TRUE(!iter->Valid());
  iter->Seek("foo");
  ASSERT_TRUE(!iter->Valid());
  delete iter;
}

// Test the empty key
TEST_F(Harness, SimpleEmptyKey) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 1, 17);
    random::SimplePhilox rnd(&philox);
    Add("", "v");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleSingle) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 2, 17);
    random::SimplePhilox rnd(&philox);
    Add("abc", "v");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleMulti) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 3, 17);
    random::SimplePhilox rnd(&philox);
    Add("abc", "v");
    Add("abcd", "v");
    Add("ac", "v2");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleMultiBigValues) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 3, 17);
    random::SimplePhilox rnd(&philox);
    Add("ainitial", "tiny");
    Add("anext", string(10000000, 'a'));
    Add("anext2", string(10000000, 'b'));
    Add("azz", "tiny");
    Test(&rnd, 100 /* num_random_access_iters */);
  }
}

TEST_F(Harness, SimpleSpecialKey) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 4, 17);
    random::SimplePhilox rnd(&philox);
    Add("\xff\xff", "v3");
    Test(&rnd);
  }
}

TEST_F(Harness, Randomized) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 5, 17);
    random::SimplePhilox rnd(&philox);
    for (int num_entries = 0; num_entries < 2000;
         num_entries += (num_entries < 50 ? 1 : 200)) {
      if ((num_entries % 10) == 0) {
        fprintf(stderr, "case %d of %d: num_entries = %d\n", (i + 1),
                int(kNumTestArgs), num_entries);
      }
      for (int e = 0; e < num_entries; e++) {
        string v;
        Add(test::RandomKey(&rnd, rnd.Skewed(4)),
            string(test::RandomString(&rnd, rnd.Skewed(5), &v)));
      }
      Test(&rnd);
    }
  }
}

static bool Between(uint64 val, uint64 low, uint64 high) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_46(mht_46_v, 848, "", "./tensorflow/core/lib/io/table_test.cc", "Between");

  bool result = (val >= low) && (val <= high);
  if (!result) {
    fprintf(stderr, "Value %llu is not in range [%llu, %llu]\n",
            static_cast<unsigned long long>(val),
            static_cast<unsigned long long>(low),
            static_cast<unsigned long long>(high));
  }
  return result;
}

class TableTest {};

TEST(TableTest, ApproximateOffsetOfPlain) {
  TableConstructor c;
  c.Add("k01", "hello");
  c.Add("k02", "hello2");
  c.Add("k03", string(10000, 'x'));
  c.Add("k04", string(200000, 'x'));
  c.Add("k05", string(300000, 'x'));
  c.Add("k06", "hello3");
  c.Add("k07", string(100000, 'x'));
  std::vector<string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kNoCompression;
  c.Finish(options, &keys, &kvmap);

  ASSERT_TRUE(Between(c.ApproximateOffsetOf("abc"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01a"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k02"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k03"), 10, 500));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04"), 10000, 11000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04a"), 210000, 211000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k05"), 210000, 211000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k06"), 510000, 511000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k07"), 510000, 511000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("xyz"), 610000, 612000));
}

static bool SnappyCompressionSupported() {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_testDTcc mht_47(mht_47_v, 893, "", "./tensorflow/core/lib/io/table_test.cc", "SnappyCompressionSupported");

  string out;
  StringPiece in = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  return port::Snappy_Compress(in.data(), in.size(), &out);
}

TEST(TableTest, ApproximateOffsetOfCompressed) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  TableConstructor c;
  string tmp;
  c.Add("k01", "hello");
  c.Add("k02", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  c.Add("k03", "hello3");
  c.Add("k04", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  std::vector<string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kSnappyCompression;
  c.Finish(options, &keys, &kvmap);
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("abc"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k02"), 10, 100));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k03"), 2000, 4000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04"), 2000, 4000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("xyz"), 4000, 7000));
}

TEST(TableTest, SeekToFirstKeyDoesNotReadTooMuch) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  string tmp;
  TableConstructor c;
  c.Add("k01", "firstvalue");
  c.Add("k03", test::CompressibleString(&rnd, 0.25, 1000000, &tmp));
  c.Add("k04", "abc");
  std::vector<string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kNoCompression;
  c.Finish(options, &keys, &kvmap);

  Iterator* iter = c.NewIterator();
  iter->Seek("k01");
  delete iter;
  // Make sure we don't read the big second block when just trying to
  // retrieve the data in the first key
  EXPECT_LT(c.BytesRead(), 200);
}

}  // namespace table
}  // namespace tensorflow
