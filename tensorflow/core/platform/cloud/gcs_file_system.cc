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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc() {
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

#include "tensorflow/core/platform/cloud/gcs_file_system.h"

#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/strcat.h"
#ifdef _WIN32
#include <io.h>  // for _mktemp
#endif
#include "absl/base/macros.h"
#include "json/json.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#include "tensorflow/core/platform/cloud/ram_file_block_cache.h"
#include "tensorflow/core/platform/cloud/time_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/retrying_utils.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#ifdef _WIN32
#ifdef DeleteFile
#undef DeleteFile
#endif
#endif

namespace tensorflow {
namespace {

constexpr char kGcsUriBase[] = "https://www.googleapis.com/storage/v1/";
constexpr char kGcsUploadUriBase[] =
    "https://www.googleapis.com/upload/storage/v1/";
constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr char kBucketMetadataLocationKey[] = "location";
constexpr size_t kReadAppendableFileBufferSize = 1024 * 1024;  // In bytes.
constexpr int kGetChildrenDefaultPageSize = 1000;
// The HTTP response code "308 Resume Incomplete".
constexpr uint64 HTTP_CODE_RESUME_INCOMPLETE = 308;
// The HTTP response code "412 Precondition Failed".
constexpr uint64 HTTP_CODE_PRECONDITION_FAILED = 412;
// The environment variable that overrides the size of the readahead buffer.
ABSL_DEPRECATED("Use GCS_READ_CACHE_BLOCK_SIZE_MB instead.")
constexpr char kReadaheadBufferSize[] = "GCS_READAHEAD_BUFFER_SIZE_BYTES";
// The environment variable that overrides the maximum age of entries in the
// Stat cache. A value of 0 means nothing is cached.
constexpr char kStatCacheMaxAge[] = "GCS_STAT_CACHE_MAX_AGE";
constexpr uint64 kStatCacheDefaultMaxAge = 5;
// The environment variable that overrides the maximum number of entries in the
// Stat cache.
constexpr char kStatCacheMaxEntries[] = "GCS_STAT_CACHE_MAX_ENTRIES";
constexpr size_t kStatCacheDefaultMaxEntries = 1024;
// The environment variable that overrides the maximum age of entries in the
// GetMatchingPaths cache. A value of 0 (the default) means nothing is cached.
constexpr char kMatchingPathsCacheMaxAge[] = "GCS_MATCHING_PATHS_CACHE_MAX_AGE";
constexpr uint64 kMatchingPathsCacheDefaultMaxAge = 0;
// The environment variable that overrides the maximum number of entries in the
// GetMatchingPaths cache.
constexpr char kMatchingPathsCacheMaxEntries[] =
    "GCS_MATCHING_PATHS_CACHE_MAX_ENTRIES";
constexpr size_t kMatchingPathsCacheDefaultMaxEntries = 1024;
// Number of bucket locations cached, most workloads wont touch more than one
// bucket so this limit is set fairly low
constexpr size_t kBucketLocationCacheMaxEntries = 10;
// ExpiringLRUCache doesnt support any "cache forever" option
constexpr size_t kCacheNeverExpire = std::numeric_limits<uint64>::max();
// The file statistics returned by Stat() for directories.
const FileStatistics DIRECTORY_STAT(0, 0, true);
// Some environments exhibit unreliable DNS resolution. Set this environment
// variable to a positive integer describing the frequency used to refresh the
// userspace DNS cache.
constexpr char kResolveCacheSecs[] = "GCS_RESOLVE_REFRESH_SECS";
// The environment variable to configure the http request's connection timeout.
constexpr char kRequestConnectionTimeout[] =
    "GCS_REQUEST_CONNECTION_TIMEOUT_SECS";
// The environment variable to configure the http request's idle timeout.
constexpr char kRequestIdleTimeout[] = "GCS_REQUEST_IDLE_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// metadata requests.
constexpr char kMetadataRequestTimeout[] = "GCS_METADATA_REQUEST_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// block reads requests.
constexpr char kReadRequestTimeout[] = "GCS_READ_REQUEST_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// upload requests.
constexpr char kWriteRequestTimeout[] = "GCS_WRITE_REQUEST_TIMEOUT_SECS";
// The environment variable to configure an additional header to send with
// all requests to GCS (format HEADERNAME:HEADERCONTENT)
constexpr char kAdditionalRequestHeader[] = "GCS_ADDITIONAL_REQUEST_HEADER";
// The environment variable to configure the throttle (format: <int64_t>)
constexpr char kThrottleRate[] = "GCS_THROTTLE_TOKEN_RATE";
// The environment variable to configure the token bucket size (format:
// <int64_t>)
constexpr char kThrottleBucket[] = "GCS_THROTTLE_BUCKET_SIZE";
// The environment variable that controls the number of tokens per request.
// (format: <int64_t>)
constexpr char kTokensPerRequest[] = "GCS_TOKENS_PER_REQUEST";
// The environment variable to configure the initial tokens (format: <int64_t>)
constexpr char kInitialTokens[] = "GCS_INITIAL_TOKENS";

// The environment variable to customize which GCS bucket locations are allowed,
// if the list is empty defaults to using the region of the zone (format, comma
// delimited list). Requires 'storage.buckets.get' permission.
constexpr char kAllowedBucketLocations[] = "GCS_ALLOWED_BUCKET_LOCATIONS";
// When this value is passed as an allowed location detects the zone tensorflow
// is running in and restricts to buckets in that region.
constexpr char kDetectZoneSentinelValue[] = "auto";

// How to upload new data when Flush() is called multiple times.
// By default the entire file is reuploaded.
constexpr char kAppendMode[] = "GCS_APPEND_MODE";
// If GCS_APPEND_MODE=compose then instead the new data is uploaded to a
// temporary object and composed with the original object. This is disabled by
// default as the multiple API calls required add a risk of stranding temporary
// objects.
constexpr char kComposeAppend[] = "compose";

Status GetTmpFilename(string* filename) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_0(mht_0_v, 318, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetTmpFilename");

  *filename = io::GetTempFilename("");
  return Status::OK();
}

/// Appends a trailing slash if the name doesn't already have one.
string MaybeAppendSlash(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_1(mht_1_v, 328, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "MaybeAppendSlash");

  if (name.empty()) {
    return "/";
  }
  if (name.back() != '/') {
    return strings::StrCat(name, "/");
  }
  return name;
}

// io::JoinPath() doesn't work in cases when we want an empty subpath
// to result in an appended slash in order for directory markers
// to be processed correctly: "gs://a/b" + "" should give "gs://a/b/".
string JoinGcsPath(const string& path, const string& subpath) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("path: \"" + path + "\"");
   mht_2_v.push_back("subpath: \"" + subpath + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_2(mht_2_v, 346, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "JoinGcsPath");

  return strings::StrCat(MaybeAppendSlash(path), subpath);
}

/// \brief Returns the given paths appending all their subfolders.
///
/// For every path X in the list, every subfolder in X is added to the
/// resulting list.
/// For example:
///  - for 'a/b/c/d' it will append 'a', 'a/b' and 'a/b/c'
///  - for 'a/b/c/' it will append 'a', 'a/b' and 'a/b/c'
///  - for 'a//b/c/' it will append 'a', 'a//b' and 'a//b/c'
///  - for '/a/b/c/' it will append '/a', '/a/b' and '/a/b/c'
std::set<string> AddAllSubpaths(const std::vector<string>& paths) {
  std::set<string> result;
  result.insert(paths.begin(), paths.end());
  for (const string& path : paths) {
    StringPiece subpath = io::Dirname(path);
    // If `path` starts with `/`, `subpath` will be `/` and then we get into an
    // infinite loop. Same behavior happens if there is a `//` pattern in
    // `path`, so we check for that and leave the loop quicker.
    while (!(subpath.empty() || subpath == "/")) {
      result.emplace(string(subpath));
      subpath = io::Dirname(subpath);
    }
  }
  return result;
}

Status ParseJson(StringPiece json, Json::Value* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_3(mht_3_v, 378, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "ParseJson");

  Json::Reader reader;
  if (!reader.parse(json.data(), json.data() + json.size(), *result)) {
    return errors::Internal("Couldn't parse JSON response from GCS.");
  }
  return Status::OK();
}

Status ParseJson(const std::vector<char>& json, Json::Value* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_4(mht_4_v, 389, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "ParseJson");

  return ParseJson(StringPiece{json.data(), json.size()}, result);
}

/// Reads a JSON value with the given name from a parent JSON value.
Status GetValue(const Json::Value& parent, const char* name,
                Json::Value* result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_5(mht_5_v, 399, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetValue");

  *result = parent.get(name, Json::Value::null);
  if (result->isNull()) {
    return errors::Internal("The field '", name,
                            "' was expected in the JSON response.");
  }
  return Status::OK();
}

/// Reads a string JSON value with the given name from a parent JSON value.
Status GetStringValue(const Json::Value& parent, const char* name,
                      string* result) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_6(mht_6_v, 414, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetStringValue");

  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isString()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a string.");
  }
  *result = result_value.asString();
  return Status::OK();
}

/// Reads a long JSON value with the given name from a parent JSON value.
Status GetInt64Value(const Json::Value& parent, const char* name,
                     int64_t* result) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_7(mht_7_v, 432, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetInt64Value");

  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (result_value.isNumeric()) {
    *result = result_value.asInt64();
    return Status::OK();
  }
  if (result_value.isString() &&
      strings::safe_strto64(result_value.asCString(), result)) {
    return Status::OK();
  }
  return errors::Internal(
      "The field '", name,
      "' in the JSON response was expected to be a number.");
}

/// Reads a boolean JSON value with the given name from a parent JSON value.
Status GetBoolValue(const Json::Value& parent, const char* name, bool* result) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_8(mht_8_v, 453, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetBoolValue");

  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isBool()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a boolean.");
  }
  *result = result_value.asBool();
  return Status::OK();
}

/// A GCS-based implementation of a random access file with an LRU block cache.
class GcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn =
      std::function<Status(const string& filename, uint64 offset, size_t n,
                           StringPiece* result, char* scratch)>;

  GcsRandomAccessFile(const string& filename, ReadFn read_fn)
      : filename_(filename), read_fn_(std::move(read_fn)) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_9(mht_9_v, 477, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsRandomAccessFile");
}

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_10(mht_10_v, 482, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  /// The implementation of reads with an LRU block cache. Thread safe.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_11(mht_11_v, 493, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Read");

    return read_fn_(filename_, offset, n, result, scratch);
  }

 private:
  /// The filename of this file.
  const string filename_;
  /// The implementation of the read operation (provided by the GCSFileSystem).
  const ReadFn read_fn_;
};

/// A GCS-based implementation of a random access file with a read buffer.
class BufferedGcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn =
      std::function<Status(const string& filename, uint64 offset, size_t n,
                           StringPiece* result, char* scratch)>;

  // Initialize the reader. Provided read_fn should be thread safe.
  BufferedGcsRandomAccessFile(const string& filename, uint64 buffer_size,
                              ReadFn read_fn)
      : filename_(filename),
        read_fn_(std::move(read_fn)),
        buffer_size_(buffer_size),
        buffer_start_(0),
        buffer_end_is_past_eof_(false) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_12(mht_12_v, 522, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "BufferedGcsRandomAccessFile");
}

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_13(mht_13_v, 527, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  /// The implementation of reads with an read buffer. Thread safe.
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_14(mht_14_v, 540, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Read");

    if (n > buffer_size_) {
      return read_fn_(filename_, offset, n, result, scratch);
    }
    {
      mutex_lock l(buffer_mutex_);
      size_t buffer_end = buffer_start_ + buffer_.size();
      size_t copy_size = 0;
      if (offset < buffer_end && offset >= buffer_start_) {
        copy_size = std::min(n, static_cast<size_t>(buffer_end - offset));
        memcpy(scratch, buffer_.data() + (offset - buffer_start_), copy_size);
        *result = StringPiece(scratch, copy_size);
      }
      bool consumed_buffer_to_eof =
          offset + copy_size >= buffer_end && buffer_end_is_past_eof_;
      if (copy_size < n && !consumed_buffer_to_eof) {
        Status status = FillBuffer(offset + copy_size);
        if (!status.ok() && status.code() != errors::Code::OUT_OF_RANGE) {
          // Empty the buffer to avoid caching bad reads.
          buffer_.resize(0);
          return status;
        }
        size_t remaining_copy = std::min(n - copy_size, buffer_.size());
        memcpy(scratch + copy_size, buffer_.data(), remaining_copy);
        copy_size += remaining_copy;
        *result = StringPiece(scratch, copy_size);
      }
      if (copy_size < n) {
        // Forget the end-of-file flag to allow for clients that poll on the
        // same file.
        buffer_end_is_past_eof_ = false;
        return errors::OutOfRange("EOF reached. Requested to read ", n,
                                  " bytes from ", offset, ".");
      }
    }
    return Status::OK();
  }

 private:
  Status FillBuffer(uint64 start) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(buffer_mutex_) {
    buffer_start_ = start;
    buffer_.resize(buffer_size_);
    StringPiece str_piece;
    Status status = read_fn_(filename_, buffer_start_, buffer_size_, &str_piece,
                             &(buffer_[0]));
    buffer_end_is_past_eof_ = status.code() == errors::Code::OUT_OF_RANGE;
    buffer_.resize(str_piece.size());
    return status;
  }

  // The filename of this file.
  const string filename_;

  // The implementation of the read operation (provided by the GCSFileSystem).
  const ReadFn read_fn_;

  // Size of buffer that we read from GCS each time we send a request.
  const uint64 buffer_size_;

  // Mutex for buffering operations that can be accessed from multiple threads.
  // The following members are mutable in order to provide a const Read.
  mutable mutex buffer_mutex_;

  // Offset of buffer from start of the file.
  mutable uint64 buffer_start_ TF_GUARDED_BY(buffer_mutex_);

  mutable bool buffer_end_is_past_eof_ TF_GUARDED_BY(buffer_mutex_);

  mutable string buffer_ TF_GUARDED_BY(buffer_mutex_);
};

// Function object declaration with params needed to create upload sessions.
typedef std::function<Status(
    uint64 start_offset, const std::string& object_to_upload,
    const std::string& bucket, uint64 file_size, const std::string& gcs_path,
    UploadSessionHandle* session_handle)>
    SessionCreator;

// Function object declaration with params needed to upload objects.
typedef std::function<Status(const std::string& session_uri,
                             uint64 start_offset, uint64 already_uploaded,
                             const std::string& tmp_content_filename,
                             uint64 file_size, const std::string& file_path)>
    ObjectUploader;

// Function object declaration with params needed to poll upload status.
typedef std::function<Status(const string& session_uri, uint64 file_size,
                             const std::string& gcs_path, bool* completed,
                             uint64* uploaded)>
    StatusPoller;

// Function object declaration with params needed to poll upload status.
typedef std::function<Status(const string& fname, const string& bucket,
                             const string& object, int64_t* generation)>
    GenerationGetter;

/// \brief GCS-based implementation of a writeable file.
///
/// Since GCS objects are immutable, this implementation writes to a local
/// tmp file and copies it to GCS on flush/close.
class GcsWritableFile : public WritableFile {
 public:
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config, bool compose_append,
                  SessionCreator session_creator,
                  ObjectUploader object_uploader, StatusPoller status_poller,
                  GenerationGetter generation_getter)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config),
        compose_append_(compose_append),
        start_offset_(0),
        session_creator_(std::move(session_creator)),
        object_uploader_(std::move(object_uploader)),
        status_poller_(std::move(status_poller)),
        generation_getter_(std::move(generation_getter)) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("bucket: \"" + bucket + "\"");
   mht_15_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_15(mht_15_v, 668, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsWritableFile");

    // TODO: to make it safer, outfile_ should be constructed from an FD
    VLOG(3) << "GcsWritableFile: " << GetGcsPath();
    if (GetTmpFilename(&tmp_content_filename_).ok()) {
      outfile_.open(tmp_content_filename_,
                    std::ofstream::binary | std::ofstream::app);
    }
  }

  /// \brief Constructs the writable file in append mode.
  ///
  /// tmp_content_filename should contain a path of an existing temporary file
  /// with the content to be appended. The class takes ownership of the
  /// specified tmp file and deletes it on close.
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem, const string& tmp_content_filename,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config, bool compose_append,
                  SessionCreator session_creator,
                  ObjectUploader object_uploader, StatusPoller status_poller,
                  GenerationGetter generation_getter)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config),
        compose_append_(compose_append),
        start_offset_(0),
        session_creator_(std::move(session_creator)),
        object_uploader_(std::move(object_uploader)),
        status_poller_(std::move(status_poller)),
        generation_getter_(std::move(generation_getter)) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("bucket: \"" + bucket + "\"");
   mht_16_v.push_back("object: \"" + object + "\"");
   mht_16_v.push_back("tmp_content_filename: \"" + tmp_content_filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_16(mht_16_v, 708, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsWritableFile");

    VLOG(3) << "GcsWritableFile: " << GetGcsPath() << "with existing file "
            << tmp_content_filename;
    tmp_content_filename_ = tmp_content_filename;
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }

  ~GcsWritableFile() override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_17(mht_17_v, 719, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "~GcsWritableFile");

    Close().IgnoreError();
    std::remove(tmp_content_filename_.c_str());
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_18(mht_18_v, 727, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Append");

    TF_RETURN_IF_ERROR(CheckWritable());
    VLOG(3) << "Append: " << GetGcsPath() << " size " << data.length();
    sync_needed_ = true;
    outfile_ << data;
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return Status::OK();
  }

  Status Close() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_19(mht_19_v, 742, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Close");

    VLOG(3) << "Close:" << GetGcsPath();
    if (outfile_.is_open()) {
      Status sync_status = Sync();
      if (sync_status.ok()) {
        outfile_.close();
      }
      return sync_status;
    }
    return Status::OK();
  }

  Status Flush() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_20(mht_20_v, 757, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Flush");

    VLOG(3) << "Flush:" << GetGcsPath();
    return Sync();
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_21(mht_21_v, 765, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Name");

    return errors::Unimplemented("GCSWritableFile does not support Name()");
  }

  Status Sync() override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_22(mht_22_v, 772, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Sync");

    VLOG(3) << "Sync started:" << GetGcsPath();
    TF_RETURN_IF_ERROR(CheckWritable());
    if (!sync_needed_) {
      return Status::OK();
    }
    Status status = SyncImpl();
    VLOG(3) << "Sync finished " << GetGcsPath();
    if (status.ok()) {
      sync_needed_ = false;
    }
    return status;
  }

  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_23(mht_23_v, 789, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "Tell");

    *position = outfile_.tellp();
    if (*position == -1) {
      return errors::Internal("tellp on the internal temporary file failed");
    }
    return Status::OK();
  }

 private:
  /// Copies the current version of the file to GCS.
  ///
  /// This SyncImpl() uploads the object to GCS.
  /// In case of a failure, it resumes failed uploads as recommended by the GCS
  /// resumable API documentation. When the whole upload needs to be
  /// restarted, Sync() returns UNAVAILABLE and relies on RetryingFileSystem.
  Status SyncImpl() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_24(mht_24_v, 807, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "SyncImpl");

    outfile_.flush();
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not write to the internal temporary file.");
    }
    UploadSessionHandle session_handle;
    uint64 start_offset = 0;
    string object_to_upload = object_;
    bool should_compose = false;
    if (compose_append_) {
      start_offset = start_offset_;
      // Only compose if the object has already been uploaded to GCS
      should_compose = start_offset > 0;
      if (should_compose) {
        object_to_upload =
            strings::StrCat(io::Dirname(object_), "/.tmpcompose/",
                            io::Basename(object_), ".", start_offset_);
      }
    }
    TF_RETURN_IF_ERROR(CreateNewUploadSession(start_offset, object_to_upload,
                                              &session_handle));
    uint64 already_uploaded = 0;
    bool first_attempt = true;
    const Status upload_status = RetryingUtils::CallWithRetries(
        [&first_attempt, &already_uploaded, &session_handle, &start_offset,
         this]() {
          if (session_handle.resumable && !first_attempt) {
            bool completed;
            TF_RETURN_IF_ERROR(RequestUploadSessionStatus(
                session_handle.session_uri, &completed, &already_uploaded));
            LOG(INFO) << "### RequestUploadSessionStatus: completed = "
                      << completed
                      << ", already_uploaded = " << already_uploaded
                      << ", file = " << GetGcsPath();
            if (completed) {
              // Erase the file from the file cache on every successful write.
              file_cache_erase_();
              // It's unclear why UploadToSession didn't return OK in the
              // previous attempt, but GCS reports that the file is fully
              // uploaded, so succeed.
              return Status::OK();
            }
          }
          first_attempt = false;
          return UploadToSession(session_handle.session_uri, start_offset,
                                 already_uploaded);
        },
        retry_config_);
    if (upload_status.code() == errors::Code::NOT_FOUND) {
      // GCS docs recommend retrying the whole upload. We're relying on the
      // RetryingFileSystem to retry the Sync() call.
      return errors::Unavailable(strings::StrCat(
          "Upload to gs://", bucket_, "/", object_,
          " failed, caused by: ", upload_status.error_message()));
    }
    if (upload_status.ok()) {
      if (should_compose) {
        TF_RETURN_IF_ERROR(AppendObject(object_to_upload));
      }
      TF_RETURN_IF_ERROR(GetCurrentFileSize(&start_offset_));
    }
    return upload_status;
  }

  Status CheckWritable() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_25(mht_25_v, 875, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "CheckWritable");

    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    return Status::OK();
  }

  Status GetCurrentFileSize(uint64* size) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_26(mht_26_v, 886, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetCurrentFileSize");

    const auto tellp = outfile_.tellp();
    if (tellp == static_cast<std::streampos>(-1)) {
      return errors::Internal(
          "Could not get the size of the internal temporary file.");
    }
    *size = tellp;
    return Status::OK();
  }

  /// Initiates a new resumable upload session.
  Status CreateNewUploadSession(uint64 start_offset,
                                std::string object_to_upload,
                                UploadSessionHandle* session_handle) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("object_to_upload: \"" + object_to_upload + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_27(mht_27_v, 903, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "CreateNewUploadSession");

    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    return session_creator_(start_offset, object_to_upload, bucket_, file_size,
                            GetGcsPath(), session_handle);
  }

  /// Appends the data of append_object to the original object and deletes
  /// append_object.
  Status AppendObject(string append_object) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("append_object: \"" + append_object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_28(mht_28_v, 916, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "AppendObject");

    const string append_object_path = GetGcsPathWithObject(append_object);
    VLOG(3) << "AppendObject: " << append_object_path << " to " << GetGcsPath();

    int64_t generation = 0;
    TF_RETURN_IF_ERROR(
        generation_getter_(GetGcsPath(), bucket_, object_, &generation));

    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&append_object, &generation, this]() {
          std::unique_ptr<HttpRequest> request;
          TF_RETURN_IF_ERROR(filesystem_->CreateHttpRequest(&request));

          request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket_, "/o/",
                                          request->EscapeString(object_),
                                          "/compose"));

          const string request_body = strings::StrCat(
              "{'sourceObjects': [{'name': '", object_,
              "','objectPrecondition':{'ifGenerationMatch':", generation,
              "}},{'name': '", append_object, "'}]}");
          request->SetTimeouts(timeouts_->connect, timeouts_->idle,
                               timeouts_->metadata);
          request->AddHeader("content-type", "application/json");
          request->SetPostFromBuffer(request_body.c_str(), request_body.size());
          TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(),
                                          " when composing to ", GetGcsPath());
          return Status::OK();
        },
        retry_config_));

    return RetryingUtils::DeleteWithRetries(
        [&append_object_path, this]() {
          return filesystem_->DeleteFile(append_object_path, nullptr);
        },
        retry_config_);
  }

  /// \brief Requests status of a previously initiated upload session.
  ///
  /// If the upload has already succeeded, sets 'completed' to true.
  /// Otherwise sets 'completed' to false and 'uploaded' to the currently
  /// uploaded size in bytes.
  Status RequestUploadSessionStatus(const string& session_uri, bool* completed,
                                    uint64* uploaded) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("session_uri: \"" + session_uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_29(mht_29_v, 964, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "RequestUploadSessionStatus");

    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    return status_poller_(session_uri, file_size, GetGcsPath(), completed,
                          uploaded);
  }

  /// Uploads data to object.
  Status UploadToSession(const string& session_uri, uint64 start_offset,
                         uint64 already_uploaded) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("session_uri: \"" + session_uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_30(mht_30_v, 977, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "UploadToSession");

    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    Status status =
        object_uploader_(session_uri, start_offset, already_uploaded,
                         tmp_content_filename_, file_size, GetGcsPath());
    if (status.ok()) {
      // Erase the file from the file cache on every successful write.
      // Note: Only local cache, this does nothing on distributed cache. The
      // distributed cache clears the cache as it is needed.
      file_cache_erase_();
    }

    return status;
  }

  string GetGcsPathWithObject(string object) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_31(mht_31_v, 997, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetGcsPathWithObject");

    return strings::StrCat("gs://", bucket_, "/", object);
  }
  string GetGcsPath() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_32(mht_32_v, 1003, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GetGcsPath");
 return GetGcsPathWithObject(object_); }

  string bucket_;
  string object_;
  GcsFileSystem* const filesystem_;  // Not owned.
  string tmp_content_filename_;
  std::ofstream outfile_;
  GcsFileSystem::TimeoutConfig* timeouts_;
  std::function<void()> file_cache_erase_;
  bool sync_needed_;  // whether there is buffered data that needs to be synced
  RetryConfig retry_config_;
  bool compose_append_;
  uint64 start_offset_;
  // Callbacks to the file system used to upload object into GCS.
  const SessionCreator session_creator_;
  const ObjectUploader object_uploader_;
  const StatusPoller status_poller_;
  const GenerationGetter generation_getter_;
};

class GcsReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  GcsReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_33(mht_33_v, 1029, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsReadOnlyMemoryRegion");
}
  const void* data() override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_34(mht_34_v, 1033, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "data");
 return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_35(mht_35_v, 1037, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "length");
 return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

bool StringPieceIdentity(StringPiece str, StringPiece* value) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_36(mht_36_v, 1047, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "StringPieceIdentity");

  *value = str;
  return true;
}

/// \brief Utility function to split a comma delimited list of strings to an
/// unordered set, lowercasing all values.
bool SplitByCommaToLowercaseSet(StringPiece list,
                                std::unordered_set<string>* set) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_37(mht_37_v, 1058, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "SplitByCommaToLowercaseSet");

  std::vector<string> vector = absl::StrSplit(absl::AsciiStrToLower(list), ',');
  *set = std::unordered_set<string>(vector.begin(), vector.end());
  return true;
}

// \brief Convert Compute Engine zone to region
string ZoneToRegion(string* zone) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_38(mht_38_v, 1068, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "ZoneToRegion");

  return zone->substr(0, zone->find_last_of('-'));
}

}  // namespace

GcsFileSystem::GcsFileSystem(bool make_default_cache) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_39(mht_39_v, 1077, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GcsFileSystem");

  uint64 value;
  block_size_ = kDefaultBlockSize;
  size_t max_bytes = kDefaultMaxCacheSize;

  uint64 max_staleness = kDefaultMaxStaleness;

  http_request_factory_ = std::make_shared<CurlHttpRequest::Factory>();
  compute_engine_metadata_client_ =
      std::make_shared<ComputeEngineMetadataClient>(http_request_factory_);
  auth_provider_ = std::unique_ptr<AuthProvider>(
      new GoogleAuthProvider(compute_engine_metadata_client_));
  zone_provider_ = std::unique_ptr<ZoneProvider>(
      new ComputeEngineZoneProvider(compute_engine_metadata_client_));

  // Apply the sys env override for the readahead buffer size if it's provided.
  if (GetEnvVar(kReadaheadBufferSize, strings::safe_strtou64, &value)) {
    block_size_ = value;
  }

  // Apply the overrides for the block size (MB), max bytes (MB), and max
  // staleness (seconds) if provided.
  if (GetEnvVar(kBlockSize, strings::safe_strtou64, &value)) {
    block_size_ = value * 1024 * 1024;
  }

  if (GetEnvVar(kMaxCacheSize, strings::safe_strtou64, &value)) {
    max_bytes = value * 1024 * 1024;
  }

  if (GetEnvVar(kMaxStaleness, strings::safe_strtou64, &value)) {
    max_staleness = value;
  }
  if (!make_default_cache) {
    max_bytes = 0;
  }
  VLOG(1) << "GCS cache max size = " << max_bytes << " ; "
          << "block size = " << block_size_ << " ; "
          << "max staleness = " << max_staleness;
  file_block_cache_ = MakeFileBlockCache(block_size_, max_bytes, max_staleness);
  // Apply overrides for the stat cache max age and max entries, if provided.
  uint64 stat_cache_max_age = kStatCacheDefaultMaxAge;
  size_t stat_cache_max_entries = kStatCacheDefaultMaxEntries;
  if (GetEnvVar(kStatCacheMaxAge, strings::safe_strtou64, &value)) {
    stat_cache_max_age = value;
  }
  if (GetEnvVar(kStatCacheMaxEntries, strings::safe_strtou64, &value)) {
    stat_cache_max_entries = value;
  }
  stat_cache_.reset(new ExpiringLRUCache<GcsFileStat>(stat_cache_max_age,
                                                      stat_cache_max_entries));
  // Apply overrides for the matching paths cache max age and max entries, if
  // provided.
  uint64 matching_paths_cache_max_age = kMatchingPathsCacheDefaultMaxAge;
  size_t matching_paths_cache_max_entries =
      kMatchingPathsCacheDefaultMaxEntries;
  if (GetEnvVar(kMatchingPathsCacheMaxAge, strings::safe_strtou64, &value)) {
    matching_paths_cache_max_age = value;
  }
  if (GetEnvVar(kMatchingPathsCacheMaxEntries, strings::safe_strtou64,
                &value)) {
    matching_paths_cache_max_entries = value;
  }
  matching_paths_cache_.reset(new ExpiringLRUCache<std::vector<string>>(
      matching_paths_cache_max_age, matching_paths_cache_max_entries));

  bucket_location_cache_.reset(new ExpiringLRUCache<string>(
      kCacheNeverExpire, kBucketLocationCacheMaxEntries));

  int64_t resolve_frequency_secs;
  if (GetEnvVar(kResolveCacheSecs, strings::safe_strto64,
                &resolve_frequency_secs)) {
    dns_cache_.reset(new GcsDnsCache(resolve_frequency_secs));
    VLOG(1) << "GCS DNS cache is enabled.  " << kResolveCacheSecs << " = "
            << resolve_frequency_secs;
  } else {
    VLOG(1) << "GCS DNS cache is disabled, because " << kResolveCacheSecs
            << " = 0 (or is not set)";
  }

  // Get the additional header
  StringPiece add_header_contents;
  if (GetEnvVar(kAdditionalRequestHeader, StringPieceIdentity,
                &add_header_contents)) {
    size_t split = add_header_contents.find(':', 0);

    if (split != StringPiece::npos) {
      StringPiece header_name = add_header_contents.substr(0, split);
      StringPiece header_value = add_header_contents.substr(split + 1);

      if (!header_name.empty() && !header_value.empty()) {
        additional_header_.reset(new std::pair<const string, const string>(
            string(header_name), string(header_value)));

        VLOG(1) << "GCS additional header ENABLED. "
                << "Name: " << additional_header_->first << ", "
                << "Value: " << additional_header_->second;
      } else {
        LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                   << add_header_contents;
      }
    } else {
      LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                 << add_header_contents;
    }
  } else {
    VLOG(1) << "GCS additional header DISABLED. No environment variable set.";
  }

  // Apply the overrides for request timeouts
  uint32 timeout_value;
  if (GetEnvVar(kRequestConnectionTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.connect = timeout_value;
  }
  if (GetEnvVar(kRequestIdleTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.idle = timeout_value;
  }
  if (GetEnvVar(kMetadataRequestTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.metadata = timeout_value;
  }
  if (GetEnvVar(kReadRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.read = timeout_value;
  }
  if (GetEnvVar(kWriteRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.write = timeout_value;
  }

  int64_t token_value;
  if (GetEnvVar(kThrottleRate, strings::safe_strto64, &token_value)) {
    GcsThrottleConfig config;
    config.enabled = true;
    config.token_rate = token_value;

    if (GetEnvVar(kThrottleBucket, strings::safe_strto64, &token_value)) {
      config.bucket_size = token_value;
    }

    if (GetEnvVar(kTokensPerRequest, strings::safe_strto64, &token_value)) {
      config.tokens_per_request = token_value;
    }

    if (GetEnvVar(kInitialTokens, strings::safe_strto64, &token_value)) {
      config.initial_tokens = token_value;
    }
    throttle_.SetConfig(config);
  }

  GetEnvVar(kAllowedBucketLocations, SplitByCommaToLowercaseSet,
            &allowed_locations_);

  StringPiece append_mode;
  GetEnvVar(kAppendMode, StringPieceIdentity, &append_mode);
  if (append_mode == kComposeAppend) {
    compose_append_ = true;
  } else {
    compose_append_ = false;
  }
}

GcsFileSystem::GcsFileSystem(
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory,
    std::unique_ptr<ZoneProvider> zone_provider, size_t block_size,
    size_t max_bytes, uint64 max_staleness, uint64 stat_cache_max_age,
    size_t stat_cache_max_entries, uint64 matching_paths_cache_max_age,
    size_t matching_paths_cache_max_entries, RetryConfig retry_config,
    TimeoutConfig timeouts, const std::unordered_set<string>& allowed_locations,
    std::pair<const string, const string>* additional_header,
    bool compose_append)
    : timeouts_(timeouts),
      retry_config_(retry_config),
      auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)),
      zone_provider_(std::move(zone_provider)),
      block_size_(block_size),
      file_block_cache_(
          MakeFileBlockCache(block_size, max_bytes, max_staleness)),
      stat_cache_(new StatCache(stat_cache_max_age, stat_cache_max_entries)),
      matching_paths_cache_(new MatchingPathsCache(
          matching_paths_cache_max_age, matching_paths_cache_max_entries)),
      bucket_location_cache_(new BucketLocationCache(
          kCacheNeverExpire, kBucketLocationCacheMaxEntries)),
      allowed_locations_(allowed_locations),
      compose_append_(compose_append),
      additional_header_(additional_header) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_40(mht_40_v, 1266, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GcsFileSystem");
}

Status GcsFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_41(mht_41_v, 1274, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::NewRandomAccessFile");

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  TF_RETURN_IF_ERROR(CheckBucketLocationConstraint(bucket));
  if (cache_enabled_) {
    result->reset(new GcsRandomAccessFile(fname, [this, bucket, object](
                                                     const string& fname,
                                                     uint64 offset, size_t n,
                                                     StringPiece* result,
                                                     char* scratch) {
      tf_shared_lock l(block_cache_lock_);
      GcsFileStat stat;
      TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
          fname, &stat,
          [this, bucket, object](const string& fname, GcsFileStat* stat) {
            return UncachedStatForObject(fname, bucket, object, stat);
          }));
      if (!file_block_cache_->ValidateAndUpdateFileSignature(
              fname, stat.generation_number)) {
        VLOG(1)
            << "File signature has been changed. Refreshing the cache. Path: "
            << fname;
      }
      *result = StringPiece();
      size_t bytes_transferred;
      TF_RETURN_IF_ERROR(file_block_cache_->Read(fname, offset, n, scratch,
                                                 &bytes_transferred));
      *result = StringPiece(scratch, bytes_transferred);
      if (bytes_transferred < n) {
        return errors::OutOfRange("EOF reached, ", result->size(),
                                  " bytes were read out of ", n,
                                  " bytes requested.");
      }
      return Status::OK();
    }));
  } else {
    result->reset(new BufferedGcsRandomAccessFile(
        fname, block_size_,
        [this, bucket, object](const string& fname, uint64 offset, size_t n,
                               StringPiece* result, char* scratch) {
          *result = StringPiece();
          size_t bytes_transferred;
          TF_RETURN_IF_ERROR(
              LoadBufferFromGCS(fname, offset, n, scratch, &bytes_transferred));
          *result = StringPiece(scratch, bytes_transferred);
          if (bytes_transferred < n) {
            return errors::OutOfRange("EOF reached, ", result->size(),
                                      " bytes were read out of ", n,
                                      " bytes requested.");
          }
          return Status::OK();
        }));
  }
  return Status::OK();
}

void GcsFileSystem::ResetFileBlockCache(size_t block_size_bytes,
                                        size_t max_bytes,
                                        uint64 max_staleness_secs) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_42(mht_42_v, 1335, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::ResetFileBlockCache");

  mutex_lock l(block_cache_lock_);
  file_block_cache_ =
      MakeFileBlockCache(block_size_bytes, max_bytes, max_staleness_secs);
  if (stats_ != nullptr) {
    stats_->Configure(this, &throttle_, file_block_cache_.get());
  }
}

// A helper function to build a FileBlockCache for GcsFileSystem.
std::unique_ptr<FileBlockCache> GcsFileSystem::MakeFileBlockCache(
    size_t block_size, size_t max_bytes, uint64 max_staleness) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_43(mht_43_v, 1349, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::MakeFileBlockCache");

  std::unique_ptr<FileBlockCache> file_block_cache(new RamFileBlockCache(
      block_size, max_bytes, max_staleness,
      [this](const string& filename, size_t offset, size_t n, char* buffer,
             size_t* bytes_transferred) {
        return LoadBufferFromGCS(filename, offset, n, buffer,
                                 bytes_transferred);
      }));

  // Check if cache is enabled here to avoid unnecessary mutex contention.
  cache_enabled_ = file_block_cache->IsCacheEnabled();
  return file_block_cache;
}

// A helper function to actually read the data from GCS.
Status GcsFileSystem::LoadBufferFromGCS(const string& fname, size_t offset,
                                        size_t n, char* buffer,
                                        size_t* bytes_transferred) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("fname: \"" + fname + "\"");
   mht_44_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_44(mht_44_v, 1371, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::LoadBufferFromGCS");

  *bytes_transferred = 0;

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  profiler::TraceMe activity(
      [fname]() { return absl::StrCat("LoadBufferFromGCS ", fname); });

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  "when reading gs://", bucket, "/", object);

  request->SetUri(strings::StrCat("https://", kStorageHost, "/", bucket, "/",
                                  request->EscapeString(object)));
  request->SetRange(offset, offset + n - 1);
  request->SetResultBufferDirect(buffer, n);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.read);

  if (stats_ != nullptr) {
    stats_->RecordBlockLoadRequest(fname, offset);
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading gs://",
                                  bucket, "/", object);

  size_t bytes_read = request->GetResultBufferDirectBytesTransferred();
  *bytes_transferred = bytes_read;
  VLOG(1) << "Successful read of gs://" << bucket << "/" << object << " @ "
          << offset << " of size: " << bytes_read;
  activity.AppendMetadata([bytes_read]() {
    return profiler::TraceMeEncode({{"block_size", bytes_read}});
  });

  if (stats_ != nullptr) {
    stats_->RecordBlockRetrieved(fname, offset, bytes_read);
  }

  throttle_.RecordResponse(bytes_read);

  if (bytes_read < n) {
    // Check stat cache to see if we encountered an interrupted read.
    GcsFileStat stat;
    if (stat_cache_->Lookup(fname, &stat)) {
      if (offset + bytes_read < stat.base.length) {
        return errors::Internal(strings::Printf(
            "File contents are inconsistent for file: %s @ %lu.", fname.c_str(),
            offset));
      }
      VLOG(2) << "Successful integrity check for: gs://" << bucket << "/"
              << object << " @ " << offset;
    }
  }

  return Status::OK();
}

/// Initiates a new upload session.
Status GcsFileSystem::CreateNewUploadSession(
    uint64 start_offset, const std::string& object_to_upload,
    const std::string& bucket, uint64 file_size, const std::string& gcs_path,
    UploadSessionHandle* session_handle) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("object_to_upload: \"" + object_to_upload + "\"");
   mht_45_v.push_back("bucket: \"" + bucket + "\"");
   mht_45_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_45(mht_45_v, 1438, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::CreateNewUploadSession");

  std::vector<char> output_buffer;
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));

  std::string uri = strings::StrCat(
      kGcsUploadUriBase, "b/", bucket,
      "/o?uploadType=resumable&name=", request->EscapeString(object_to_upload));
  request->SetUri(uri);
  request->AddHeader("X-Upload-Content-Length",
                     absl::StrCat(file_size - start_offset));
  request->SetPostEmptyBody();
  request->SetResultBuffer(&output_buffer);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(),
                                  " when initiating an upload to ", gcs_path);
  if (session_handle != nullptr) {
    session_handle->resumable = true;
    session_handle->session_uri = request->GetResponseHeader("Location");
    if (session_handle->session_uri.empty()) {
      return errors::Internal("Unexpected response from GCS when writing to ",
                              gcs_path, ": 'Location' header not returned.");
    }
  }
  return Status::OK();
}

Status GcsFileSystem::UploadToSession(const std::string& session_uri,
                                      uint64 start_offset,
                                      uint64 already_uploaded,
                                      const std::string& tmp_content_filename,
                                      uint64 file_size,
                                      const std::string& file_path) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_46_v.push_back("tmp_content_filename: \"" + tmp_content_filename + "\"");
   mht_46_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_46(mht_46_v, 1476, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::UploadToSession");

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(session_uri);
  if (file_size > 0) {
    request->AddHeader("Content-Range",
                       strings::StrCat("bytes ", already_uploaded, "-",
                                       file_size - start_offset - 1, "/",
                                       file_size - start_offset));
  }
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.write);

  TF_RETURN_IF_ERROR(request->SetPutFromFile(tmp_content_filename,
                                             start_offset + already_uploaded));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when uploading ",
                                  file_path);
  return Status::OK();
}

Status GcsFileSystem::RequestUploadSessionStatus(const string& session_uri,
                                                 uint64 file_size,
                                                 const std::string& gcs_path,
                                                 bool* completed,
                                                 uint64* uploaded) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_47_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_47(mht_47_v, 1504, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::RequestUploadSessionStatus");

  CHECK(completed != nullptr) << "RequestUploadSessionStatus() called with out "
                                 "param 'completed' == nullptr.";  // Crash ok
  CHECK(uploaded != nullptr) << "RequestUploadSessionStatus() called with out "
                                "param 'uploaded' == nullptr.";  // Crash ok
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(session_uri);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  request->AddHeader("Content-Range", strings::StrCat("bytes */", file_size));
  request->SetPutEmptyBody();
  Status status = request->Send();
  if (status.ok()) {
    *completed = true;
    return Status::OK();
  }
  *completed = false;
  if (request->GetResponseCode() != HTTP_CODE_RESUME_INCOMPLETE) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(status, " when resuming upload ", gcs_path);
  }
  const std::string received_range = request->GetResponseHeader("Range");
  if (received_range.empty()) {
    // This means GCS doesn't have any bytes of the file yet.
    *uploaded = 0;
  } else {
    StringPiece range_piece(received_range);
    absl::ConsumePrefix(&range_piece,
                        "bytes=");  // May or may not be present.

    auto return_error = [](const std::string& gcs_path,
                           const std::string& error_message) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("gcs_path: \"" + gcs_path + "\"");
   mht_48_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_48(mht_48_v, 1539, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

      return errors::Internal("Unexpected response from GCS when writing ",
                              gcs_path, ": ", error_message);
    };

    std::vector<string> range_strs = str_util::Split(range_piece, '-');
    if (range_strs.size() != 2) {
      return return_error(gcs_path, "Range header '" + received_range +
                                        "' could not be parsed.");
    }

    std::vector<int64_t> range_parts;
    for (const std::string& range_str : range_strs) {
      int64_t tmp;
      if (strings::safe_strto64(range_str, &tmp)) {
        range_parts.push_back(tmp);
      } else {
        return return_error(gcs_path, "Range header '" + received_range +
                                          "' could not be parsed.");
      }
    }

    if (range_parts[0] != 0) {
      return return_error(gcs_path, "The returned range '" + received_range +
                                        "' does not start at zero.");
    }
    // If GCS returned "Range: 0-10", this means 11 bytes were uploaded.
    *uploaded = range_parts[1] + 1;
  }
  return Status::OK();
}

Status GcsFileSystem::ParseGcsPathForScheme(StringPiece fname, string scheme,
                                            bool empty_object_ok,
                                            string* bucket, string* object) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_49(mht_49_v, 1577, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::ParseGcsPathForScheme");

  StringPiece parsed_scheme, bucketp, objectp;
  io::ParseURI(fname, &parsed_scheme, &bucketp, &objectp);
  if (parsed_scheme != scheme) {
    return errors::InvalidArgument("GCS path doesn't start with 'gs://': ",
                                   fname);
  }
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("GCS path doesn't contain a bucket name: ",
                                   fname);
  }
  absl::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("GCS path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

Status GcsFileSystem::ParseGcsPath(StringPiece fname, bool empty_object_ok,
                                   string* bucket, string* object) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_50(mht_50_v, 1602, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::ParseGcsPath");

  return ParseGcsPathForScheme(fname, "gs", empty_object_ok, bucket, object);
}

void GcsFileSystem::ClearFileCaches(const string& fname) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_51(mht_51_v, 1610, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::ClearFileCaches");

  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->RemoveFile(fname);
  stat_cache_->Delete(fname);
  // TODO(rxsang): Remove the patterns that matche the file in
  // MatchingPathsCache as well.
}

Status GcsFileSystem::NewWritableFile(const string& fname,
                                      TransactionToken* token,
                                      std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_52(mht_52_v, 1624, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::NewWritableFile");

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  auto session_creator =
      [this](uint64 start_offset, const std::string& object_to_upload,
             const std::string& bucket, uint64 file_size,
             const std::string& gcs_path, UploadSessionHandle* session_handle) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("object_to_upload: \"" + object_to_upload + "\"");
   mht_53_v.push_back("bucket: \"" + bucket + "\"");
   mht_53_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_53(mht_53_v, 1637, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

        return CreateNewUploadSession(start_offset, object_to_upload, bucket,
                                      file_size, gcs_path, session_handle);
      };
  auto object_uploader =
      [this](const std::string& session_uri, uint64 start_offset,
             uint64 already_uploaded, const std::string& tmp_content_filename,
             uint64 file_size, const std::string& file_path) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_54_v.push_back("tmp_content_filename: \"" + tmp_content_filename + "\"");
   mht_54_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_54(mht_54_v, 1650, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

        return UploadToSession(session_uri, start_offset, already_uploaded,
                               tmp_content_filename, file_size, file_path);
      };
  auto status_poller = [this](const string& session_uri, uint64 file_size,
                              const std::string& gcs_path, bool* completed,
                              uint64* uploaded) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_55_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_55(mht_55_v, 1661, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    return RequestUploadSessionStatus(session_uri, file_size, gcs_path,
                                      completed, uploaded);
  };

  auto generation_getter = [this](const string& fname, const string& bucket,
                                  const string& object, int64* generation) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("fname: \"" + fname + "\"");
   mht_56_v.push_back("bucket: \"" + bucket + "\"");
   mht_56_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_56(mht_56_v, 1673, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    GcsFileStat stat;
    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&fname, &bucket, &object, &stat, this]() {
          return UncachedStatForObject(fname, bucket, object, &stat);
        },
        retry_config_));
    *generation = stat.generation_number;
    return Status::OK();
  };

  result->reset(new GcsWritableFile(
      bucket, object, this, &timeouts_,
      [this, fname]() { ClearFileCaches(fname); }, retry_config_,
      compose_append_, session_creator, object_uploader, status_poller,
      generation_getter));
  return Status::OK();
}

// Reads the file from GCS in chunks and stores it in a tmp file,
// which is then passed to GcsWritableFile.
Status GcsFileSystem::NewAppendableFile(const string& fname,
                                        TransactionToken* token,
                                        std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_57(mht_57_v, 1700, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::NewAppendableFile");

  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, token, &reader));
  std::unique_ptr<char[]> buffer(new char[kReadAppendableFileBufferSize]);
  Status status;
  uint64 offset = 0;
  StringPiece read_chunk;

  // Read the file from GCS in chunks and save it to a tmp file.
  string old_content_filename;
  TF_RETURN_IF_ERROR(GetTmpFilename(&old_content_filename));
  std::ofstream old_content(old_content_filename, std::ofstream::binary);
  while (true) {
    status = reader->Read(offset, kReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      old_content << read_chunk;
      offset += kReadAppendableFileBufferSize;
    } else if (status.code() == error::NOT_FOUND) {
      // New file, there is no existing content in it.
      break;
    } else if (status.code() == error::OUT_OF_RANGE) {
      // Expected, this means we reached EOF.
      old_content << read_chunk;
      break;
    } else {
      return status;
    }
  }
  old_content.close();

  auto session_creator =
      [this](uint64 start_offset, const std::string& object_to_upload,
             const std::string& bucket, uint64 file_size,
             const std::string& gcs_path, UploadSessionHandle* session_handle) {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("object_to_upload: \"" + object_to_upload + "\"");
   mht_58_v.push_back("bucket: \"" + bucket + "\"");
   mht_58_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_58(mht_58_v, 1740, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

        return CreateNewUploadSession(start_offset, object_to_upload, bucket,
                                      file_size, gcs_path, session_handle);
      };
  auto object_uploader =
      [this](const std::string& session_uri, uint64 start_offset,
             uint64 already_uploaded, const std::string& tmp_content_filename,
             uint64 file_size, const std::string& file_path) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_59_v.push_back("tmp_content_filename: \"" + tmp_content_filename + "\"");
   mht_59_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_59(mht_59_v, 1753, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

        return UploadToSession(session_uri, start_offset, already_uploaded,
                               tmp_content_filename, file_size, file_path);
      };

  auto status_poller = [this](const string& session_uri, uint64 file_size,
                              const std::string& gcs_path, bool* completed,
                              uint64* uploaded) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("session_uri: \"" + session_uri + "\"");
   mht_60_v.push_back("gcs_path: \"" + gcs_path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_60(mht_60_v, 1765, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    return RequestUploadSessionStatus(session_uri, file_size, gcs_path,
                                      completed, uploaded);
  };

  auto generation_getter = [this](const string& fname, const string& bucket,
                                  const string& object, int64* generation) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("fname: \"" + fname + "\"");
   mht_61_v.push_back("bucket: \"" + bucket + "\"");
   mht_61_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_61(mht_61_v, 1777, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    GcsFileStat stat;
    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&fname, &bucket, &object, &stat, this]() {
          return UncachedStatForObject(fname, bucket, object, &stat);
        },
        retry_config_));
    *generation = stat.generation_number;
    return Status::OK();
  };

  // Create a writable file and pass the old content to it.
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsWritableFile(
      bucket, object, this, old_content_filename, &timeouts_,
      [this, fname]() { ClearFileCaches(fname); }, retry_config_,
      compose_append_, session_creator, object_uploader, status_poller,
      generation_getter));
  return Status::OK();
}

Status GcsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_62(mht_62_v, 1805, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::NewReadOnlyMemoryRegionFromFile");

  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(fname, token, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, token, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new GcsReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status GcsFileSystem::FileExists(const string& fname, TransactionToken* token) {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_63(mht_63_v, 1824, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::FileExists");

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool result;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &result));
    if (result) {
      return Status::OK();
    }
  }

  // Check if the object exists.
  GcsFileStat stat;
  const Status status = StatForObject(fname, bucket, object, &stat);
  if (status.code() != errors::Code::NOT_FOUND) {
    return status;
  }

  // Check if the folder exists.
  bool result;
  TF_RETURN_IF_ERROR(FolderExists(fname, &result));
  if (result) {
    return Status::OK();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::ObjectExists(const string& fname, const string& bucket,
                                   const string& object, bool* result) {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("fname: \"" + fname + "\"");
   mht_64_v.push_back("bucket: \"" + bucket + "\"");
   mht_64_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_64(mht_64_v, 1858, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::ObjectExists");

  GcsFileStat stat;
  const Status status = StatForObject(fname, bucket, object, &stat);
  switch (status.code()) {
    case errors::Code::OK:
      *result = !stat.base.is_directory;
      return Status::OK();
    case errors::Code::NOT_FOUND:
      *result = false;
      return Status::OK();
    default:
      return status;
  }
}

Status GcsFileSystem::UncachedStatForObject(const string& fname,
                                            const string& bucket,
                                            const string& object,
                                            GcsFileStat* stat) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("fname: \"" + fname + "\"");
   mht_65_v.push_back("bucket: \"" + bucket + "\"");
   mht_65_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_65(mht_65_v, 1882, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::UncachedStatForObject");

  std::vector<char> output_buffer;
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  " when reading metadata of gs://", bucket,
                                  "/", object);

  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object),
                                  "?fields=size%2Cgeneration%2Cupdated"));
  request->SetResultBuffer(&output_buffer);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);

  if (stats_ != nullptr) {
    stats_->RecordStatObjectRequest();
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      request->Send(), " when reading metadata of gs://", bucket, "/", object);

  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));

  // Parse file size.
  TF_RETURN_IF_ERROR(GetInt64Value(root, "size", &stat->base.length));

  // Parse generation number.
  TF_RETURN_IF_ERROR(
      GetInt64Value(root, "generation", &stat->generation_number));

  // Parse file modification time.
  string updated;
  TF_RETURN_IF_ERROR(GetStringValue(root, "updated", &updated));
  TF_RETURN_IF_ERROR(ParseRfc3339Time(updated, &(stat->base.mtime_nsec)));

  VLOG(1) << "Stat of: gs://" << bucket << "/" << object << " -- "
          << " length: " << stat->base.length
          << " generation: " << stat->generation_number
          << "; mtime_nsec: " << stat->base.mtime_nsec
          << "; updated: " << updated;

  if (str_util::EndsWith(fname, "/")) {
    // In GCS a path can be both a directory and a file, both it is uncommon for
    // other file systems. To avoid the ambiguity, if a path ends with "/" in
    // GCS, we always regard it as a directory mark or a virtual directory.
    stat->base.is_directory = true;
  } else {
    stat->base.is_directory = false;
  }
  return Status::OK();
}

Status GcsFileSystem::StatForObject(const string& fname, const string& bucket,
                                    const string& object, GcsFileStat* stat) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("fname: \"" + fname + "\"");
   mht_66_v.push_back("bucket: \"" + bucket + "\"");
   mht_66_v.push_back("object: \"" + object + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_66(mht_66_v, 1941, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::StatForObject");

  if (object.empty()) {
    return errors::InvalidArgument(strings::Printf(
        "'object' must be a non-empty string. (File: %s)", fname.c_str()));
  }

  TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
      fname, stat,
      [this, &bucket, &object](const string& fname, GcsFileStat* stat) {
        return UncachedStatForObject(fname, bucket, object, stat);
      }));
  return Status::OK();
}

Status GcsFileSystem::BucketExists(const string& bucket, bool* result) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("bucket: \"" + bucket + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_67(mht_67_v, 1959, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::BucketExists");

  const Status status = GetBucketMetadata(bucket, nullptr);
  switch (status.code()) {
    case errors::Code::OK:
      *result = true;
      return Status::OK();
    case errors::Code::NOT_FOUND:
      *result = false;
      return Status::OK();
    default:
      return status;
  }
}

Status GcsFileSystem::CheckBucketLocationConstraint(const string& bucket) {
   std::vector<std::string> mht_68_v;
   mht_68_v.push_back("bucket: \"" + bucket + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_68(mht_68_v, 1977, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::CheckBucketLocationConstraint");

  if (allowed_locations_.empty()) {
    return Status::OK();
  }

  // Avoid calling external API's in the constructor
  if (allowed_locations_.erase(kDetectZoneSentinelValue) == 1) {
    string zone;
    TF_RETURN_IF_ERROR(zone_provider_->GetZone(&zone));
    allowed_locations_.insert(ZoneToRegion(&zone));
  }

  string location;
  TF_RETURN_IF_ERROR(GetBucketLocation(bucket, &location));
  if (allowed_locations_.find(location) != allowed_locations_.end()) {
    return Status::OK();
  }

  return errors::FailedPrecondition(strings::Printf(
      "Bucket '%s' is in '%s' location, allowed locations are: (%s).",
      bucket.c_str(), location.c_str(),
      absl::StrJoin(allowed_locations_, ", ").c_str()));
}

Status GcsFileSystem::GetBucketLocation(const string& bucket,
                                        string* location) {
   std::vector<std::string> mht_69_v;
   mht_69_v.push_back("bucket: \"" + bucket + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_69(mht_69_v, 2006, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetBucketLocation");

  auto compute_func = [this](const string& bucket, string* location) {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("bucket: \"" + bucket + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_70(mht_70_v, 2011, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    std::vector<char> result_buffer;
    Status status = GetBucketMetadata(bucket, &result_buffer);
    Json::Value result;
    TF_RETURN_IF_ERROR(ParseJson(result_buffer, &result));
    string bucket_location;
    TF_RETURN_IF_ERROR(
        GetStringValue(result, kBucketMetadataLocationKey, &bucket_location));
    // Lowercase the GCS location to be case insensitive for allowed locations.
    *location = absl::AsciiStrToLower(bucket_location);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(
      bucket_location_cache_->LookupOrCompute(bucket, location, compute_func));

  return Status::OK();
}

Status GcsFileSystem::GetBucketMetadata(const string& bucket,
                                        std::vector<char>* result_buffer) {
   std::vector<std::string> mht_71_v;
   mht_71_v.push_back("bucket: \"" + bucket + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_71(mht_71_v, 2035, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetBucketMetadata");

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket));

  if (result_buffer != nullptr) {
    request->SetResultBuffer(result_buffer);
  }

  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  return request->Send();
}

Status GcsFileSystem::FolderExists(const string& dirname, bool* result) {
   std::vector<std::string> mht_72_v;
   mht_72_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_72(mht_72_v, 2052, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::FolderExists");

  StatCache::ComputeFunc compute_func = [this](const string& dirname,
                                               GcsFileStat* stat) {
   std::vector<std::string> mht_73_v;
   mht_73_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_73(mht_73_v, 2058, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

    std::vector<string> children;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dirname, 1, &children, true /* recursively */,
                           true /* include_self_directory_marker */));
    if (!children.empty()) {
      stat->base = DIRECTORY_STAT;
      return Status::OK();
    } else {
      return errors::InvalidArgument("Not a directory!");
    }
  };
  GcsFileStat stat;
  Status s = stat_cache_->LookupOrCompute(MaybeAppendSlash(dirname), &stat,
                                          compute_func);
  if (s.ok()) {
    *result = stat.base.is_directory;
    return Status::OK();
  }
  if (errors::IsInvalidArgument(s)) {
    *result = false;
    return Status::OK();
  }
  return s;
}

Status GcsFileSystem::GetChildren(const string& dirname,
                                  TransactionToken* token,
                                  std::vector<string>* result) {
   std::vector<std::string> mht_74_v;
   mht_74_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_74(mht_74_v, 2090, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetChildren");

  return GetChildrenBounded(dirname, UINT64_MAX, result,
                            false /* recursively */,
                            false /* include_self_directory_marker */);
}

Status GcsFileSystem::GetMatchingPaths(const string& pattern,
                                       TransactionToken* token,
                                       std::vector<string>* results) {
   std::vector<std::string> mht_75_v;
   mht_75_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_75(mht_75_v, 2102, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetMatchingPaths");

  MatchingPathsCache::ComputeFunc compute_func =
      [this](const string& pattern, std::vector<string>* results) {
   std::vector<std::string> mht_76_v;
   mht_76_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_76(mht_76_v, 2108, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "lambda");

        results->clear();
        // Find the fixed prefix by looking for the first wildcard.
        const string& fixed_prefix =
            pattern.substr(0, pattern.find_first_of("*?[\\"));
        const string dir(this->Dirname(fixed_prefix));
        if (dir.empty()) {
          return errors::InvalidArgument(
              "A GCS pattern doesn't have a bucket name: ", pattern);
        }
        std::vector<string> all_files;
        TF_RETURN_IF_ERROR(GetChildrenBounded(
            dir, UINT64_MAX, &all_files, true /* recursively */,
            false /* include_self_directory_marker */));

        const auto& files_and_folders = AddAllSubpaths(all_files);

        // To handle `/` in the object names, we need to remove it from `dir`
        // and then use `StrCat` to insert it back.
        const StringPiece dir_no_slash = str_util::StripSuffix(dir, "/");

        // Match all obtained paths to the input pattern.
        for (const auto& path : files_and_folders) {
          // Manually construct the path instead of using `JoinPath` for the
          // cases where `path` starts with a `/` (which is a valid character in
          // the filenames of GCS objects). `JoinPath` canonicalizes the result,
          // removing duplicate slashes. We know that `dir_no_slash` does not
          // end in `/`, so we are safe inserting the new `/` here as the path
          // separator.
          const string full_path = strings::StrCat(dir_no_slash, "/", path);
          if (this->Match(full_path, pattern)) {
            results->push_back(full_path);
          }
        }
        return Status::OK();
      };
  TF_RETURN_IF_ERROR(
      matching_paths_cache_->LookupOrCompute(pattern, results, compute_func));
  return Status::OK();
}

Status GcsFileSystem::GetChildrenBounded(const string& dirname,
                                         uint64 max_results,
                                         std::vector<string>* result,
                                         bool recursive,
                                         bool include_self_directory_marker) {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_77(mht_77_v, 2157, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetChildrenBounded");

  if (!result) {
    return errors::InvalidArgument("'result' cannot be null");
  }
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(
      ParseGcsPath(MaybeAppendSlash(dirname), true, &bucket, &object_prefix));

  string nextPageToken;
  uint64 retrieved_results = 0;
  while (true) {  // A loop over multiple result pages.
    std::vector<char> output_buffer;
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
    auto uri = strings::StrCat(kGcsUriBase, "b/", bucket, "/o");
    if (recursive) {
      uri = strings::StrCat(uri, "?fields=items%2Fname%2CnextPageToken");
    } else {
      // Set "/" as a delimiter to ask GCS to treat subfolders as children
      // and return them in "prefixes".
      uri = strings::StrCat(uri,
                            "?fields=items%2Fname%2Cprefixes%2CnextPageToken");
      uri = strings::StrCat(uri, "&delimiter=%2F");
    }
    if (!object_prefix.empty()) {
      uri = strings::StrCat(uri,
                            "&prefix=", request->EscapeString(object_prefix));
    }
    if (!nextPageToken.empty()) {
      uri = strings::StrCat(
          uri, "&pageToken=", request->EscapeString(nextPageToken));
    }
    if (max_results - retrieved_results < kGetChildrenDefaultPageSize) {
      uri =
          strings::StrCat(uri, "&maxResults=", max_results - retrieved_results);
    }
    request->SetUri(uri);
    request->SetResultBuffer(&output_buffer);
    request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading ", dirname);
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
    const auto items = root.get("items", Json::Value::null);
    if (!items.isNull()) {
      if (!items.isArray()) {
        return errors::Internal(
            "Expected an array 'items' in the GCS response.");
      }
      for (size_t i = 0; i < items.size(); i++) {
        const auto item = items.get(i, Json::Value::null);
        if (!item.isObject()) {
          return errors::Internal(
              "Unexpected JSON format: 'items' should be a list of objects.");
        }
        string name;
        TF_RETURN_IF_ERROR(GetStringValue(item, "name", &name));
        // The names should be relative to the 'dirname'. That means the
        // 'object_prefix', which is part of 'dirname', should be removed from
        // the beginning of 'name'.
        StringPiece relative_path(name);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(strings::StrCat(
              "Unexpected response: the returned file name ", name,
              " doesn't match the prefix ", object_prefix));
        }
        if (!relative_path.empty() || include_self_directory_marker) {
          result->emplace_back(relative_path);
        }
        if (++retrieved_results >= max_results) {
          return Status::OK();
        }
      }
    }
    const auto prefixes = root.get("prefixes", Json::Value::null);
    if (!prefixes.isNull()) {
      // Subfolders are returned for the non-recursive mode.
      if (!prefixes.isArray()) {
        return errors::Internal(
            "'prefixes' was expected to be an array in the GCS response.");
      }
      for (size_t i = 0; i < prefixes.size(); i++) {
        const auto prefix = prefixes.get(i, Json::Value::null);
        if (prefix.isNull() || !prefix.isString()) {
          return errors::Internal(
              "'prefixes' was expected to be an array of strings in the GCS "
              "response.");
        }
        const string& prefix_str = prefix.asString();
        StringPiece relative_path(prefix_str);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(
              "Unexpected response: the returned folder name ", prefix_str,
              " doesn't match the prefix ", object_prefix);
        }
        result->emplace_back(relative_path);
        if (++retrieved_results >= max_results) {
          return Status::OK();
        }
      }
    }
    const auto token = root.get("nextPageToken", Json::Value::null);
    if (token.isNull()) {
      return Status::OK();
    }
    if (!token.isString()) {
      return errors::Internal(
          "Unexpected response: nextPageToken is not a string");
    }
    nextPageToken = token.asString();
  }
}

Status GcsFileSystem::Stat(const string& fname, TransactionToken* token,
                           FileStatistics* stat) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_78(mht_78_v, 2275, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::Stat");

  if (!stat) {
    return errors::Internal("'stat' cannot be nullptr.");
  }
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      *stat = DIRECTORY_STAT;
      return Status::OK();
    }
    return errors::NotFound("The specified bucket ", fname, " was not found.");
  }

  GcsFileStat gcs_stat;
  const Status status = StatForObject(fname, bucket, object, &gcs_stat);
  if (status.ok()) {
    *stat = gcs_stat.base;
    return Status::OK();
  }
  if (status.code() != errors::Code::NOT_FOUND) {
    return status;
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    *stat = DIRECTORY_STAT;
    return Status::OK();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::DeleteFile(const string& fname, TransactionToken* token) {
   std::vector<std::string> mht_79_v;
   mht_79_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_79(mht_79_v, 2313, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::DeleteFile");

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object)));
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  request->SetDeleteRequest();

  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when deleting ", fname);
  ClearFileCaches(fname);
  return Status::OK();
}

Status GcsFileSystem::CreateDir(const string& dirname,
                                TransactionToken* token) {
   std::vector<std::string> mht_80_v;
   mht_80_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_80(mht_80_v, 2334, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::CreateDir");

  string dirname_with_slash = MaybeAppendSlash(dirname);
  VLOG(3) << "CreateDir: creating directory with dirname: " << dirname
          << " and dirname_with_slash: " << dirname_with_slash;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(dirname_with_slash, /*empty_object_ok=*/true,
                                  &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    return is_bucket ? Status::OK()
                     : errors::NotFound("The specified bucket ",
                                        dirname_with_slash, " was not found.");
  }

  if (FileExists(dirname_with_slash, token).ok()) {
    // Use the original name for a correct error here.
    VLOG(3) << "CreateDir: directory already exists, not uploading " << dirname;
    return errors::AlreadyExists(dirname);
  }

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));

  request->SetUri(strings::StrCat(
      kGcsUploadUriBase, "b/", bucket,
      "/o?uploadType=media&name=", request->EscapeString(object),
      // Adding this parameter means HTTP_CODE_PRECONDITION_FAILED
      // will be returned if the object already exists, so avoid reuploading.
      "&ifGenerationMatch=0"));

  request->SetPostEmptyBody();
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  const Status& status = request->Send();
  if (status.ok()) {
    VLOG(3) << "CreateDir: finished uploading directory " << dirname;
    return Status::OK();
  }
  if (request->GetResponseCode() != HTTP_CODE_PRECONDITION_FAILED) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(status, " when uploading ",
                                    dirname_with_slash);
  }
  VLOG(3) << "Ignoring directory already exists on object "
          << dirname_with_slash;
  return errors::AlreadyExists(dirname);
}

// Checks that the directory is empty (i.e no objects with this prefix exist).
// Deletes the GCS directory marker if it exists.
Status GcsFileSystem::DeleteDir(const string& dirname,
                                TransactionToken* token) {
   std::vector<std::string> mht_81_v;
   mht_81_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_81(mht_81_v, 2388, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::DeleteDir");

  std::vector<string> children;
  // A directory is considered empty either if there are no matching objects
  // with the corresponding name prefix or if there is exactly one matching
  // object and it is the directory marker. Therefore we need to retrieve
  // at most two children for the prefix to detect if a directory is empty.
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(dirname, 2, &children, true /* recursively */,
                         true /* include_self_directory_marker */));

  if (children.size() > 1 || (children.size() == 1 && !children[0].empty())) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }
  if (children.size() == 1 && children[0].empty()) {
    // This is the directory marker object. Delete it.
    return DeleteFile(MaybeAppendSlash(dirname), token);
  }
  return Status::OK();
}

Status GcsFileSystem::GetFileSize(const string& fname, TransactionToken* token,
                                  uint64* file_size) {
   std::vector<std::string> mht_82_v;
   mht_82_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_82(mht_82_v, 2413, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::GetFileSize");

  if (!file_size) {
    return errors::Internal("'file_size' cannot be nullptr.");
  }

  // Only validate the name.
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, token, &stat));
  *file_size = stat.length;
  return Status::OK();
}

Status GcsFileSystem::RenameFile(const string& src, const string& target,
                                 TransactionToken* token) {
   std::vector<std::string> mht_83_v;
   mht_83_v.push_back("src: \"" + src + "\"");
   mht_83_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_83(mht_83_v, 2434, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::RenameFile");

  if (!IsDirectory(src, token).ok()) {
    return RenameObject(src, target);
  }
  // Rename all individual objects in the directory one by one.
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(src, UINT64_MAX, &children, true /* recursively */,
                         true /* include_self_directory_marker */));
  for (const string& subpath : children) {
    TF_RETURN_IF_ERROR(
        RenameObject(JoinGcsPath(src, subpath), JoinGcsPath(target, subpath)));
  }
  return Status::OK();
}

// Uses a GCS API command to copy the object and then deletes the old one.
Status GcsFileSystem::RenameObject(const string& src, const string& target) {
   std::vector<std::string> mht_84_v;
   mht_84_v.push_back("src: \"" + src + "\"");
   mht_84_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_84(mht_84_v, 2456, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::RenameObject");

  VLOG(3) << "RenameObject: started gs://" << src << " to " << target;
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseGcsPath(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseGcsPath(target, false, &target_bucket, &target_object));

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", src_bucket, "/o/",
                                  request->EscapeString(src_object),
                                  "/rewriteTo/b/", target_bucket, "/o/",
                                  request->EscapeString(target_object)));
  request->SetPostEmptyBody();
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  std::vector<char> output_buffer;
  request->SetResultBuffer(&output_buffer);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when renaming ", src,
                                  " to ", target);
  // Flush the target from the caches.  The source will be flushed in the
  // DeleteFile call below.
  ClearFileCaches(target);
  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
  bool done;
  TF_RETURN_IF_ERROR(GetBoolValue(root, "done", &done));
  if (!done) {
    // If GCS didn't complete rewrite in one call, this means that a large file
    // is being copied to a bucket with a different storage class or location,
    // which requires multiple rewrite calls.
    // TODO(surkov): implement multi-step rewrites.
    return errors::Unimplemented(
        "Couldn't rename ", src, " to ", target,
        ": moving large files between buckets with different "
        "locations or storage classes is not supported.");
  }

  VLOG(3) << "RenameObject: finished from: gs://" << src << " to " << target;
  // In case the delete API call failed, but the deletion actually happened
  // on the server side, we can't just retry the whole RenameFile operation
  // because the source object is already gone.
  return RetryingUtils::DeleteWithRetries(
      [this, &src]() { return DeleteFile(src, nullptr); }, retry_config_);
}

Status GcsFileSystem::IsDirectory(const string& fname,
                                  TransactionToken* token) {
   std::vector<std::string> mht_85_v;
   mht_85_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_85(mht_85_v, 2506, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::IsDirectory");

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      return Status::OK();
    }
    return errors::NotFound("The specified bucket gs://", bucket,
                            " was not found.");
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    return Status::OK();
  }
  bool is_object;
  TF_RETURN_IF_ERROR(ObjectExists(fname, bucket, object, &is_object));
  if (is_object) {
    return errors::FailedPrecondition("The specified path ", fname,
                                      " is not a directory.");
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::DeleteRecursively(const string& dirname,
                                        TransactionToken* token,
                                        int64_t* undeleted_files,
                                        int64_t* undeleted_dirs) {
   std::vector<std::string> mht_86_v;
   mht_86_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_86(mht_86_v, 2539, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::DeleteRecursively");

  if (!undeleted_files || !undeleted_dirs) {
    return errors::Internal(
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  }
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  if (!IsDirectory(dirname, token).ok()) {
    *undeleted_dirs = 1;
    return Status(
        error::NOT_FOUND,
        strings::StrCat(dirname, " doesn't exist or not a directory."));
  }
  std::vector<string> all_objects;
  // Get all children in the directory recursively.
  TF_RETURN_IF_ERROR(GetChildrenBounded(
      dirname, UINT64_MAX, &all_objects, true /* recursively */,
      true /* include_self_directory_marker */));
  for (const string& object : all_objects) {
    const string& full_path = JoinGcsPath(dirname, object);
    // Delete all objects including directory markers for subfolders.
    // Since DeleteRecursively returns OK if individual file deletions fail,
    // and therefore RetryingFileSystem won't pay attention to the failures,
    // we need to make sure these failures are properly retried.
    const auto& delete_file_status = RetryingUtils::DeleteWithRetries(
        [this, &full_path, token]() { return DeleteFile(full_path, token); },
        retry_config_);
    if (!delete_file_status.ok()) {
      if (IsDirectory(full_path, token).ok()) {
        // The object is a directory marker.
        (*undeleted_dirs)++;
      } else {
        (*undeleted_files)++;
      }
    }
  }
  return Status::OK();
}

// Flushes all caches for filesystem metadata and file contents. Useful for
// reclaiming memory once filesystem operations are done (e.g. model is loaded),
// or for resetting the filesystem to a consistent state.
void GcsFileSystem::FlushCaches(TransactionToken* token) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_87(mht_87_v, 2584, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::FlushCaches");

  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->Flush();
  stat_cache_->Clear();
  matching_paths_cache_->Clear();
  bucket_location_cache_->Clear();
}

void GcsFileSystem::SetStats(GcsStatsInterface* stats) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_88(mht_88_v, 2595, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::SetStats");

  CHECK(stats_ == nullptr) << "SetStats() has already been called.";
  CHECK(stats != nullptr);
  mutex_lock l(block_cache_lock_);
  stats_ = stats;
  stats_->Configure(this, &throttle_, file_block_cache_.get());
}

void GcsFileSystem::SetCacheStats(FileBlockCacheStatsInterface* cache_stats) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_89(mht_89_v, 2606, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::SetCacheStats");

  tf_shared_lock l(block_cache_lock_);
  if (file_block_cache_ == nullptr) {
    LOG(ERROR) << "Tried to set cache stats of non-initialized file block "
                  "cache object. This may result in not exporting the intended "
                  "monitoring data";
    return;
  }
  file_block_cache_->SetStats(cache_stats);
}

void GcsFileSystem::SetAuthProvider(
    std::unique_ptr<AuthProvider> auth_provider) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_90(mht_90_v, 2621, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::SetAuthProvider");

  mutex_lock l(mu_);
  auth_provider_ = std::move(auth_provider);
}

// Creates an HttpRequest and sets several parameters that are common to all
// requests.  All code (in GcsFileSystem) that creates an HttpRequest should
// go through this method, rather than directly using http_request_factory_.
Status GcsFileSystem::CreateHttpRequest(std::unique_ptr<HttpRequest>* request) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_file_systemDTcc mht_91(mht_91_v, 2632, "", "./tensorflow/core/platform/cloud/gcs_file_system.cc", "GcsFileSystem::CreateHttpRequest");

  std::unique_ptr<HttpRequest> new_request{http_request_factory_->Create()};
  if (dns_cache_) {
    dns_cache_->AnnotateRequest(new_request.get());
  }

  string auth_token;
  {
    tf_shared_lock l(mu_);
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));
  }

  new_request->AddAuthBearerHeader(auth_token);

  if (additional_header_) {
    new_request->AddHeader(additional_header_->first,
                           additional_header_->second);
  }

  if (stats_ != nullptr) {
    new_request->SetRequestStats(stats_->HttpStats());
  }

  if (!throttle_.AdmitRequest()) {
    return errors::Unavailable("Request throttled");
  }

  *request = std::move(new_request);
  return Status::OK();
}

}  // namespace tensorflow

// The TPU_GCS_FS option sets a TPU-on-GCS optimized file system that allows
// TPU pods to function more optimally. When TPU_GCS_FS is enabled then
// gcs_file_system will not be registered as a file system since the
// tpu_gcs_file_system is going to take over its responsibilities. The tpu file
// system is a child of gcs file system with TPU-pod on GCS optimizations.
// This option is set ON/OFF in the GCP TPU tensorflow config.
// Initialize gcs_file_system
REGISTER_LEGACY_FILE_SYSTEM("gs", ::tensorflow::RetryingGcsFileSystem);
