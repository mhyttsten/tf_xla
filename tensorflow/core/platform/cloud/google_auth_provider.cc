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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc() {
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

#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#else
#include <sys/types.h>
#endif
#include <fstream>
#include <utility>

#include "absl/strings/match.h"
#include "json/json.h"
#include "tensorflow/core/platform/base64.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/retrying_utils.h"

namespace tensorflow {

namespace {

// The environment variable pointing to the file with local
// Application Default Credentials.
constexpr char kGoogleApplicationCredentials[] =
    "GOOGLE_APPLICATION_CREDENTIALS";

// The environment variable to override token generation for testing.
constexpr char kGoogleAuthTokenForTesting[] = "GOOGLE_AUTH_TOKEN_FOR_TESTING";

// The environment variable which can override '~/.config/gcloud' if set.
constexpr char kCloudSdkConfig[] = "CLOUDSDK_CONFIG";

// The environment variable used to skip attempting to fetch GCE credentials:
// setting this to 'true' (case insensitive) will skip attempting to contact
// the GCE metadata service.
constexpr char kNoGceCheck[] = "NO_GCE_CHECK";

// The default path to the gcloud config folder, relative to the home folder.
constexpr char kGCloudConfigFolder[] = ".config/gcloud/";

// The name of the well-known credentials JSON file in the gcloud config folder.
constexpr char kWellKnownCredentialsFile[] =
    "application_default_credentials.json";

// The minimum time delta between now and the token expiration time
// for the token to be re-used.
constexpr int kExpirationTimeMarginSec = 60;

// The URL to retrieve the auth bearer token via OAuth with a refresh token.
constexpr char kOAuthV3Url[] = "https://www.googleapis.com/oauth2/v3/token";

// The URL to retrieve the auth bearer token via OAuth with a private key.
constexpr char kOAuthV4Url[] = "https://www.googleapis.com/oauth2/v4/token";

// The URL to retrieve the auth bearer token when running in Google Compute
// Engine.
constexpr char kGceTokenPath[] = "instance/service-accounts/default/token";

// The authentication token scope to request.
constexpr char kOAuthScope[] = "https://www.googleapis.com/auth/cloud-platform";

/// Returns whether the given path points to a readable file.
bool IsFile(const string& filename) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "IsFile");

  std::ifstream fstream(filename.c_str());
  return fstream.good();
}

/// Returns the credentials file name from the env variable.
Status GetEnvironmentVariableFileName(string* filename) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GetEnvironmentVariableFileName");

  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  const char* result = std::getenv(kGoogleApplicationCredentials);
  if (!result || !IsFile(result)) {
    return errors::NotFound(strings::StrCat("$", kGoogleApplicationCredentials,
                                            " is not set or corrupt."));
  }
  *filename = result;
  return Status::OK();
}

/// Returns the well known file produced by command 'gcloud auth login'.
Status GetWellKnownFileName(string* filename) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_2(mht_2_v, 275, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GetWellKnownFileName");

  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  string config_dir;
  const char* config_dir_override = std::getenv(kCloudSdkConfig);
  if (config_dir_override) {
    config_dir = config_dir_override;
  } else {
    // Determine the home dir path.
    const char* home_dir = std::getenv("HOME");
    if (!home_dir) {
      return errors::FailedPrecondition("Could not read $HOME.");
    }
    config_dir = io::JoinPath(home_dir, kGCloudConfigFolder);
  }
  auto result = io::JoinPath(config_dir, kWellKnownCredentialsFile);
  if (!IsFile(result)) {
    return errors::NotFound(
        "Could not find the credentials file in the standard gcloud location.");
  }
  *filename = result;
  return Status::OK();
}

}  // namespace

GoogleAuthProvider::GoogleAuthProvider(
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client)
    : GoogleAuthProvider(std::unique_ptr<OAuthClient>(new OAuthClient()),
                         std::move(compute_engine_metadata_client),
                         Env::Default()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_3(mht_3_v, 309, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GoogleAuthProvider");
}

GoogleAuthProvider::GoogleAuthProvider(
    std::unique_ptr<OAuthClient> oauth_client,
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client,
    Env* env)
    : oauth_client_(std::move(oauth_client)),
      compute_engine_metadata_client_(
          std::move(compute_engine_metadata_client)),
      env_(env) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_4(mht_4_v, 321, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GoogleAuthProvider");
}

Status GoogleAuthProvider::GetToken(string* t) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_5(mht_5_v, 326, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GetToken");

  mutex_lock lock(mu_);
  const uint64 now_sec = env_->NowSeconds();

  if (now_sec + kExpirationTimeMarginSec < expiration_timestamp_sec_) {
    *t = current_token_;
    return Status::OK();
  }

  if (GetTokenForTesting().ok()) {
    *t = current_token_;
    return Status::OK();
  }

  auto token_from_files_status = GetTokenFromFiles();
  if (token_from_files_status.ok()) {
    *t = current_token_;
    return Status::OK();
  }

  char* no_gce_check_var = std::getenv(kNoGceCheck);
  bool skip_gce_check = no_gce_check_var != nullptr &&
                        absl::EqualsIgnoreCase(no_gce_check_var, "true");
  Status token_from_gce_status;
  if (skip_gce_check) {
    token_from_gce_status =
        Status(error::CANCELLED,
               strings::StrCat("GCE check skipped due to presence of $",
                               kNoGceCheck, " environment variable."));
  } else {
    token_from_gce_status = GetTokenFromGce();
  }

  if (token_from_gce_status.ok()) {
    *t = current_token_;
    return Status::OK();
  }

  if (skip_gce_check) {
    LOG(INFO)
        << "Attempting an empty bearer token since no token was retrieved "
        << "from files, and GCE metadata check was skipped.";
  } else {
    LOG(WARNING)
        << "All attempts to get a Google authentication bearer token failed, "
        << "returning an empty token. Retrieving token from files failed with "
           "\""
        << token_from_files_status.ToString() << "\"."
        << " Retrieving token from GCE failed with \""
        << token_from_gce_status.ToString() << "\".";
  }

  // Public objects can still be accessed with an empty bearer token,
  // so return an empty token instead of failing.
  *t = "";

  // We only want to keep returning our empty token if we've tried and failed
  // the (potentially slow) task of detecting GCE.
  if (skip_gce_check) {
    expiration_timestamp_sec_ = 0;
  } else {
    expiration_timestamp_sec_ = UINT64_MAX;
  }
  current_token_ = "";

  return Status::OK();
}

Status GoogleAuthProvider::GetTokenFromFiles() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_6(mht_6_v, 397, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GetTokenFromFiles");

  string credentials_filename;
  if (!GetEnvironmentVariableFileName(&credentials_filename).ok() &&
      !GetWellKnownFileName(&credentials_filename).ok()) {
    return errors::NotFound("Could not locate the credentials file.");
  }

  Json::Value json;
  Json::Reader reader;
  std::ifstream credentials_fstream(credentials_filename);
  if (!reader.parse(credentials_fstream, json)) {
    return errors::FailedPrecondition(
        "Couldn't parse the JSON credentials file.");
  }
  if (json.isMember("refresh_token")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromRefreshTokenJson(
        json, kOAuthV3Url, &current_token_, &expiration_timestamp_sec_));
  } else if (json.isMember("private_key")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromServiceAccountJson(
        json, kOAuthV4Url, kOAuthScope, &current_token_,
        &expiration_timestamp_sec_));
  } else {
    return errors::FailedPrecondition(
        "Unexpected content of the JSON credentials file.");
  }
  return Status::OK();
}

Status GoogleAuthProvider::GetTokenFromGce() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_7(mht_7_v, 428, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GetTokenFromGce");

  std::vector<char> response_buffer;
  const uint64 request_timestamp_sec = env_->NowSeconds();

  TF_RETURN_IF_ERROR(compute_engine_metadata_client_->GetMetadata(
      kGceTokenPath, &response_buffer));
  StringPiece response =
      StringPiece(&response_buffer[0], response_buffer.size());

  TF_RETURN_IF_ERROR(oauth_client_->ParseOAuthResponse(
      response, request_timestamp_sec, &current_token_,
      &expiration_timestamp_sec_));

  return Status::OK();
}

Status GoogleAuthProvider::GetTokenForTesting() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_providerDTcc mht_8(mht_8_v, 447, "", "./tensorflow/core/platform/cloud/google_auth_provider.cc", "GoogleAuthProvider::GetTokenForTesting");

  const char* token = std::getenv(kGoogleAuthTokenForTesting);
  if (!token) {
    return errors::NotFound("The env variable for testing was not set.");
  }
  expiration_timestamp_sec_ = UINT64_MAX;
  current_token_ = token;
  return Status::OK();
}

}  // namespace tensorflow
