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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.h"

#include <algorithm>

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IntegerSet.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

Status CheckLayoutIsSupported(const Layout& layout) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc mht_0(mht_0_v, 206, "", "./tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.cc", "CheckLayoutIsSupported");

  // Currently we support small mesh rank for arbitrary layout.
  if (layout.mesh().rank() > 3)
    return errors::InvalidArgument("Large mesh rank size is not supported",
                                   layout.ToString());

  return Status::OK();
}

Status ValidateShapeAndGetNewShape(
    const llvm::SmallVector<int64_t, 4>& op_shape, const Layout& layout,
    llvm::SmallVectorImpl<int64_t>& new_random_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc mht_1(mht_1_v, 220, "", "./tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.cc", "ValidateShapeAndGetNewShape");

  TF_RETURN_IF_ERROR(CheckLayoutIsSupported(layout));

  // Validate that sharding of random op is compatible with it's user defined
  // shape and calculate new shape of local random op.
  const auto op_sharding = layout.num_shards();
  new_random_shape.reserve(op_shape.size());

  if (op_sharding.size() != op_shape.size())
    return errors::InvalidArgument(
        "Sharding dimension of random op does not match rank of the "
        "random op. Received sharding: ",
        layout.ToString());

  for (int i = 0; i < op_sharding.size(); ++i) {
    const auto dimension_sharding = op_sharding[i];
    const auto op_dimension_size = op_shape[i];
    if (op_dimension_size % dimension_sharding != 0) {
      return errors::InvalidArgument(
          "Sharding of random op incompatible with shape. Received "
          "sharding: ",
          layout.ToString());
    }
    new_random_shape.emplace_back(op_dimension_size / dimension_sharding);
  }
  return Status::OK();
}

// Get a device seed for this layout and device_id.
//
// The computation will be inserted directly after the mesh coordinate
// computation in the current cluster. First we search for a Squeeze with the
// attribute kDeviceSeedForMeshDims = layout.mesh_dims
// If it exists, we return that, otherwise we insert the ops to compute a device
// seed.
StatusOr<mlir::Value> GetDeviceSeed(const Layout& layout, mlir::Operation* op) {
  // We need both a set, to check for membership and a vector that we sort
  // to use as the attribute attached to the squeeze op.
  llvm::SmallVector<int32_t, 4> layout_dims;
  llvm::SmallSet<int32_t, 4> layout_dims_set;
  for (const ShardingSpec& spec : layout.sharding_specs()) {
    if (Layout::IsUnshardedSpec(spec)) continue;
    layout_dims.emplace_back(
        layout.mesh().GetMeshDimIndexWithName(spec.sharding_spec()));
    layout_dims_set.insert(layout_dims.back());
  }
  llvm::sort(layout_dims);

  mlir::tf_device::ClusterOp cluster =
      op->getParentOfType<mlir::tf_device::ClusterOp>();
  if (!cluster)
    return errors::InvalidArgument(
        "random op not in ClusterOp when it should be");

  for (mlir::TF::SqueezeOp squeeze : cluster.getOps<mlir::TF::SqueezeOp>())
    if (squeeze->hasAttrOfType<mlir::DenseIntElementsAttr>(
            kDeviceSeedForMeshDims) &&
        std::equal(layout_dims.begin(), layout_dims.end(),
                   squeeze
                       ->getAttrOfType<mlir::DenseIntElementsAttr>(
                           kDeviceSeedForMeshDims)
                       .getValues<uint32_t>()
                       .begin()))
      return squeeze.output();

  TF_ASSIGN_OR_RETURN(mlir::Value mesh_coordinates,
                      GetMeshCoordinatesFromCluster(cluster));

  mlir::OpBuilder builder(cluster.getContext());
  builder.setInsertionPointAfterValue(mesh_coordinates);

  // mesh_coordinates is a [1, mesh.rank()] shaped tensor containing the current
  // mesh coordinates of the device.
  // If there are 4 mesh dimensions [w, x, y, z] and only [w, x, z] are used in
  // this layout then one way of getting the device id would be
  // w_coord + x_coord*size_w + z_coord*size_x*size_w
  // Note that only the dims in layout_dims count.
  llvm::SmallVector<uint32_t, 4> multipliers(layout.mesh().rank(), 0);

  // By starting with 65536, we effective perform a left shift of the id by
  // 16 bits.
  int32_t running_product = 65536;
  for (int i = 0; i < layout.mesh().rank(); ++i) {
    if (layout_dims_set.contains(i)) {
      multipliers[i] = running_product;
      running_product = running_product * layout.mesh().dim_sizes()[i];
    }
  }

  mlir::RankedTensorType const_type = mlir::RankedTensorType::get(
      {static_cast<int64>(multipliers.size()), 1}, builder.getIntegerType(32));
  mlir::Attribute const_attr =
      mlir::DenseIntElementsAttr::get(const_type, multipliers);
  mlir::Value multiplier =
      builder.create<mlir::TF::ConstOp>(cluster.getLoc(), const_attr).output();

  const mlir::RankedTensorType one_by_one =
      mlir::RankedTensorType::get({1, 1}, builder.getIntegerType(32));

  mlir::Value seed = builder.create<mlir::TF::MatMulOp>(
      cluster.getLoc(), one_by_one, mesh_coordinates, multiplier);

  // Largest prime in 16 bits.
  mlir::Value prime = CreateIntScalarConst(
      /*value=*/65521, builder, cluster.getLoc(), /*use_int64=*/false);

  mlir::Value seed_plus_prime =
      builder
          .create<mlir::TF::AddV2Op>(cluster.getLoc(), one_by_one, seed, prime)
          .z();

  mlir::TF::SqueezeOp squeeze = builder.create<mlir::TF::SqueezeOp>(
      cluster.getLoc(),
      mlir::RankedTensorType::get({}, builder.getIntegerType(32)),
      seed_plus_prime, builder.getI64ArrayAttr({0, 1}));

  squeeze->setAttr(kDeviceSeedForMeshDims,
                   builder.getI32TensorAttr(layout_dims));

  return squeeze.output();
}

// Compute the new local shape for SPMD expansion and ensure it is valid.
template <typename RandomOp>
StatusOr<llvm::SmallVector<int64_t, 4>> GetNewLocalShape(mlir::Operation* op,
                                                         const Layout& layout) {
  auto random_op = llvm::cast<RandomOp>(op);
  llvm::SmallVector<int64_t, 4> op_shape;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(random_op.shape(), &op_shape));

  // Validate that sharding of random op is compatible with it's user defined
  // shape and calculate new shape of local random op.
  llvm::SmallVector<int64_t, 4> new_random_shape;
  TF_RETURN_IF_ERROR(
      ValidateShapeAndGetNewShape(op_shape, layout, new_random_shape));
  return new_random_shape;
}

// Calculate the new local seed
template <typename RandomOp>
StatusOr<mlir::Value> ComputeNewSeed(mlir::OpBuilder& builder,
                                     mlir::Operation* op, const Layout& layout,
                                     mlir::Location& location,
                                     mlir::Value op_seed) {
  TF_ASSIGN_OR_RETURN(auto device_id_seed, GetDeviceSeed(layout, op));
  mlir::Type seed_type =
      op_seed.getType().cast<mlir::TensorType>().getElementType();

  device_id_seed = builder.create<mlir::TF::CastOp>(
      location, mlir::RankedTensorType::get({}, seed_type), device_id_seed);

  mlir::Value seed_xor =
      builder.create<mlir::TF::BitwiseXorOp>(location, op_seed, device_id_seed);
  return seed_xor;
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV1(const Layout& layout,
                                                         mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(auto seed_xor,
                      ComputeNewSeed<RandomOp>(builder, op, layout, location,
                                               random_op.seed()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape,
      op->getResult(0).getType().cast<mlir::TensorType>().getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);
  // TODO(zhonglinhan) : check different input for StatelessRandomUniformInt
  auto local_random = builder.create<RandomOp>(location, new_random_type,
                                               new_shape_value, seed_xor);
  op->getResult(0).replaceAllUsesWith(local_random.output());
  op->erase();
  return local_random.getOperation();
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV2(const Layout& layout,
                                                         mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(
      auto seed_xor,
      ComputeNewSeed<RandomOp>(builder, op, layout, location, random_op.key()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape,
      op->getResult(0).getType().cast<mlir::TensorType>().getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);

  auto local_random =
      builder.create<RandomOp>(location, new_random_type, new_shape_value,
                               seed_xor, random_op.counter(), random_op.alg());
  op->getResult(0).replaceAllUsesWith(local_random.output());
  op->erase();
  return local_random.getOperation();
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV2Range(
    const Layout& layout, mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(
      auto seed_xor,
      ComputeNewSeed<RandomOp>(builder, op, layout, location, random_op.key()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape,
      op->getResult(0).getType().cast<mlir::TensorType>().getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);

  auto local_random = builder.create<RandomOp>(
      location, new_random_type, new_shape_value, seed_xor, random_op.counter(),
      random_op.alg(), random_op.minval(), random_op.maxval());
  op->getResult(0).replaceAllUsesWith(local_random.output());
  op->erase();
  return local_random.getOperation();
}

}  // namespace

StatusOr<mlir::Operation*> RandomOpSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSrandom_op_spmd_expanderDTcc mht_2(mht_2_v, 479, "", "./tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.cc", "RandomOpSPMDExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout)
    return errors::InvalidArgument(
        "layout of Random op must be known before SPMD expansion.");

  // For fully replicated random ops, all devices have the same random
  // value. As so, SPMD expansion is a no-op.
  if (layout->IsFullyReplicated()) return op;
  if (llvm::isa<mlir::TF::StatelessRandomUniformOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessRandomUniformOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformFullIntOp>(op)) {
    return CreatedShardedLocalRandomOpV1<
        mlir::TF::StatelessRandomUniformFullIntOp>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomNormalOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessRandomNormalOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessTruncatedNormalOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessTruncatedNormalOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformFullIntV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<
        mlir::TF::StatelessRandomUniformFullIntV2Op>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomNormalV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<mlir::TF::StatelessRandomNormalV2Op>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessTruncatedNormalV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<
        mlir::TF::StatelessTruncatedNormalV2Op>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<mlir::TF::StatelessRandomUniformV2Op>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformIntV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2Range<
        mlir::TF::StatelessRandomUniformIntV2Op>(*layout, op);
  }
  return errors::Unimplemented(absl::StrCat(
      "SPMD expansion for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
RandomOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> output_layouts;

  // For random op, input is always replicated and we always respect layouts
  // from consumers.
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
RandomOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> input_layouts;

  // For random op, default the input layout as replicated layout.
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(i)));
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
