// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/utils/cycle_detector.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

#define DEBUG_TYPE "disc-dot-merge"

// TODO: this pass shares some common functions with lhlo-fusion-pass. The dot
// merge can actually be regarded as a kind of horizontal fusion. We will
// reassess the necessity of merging this pass into the fusion-pass after fixing
// the bugs in horizontal fusion functions in lhlo-fusion-pass.

namespace mlir {
namespace disc_ral {
namespace {
  
llvm::Optional<int32_t> TryMergeNode(GraphCycles* graph_cycles, int32_t a,
                                     int32_t b) {
  if (graph_cycles == nullptr) {
    return llvm::None;
  }
  bool has_edge_inserted_a2b = false;
  if (!graph_cycles->HasEdge(a, b) && !graph_cycles->HasEdge(b, a)) {
    has_edge_inserted_a2b = graph_cycles->InsertEdge(a, b);
    if (!has_edge_inserted_a2b) {
      // Cannot merge a and b as we cannot even insert an edge between a and b.
      return llvm::None;
    }
  }
  int32_t from = graph_cycles->HasEdge(a, b) ? a : b;
  int32_t to = (from == a) ? b : a;
  auto result = graph_cycles->ContractEdge(from, to);
  if (!result.hasValue() && has_edge_inserted_a2b) {
    // Restore the graph.
    graph_cycles->RemoveEdge(a, b);
  }
  return result;
}


// NOTE: this function is copied from `lhlo_fusion.cc`.
// Returns all the values touched by this op or its nested ops.
SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op) {
  SmallVector<Value, 4> values;
  op->walk([&](Operation* nest_op) {
    for (Value v : nest_op->getOperands()) {
      values.push_back(v);
    }
  });
  return values;
}

// Arrange the insert-point of `op`'s operands in the same block. Do not deal
// with cross-block operands.
void arrangeOperandsInsertPointInBlock(Operation* op) {
  for (auto operand : GetAllPossibleUsedValues(op)) {
    auto operandOp = operand.getDefiningOp();
    // Note that `isBeforeInBlock` also check whether `op` and `operandOp` are
    // in the same block.
    if ((operandOp != nullptr) && !operandOp->isBeforeInBlock(op)) {
      operandOp->moveBefore(op);
      arrangeOperandsInsertPointInBlock(operandOp);
    }
  }
}


class DotBatchingConverter {
 public:
  DotBatchingConverter(FuncOp func) : func_(func){};
  void run();

 private:
  struct DotShape {
    int64_t m_dimension;
    int64_t n_dimension;
    DimValue m_dim;
    DimValue n_dim;
    DimValue contracting_dim;
    SmallVector<DimValue> batching_dims;
    mhlo::DotDimensionNumbersAttr dimension_numbers;
    Type element_type;
  };
  
  struct DotShareInfo {
    Value shared_operand;
    int64_t lhs_contracting_dim;
    int64_t rhs_contracting_dim;

    DotShareInfo(mhlo::DotGeneralOp dot, bool shared_type){
      shared_operand = shared_type ? dot.lhs() : dot.rhs();
      auto& dim_numbers = dot.dot_dimension_numbers();
      lhs_contracting_dim = dim_numbers.getLhsBatchingDimensions();
      rhs_contracting_dim = dim_numbers.getRhsBatchingDimensions();
    }
    
    // two dot op map condition:
    //   @1 have the same one-hand-side operand
    //   @2 have the same contracting_dim of the above operand
    //   @3 have the same contracting_dim of the other hand-side operand
    // TODO: we can only use @1 & @2 as the key of the map, cause the @3 can be transposed to match.
    bool operator==(const DotShareInfo& other) const {
      return (shared_operand == other.shared_operand) &&
             (lhs_contracting_dim == other.lhs_contracting_dim) &&
             (rhs_contracting_dim == other.rhs_contracting_dim);
    }
  };
  struct DotShareInfoHash {
    std::size_t operator()(const DotShareInfo& dotShareInfo) const {
      std::size_t hash = dotShareInfo.shared_operand.hash();
      const auto& dimension_numbers = dotShareInfo.dimension_numbers;
      const auto& lhs_contracting_dims =
          dimension_numbers.getLhsContractingDimensions();
      const auto& rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(lhs_contracting_dims.begin(),
                                         lhs_contracting_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(rhs_contracting_dims.begin(),
                                         rhs_contracting_dims.end()));
      return hash;
    }
  };

  struct MergeDotCluster {
    MergeDotCluster(Operation* op, int shared_tp, int id) : shared_type(shared_tp), 
                                                            leader_op_id(id) {
      ops.push_back(op);
    }

    // Merges `other` into this cluster, and clears `other`.
    void merge(MergeDotCluster& other) {
      ops.insert(ops.end(), other.ops.begin(), other.ops.end());
      other.ops.clear();
    }

    // true for shared_lhs, false for shared_rhs
    bool shared_type;

    // ID of the representative node of this cluster.
    int leader_op_id;

    // Dot ops to be Merged.
    SmallVector<Operation*> ops;

    bool is_share_lhs(){return shared_type == true}
    bool is_share_rhs(){return shared_type == false}
  };

  DotShape getDotShape(mhlo::DotGeneralOp op);
  bool buildSharedOperandDotMap();
  bool mergeDots();
  bool applyMerge(MergeDotCluster& cluster);

 private:
  FuncOp func_;
  // A map to help to cluster dot that have shared lhs together.
  std::unordered_map<DotShareInfo, SmallVector<mhlo::DotGeneralOp>, DotShareInfoHash>
      share_lhs_dot_map;
  // A map to help to cluster dot that have shared rhs together.
  std::unordered_map<DotShareInfo, SmallVector<mhlo::DotGeneralOp>, DotShareInfoHash>
      share_rhs_dot_map;
};

void DotBatchingConverter::run() {
  if (!buildSharedOperandDotMap()) {
    return;
  }
  mergeDots();
}


DotShape DotBatchingConverter::getDotShape(mhlo::DotGeneral op){
  ShapeAnalysis analysis(func_);
  if (failed(analysis.run())) {
    // error message should be generated inside the above function call.
    return false;
  }
  DotShape dot_shape;
  Value lhs = op.lhs();
  Value rhs = op.rhs();
  // Initialize `dimension_numbers`.
  dot_shape.dimension_numbers = op.dot_dimension_numbers();
  auto lhs_batch_dims =
      dot_shape.dimension_numbers.getLhsBatchingDimensions();
  auto rhs_batch_dims =
      dot_shape.dimension_numbers.getRhsBatchingDimensions();
  assert(lhs_batch_dims.size() == rhs_batch_dims.size());
  // Initialize `batching_dims`.
  SmallVector<DimValue>& batching_dims = dot_shape.batching_dims;
  for (auto dim : lhs_batch_dims) {
    batching_dims.emplace_back(std::move(analysis.getDimValue(lhs, dim)));
  }
  // Initialize `lhs_contracting_dim`.
  auto lhs_contracting_dims =
      dot_shape.dimension_numbers.getLhsContractingDimensions();
  assert(lhs_contracting_dims.size() == 1);
  dot_shape.lhs_contracting_dim =
      analysis.getDimValue(lhs, lhs_contracting_dims[0]);
  // Initialize `rhs_contracting_dim`.
  auto rhs_contracting_dims =
      dot_shape.dimension_numbers.getRhsContractingDimensions();
  assert(rhs_contracting_dims.size() == 1);
  dot_shape.rhs_contracting_dim =
      analysis.getDimValue(rhs, rhs_contracting_dims[0]);
  // Initialize `m_dim`.
  int64_t lhs_rank = lhs.getType().cast<RankedTensorType>().getRank();
  assert(lhs_batch_dims.size() + 2 == lhs_rank);
  DenseSet<int64_t> lhs_batch_dims_set(lhs_batch_dims.begin(),
                                        lhs_batch_dims.end());
  for (int64_t i = 0; i < lhs_rank; i++) {
    if ((lhs_batch_dims_set.find(i) == lhs_batch_dims_set.end()) &&
        (i != lhs_contracting_dims[0])) {
      dot_shape.m_dim = analysis.getDimValue(lhs, i);
      dot_shape.n_dimension = i; // ! only need this value 
      break;
    }
  }
  // Initialize `n_dim`.
  int64_t rhs_rank = rhs.getType().cast<RankedTensorType>().getRank();
  assert(rhs_batch_dims.size() + 2 == rhs_rank);
  DenseSet<int64_t> rhs_batch_dims_set(rhs_batch_dims.begin(),
                                        rhs_batch_dims.end());
  auto rhs_contracting_dims =
      dot_shape.dimension_numbers.getRhsContractingDimensions();
  assert(rhs_contracting_dims.size() == 1);
  for (int64_t i = 0; i < rhs_rank; i++) {
    if ((rhs_batch_dims_set.find(i) == rhs_batch_dims_set.end()) &&
        (i != rhs_contracting_dims[0])) {
      dot_shape.n_dim = analysis.getDimValue(rhs, i);
      dot_shape.n_dimension = i; // ! only need this value
      break;
    }
  }
  // Initialize `element_type`.
  dot_shape.element_type =
      op.getType().cast<RankedTensorType>().getElementType();
  return dot_shape;
}


bool DotBatchingConverter::buildSharedOperandDotMap() {
  func_.walk([&](mhlo::DotGeneralOp op) {
    bool share_lhs = true;
    bool share_rhs = false;
    DotShareInfo lhs_share_info = DotShareInfo(op, share_lhs);
    DotShareInfo rhs_share_info = DotShareInfo(op, share_rhs);
    auto& shared_lhs_op_list = share_lhs_dot_map[lhs_share_info];
    shared_lhs_op_list.push_back(op);
    auto& shared_rhs_op_list = share_rhs_dot_map[rhs_share_info];
    shared_rhs_op_list.push_back(op);

  });
  return true;
}

bool DotBatchingConverter::applyMerge(MergeDotCluster& cluster) {
  auto& ops = cluster.ops;
  auto loc = ops.front()->getLoc();
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  auto foremost_dot = dyn_cast<mhlo::DotGeneralOp>(foremost);
  // We use the foremost dot to create the builder. Thus we only need to reorder
  // the operands of some newly created ops, rather users of them.
  OpBuilder builder(foremost_dot);
  auto orig_lhs_type =
      foremost_dot.lhs().getType().dyn_cast<RankedTensorType>();
  auto orig_rhs_type =
      foremost_dot.rhs().getType().dyn_cast<RankedTensorType>();
  auto orig_result_type = foremost_dot.getType().dyn_cast<RankedTensorType>();
  SmallVector<Value, 4> lhs_operands;
  SmallVector<Value, 4> rhs_operands;
  int64_t concat_dim = cluster.is_share_lhs() ? 
                        getDotShape(foremost_dot).n_dimension : 
                        getDotShape(foremost_dot).m_dimension;
  bool is_dynamic_shape = false;
  int64_t concat_dim_sum = 0;
  Value lhs;
  Value rhs;
  if(cluster.is_share_lhs()){
    for (auto op : ops) {
      mhlo::DotGeneralOp dot = dyn_cast<mhlo::DotGeneralOp>(op);
      DotShareInfo dot_shape = getDotShape(dot);
      auto rhs_type = op.rhs().getType().dyn_cast<RankedTensorType>();
      auto concat_dim_size = rhs_type.getDimSize(concat_dim)
      if (concat_dim_size == -1) {
        is_dynamic_shape = true;
      } else {
        concat_dim_sum += concat_dim_size;

      }
      lhs_operands.push_back(dot.lhs());
    }
    //concat rhs
    auto rhs_rank = orig_rhs_type.getRank();
    SmallVector<int64_t, 4> rhs_shapes(rhs_rank, ShapedType::kDynamicSize);
    for (int64_t i = 0; i < rhs_rank; i++) {
      if(i == concat_dim) {
        if (is_dynamic_shape) {
          result_shapes[i] = -1;
        } else {
          result_shapes[i] = concat_dim_sum;
        }
      } else {
        result_shapes[i] = orig_rhs_type.getDimSize(i);  
      } 
    }
    auto rhs_type =
        RankedTensorType::get(rhs_shapes, orig_rhs_type.getElementType());
    rhs = builder.create<mhlo::ConcatenateOp>(loc, rhs_type, rhs_operands,
                                                    concat_dim);
    lhs = formost_dot.lhs();
  } else if (cluster.is_share_rhs()) {
    for (auto op : ops) {
      mhlo::DotGeneralOp dot = dyn_cast<mhlo::DotGeneralOp>(op);
      DotShareInfo dot_shape = getDotShape(dot);
      auto lhs_type = op.lhs().getType().dyn_cast<RankedTensorType>();
      auto concat_dim_size = lhs_type.getDimSize(concat_dim)
      if (concat_dim_size == -1) {
        is_dynamic_shape = true;
      } else {
        concat_dim_sum += concat_dim_size;
      }
      lhs_operands.push_back(dot.lhs());
    }
    //concat lhs
    auto lhs_rank = orig_lhs_type.getRank();
    SmallVector<int64_t, 4> lhs_shapes(lhs_rank, ShapedType::kDynamicSize);
    for (int64_t i = 0; i < lhs_rank; i++) {
      if (i == concat_dim) {
        if (is_dynamic_shape) {
          result_shapes[i] = -1;
        } else {
          result_shapes[i] = concat_dim_sum;
        }
      } else {
        result_shapes[i] = orig_lhs_type.getDimSize(i);  
      } 
    }
    auto lhs_type =
        RankedTensorType::get(lhs_shapes, orig_lhs_type.getElementType());
    lhs = builder.create<mhlo::ConcatenateOp>(loc, lhs_type, lhs_operands,
                                                    concat_dim);
    rhs = formost_dot.rhs();
  }
  
  // Result type.
  auto result_rank = orig_result_type.getRank();
  SmallVector<int64_t, 4> result_shapes(result_rank, ShapedType::kDynamicSize);
  for (int64_t i = 0; i < result_rank; i++) {
    if (i == concat_dim) {
      if (is_dynamic_shape) {
        result_shapes[i] = -1;
      } else {
        result_shapes[i] = concat_dim_sum;
      }
    } else {
      result_shapes[i] = orig_result_type.getDimSize(i);  
    }   
  }
  auto result_type =
      RankedTensorType::get(result_shapes, orig_result_type.getElementType());
  // Build dot dimension numbers.
  auto dim_numbers = foremost_dot.dot_dimension_numbers();

  SmallVector<int64_t> lhs_batching_dims;
  auto lhs_batch = dim_numbers.getLhsBatchingDimensions();
  lhs_batching_dims.insert(lhs_batching_dims.end(), lhs_batch.begin(),
                           lhs_batch.end());

  SmallVector<int64_t> rhs_batching_dims;
  auto rhs_batch = dim_numbers.getRhsBatchingDimensions();
  rhs_batching_dims.insert(rhs_batching_dims.end(), rhs_batch.begin(),
                           rhs_batch.end());

  SmallVector<int64_t> lhs_contracting_dims;
  auto lhs_contract = dim_numbers.getLhsContractingDimensions();
  for (auto& val : lhs_contract) {
    lhs_contracting_dims.push_back(val);
  }

  SmallVector<int64_t> rhs_contracting_dims;
  auto rhs_contract = dim_numbers.getRhsContractingDimensions();
  for (auto& val : rhs_contract) {
    rhs_contracting_dims.push_back(val);
  }

  auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), lhs_batching_dims, rhs_batching_dims,
      lhs_contracting_dims, rhs_contracting_dims);
  
  // Build DotGeneralOp
  Value merged_dot = builder.create<mhlo::DotGeneralOp>(
      loc, result_type, lhs, rhs, dot_dimension_attr, nullptr);  

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // Build slice and reshape op for each of the original dot op, and replace the
  // dot.
  for (int64_t i = 0; i < ops.size(); i++) {
    mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
    if (!is_dynamic_shape) {
      // Use static-dim ops.
      int64_t concat_dim_start = 0;
      SmallVector<int64_t> starts(result_rank);
      SmallVector<int64_t> ends(result_rank);
      for (int64_t i = 0; i < result_rank; i++) {
        if (i == concat_dim) {
          starts[i] = concat_dim_start;
          ends[i] = concat_dim_start +
                    op.getType().dyn_cast<RankedTensorType>().getDimSize(concat_dim);
          concat_dim_start = ends[i];
        } else {
          starts[i] = 0;
          ends[i] = orig_dot_type.getDimSize(i);
        }
      }
      SmallVector<int64_t> strides(result_rank, 1);
      auto slice = builder.create<mhlo::SliceOp>(
          loc, merged_dot, GetI64ElementsAttr(starts, &builder),
          GetI64ElementsAttr(ends, &builder),
          GetI64ElementsAttr(strides, &builder));
      op->replaceAllUsesWith(slice);
    } else {
      // Use dynamic-dim ops.
      Value concat_dim_start = builder.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value, 4> stride_values(result_rank, one);
      SmallVector<Value, 4> begin_values(result_rank, zero);
      SmallVector<Value, 4> end_values;
      for (int64_t i = 0; i < result_rank; j++) {
        if (i == concat_dim) {
          begin_values[i] = concat_dim_start;
          end_values[i] = 
            builder.create<arith::AddIOp>(loc, 
                                          begin_values[i], 
                                          builder.create<tensor::DimOp>(loc, op, concat_dim));
          concat_dim_start = end_values[i];
        } else {
          begin_values[i] = zero;
          end_values[i] = builder.create<tensor::DimOp>(loc, op, i);
        }
      }

      
      auto index_ty = builder.getIndexType();
      auto start_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(begin_values.size())},
                                index_ty),
          begin_values);
      auto end_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(end_values.size())},
                                index_ty),
          end_values);
      auto stride_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(stride_values.size())},
                                index_ty),
          stride_values);
      SmallVector<int64_t, 4> slice_shapes(result_rank,
                                           ShapedType::kDynamicSize);
      // slice_shapes is exactly the same as the op which has been concatenated.
      for (int64_t i = 0; i < result_rank; i++) {
        slice_shapes[i] = op.getType().dyn_cast<RankedTensorType>().getDimSize(i);
      }
      auto slice_type =
          RankedTensorType::get(slice_shapes, orig_dot_type.getElementType());
      auto dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, merged_dot, start_indices, end_indices,
          stride_indices);
      
      op->replaceAllUsesWith(dyn_slice);
    }
  }

  // No longer need the original dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }

  return true;
}

bool DotBatchingConverter::mergeDots() {
  // Dot ops in `shared_operand_dot_map` can be merged together if the merge
  // does not introduce cycle.

  // Form cycle detector.
  std::vector<Operation*> func_op_list;
  DenseMap<Operation*, int64_t> func_op_to_id;
  func_.walk([&](Operation* op) {
    func_op_to_id.try_emplace(op, func_op_list.size());
    func_op_list.push_back(op);
  });
  GraphCycles cycle_detector(func_op_list.size());
  for (int node_id = 0; node_id < func_op_list.size(); ++node_id) {
    Operation* op = func_op_list[node_id];
    for (Value operand : op->getOperands()) {
      Operation* operand_op = operand.getDefiningOp();
      if (operand_op == nullptr) {
        // skip block argument
        continue;
      }
      auto iter = func_op_to_id.find(operand_op);
      assert(iter != func_op_to_id.end());
      cycle_detector.InsertEdge(iter->second, node_id);
    }
  }

  bool share_lhs = true;
  bool share_rhs = false;
  
  // Find share lhs operand clusters.
  SmallVector<MergeDotCluster> share_lhs_cluster;
  for (auto& share_dots : share_lhs_dot_map) {
    auto ops = share_dots.second;
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<MergeDotCluster> clusters;
    for (auto op : ops) {
      clusters.emplace_back(op.getOperation(),
                            share_lhs,
                            func_op_to_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& merged = clusters[i];
      if (merged.ops.empty()) {
        continue;
      }
      for (int64_t j = 0; j < clusters.size(); j++) {
        auto& to_merge = clusters[j];
        if (to_merge.ops.empty()) {
          continue;
        }
        // Only merge `dot`s within the same block.
        if (merged.ops.front()->getBlock() !=
            to_merge.ops.front()->getBlock()) {
          continue;
        }
        // Try merge.
        int64_t merged_id = merged.leader_op_id;
        int64_t to_merge_id = to_merge.leader_op_id;
        auto optional_merged_id =
            TryMergeNode(&cycle_detector, merged_id, to_merge_id);
        if (!optional_merged_id.hasValue()) {
          // It forms a cycle.
          continue;  // commented for debugging.
        }
        merged.merge(to_merge);
        merged.leader_op_id = *optional_merged_id;
      }
    }
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        share_lhs_cluster.push_back(cluster);
      }
    }
  }

  // Find share rhs operand clusters.
  SmallVector<MergeDotCluster> share_rhs_cluster;
  for (auto& share_dots : share_rhs_dot_map) {
    auto ops = share_dots.second;
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<MergeDotCluster> clusters;
    for (auto op : ops) {
      clusters.emplace_back(op.getOperation(),
                            share_rhs,
                            func_op_to_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& merged = clusters[i];
      if (merged.ops.empty()) {
        continue;
      }
      for (int64_t j = 0; j < clusters.size(); j++) {
        auto& to_merge = clusters[j];
        if (to_merge.ops.empty()) {
          continue;
        }
        // Only merge `dot`s within the same block.
        if (merged.ops.front()->getBlock() !=
            to_merge.ops.front()->getBlock()) {
          continue;
        }
        // Try merge.
        int64_t merged_id = merged.leader_op_id;
        int64_t to_merge_id = to_merge.leader_op_id;
        auto optional_merged_id =
            TryMergeNode(&cycle_detector, merged_id, to_merge_id);
        if (!optional_merged_id.hasValue()) {
          // It forms a cycle.
          continue;  // commented for debugging.
        }
        merged.merge(to_merge);
        merged.leader_op_id = *optional_merged_id;
      }
    }
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        share_rhs_cluster.push_back(cluster);
      }
    }
  }
  // Apply lhs merge
  for (auto& cluster : share_lhs_cluster) {
    applyMerge(cluster);
  }
  // Apply rhs merge.
  for (auto& cluster : share_rhs_cluster) {
    applyMerge(cluster);
  }
  
  
  return true;
}

struct DiscDotMergePass : public DiscDotMergePassBase<DiscDotMergePass> {
  DiscDotMergePass()
      : DiscDotMergePassBase<DiscDotMergePass>::DiscDotMergePassBase() {}
  void runOnOperation() override;

 private:
  void dotBatchingSimplifier(FuncOp& func);
};

void DiscDotMergePass::runOnOperation() {
  FuncOp func = getOperation();

  dotBatchingSimplifier(func);
}

void DiscDotMergePass::dotBatchingSimplifier(FuncOp& func) {
  DotBatchingConverter(func).run();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscDotMergePass() {
  return std::make_unique<DiscDotMergePass>();
}

}  // namespace disc_ral
}  // namespace mlir