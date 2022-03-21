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

// NOTE: this pass shares some common functions with lhlo-fusion-pass. The dot
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

void buildBlockGraphCycles(Block* block,
                           std::unique_ptr<GraphCycles> cycle_detector,
                           DenseMap<Operation*, int64_t>& op_to_id) {
  std::vector<Operation*> op_list;
  op_to_id.clear();
  for (Operation& op : *block) {
    op_to_id.try_emplace(&op, op_list.size());
    op_list.push_back(&op);
  }
  cycle_detector.reset(new GraphCycles(op_list.size()));
  for (int64_t node_id = 0; node_id < op_list.size(); node_id++) {
    Operation* op = op_list[node_id];
    for (Value operand : GetAllPossibleUsedValues(op)) {
      Operation* operand_op = operand.getDefiningOp();
      // Only consider the operand_op inside the target block.
      auto iter = op_to_id.find(operand_op);
      if (iter == op_to_id.end()) {
        continue;
      }
      cycle_detector->InsertEdge(iter->second, node_id);
    }
  }
}

struct DotCluster {
  DotCluster(Operation* op, int op_id) : leader_op_id(op_id) {
    ops.push_back(op);
  }

  // Merges `other` into this cluster, and clears `other`.
  void merge(DotCluster& other) {
    ops.insert(ops.end(), other.ops.begin(), other.ops.end());
    other.ops.clear();
  }

  // ID of the representative node of this cluster.
  int leader_op_id;

  // Dot ops to be batched.
  SmallVector<Operation*> ops;
};

class DotBatchingConverter {
 public:
  DotBatchingConverter(FuncOp func) : func_(func){};
  bool run();

 public:
  struct DotShape {
    DimValue m_dim;
    DimValue n_dim;
    DimValue contracting_dim;
    SmallVector<DimValue> batching_dims;
    mhlo::DotDimensionNumbersAttr dimension_numbers;
    Type element_type;
    bool operator==(const DotShape& other) const {
      return (m_dim == other.m_dim) && (n_dim == other.n_dim) &&
             (contracting_dim == other.contracting_dim) &&
             (batching_dims == other.batching_dims) &&
             (dimension_numbers == other.dimension_numbers) &&
             (element_type == other.element_type);
    }
  };

  struct DotShapeHash {
    std::size_t operator()(const DotShape& dotShape) const {
      std::size_t hash = dotShape.m_dim.hash();
      hash = llvm::hash_combine(hash, dotShape.n_dim.hash());
      hash = llvm::hash_combine(hash, dotShape.contracting_dim.hash());
      for (const auto& d : dotShape.batching_dims) {
        hash = llvm::hash_combine(hash, d.hash());
      }
      const auto& dimension_numbers = dotShape.dimension_numbers;
      const auto& lhs_batch_dims = dimension_numbers.getLhsBatchingDimensions();
      const auto& rhs_batch_dims = dimension_numbers.getRhsBatchingDimensions();
      const auto& lhs_contracting_dims =
          dimension_numbers.getLhsContractingDimensions();
      const auto& rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(lhs_batch_dims.begin(),
                                                         lhs_batch_dims.end()));
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(rhs_batch_dims.begin(),
                                                         rhs_batch_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(lhs_contracting_dims.begin(),
                                         lhs_contracting_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(rhs_contracting_dims.begin(),
                                         rhs_contracting_dims.end()));
      hash = llvm::hash_combine(hash, mlir::hash_value(dotShape.element_type));
      return hash;
    }
  };

  using DotShapeEqualMap =
      std::unordered_map<DotShape, SmallVector<mhlo::DotGeneralOp>,
                         DotShapeHash>;

 private:
  bool buildShapedDotMap(Block* block, ShapeAnalysis& analysis,
                         DotShapeEqualMap& equal_shape_map);
  bool batchingDots(Block* block, const DotShapeEqualMap& equal_shape_map);
  Value expandDim0(OpBuilder& builder, Location& loc, Value value);
  bool applyBatching(DotCluster& cluster);

 private:
  FuncOp func_;
};

bool DotBatchingConverter::run() {
  ShapeAnalysis analysis(func_);
  if (failed(analysis.run())) {
    LLVM_DEBUG(llvm::dbgs() << "ShapeAnalysis failes for dot merge.\n");
    return false;
  }

  SmallVector<Block*> blocks;
  func_.walk([&](Block* block) { blocks.push_back(block); });

  for (Block* block : blocks) {
    // A map to help to cluster dots with same shape and dim-numbers together.
    DotShapeEqualMap equal_shape_map;
    if (!buildShapedDotMap(block, analysis, equal_shape_map)) {
      continue;
    }
    batchingDots(block, equal_shape_map);
  }

  return true;
}

bool DotBatchingConverter::buildShapedDotMap(
    Block* block, ShapeAnalysis& analysis, DotShapeEqualMap& equal_shape_map) {
  block->walk([&](mhlo::DotGeneralOp op) {
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
    // Initialize `contracting_dim`.
    auto lhs_contracting_dims =
        dot_shape.dimension_numbers.getLhsContractingDimensions();
    assert(lhs_contracting_dims.size() == 1);
    dot_shape.contracting_dim =
        analysis.getDimValue(lhs, lhs_contracting_dims[0]);
    // Initialize `m_dim`.
    int64_t lhs_rank = lhs.getType().cast<RankedTensorType>().getRank();
    assert(lhs_batch_dims.size() + 2 == lhs_rank);
    DenseSet<int64_t> lhs_batch_dims_set(lhs_batch_dims.begin(),
                                         lhs_batch_dims.end());
    for (int64_t i = 0; i < lhs_rank; i++) {
      if ((lhs_batch_dims_set.find(i) == lhs_batch_dims_set.end()) &&
          (i != lhs_contracting_dims[0])) {
        dot_shape.m_dim = analysis.getDimValue(lhs, i);
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
        break;
      }
    }
    // Initialize `element_type`.
    dot_shape.element_type =
        op.getType().cast<RankedTensorType>().getElementType();

    auto& op_list = equal_shape_map[std::move(dot_shape)];
    op_list.push_back(op);
  });
  return true;
}

// Expand dim 0. We use reshape op to expand it currently. We plan to use
// tensor.ExpandOp in the future.
Value DotBatchingConverter::expandDim0(OpBuilder& builder, Location& loc,
                                       Value value) {
  auto type = value.getType().dyn_cast<RankedTensorType>();
  if (!type) {
    return nullptr;
  }
  SmallVector<int64_t, 4> result_dims;
  result_dims.push_back(1);
  bool is_static = true;
  for (int64_t i = 0; i < type.getRank(); i++) {
    int64_t dim = type.getDimSize(i);
    is_static &= (dim != ShapedType::kDynamicSize);
    result_dims.push_back(dim);
  }
  auto result_type = RankedTensorType::get(result_dims, type.getElementType());
  if (is_static) {
    return builder.create<mhlo::ReshapeOp>(loc, result_type, value);
  }
  SmallVector<Value, 4> dims;
  dims.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
  for (int64_t i = 0; i < type.getRank(); i++) {
    dims.push_back(builder.create<tensor::DimOp>(loc, value, i));
  }
  auto result_shape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(dims.size())},
                            builder.getIndexType()),
      dims);
  auto dyn_reshape = builder.create<mhlo::DynamicReshapeOp>(
      loc, result_type, value, result_shape);
  return dyn_reshape;
}

bool DotBatchingConverter::applyBatching(DotCluster& cluster) {
  auto& ops = cluster.ops;
  auto loc = ops.front()->getLoc();
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  // Move all dot ops, and their consumers if necessary, before the original
  // foremost dot. This makes sure that the newly created ops in this function
  // dominates their uses.
  for (auto op : ops) {
    if (foremost == op) {
      continue;
    }
    op->moveBefore(foremost);
    arrangeOperandsInsertPointInBlock(op);
  }
  auto last_dot = dyn_cast<mhlo::DotGeneralOp>(foremost);
  // We use the foremost dot to create the builder. Thus we only need to reorder
  // the operands of some newly created ops, rather users of them.
  OpBuilder builder(last_dot);
  auto orig_lhs_type = last_dot.lhs().getType().dyn_cast<RankedTensorType>();
  auto orig_rhs_type = last_dot.rhs().getType().dyn_cast<RankedTensorType>();
  auto orig_dot_type = last_dot.getType().dyn_cast<RankedTensorType>();
  SmallVector<Value, 4> lhs_operands;
  SmallVector<Value, 4> rhs_operands;
  for (auto op : ops) {
    mhlo::DotGeneralOp dot = dyn_cast<mhlo::DotGeneralOp>(op);
    auto lhs_expand = expandDim0(builder, loc, dot.lhs());
    auto rhs_expand = expandDim0(builder, loc, dot.rhs());
    if (!lhs_expand || !rhs_expand) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to expand dim for dot merge.\n");
      return false;
    }
    lhs_operands.push_back(lhs_expand);
    rhs_operands.push_back(rhs_expand);
  }
  auto concat_dim = builder.getI64IntegerAttr(0);
  // Concat lhs.
  auto lhs_rank = orig_lhs_type.getRank() + 1;
  SmallVector<int64_t, 4> lhs_shapes(lhs_rank, ShapedType::kDynamicSize);
  lhs_shapes[0] = ops.size();
  for (int64_t i = 1; i < lhs_rank; i++) {
    lhs_shapes[i] = orig_lhs_type.getDimSize(i - 1);
  }
  auto lhs_type =
      RankedTensorType::get(lhs_shapes, orig_lhs_type.getElementType());
  Value lhs = builder.create<mhlo::ConcatenateOp>(loc, lhs_type, lhs_operands,
                                                  concat_dim);
  // Concat rhs.
  auto rhs_rank = orig_rhs_type.getRank() + 1;
  SmallVector<int64_t, 4> rhs_shapes(rhs_rank, ShapedType::kDynamicSize);
  rhs_shapes[0] = ops.size();
  for (int64_t i = 1; i < rhs_rank; i++) {
    rhs_shapes[i] = orig_rhs_type.getDimSize(i - 1);
  }
  auto rhs_type =
      RankedTensorType::get(rhs_shapes, orig_rhs_type.getElementType());
  Value rhs = builder.create<mhlo::ConcatenateOp>(loc, rhs_type, rhs_operands,
                                                  concat_dim);
  // Result type.
  auto result_rank = orig_dot_type.getRank() + 1;
  SmallVector<int64_t, 4> result_shapes(result_rank, ShapedType::kDynamicSize);
  result_shapes[0] = ops.size();
  for (int64_t i = 1; i < result_rank; i++) {
    result_shapes[i] = orig_dot_type.getDimSize(i - 1);
  }
  auto result_type =
      RankedTensorType::get(result_shapes, orig_dot_type.getElementType());
  // Build dot dimension numbers.
  auto dim_numbers = last_dot.dot_dimension_numbers();

  SmallVector<int64_t> lhs_batching_dims;
  auto lhs_batch = dim_numbers.getLhsBatchingDimensions();
  lhs_batching_dims.push_back(0);
  lhs_batching_dims.insert(lhs_batching_dims.end(), lhs_batch.begin(),
                           lhs_batch.end());

  SmallVector<int64_t> rhs_batching_dims;
  auto rhs_batch = dim_numbers.getRhsBatchingDimensions();
  rhs_batching_dims.push_back(0);
  rhs_batching_dims.insert(rhs_batching_dims.end(), rhs_batch.begin(),
                           rhs_batch.end());

  SmallVector<int64_t> lhs_contracting_dims;
  auto lhs_contract = dim_numbers.getLhsContractingDimensions();
  for (auto& val : lhs_contract) {
    lhs_contracting_dims.push_back(val + 1);
  }

  SmallVector<int64_t> rhs_contracting_dims;
  auto rhs_contract = dim_numbers.getRhsContractingDimensions();
  for (auto& val : rhs_contract) {
    rhs_contracting_dims.push_back(val + 1);
  }

  auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), lhs_batching_dims, rhs_batching_dims,
      lhs_contracting_dims, rhs_contracting_dims);

  // Create batched dot.
  Value batched_dot = builder.create<mhlo::DotGeneralOp>(
      loc, result_type, lhs, rhs, dot_dimension_attr, nullptr);

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // Build slice and reshape op for each of the original dot op, and replace the
  // dot.
  for (int64_t i = 0; i < ops.size(); i++) {
    mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
    if (orig_dot_type.getNumDynamicDims() == 0) {
      // Use static-dim ops.
      SmallVector<int64_t> starts(result_rank, 0);
      starts[0] = i;
      SmallVector<int64_t> ends(result_rank);
      ends[0] = i + 1;
      for (int64_t i = 1; i < result_rank; i++) {
        ends[i] = orig_dot_type.getDimSize(i - 1);
      }
      SmallVector<int64_t> strides(result_rank, 1);
      auto slice = builder.create<mhlo::SliceOp>(
          loc, batched_dot, GetI64ElementsAttr(starts, &builder),
          GetI64ElementsAttr(ends, &builder),
          GetI64ElementsAttr(strides, &builder));
      auto reshape = builder.create<mhlo::ReshapeOp>(loc, orig_dot_type, slice);
      op->replaceAllUsesWith(reshape);
    } else {
      // Use dynamic-dim ops.
      SmallVector<Value, 4> stride_values(result_rank, one);
      SmallVector<Value, 4> begin_values(result_rank, zero);
      begin_values[0] = builder.create<arith::ConstantIndexOp>(loc, i);
      SmallVector<Value, 4> end_values;
      end_values.push_back(builder.create<arith::ConstantIndexOp>(loc, i + 1));
      for (int64_t j = 1; j < result_rank; j++) {
        end_values.push_back(
            builder.create<tensor::DimOp>(loc, batched_dot, j));
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
      slice_shapes[0] = 1;
      for (int64_t i = 1; i < result_rank; i++) {
        slice_shapes[i] = orig_dot_type.getDimSize(i - 1);
      }
      auto slice_type =
          RankedTensorType::get(slice_shapes, orig_dot_type.getElementType());
      auto dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, batched_dot, start_indices, end_indices,
          stride_indices);
      SmallVector<Value, 4> reshape_shape_values;
      for (int64_t j = 1; j < slice_type.getRank(); j++) {
        reshape_shape_values.push_back(
            builder.create<tensor::DimOp>(loc, dyn_slice, j));
      }
      auto reshape_shape = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get(
              {static_cast<int64_t>(reshape_shape_values.size())}, index_ty),
          reshape_shape_values);
      auto dyn_reshape = builder.create<mhlo::DynamicReshapeOp>(
          loc, orig_dot_type, dyn_slice, reshape_shape);
      op->replaceAllUsesWith(dyn_reshape);
    }
  }

  // No longer need the original dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }

  return true;
}

bool DotBatchingConverter::batchingDots(
    Block* block, const DotShapeEqualMap& equal_shape_map) {
  // Dot ops in `equal_shape_map` can be batched together if the batching
  // does not introduce cycle.

  // Form cycle detector.
  std::unique_ptr<GraphCycles> cycle_detector(new GraphCycles(0));
  DenseMap<Operation*, int64_t> op_to_id;
  buildBlockGraphCycles(block, std::move(cycle_detector), op_to_id);

  // Find batch clusters.
  SmallVector<DotCluster> batch_clusters;
  for (auto& equal_dots : equal_shape_map) {
    auto ops = equal_dots.second;
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<DotCluster> clusters;
    for (auto op : ops) {
      clusters.emplace_back(op.getOperation(), op_to_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& batched = clusters[i];
      if (batched.ops.empty()) {
        continue;
      }
      for (int64_t j = 0; j < clusters.size(); j++) {
        auto& to_batch = clusters[j];
        if (to_batch.ops.empty()) {
          continue;
        }
        // Try merge.
        int64_t batched_id = batched.leader_op_id;
        int64_t to_batch_id = to_batch.leader_op_id;
        auto optional_merged_id =
            TryMergeNode(cycle_detector.get(), batched_id, to_batch_id);
        if (!optional_merged_id.hasValue()) {
          // It forms a cycle.
          continue;
        }
        batched.merge(to_batch);
        batched.leader_op_id = *optional_merged_id;
      }
    }
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        batch_clusters.push_back(cluster);
      }
    }
  }

  // Apply batching.
  for (auto& cluster : batch_clusters) {
    applyBatching(cluster);
  }
  return true;
}

struct DiscDotMergePass : public DiscDotMergePassBase<DiscDotMergePass> {
  DiscDotMergePass()
      : DiscDotMergePassBase<DiscDotMergePass>::DiscDotMergePassBase() {}
  void runOnOperation() override;

 private:
  bool dotBatchingSimplifier(FuncOp& func);
};

void DiscDotMergePass::runOnOperation() {
  FuncOp func = getOperation();

  if (!dotBatchingSimplifier(func)) {
    signalPassFailure();
  }
}

bool DiscDotMergePass::dotBatchingSimplifier(FuncOp& func) {
  return DotBatchingConverter(func).run();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscDotMergePass() {
  return std::make_unique<DiscDotMergePass>();
}

}  // namespace disc_ral
}  // namespace mlir