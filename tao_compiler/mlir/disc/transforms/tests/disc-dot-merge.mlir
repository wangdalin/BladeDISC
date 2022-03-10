// RUN: disc-opt -disc-dot-merge -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @dot_batching
// This UT builds the tensor shape implicitly because current implement of
// ShapeAnalysis relies on them to build DimValue. ShapeAnalysisV2, which is on
// going for development, will solve this problem.
func @dot_batching(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m= tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: mhlo.concatenate
  // CHECK: mhlo.concatenate
  // CHECK: mhlo.dot_general
  // CHECK-DAG: lhs_batching_dimensions = [0]
  // CHECK-DAG: rhs_batching_dimensions = [0]
  // CHECK-DAG: lhs_contracting_dimensions = [2]
  // CHECK-DAG: rhs_contracting_dimensions = [1]
  // CHECK: -> tensor<2x?x?xf32>
  // CHECK-NOT: mhlo.dot_general
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_not_batching_diff_dtype
func @dot_not_batching_diff_dtype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m = tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.convert"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %4 = "mhlo.convert"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %6 = "mhlo.convert"(%5) : (tensor<?x?xf16>) -> tensor<?x?xf32>
  %7 = "mhlo.add"(%2, %6) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK: mhlo.dot_general
  // CHECK: mhlo.dot_general
  // CHECK-NOT: lhs_batching_dimensions = [0]
  // CHECK-NOT: rhs_batching_dimensions = [0]
  return %7: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_not_batching_cycle
func @dot_not_batching_cycle(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.extract %arg2[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK: mhlo.dot_general
  // CHECK: mhlo.dot_general
  // CHECK-NOT: lhs_batching_dimensions = [0]
  // CHECK-NOT: rhs_batching_dimensions = [0]
  return %5: tensor<?x?xf32>
}
