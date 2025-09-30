#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - BaB Spec Refinement       ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import torch
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import path_config
from spec_parser.type import VerificationStatus

@dataclass
class Counterexample:
    input_point: torch.Tensor
    abstract_output: torch.Tensor
    concrete_output: Optional[torch.Tensor] = None
    is_spurious: Optional[bool] = None

@dataclass
class VerificationSubproblem:
    input_lb: torch.Tensor
    input_ub: torch.Tensor
    depth: int
    subproblem_id: Optional[int] = None
    parent_id: Optional[int] = None
    split_dimension: Optional[int] = None
    priority: float = 0.0
    counterexample: Optional[Counterexample] = None
    relu_constraints: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.relu_constraints is None:
            self.relu_constraints = []

@dataclass
class SpecRefinementResult:
    status: VerificationStatus
    total_time: float
    total_subproblems: int
    max_depth: int
    verified_regions: List[VerificationSubproblem]
    spurious_counterexamples: List[Counterexample]
    real_counterexample: Optional[Counterexample] = None
    stats: Optional[Dict] = None

class SpecRefinement:

    def __init__(self,
                 max_depth: int = 10,
                 max_subproblems: int = 1000,
                 time_limit: float = 1500.0,
                 spurious_check_enabled: bool = True,
                 verbose: bool = True):

        self.max_depth = max_depth
        self.max_subproblems = max_subproblems
        self.time_limit = time_limit
        self.spurious_check_enabled = spurious_check_enabled
        self.verbose = verbose

        self.stats = {
            'total_subproblems': 0,
            'sat_subproblems': 0,
            'unsat_subproblems': 0,
            'unknown_subproblems': 0,
            'spurious_counterexamples': 0,
            'real_counterexamples': 0,
            'refinement_splits': 0,
            'max_depth_reached': 0
        }

    def search(self,
               initial_input_lb: torch.Tensor,
               initial_input_ub: torch.Tensor,
               incomplete_verifier,
               concrete_network: torch.nn.Module) -> SpecRefinementResult:

        if self.verbose:
            print("🌳 Starting ReLU BaB specification refinement verification")

            print(f"Time limit: {self.time_limit}s")

        start_time = time.time()

        self._reset_stats()

        initial_subproblem = VerificationSubproblem(
            input_lb=initial_input_lb,
            input_ub=initial_input_ub,
            depth=0
        )

        result = self._spec_refinement_search(
            initial_subproblem,
            incomplete_verifier,
            concrete_network,
            start_time
        )

        total_time = time.time() - start_time
        result.total_time = total_time
        result.stats = self.stats.copy()

        if self.verbose:
            self._print_statistics(result)

        return result

    def _reset_stats(self):
        self.stats = {
            'total_subproblems': 0,
            'sat_subproblems': 0,
            'unsat_subproblems': 0,
            'unknown_subproblems': 0,
            'spurious_counterexamples': 0,
            'real_counterexamples': 0,
            'refinement_splits': 0,
            'max_depth_reached': 0
        }

    def _spec_refinement_search(self,
                               initial_subproblem: VerificationSubproblem,
                               incomplete_verifier,
                               concrete_network: torch.nn.Module,
                               start_time: float) -> SpecRefinementResult:

        subproblem_counter = 0
        initial_subproblem.subproblem_id = subproblem_counter
        subproblem_counter += 1

        work_queue = [initial_subproblem]
        verified_regions = []
        spurious_counterexamples = []
        real_counterexample = None
        max_depth = 0

        if self.verbose:
            print(f"🌳 Starting BaB search - initial subproblem #{initial_subproblem.subproblem_id}")

        while work_queue and len(verified_regions) + len(work_queue) < self.max_subproblems:

            if time.time() - start_time > self.time_limit:
                if self.verbose:
                    print(f"⏱️  Time limit {self.time_limit}s reached, stopping search")
                break

            current_subproblem = work_queue.pop(0)
            max_depth = max(max_depth, current_subproblem.depth)

            if self.verbose:
                print(f"\n� Processing subproblem #{current_subproblem.subproblem_id}")
                print(f"📊 Current status: queued={len(work_queue)}, verified={len(verified_regions)}, depth={current_subproblem.depth}")
                print(f"📋 Subproblem details:")
                print(f"depth: {current_subproblem.depth}")

                if hasattr(current_subproblem, 'parent_id') and current_subproblem.parent_id is not None:
                    parent_info = f"Parent problem ID={current_subproblem.parent_id}"
                else:
                    parent_info = "Root problem"
                print(f"Relationship: {parent_info}")
                if current_subproblem.relu_constraints:
                    print(f"ReLUconstraint count: {len(current_subproblem.relu_constraints)}")
                    for i, constraint in enumerate(current_subproblem.relu_constraints[-2:]):
                        print(f"constraint{i+1}: {constraint['layer']}[{constraint['neuron_idx']}] = {constraint['constraint_type']}")
                else:
                    print(f"ReLUconstraint: None")
                print("-" * 60)

            incomplete_verifier.input_lb = current_subproblem.input_lb.unsqueeze(0)
            incomplete_verifier.input_ub = current_subproblem.input_ub.unsqueeze(0)

            if hasattr(incomplete_verifier, 'set_relu_constraints'):
                incomplete_verifier.set_relu_constraints(current_subproblem.relu_constraints)

            bounding_status = incomplete_verifier._abstract_constraint_solving(
                current_subproblem.input_lb,
                current_subproblem.input_ub,
                0
            )

            potential_counterexample = None
            if bounding_status == VerificationStatus.UNSAT and self.spurious_check_enabled:
                if hasattr(incomplete_verifier, 'get_counterexample'):

                    potential_counterexample = incomplete_verifier.get_counterexample()
                else:

                    potential_counterexample = None

            self.stats['total_subproblems'] += 1

            if bounding_status == VerificationStatus.SAT:

                verified_regions.append(current_subproblem)
                self.stats['sat_subproblems'] += 1
                if self.verbose:
                    print(f"✅ Subproblem #{current_subproblem.subproblem_id} abstract verification successful (depth={current_subproblem.depth})")
                    print(f"Result: region is safe, added to verified regions")

            elif bounding_status == VerificationStatus.UNSAT:

                if self.verbose:
                    print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} abstract verification found violation possibility (depth={current_subproblem.depth})")

                if self.spurious_check_enabled and potential_counterexample is not None:

                    if self.verbose:

                        pass

                    is_spurious, concrete_output = self._spurious_check(
                        potential_counterexample,
                        concrete_network,
                        incomplete_verifier
                    )

                    counterexample = Counterexample(
                        input_point=potential_counterexample,
                        abstract_output=None,
                        concrete_output=concrete_output,
                        is_spurious=is_spurious
                    )

                    if is_spurious:

                        spurious_counterexamples.append(counterexample)
                        self.stats['spurious_counterexamples'] += 1
                        if self.verbose:
                            print(f"🟡 Spurious counterexample: 𝒩(ce) satisfies ψ, need to refine abstract domain")

                        if current_subproblem.depth < self.max_depth:
                            child_subproblems = self._spec_refinement_core(
                                current_subproblem, counterexample, incomplete_verifier
                            )
                            if child_subproblems:

                                for child in child_subproblems:
                                    child.subproblem_id = subproblem_counter
                                    child.parent_id = current_subproblem.subproblem_id
                                    subproblem_counter += 1
                                work_queue.extend(child_subproblems)
                                self.stats['refinement_splits'] += 1
                                if self.verbose:
                                    print(f"🔀 Subproblem #{current_subproblem.subproblem_id} abstract domain refined to {len(child_subproblems)} subproblems:")

                        else:
                            if self.verbose:
                                print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} reached maximum depth, stopping refinement")
                    else:

                        real_counterexample = counterexample
                        self.stats['real_counterexamples'] += 1
                        if self.verbose:
                            print(f"❌ Subproblem #{current_subproblem.subproblem_id} real counterexample: 𝒩(ce) violates ψ, verification failed")

                        return SpecRefinementResult(
                            status=VerificationStatus.UNSAT,
                            total_time=0.0,
                            total_subproblems=self.stats['total_subproblems'],
                            max_depth=max_depth,
                            verified_regions=verified_regions,
                            spurious_counterexamples=spurious_counterexamples,
                            real_counterexample=real_counterexample
                        )

                else:

                    if self.verbose:
                        print(f"Subproblem #{current_subproblem.subproblem_id} no counterexample or skipped spurious check, proceeding with direct ReLU BaB refinement")

                    if current_subproblem.depth < self.max_depth:
                        child_subproblems = self._spec_refinement_core(
                            current_subproblem, None, incomplete_verifier
                        )
                        if child_subproblems:

                            for child in child_subproblems:
                                child.subproblem_id = subproblem_counter
                                child.parent_id = current_subproblem.subproblem_id
                                subproblem_counter += 1
                            work_queue.extend(child_subproblems)
                            self.stats['refinement_splits'] += 1
                            if self.verbose:
                                print(f"🔀 Subproblem #{current_subproblem.subproblem_id} ReLU BaB refined to {len(child_subproblems)} subproblems:")

                        else:
                            if self.verbose:
                                print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} cannot be further refined in abstract domain")
                    else:
                        if self.verbose:
                            print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} reached maximum depth, stopping refinement")

                self.stats['unsat_subproblems'] += 1

            else:

                self.stats['unknown_subproblems'] += 1
                if self.verbose:
                    print(f"❓ Subproblem #{current_subproblem.subproblem_id} abstract verification result unknown (depth={current_subproblem.depth})")

                if current_subproblem.depth < self.max_depth:

                    child_subproblems = self._spec_refinement_core(
                        current_subproblem,
                        None,
                        incomplete_verifier
                    )
                    if child_subproblems:

                        for child in child_subproblems:
                            child.subproblem_id = subproblem_counter
                            child.parent_id = current_subproblem.subproblem_id
                            subproblem_counter += 1
                        work_queue.extend(child_subproblems)
                        self.stats['refinement_splits'] += 1
                        if self.verbose:
                            print(f"🔀 Subproblem #{current_subproblem.subproblem_id} abstract domain refined to {len(child_subproblems)} subproblems:")

                else:
                    if self.verbose:
                        print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} reached maximum depth {self.max_depth}, stopping refinement")

        if not work_queue:
            if self.verbose:
                print("✅ All subproblems verified successfully, property holds")
            status = VerificationStatus.SAT
        else:
            if self.verbose:
                print("❓ Reached search limits, result unknown")
            status = VerificationStatus.UNKNOWN

        self.stats['max_depth_reached'] = max_depth

        return SpecRefinementResult(
            status=status,
            total_time=0.0,
            total_subproblems=self.stats['total_subproblems'],
            max_depth=max_depth,
            verified_regions=verified_regions,
            spurious_counterexamples=spurious_counterexamples,
            real_counterexample=real_counterexample
        )

    def _spurious_check(self,
                       counterexample_input: torch.Tensor,
                       concrete_network: torch.nn.Module,
                       incomplete_verifier) -> Tuple[bool, torch.Tensor]:

        try:

            with torch.no_grad():
                if counterexample_input.dim() == 1:

                    input_batch = counterexample_input.unsqueeze(0)
                else:
                    input_batch = counterexample_input

                concrete_output = concrete_network(input_batch)

                if concrete_output.dim() > 1:
                    concrete_output = concrete_output.squeeze(0)

            output_constraints = None
            true_label = None

            if hasattr(incomplete_verifier.spec.output_spec, 'output_constraints'):
                output_constraints = incomplete_verifier.spec.output_spec.output_constraints

            if hasattr(incomplete_verifier.spec.output_spec, 'labels') and incomplete_verifier.spec.output_spec.labels is not None:
                true_label = incomplete_verifier.spec.output_spec.labels[0].item()

            from abstract_constraint_solver.base_verifier import BaseVerifier
            verdict = BaseVerifier._single_result_verdict(
                concrete_output, concrete_output,
                output_constraints,
                true_label
            )

            satisfies_property = (verdict == VerificationStatus.SAT)

            if self.verbose:
                print(f"🔍 Spurious Check:")

            is_spurious = satisfies_property

            return is_spurious, concrete_output

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Spurious check execution error: {e}")

            return False, torch.zeros_like(counterexample_input)

    def _spec_refinement_core(self,
                               subproblem: VerificationSubproblem,
                               counterexample: Optional[Counterexample],
                               incomplete_verifier) -> List[VerificationSubproblem]:

        relu_subproblems = self._split_relu_constraints(subproblem, counterexample, incomplete_verifier)
        if relu_subproblems:
            if self.verbose:
                print(f"🔀 Subproblem #{subproblem.subproblem_id} ReLU BaB split: generated {len(relu_subproblems)} subproblems")
                for i, child in enumerate(relu_subproblems):
                    constraint_info = "no constraint"
                    if child.relu_constraints:
                        latest_constraint = child.relu_constraints[-1]
                        constraint_info = f"{latest_constraint['layer']}[{latest_constraint['neuron_idx']}]={latest_constraint['constraint_type']}"
                    print(f"Subproblem {i+1}: {constraint_info}")
            return relu_subproblems

        if self.verbose:
            print(f"⚠️  Subproblem #{subproblem.subproblem_id} cannot perform ReLU constraint splitting, stopping refinement")
        return []

    def _split_relu_constraints(self,
                               subproblem: VerificationSubproblem,
                               counterexample: Optional[Counterexample],
                               incomplete_verifier) -> List[VerificationSubproblem]:

        try:
            if self.verbose:
                print(f"🔀 Subproblem #{subproblem.subproblem_id} executing ReLU constraint splitting (true ReLU BaB)")

            relu_bounds = self._get_relu_activation_bounds(subproblem, incomplete_verifier)
            if not relu_bounds:
                if self.verbose:
                    print(f"⚠️  Subproblem #{subproblem.subproblem_id} cannot obtain ReLU bounds, unable to perform splitting")
                return []

            target_relu = self._select_target_relu_for_splitting(relu_bounds)
            if not target_relu:
                if self.verbose:
                    print(f"⚠️  Subproblem #{subproblem.subproblem_id} no splittable unstable ReLU neurons found")
                return []

            layer_name, neuron_idx, lb_val, ub_val, instability = target_relu

            if self.verbose:
                print(f"Subproblem #{subproblem.subproblem_id} selected splitting target:")
                print(f"Layer: {layer_name}")
                print(f"Neuron: {neuron_idx}")

            child_subproblems = []

            child1 = VerificationSubproblem(
                input_lb=subproblem.input_lb.clone(),
                input_ub=subproblem.input_ub.clone(),
                depth=subproblem.depth + 1,
                parent_id=id(subproblem),
                split_dimension=None,
                counterexample=counterexample
            )

            child1.relu_constraints = subproblem.relu_constraints.copy()
            child1.relu_constraints.append({
                'layer': layer_name,
                'neuron_idx': neuron_idx,
                'constraint_type': 'inactive',
                'preactivation_ub': 0.0
            })
            child_subproblems.append(child1)

            child2 = VerificationSubproblem(
                input_lb=subproblem.input_lb.clone(),
                input_ub=subproblem.input_ub.clone(),
                depth=subproblem.depth + 1,
                parent_id=id(subproblem),
                split_dimension=None,
                counterexample=counterexample
            )

            child2.relu_constraints = subproblem.relu_constraints.copy()
            child2.relu_constraints.append({
                'layer': layer_name,
                'neuron_idx': neuron_idx,
                'constraint_type': 'active',
                'preactivation_lb': 0.0
            })
            child_subproblems.append(child2)

            if self.verbose:
                print(f"✅ Subproblem #{subproblem.subproblem_id} ReLU BaB splitting completed: generated {len(child_subproblems)} subproblems")
                print(f"Subproblem 1: ReLU[{layer_name}:{neuron_idx}] = inactive")
                print(f"Subproblem 2: ReLU[{layer_name}:{neuron_idx}] = active")

            return child_subproblems

        except Exception as e:
            if self.verbose:
                print(f"❌ Subproblem #{subproblem.subproblem_id} ReLU constraint splitting failed: {e}")

            return []

    def _get_relu_activation_bounds(self, subproblem: VerificationSubproblem, incomplete_verifier):

        try:
            if self.verbose:
                constraint_count = len(subproblem.relu_constraints) if subproblem.relu_constraints else 0
                print(f"🔍 [BaB Bound Retrieval] Getting ReLU preactivation bounds (constraint count: {constraint_count})...")

            relu_bounds = {}

            if hasattr(incomplete_verifier, 'hz_layer_bounds') and incomplete_verifier.hz_layer_bounds:

                layer_names = list(incomplete_verifier.hz_layer_bounds.keys())

                for i, layer_name in enumerate(layer_names):

                    if 'relu' in layer_name.lower():

                        if i > 0:
                            prev_layer_name = layer_names[i-1]
                            prev_layer_data = incomplete_verifier.hz_layer_bounds[prev_layer_name]
                            lb = prev_layer_data.get('lb')
                            ub = prev_layer_data.get('ub')

                            if lb is not None and ub is not None:

                                relu_bounds[layer_name] = (lb.clone(), ub.clone())

                                if self.verbose:
                                    unstable_count = ((lb < 0) & (ub > 0)).sum().item()
                                    print(f"🔍 {layer_name} (from {prev_layer_name}): {unstable_count} unstable ReLU")

                                    for constraint in subproblem.relu_constraints:
                                        if constraint['layer'] == layer_name:
                                            neuron_idx = constraint['neuron_idx']
                                            if neuron_idx < len(lb.view(-1)):
                                                lb_val = lb.view(-1)[neuron_idx].item()
                                                ub_val = ub.view(-1)[neuron_idx].item()
                                                print(f"Constraint neuron {neuron_idx}: [{lb_val:.6f}, {ub_val:.6f}] ({constraint['constraint_type']})")

            elif hasattr(incomplete_verifier, 'autolirpa_layer_bounds') and incomplete_verifier.autolirpa_layer_bounds:
                layer_names = list(incomplete_verifier.autolirpa_layer_bounds.keys())

                for i, layer_name in enumerate(layer_names):
                    if 'relu' in layer_name.lower() and i > 0:
                        prev_layer_name = layer_names[i-1]
                        prev_layer_data = incomplete_verifier.autolirpa_layer_bounds[prev_layer_name]
                        lb = prev_layer_data.get('lb')
                        ub = prev_layer_data.get('ub')

                        if lb is not None and ub is not None:
                            relu_bounds[layer_name] = (lb.clone(), ub.clone())

                            if self.verbose:
                                unstable_count = ((lb < 0) & (ub > 0)).sum().item()
                                print(f"🔍 {layer_name}: {unstable_count} unstable ReLU")

            else:
                if self.verbose:
                    print(f"⚠️  Verifier has not recorded layer boundary information, unable to obtain ReLU bounds")
                return None

            if self.verbose:
                total_layers = len(relu_bounds)
                total_unstable = sum(((lb < 0) & (ub > 0)).sum().item() for lb, ub in relu_bounds.values())
                print(f"🔍 [BaB Bound Retrieval] Successfully obtained {total_layers} layer ReLU bounds, total {total_unstable} unstable neurons")

            return relu_bounds if relu_bounds else None

        except Exception as e:
            if self.verbose:
                print(f"❌ Error occurred while obtaining ReLU preactivation bounds: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _select_target_relu_for_splitting(self, relu_bounds):

        try:
            if not relu_bounds:
                return None

            best_candidate = None
            max_instability = -1.0

            for layer_name, (lb, ub) in relu_bounds.items():

                unstable_mask = (lb < 0) & (ub > 0)
                if unstable_mask.any():
                    unstable_indices = unstable_mask.nonzero(as_tuple=True)[0]
                    for idx in unstable_indices:
                        instability = (ub[idx] - lb[idx]).item()
                        if instability > max_instability:
                            max_instability = instability
                            best_candidate = (
                                layer_name,
                                idx.item(),
                                lb[idx].item(),
                                ub[idx].item(),
                                instability
                            )

            if self.verbose and best_candidate:
                layer_name, neuron_idx, lb_val, ub_val, instability = best_candidate

            return best_candidate

        except Exception as e:
            if self.verbose:
                print(f"❌ Target ReLU selection failed: {e}")
            return None

    def _print_statistics(self, result: SpecRefinementResult):
        print("\n" + "="*60)
        print("📊 Specification Refinement Verification Statistics")
        print("="*60)
        print(f"Final result: {result.status.name}")
        print(f"Total time: {result.total_time:.2f} seconds")
        print(f"Total subproblems: {result.total_subproblems}")
        print(f"Maximum search depth: {result.max_depth}")
        print(f"Successfully verified subproblems: {self.stats['sat_subproblems']}")
        print(f"Counterexample found subproblems: {self.stats['unsat_subproblems']}")
        print(f"Unknown result subproblems: {self.stats['unknown_subproblems']}")

        print(f"\n🔍 Spurious Check statistics:")
        print(f"Spurious counterexamples: {self.stats['spurious_counterexamples']}")
        print(f"Real counterexamples: {self.stats['real_counterexamples']}")
        print(f"Abstract domain refinement splits: {self.stats['refinement_splits']}")

        print(f"\n📊 Verified regions: {len(result.verified_regions)}")
        print(f"📊 Spurious counterexamples: {len(result.spurious_counterexamples)}")

        if result.real_counterexample is not None:
            print(f"💥 Real counterexample dimensions: {result.real_counterexample.input_point.shape}")

        print("="*60)

def create_spec_refinement_core(max_depth=8, max_subproblems=500, time_limit=1500.0,
                               spurious_check_enabled=True,
                               verbose=True) -> SpecRefinement:

    return SpecRefinement(
        max_depth=max_depth,
        max_subproblems=max_subproblems,
        time_limit=time_limit,
        spurious_check_enabled=spurious_check_enabled,
        verbose=verbose
    )

if __name__ == "__main__":
    print("Specification Refinement ReLU BaB Core Module")
    print("Core module focused on ReLU Branch and Bound specification refinement verification")
    print("\nCore Workflow:")
    print("1️⃣ Abstract Constraint Solving → Abstract verification")
    print("2️⃣ ReLU BaB Splitting → Identify unstable ReLUs and split")
    print("3️⃣ [Optional] Spurious Check → Check if 𝒩(ce) satisfies ψ")
    print("4️⃣ Refinement Loop → Recursive refinement until verification complete")
    print("\nCore Features:")
    print("✅ Focused on ReLU BaB splitting strategies")
    print("✅ Support for optional counterexample mode")
    print("✅ Compatible with HybridZonotope relaxed verification")
    print("✅ Simplified design, removing unnecessary complexity")
    print("✅ Efficient and reliable ReLU constraint refinement")
    print("✅ Support for pure BaB mode without counterexamples")