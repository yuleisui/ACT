#===- verifier.bab_refinement.bab_spec_refinement.py - BaB Refinement ----#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===---------------------------------------------------------------------===#

import torch
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from input_parser.type import VerifyResult

@dataclass
class Counterexample:
    """Represents a potential counterexample found during verification.
    
    Attributes:
        input_point: The input tensor that generates the counterexample.
        abstract_output: The abstract verifier's output for this input.
        concrete_output: The concrete network's actual output (if computed).
        is_spurious: Whether this counterexample is spurious (satisfies property).
    """
    input_point: torch.Tensor
    abstract_output: torch.Tensor
    concrete_output: Optional[torch.Tensor] = None
    is_spurious: Optional[bool] = None

@dataclass
class Subproblem:
    """Represents a verification subproblem in the BaB search tree.
    
    Attributes:
        input_lb: Lower bounds for input variables.
        input_ub: Upper bounds for input variables.
        depth: Depth in the BaB search tree.
        subproblem_id: Unique identifier for this subproblem.
        parent_id: ID of parent subproblem in search tree.
        split_dimension: Dimension used for splitting (if applicable).
        priority: Priority for queue ordering (lower = higher priority).
        counterexample: Associated counterexample (if any).
        relu_constraints: List of ReLU constraints for this subproblem.
    """
    input_lb: torch.Tensor
    input_ub: torch.Tensor
    depth: int
    subproblem_id: Optional[int] = None
    parent_id: Optional[int] = None
    split_dimension: Optional[int] = None
    priority: float = 0.0
    counterexample: Optional[Counterexample] = None
    relu_constraints: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BaBStats:
    """Statistics collected during BaB refinement process.
    
    Attributes:
        total_subproblems: Total number of subproblems processed.
        sat_subproblems: Number of subproblems verified as safe.
        unsat_subproblems: Number of subproblems with counterexamples.
        unknown_subproblems: Number of subproblems with unknown results.
        spurious_counterexamples: Number of spurious counterexamples found.
        real_counterexamples: Number of real counterexamples found.
        refinement_splits: Number of successful BaB splits performed.
        max_depth_reached: Maximum depth reached in search tree.
    """
    total_subproblems: int = 0
    sat_subproblems: int = 0
    unsat_subproblems: int = 0
    unknown_subproblems: int = 0
    spurious_counterexamples: int = 0
    real_counterexamples: int = 0
    refinement_splits: int = 0
    max_depth_reached: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert stats to dictionary format."""
        return {
            'total_subproblems': self.total_subproblems,
            'sat_subproblems': self.sat_subproblems,
            'unsat_subproblems': self.unsat_subproblems,
            'unknown_subproblems': self.unknown_subproblems,
            'spurious_counterexamples': self.spurious_counterexamples,
            'real_counterexamples': self.real_counterexamples,
            'refinement_splits': self.refinement_splits,
            'max_depth_reached': self.max_depth_reached
        }

@dataclass
class RefinementResult:
    """Result of the BaB refinement verification process.
    
    Attributes:
        status: Final verification status (SAT/UNSAT/UNKNOWN).
        total_time: Total time spent on verification.
        total_subproblems: Total number of subproblems processed.
        max_depth: Maximum depth reached in search tree.
        verified_regions: List of successfully verified subproblems.
        spurious_counterexamples: List of spurious counterexamples found.
        real_counterexample: Real counterexample (if found).
        stats: Detailed statistics from the verification process.
    """
    status: VerifyResult
    total_time: float
    total_subproblems: int
    max_depth: int
    verified_regions: List[Subproblem]
    spurious_counterexamples: List[Counterexample]
    real_counterexample: Optional[Counterexample] = None
    stats: Optional[BaBStats] = None

class BaBRefinement:
    """Branch-and-Bound refinement for neural network verification.
    
    Implements ReLU constraint splitting to eliminate spurious counterexamples
    and refine abstract domains for precise verification.
    """

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    def __init__(self,
                 max_depth: int = 10,
                 max_subproblems: int = 1000,
                 time_limit: float = 1500.0,
                 spurious_check_enabled: bool = True,
                 verbose: bool = True) -> None:
        """Initialize BaB refinement with configuration parameters.
        
        Args:
            max_depth: Maximum depth for BaB search tree.
            max_subproblems: Maximum number of subproblems to process.
            time_limit: Maximum time in seconds for verification.
            spurious_check_enabled: Whether to perform spurious counterexample checking.
            verbose: Whether to print detailed progress information.
        """
        self.max_depth = max_depth
        self.max_subproblems = max_subproblems
        self.time_limit = time_limit
        self.spurious_check_enabled = spurious_check_enabled
        self.verbose = verbose
        self.stats = BaBStats()

    def verify(self,
               initial_input_lb: torch.Tensor,
               initial_input_ub: torch.Tensor,
               incomplete_verifier,
               concrete_network: torch.nn.Module) -> RefinementResult:
        """Main verification entry point using BaB refinement.
        
        Args:
            initial_input_lb: Lower bounds for input variables.
            initial_input_ub: Upper bounds for input variables.
            incomplete_verifier: Abstract verifier instance.
            concrete_network: Concrete neural network for spurious checking.
            
        Returns:
            RefinementResult containing verification status and statistics.
        """
        if self.verbose:
            print("🌳 Starting ReLU BaB specification refinement verification")
            print(f"Time limit: {self.time_limit}s")

        start_time = time.time()
        self._reset_stats()

        initial_subproblem = Subproblem(
            input_lb=initial_input_lb,
            input_ub=initial_input_ub,
            depth=0
        )

        result = self._bab_search(
            initial_subproblem,
            incomplete_verifier,
            concrete_network,
            start_time
        )

        total_time = time.time() - start_time
        result.total_time = total_time
        result.stats = self.stats

        if self.verbose:
            self._print_stats(result)

        return result

    # ============================================================================
    # CORE SEARCH ALGORITHM (PRIVATE)
    # ============================================================================

    def _bab_search(self,
                    initial_subproblem: Subproblem,
                    incomplete_verifier,
                    concrete_network: torch.nn.Module,
                    start_time: float) -> RefinementResult:
        """Core BaB search algorithm with ReLU constraint splitting.
        
        Args:
            initial_subproblem: Initial subproblem to start search from.
            incomplete_verifier: Abstract verifier for bound computation.
            concrete_network: Concrete network for spurious checking.
            start_time: Start time for timeout checking.
            
        Returns:
            RefinementResult with verification outcome and statistics.
        """
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

            self._log_subproblem_progress(current_subproblem, len(work_queue), len(verified_regions))

            # Configure verifier with current subproblem bounds and constraints
            incomplete_verifier.input_lb = current_subproblem.input_lb.unsqueeze(0)
            incomplete_verifier.input_ub = current_subproblem.input_ub.unsqueeze(0)

            if hasattr(incomplete_verifier, 'set_relu_constraints'):
                incomplete_verifier.set_relu_constraints(current_subproblem.relu_constraints)

            # Perform abstract verification on current subproblem
            verification_status = incomplete_verifier._abstract_constraint_solving(
                current_subproblem.input_lb,
                current_subproblem.input_ub,
                0
            )

            # Extract potential counterexample if available
            potential_counterexample = self._extract_counterexample(
                verification_status, incomplete_verifier
            )

            self.stats.total_subproblems += 1

            # Process verification result based on status
            if verification_status == VerifyResult.SAT:
                # Region verified as safe
                verified_regions.append(current_subproblem)
                self.stats.sat_subproblems += 1
                if self.verbose:
                    print(f"✅ Subproblem #{current_subproblem.subproblem_id} verified safe (depth={current_subproblem.depth})")

            elif verification_status == VerifyResult.UNSAT:
                # Potential counterexample found - need spurious check or refinement
                if self.verbose:
                    print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} found potential violation (depth={current_subproblem.depth})")

                real_ce = self._handle_unsat_subproblem(
                    current_subproblem, potential_counterexample, concrete_network, 
                    incomplete_verifier, spurious_counterexamples, work_queue, 
                    subproblem_counter
                )
                
                if real_ce:
                    # Real counterexample found - verification fails
                    return RefinementResult(
                        status=VerifyResult.UNSAT,
                        total_time=0.0,
                        total_subproblems=self.stats.total_subproblems,
                        max_depth=max_depth,
                        verified_regions=verified_regions,
                        spurious_counterexamples=spurious_counterexamples,
                        real_counterexample=real_ce
                    )

                self.stats.unsat_subproblems += 1

            else:
                # Unknown result - try refinement if depth allows
                self.stats.unknown_subproblems += 1
                if self.verbose:
                    print(f"❓ Subproblem #{current_subproblem.subproblem_id} unknown result (depth={current_subproblem.depth})")

                if current_subproblem.depth < self.max_depth:
                    child_subproblems = self._split_relu(current_subproblem, None, incomplete_verifier)
                    if child_subproblems:
                        self._add_child_subproblems(child_subproblems, current_subproblem, 
                                                   work_queue, subproblem_counter)
                        self.stats.refinement_splits += 1
                else:
                    if self.verbose:
                        print(f"⚠️  Subproblem #{current_subproblem.subproblem_id} reached max depth {self.max_depth}")

        # Determine final verification status
        if not work_queue:
            if self.verbose:
                print("✅ All subproblems verified successfully, property holds")
            status = VerifyResult.SAT
        else:
            if self.verbose:
                print("❓ Reached search limits, result unknown")
            status = VerifyResult.UNKNOWN

        self.stats.max_depth_reached = max_depth

        return RefinementResult(
            status=status,
            total_time=0.0,
            total_subproblems=self.stats.total_subproblems,
            max_depth=max_depth,
            verified_regions=verified_regions,
            spurious_counterexamples=spurious_counterexamples,
            real_counterexample=real_counterexample
        )

    # ============================================================================
    # SUBPROBLEM MANAGEMENT (PRIVATE)
    # ============================================================================

    def _handle_unsat_subproblem(self, subproblem: Subproblem, potential_ce: Optional[torch.Tensor],
                                concrete_network: torch.nn.Module, verifier, spurious_list: List[Counterexample],
                                work_queue: List[Subproblem], subproblem_counter: int) -> Optional[Counterexample]:
        """Handle UNSAT subproblem with spurious checking and refinement."""
        if self.spurious_check_enabled and potential_ce is not None:
            # Perform spurious check
            is_spurious, concrete_output = self._check_spurious(potential_ce, concrete_network, verifier)
            
            counterexample = Counterexample(
                input_point=potential_ce,
                abstract_output=None,
                concrete_output=concrete_output,
                is_spurious=is_spurious
            )

            if is_spurious:
                # Spurious counterexample - refine if possible
                spurious_list.append(counterexample)
                self.stats.spurious_counterexamples += 1
                if self.verbose:
                    print(f"🟡 Spurious counterexample: need to refine abstract domain")

                if subproblem.depth < self.max_depth:
                    child_subproblems = self._split_relu(subproblem, counterexample, verifier)
                    if child_subproblems:
                        self._add_child_subproblems(child_subproblems, subproblem, work_queue, subproblem_counter)
                        self.stats.refinement_splits += 1
                else:
                    if self.verbose:
                        print(f"⚠️  Max depth reached, stopping refinement")
                return None
            else:
                # Real counterexample found
                self.stats.real_counterexamples += 1
                if self.verbose:
                    print(f"❌ Real counterexample found: verification failed")
                return counterexample
        else:
            # No counterexample or spurious check disabled - try direct refinement
            if self.verbose:
                print("📝 No counterexample available, trying direct ReLU refinement")
                
            if subproblem.depth < self.max_depth:
                child_subproblems = self._split_relu(subproblem, None, verifier)
                if child_subproblems:
                    self._add_child_subproblems(child_subproblems, subproblem, work_queue, subproblem_counter)
                    self.stats.refinement_splits += 1
                else:
                    if self.verbose:
                        print(f"⚠️  Cannot refine subproblem #{subproblem.subproblem_id}")
            else:
                if self.verbose:
                    print(f"⚠️  Max depth reached for subproblem #{subproblem.subproblem_id}")
        return None

    def _add_child_subproblems(self, children: List[Subproblem], parent: Subproblem, 
                              work_queue: List[Subproblem], subproblem_counter: int) -> None:
        """Add child subproblems to work queue with proper ID assignment."""
        for child in children:
            child.subproblem_id = subproblem_counter
            child.parent_id = parent.subproblem_id
            subproblem_counter += 1
        work_queue.extend(children)
        
        if self.verbose:
            print(f"🔀 Split subproblem #{parent.subproblem_id} into {len(children)} children")

    def _extract_counterexample(self, status: VerifyResult, verifier) -> Optional[torch.Tensor]:
        """Extract potential counterexample from verifier if available."""
        if status == VerifyResult.UNSAT and self.spurious_check_enabled:
            if hasattr(verifier, 'get_counterexample'):
                return verifier.get_counterexample()
        return None

    # ============================================================================
    # SPURIOUS COUNTEREXAMPLE CHECKING (PRIVATE)
    # ============================================================================

    def _check_spurious(self, counterexample_input: torch.Tensor, concrete_network: torch.nn.Module,
                       incomplete_verifier) -> Tuple[bool, torch.Tensor]:
        """Check if counterexample is spurious by evaluating on concrete network.
        
        Args:
            counterexample_input: Input tensor to check.
            concrete_network: Concrete neural network.
            incomplete_verifier: Verifier with specification.
            
        Returns:
            Tuple of (is_spurious, concrete_output).
        """
        try:
            # Evaluate concrete network on counterexample
            with torch.no_grad():
                input_batch = counterexample_input.unsqueeze(0) if counterexample_input.dim() == 1 else counterexample_input
                concrete_output = concrete_network(input_batch)
                
                if concrete_output.dim() > 1:
                    concrete_output = concrete_output.squeeze(0)

            # Extract specification constraints
            output_constraints = None
            true_label = None

            if hasattr(incomplete_verifier.spec.output_spec, 'output_constraints'):
                output_constraints = incomplete_verifier.spec.output_spec.output_constraints

            if hasattr(incomplete_verifier.spec.output_spec, 'labels') and incomplete_verifier.spec.output_spec.labels is not None:
                true_label = incomplete_verifier.spec.output_spec.labels[0].item()

            # Use temporary verifier to evaluate output bounds
            from abstract_constraint_solver.interval.interval_verifier import IntervalVerifier
            temp_verifier = IntervalVerifier(incomplete_verifier.dataset, 'interval', 
                                           incomplete_verifier.spec, incomplete_verifier.device)
            verdict = temp_verifier._evaluate_output_bounds(
                concrete_output, concrete_output, output_constraints, true_label
            )

            is_spurious = (verdict == VerifyResult.SAT)
            
            if self.verbose:
                print(f"🔍 Spurious check: {'spurious' if is_spurious else 'real'}")

            return is_spurious, concrete_output

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Spurious check error: {e}")
            return False, torch.zeros_like(counterexample_input)

    # ============================================================================
    # RELU CONSTRAINT SPLITTING (PRIVATE)
    # ============================================================================

    def _split_relu(self, subproblem: Subproblem, counterexample: Optional[Counterexample],
                   incomplete_verifier) -> List[Subproblem]:
        """Split ReLU constraints to create child subproblems.
        
        Args:
            subproblem: Current subproblem to split.
            counterexample: Associated counterexample (if any).
            incomplete_verifier: Verifier for bound computation.
            
        Returns:
            List of child subproblems with ReLU constraints.
        """
        try:
            if self.verbose:
                print(f"🔀 Splitting ReLU constraints for subproblem #{subproblem.subproblem_id}")

            # Get ReLU activation bounds for splitting decision
            relu_bounds = self._get_relu_bounds(subproblem, incomplete_verifier)
            if not relu_bounds:
                if self.verbose:
                    print(f"⚠️  Cannot obtain ReLU bounds for splitting")
                return []

            # Select target ReLU neuron for splitting
            target_relu = self._select_split_target(relu_bounds)
            if not target_relu:
                if self.verbose:
                    print(f"⚠️  No unstable ReLU neurons found for splitting")
                return []

            layer_name, neuron_idx, lb_val, ub_val, instability = target_relu

            if self.verbose:
                print(f"Selected split target: {layer_name}[{neuron_idx}] with instability {instability:.6f}")

            # Create child subproblems with ReLU constraints
            child_subproblems = []

            # Child 1: ReLU inactive (output = 0)
            child1 = Subproblem(
                input_lb=subproblem.input_lb.clone(),
                input_ub=subproblem.input_ub.clone(),
                depth=subproblem.depth + 1,
                parent_id=id(subproblem),
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

            # Child 2: ReLU active (output = input)
            child2 = Subproblem(
                input_lb=subproblem.input_lb.clone(),
                input_ub=subproblem.input_ub.clone(),
                depth=subproblem.depth + 1,
                parent_id=id(subproblem),
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
                print(f"✅ ReLU splitting complete: {len(child_subproblems)} children created")
                print(f"  Child 1: ReLU[{layer_name}:{neuron_idx}] = inactive")
                print(f"  Child 2: ReLU[{layer_name}:{neuron_idx}] = active")

            return child_subproblems

        except Exception as e:
            if self.verbose:
                print(f"❌ ReLU splitting failed: {e}")
            return []

    def _get_relu_bounds(self, subproblem: Subproblem, incomplete_verifier) -> Optional[Dict]:
        """Get ReLU activation bounds for splitting decisions.
        
        Args:
            subproblem: Current subproblem.
            incomplete_verifier: Verifier with layer bounds.
            
        Returns:
            Dictionary mapping layer names to (lower_bound, upper_bound) tuples.
        """
        try:
            if self.verbose:
                constraint_count = len(subproblem.relu_constraints)
                print(f"🔍 Getting ReLU bounds (current constraints: {constraint_count})")

            relu_bounds = {}

            # Try HybridZonotope bounds first
            if hasattr(incomplete_verifier, 'hz_layer_bounds') and incomplete_verifier.hz_layer_bounds:
                layer_names = list(incomplete_verifier.hz_layer_bounds.keys())
                
                for i, layer_name in enumerate(layer_names):
                    if 'relu' in layer_name.lower() and i > 0:
                        prev_layer_name = layer_names[i-1]
                        prev_layer_data = incomplete_verifier.hz_layer_bounds[prev_layer_name]
                        lb = prev_layer_data.get('lb')
                        ub = prev_layer_data.get('ub')

                        if lb is not None and ub is not None:
                            relu_bounds[layer_name] = (lb.clone(), ub.clone())
                            
                            if self.verbose:
                                unstable_count = ((lb < 0) & (ub > 0)).sum().item()
                                print(f"🔍 {layer_name}: {unstable_count} unstable ReLU neurons")

            # Try AutoLiRPA bounds as fallback
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
                                print(f"🔍 {layer_name}: {unstable_count} unstable ReLU neurons")
            else:
                if self.verbose:
                    print(f"⚠️  No layer bounds available in verifier")
                return None

            if self.verbose and relu_bounds:
                total_layers = len(relu_bounds)
                total_unstable = sum(((lb < 0) & (ub > 0)).sum().item() for lb, ub in relu_bounds.values())
                print(f"🔍 Found {total_layers} ReLU layers with {total_unstable} total unstable neurons")

            return relu_bounds if relu_bounds else None

        except Exception as e:
            if self.verbose:
                print(f"❌ Error getting ReLU bounds: {e}")
            return None

    def _select_split_target(self, relu_bounds: Dict) -> Optional[Tuple]:
        """Select target ReLU neuron for splitting based on instability.
        
        Args:
            relu_bounds: Dictionary of layer bounds.
            
        Returns:
            Tuple of (layer_name, neuron_idx, lb_val, ub_val, instability) or None.
        """
        try:
            if not relu_bounds:
                return None

            best_candidate = None
            max_instability = -1.0

            for layer_name, (lb, ub) in relu_bounds.items():
                # Find unstable ReLU neurons (lb < 0 < ub)
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

            return best_candidate

        except Exception as e:
            if self.verbose:
                print(f"❌ Target selection failed: {e}")
            return None

    # ============================================================================
    # UTILITY METHODS (PRIVATE)
    # ============================================================================

    def _reset_stats(self) -> None:
        """Reset statistics counters for new verification run."""
        self.stats = BaBStats()

    def _log_subproblem_progress(self, subproblem: Subproblem, queue_size: int, verified_count: int) -> None:
        """Log detailed progress information for current subproblem."""
        if not self.verbose:
            return
            
        print(f"\n🔍 Processing subproblem #{subproblem.subproblem_id}")
        print(f"📊 Status: queued={queue_size}, verified={verified_count}, depth={subproblem.depth}")
        
        parent_info = f"Parent ID={subproblem.parent_id}" if subproblem.parent_id else "Root problem"
        print(f"🔗 {parent_info}")
        
        if subproblem.relu_constraints:
            print(f"🧠 ReLU constraints: {len(subproblem.relu_constraints)}")
            for i, constraint in enumerate(subproblem.relu_constraints[-2:], 1):
                print(f"  constraint{i}: {constraint['layer']}[{constraint['neuron_idx']}] = {constraint['constraint_type']}")
        else:
            print("🧠 ReLU constraints: None")
        print("-" * 60)

    def _print_stats(self, result: RefinementResult) -> None:
        """Print comprehensive verification statistics.
        
        Args:
            result: Refinement result with statistics.
        """
        print("\n" + "="*60)
        print("📊 BaB Refinement Verification Statistics")
        print("="*60)
        print(f"Final result: {result.status.name}")
        print(f"Total time: {result.total_time:.2f} seconds")
        print(f"Total subproblems: {result.total_subproblems}")
        print(f"Maximum search depth: {result.max_depth}")
        print(f"Successfully verified: {self.stats.sat_subproblems}")
        print(f"Counterexample found: {self.stats.unsat_subproblems}")
        print(f"Unknown results: {self.stats.unknown_subproblems}")

        print(f"\n🔍 Spurious checking:")
        print(f"Spurious counterexamples: {self.stats.spurious_counterexamples}")
        print(f"Real counterexamples: {self.stats.real_counterexamples}")
        print(f"Refinement splits: {self.stats.refinement_splits}")

        print(f"\n📊 Final counts:")
        print(f"Verified regions: {len(result.verified_regions)}")
        print(f"Spurious counterexamples: {len(result.spurious_counterexamples)}")

        if result.real_counterexample is not None:
            print(f"💥 Real counterexample shape: {result.real_counterexample.input_point.shape}")

        print("="*60)


def create_bab_refinement(max_depth: int = 8, max_subproblems: int = 500, 
                         time_limit: float = 1500.0, spurious_check_enabled: bool = True,
                         verbose: bool = True) -> BaBRefinement:
    """Create BaB refinement instance with specified configuration.
    
    Args:
        max_depth: Maximum depth for BaB search tree.
        max_subproblems: Maximum number of subproblems to process.
        time_limit: Maximum time in seconds for verification.
        spurious_check_enabled: Whether to perform spurious counterexample checking.
        verbose: Whether to print detailed progress information.
        
    Returns:
        Configured BaBRefinement instance.
    """
    return BaBRefinement(
        max_depth=max_depth,
        max_subproblems=max_subproblems,
        time_limit=time_limit,
        spurious_check_enabled=spurious_check_enabled,
        verbose=verbose
    )

if __name__ == "__main__":
    print("BaB Refinement Module for Neural Network Verification")
    print("Implements Branch-and-Bound ReLU constraint splitting for precise verification")
    print("\nCore Workflow:")
    print("1️⃣ Abstract Verification → Check subproblem with current constraints")
    print("2️⃣ Spurious Checking → Validate counterexamples on concrete network")
    print("3️⃣ ReLU Splitting → Split unstable ReLU neurons into active/inactive cases")
    print("4️⃣ Refinement Loop → Recursively refine until verification complete")
    print("\nKey Features:")
    print("✅ Efficient ReLU constraint splitting strategies")
    print("✅ Optional spurious counterexample detection")
    print("✅ Compatible with HybridZonotope and AutoLiRPA verifiers")
    print("✅ Comprehensive statistics and progress tracking")
    print("✅ Configurable depth and time limits")
    print("✅ Clean API with backward compatibility")
    
    # Demo usage
    print("\nUsage Example:")
    print("```python")
    print("# Create BaB refinement instance")
    print("refinement = create_bab_refinement(max_depth=10, verbose=True)")
    print("")
    print("# Run verification with BaB refinement")
    print("result = refinement.verify(input_lb, input_ub, verifier, network)")
    print("```")