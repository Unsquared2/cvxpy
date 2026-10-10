"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from types import SimpleNamespace

import clarabel
import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL, triu_csc
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.tests.base_test import BaseTest


class ClarabelTest(BaseTest):
    """
    Tests for Clarabel solver interface.
    """

    def setUp(self):
        """
        Sets up a problem as in test_compile.py function test_quad_form.
        """
        np.random.seed(42)
        P = np.random.randn(3, 3) - 1j*np.random.randn(3, 3)
        P = np.conj(P.T).dot(P)
        b = np.arange(3) + 3j*(np.arange(3) + 10)
        x = Variable(3, complex=True)
        self.value = cp.quad_form(b, P).value
        self.prob = Problem(cp.Minimize(cp.quad_form(x, P)), [x == b])

        # A dummy solution object (clarabel.DefaultSolution has no python constructor).
        self.solution: clarabel.DefaultSolution = type("DefaultSolution", (object,), {
            "x": [5.263500536056257e-12, 0.9999999999975893, 1.999999999997496,],
            "s": [0.0, 0.0, 0.0, ],
            "z": [-491.42397588400473, 279.36833425936913, 191.11806896900944, ],
            "status": CLARABEL.SOLVED,
            "obj_val": 21434.491423910207,
            "obj_val_dual": 21434.491423919164,
            "solve_time": 9.79e-5,
            "iterations": 0,
            "r_prim": 5.5338076549567106e-14,
            "r_dual": 1.5829159225003945e-15,
        })

        return super().setUp()

    def test_invert_when_solved(self):
        """Tests invert when a solution is present and solver status from clarabel is SOLVED."""
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        solver_inverse_data = SolverInverseData(inverse_data[-1], solver_options={})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.OPTIMAL, solution.status)

    def test_invert_when_insufficient_progress_should_fail(self):
        """
        Tests invert when a solution is present and solver status from clarabel is
        InsufficientProgress.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solution = solver.invert(self.solution, SolverInverseData(inverse_data[-1], {}))
        self.assertEqual(s.SOLVER_ERROR, solution.status)

    def test_invert_when_insufficient_progress_but_accept_unknown(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solver_inverse_data = SolverInverseData(inverse_data[-1],
                                                solver_options={CLARABEL.ACCEPT_UNKNOWN: True})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.OPTIMAL_INACCURATE, solution.status)

    def test_invert_when_insufficient_progress_but_accept_unknown_and_no_solution(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        Nevertheless, clarabel did not return a solution and therefore the resulting
        status should be SolverError.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        self.solution.x = None
        self.solution.z = None
        solver_inverse_data = SolverInverseData(inverse_data[-1],
                                                solver_options={"accept_unknown": True})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.SOLVER_ERROR, solution.status)

    def test_unsolved_status_handling(self):
        """
        Test that the Clarabel interface correctly handles the 'Unsolved'
        status without raising a KeyError.
        """
        solver = CLARABEL()
        mock_solution = SimpleNamespace(
            status=CLARABEL.UNSOLVED,
            solve_time=0.01,
            iterations=0,
            x=None,
            z=None,
            attr={}
        )
        mock_inverse_data = SimpleNamespace(solver_options={})
        sol = solver.invert(mock_solution, mock_inverse_data)
        self.assertEqual(sol.status, s.SOLVER_ERROR)

    def test_unpack_results_with_problem_data(self):
        """
        Results of a solve through get_problem_data() / solve_via_data()
        unpack, although that path records no solver options.
        """
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0, cp.sum(x) <= 1])
        data, chain, inverse_data = prob.get_problem_data(cp.CLARABEL)
        solution = chain.solver.solve_via_data(data, False, False, {})
        prob.unpack_results(solution, chain, inverse_data)
        self.assertEqual(prob.status, s.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, [0.5, 0.5], places=5)

    def test_triu_csc(self):
        """
        The upper triangle passed to Clarabel matches scipy.sparse.triu, in
        canonical CSC form, whatever the format of P.
        """
        rng = np.random.default_rng(0)
        M = sp.random(12, 12, density=0.4, random_state=1)
        M = M + M.T
        triu = sp.csc_array(sp.triu(M))
        # an upper triangular CSC matrix with unsorted indices
        perm = np.concatenate([np.arange(a, b)[::-1] for a, b in
                               zip(triu.indptr[:-1], triu.indptr[1:])])
        unsorted = sp.csc_array((triu.data[perm], triu.indices[perm], triu.indptr),
                                shape=triu.shape)
        for P in [M.tocsc(), M.tocsr(), M.tocoo(), triu, unsorted,
                  sp.csc_array((12, 12)), sp.csc_array(rng.standard_normal((12, 12)))]:
            U = triu_csc(P)
            self.assertEqual(U.format, "csc")
            self.assertTrue(U.has_canonical_format)
            self.assertEqual((U != sp.triu(P)).nnz, 0)
        # already upper triangular and canonical: passed through
        self.assertIs(triu_csc(triu), triu)
