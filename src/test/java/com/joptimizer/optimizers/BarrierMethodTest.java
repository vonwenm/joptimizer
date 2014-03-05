/*
 * Copyright 2011-2013 JOptimizer
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
package com.joptimizer.optimizers;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.functions.BarrierFunction;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.FunctionsUtils;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.LogarithmicBarrier;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.functions.PSDQuadraticMultivariateRealFunction;
import com.joptimizer.functions.QuadraticMultivariateRealFunction;
import com.joptimizer.functions.SDPLogarithmicBarrier;
import com.joptimizer.functions.SOCPLogarithmicBarrier;
import com.joptimizer.functions.SOCPLogarithmicBarrier.SOCPConstraintParameters;
import com.joptimizer.functions.StrictlyConvexMultivariateRealFunction;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BarrierMethodTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Quadratic objective with linear eq and ineq.
	 */
	public void testOptimize() throws Exception {
		log.debug("testOptimize");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function
		double theta = 0.01522;
		DoubleMatrix2D P = pMatrix.assign(Mult.mult(theta));
		DoubleMatrix1D q = qVector.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[3];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] {-1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] {	0, 0, -1 }, 0);
	
		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setCheckProgressConditions(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.6, 0.2, 0.2 });
		// equalities
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 3);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.000001);
		assertEquals(0.5086308460954377,  sol[1], 0.000001);
		assertEquals(0.44504603834467693, sol[2], 0.000001);
	}
	
	/**
	 * Quadratic objective with linear eq and ineq
	 * with not-feasible initial point.
	 */
	public void testOptimize2() throws Exception {
		log.debug("testOptimize2");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function.
		double theta = 0.01522;
		DoubleMatrix2D P = pMatrix.assign(Mult.mult(theta));
		DoubleMatrix1D q = qVector.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[3];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] {-1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] {	0, 0, -1 }, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setNotFeasibleInitialPoint(new double[] { -0.2, 1.0, 0.2 });
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		// equalities
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 3);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.000001);
		assertEquals(0.5086308460954377,  sol[1], 0.000001);
		assertEquals(0.44504603834467693, sol[2], 0.000001);
	}

	/**
	 * Quadratic objective with linear eq and ineq
	 * without initial point.
	 */
	public void testOptimize3() throws Exception {
		log.debug("testOptimize3");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function.
		double theta = 0.01522;
		DoubleMatrix2D P = pMatrix.assign(Mult.mult(theta));
		DoubleMatrix1D q = qVector.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[3];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] {-1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] {	0, 0, -1 }, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		// equalities
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 3);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.000001);
		assertEquals(0.5086308460954377,  sol[1], 0.000001);
		assertEquals(0.44504603834467693, sol[2], 0.000001);
	}

	/**
	 * Quadratic objective with linear eq and quadratic ineq.
	 */
	public void testOptimize4() throws Exception {
		log.debug("testOptimize4");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function (Risk-Aversion).
		double theta = 0.01522;
		DoubleMatrix2D P = pMatrix.assign(Mult.mult(theta));
		DoubleMatrix1D q = qVector.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[4];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] { -1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] { 0, 0, -1 }, 0);
		inequalities[3] = FunctionsUtils.createCircle(3, 5);//not linear

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.2, 0.6, 0.2 });
		or.setInitialLagrangian(new double[]{0.5,0.5,0.5,0.5});
		// Equality constraints
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 3);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.000001);
		assertEquals(0.5086308460954377,  sol[1], 0.000001);
		assertEquals(0.44504603834467693, sol[2], 0.000001);
	}

	/**
	 * Linear objective with quadratic ineq.
	 */
	public void testOptimize1D() throws Exception {
		log.debug("testOptimize1D");

        // Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] {1}, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(1, 1);//dim=1, radius=1, center=(0,0)

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] {0});
		//or.setTolerance(1.E-10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 1);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-1., sol[0], 0.000001);
    assertEquals(-1., value,  0.000001);
	}
	
	/**
	 * Linear objective with quadratic ineq.
	 */
	public void testOptimize5() throws Exception {
		log.debug("testOptimize5");
		// START SNIPPET: BarrierMethod-1

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 1, 1 }, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 1);//dim=2, radius=1, center=(0,0)

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0, 0 });
		or.setTolerance(1.E-11);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: BarrierMethod-1
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-Math.sqrt(2)/2, sol[0], 0.000000001);//-0,70710678119
        assertEquals(-Math.sqrt(2)/2, sol[1], 0.000000001);//-0,70710678119
        assertEquals(-Math.sqrt(2),   value,  0.000000001);//-1,41421356237
	}
	
	/**
	 * Very simple linear.
	 */
	public void testSimpleLinear() throws Exception{
		log.debug("testSimpleLinear");
		// START SNIPPET: BarrierMethod-2
		
	    // Objective function (plane)
		double[] C = new double[] { 1., 1. };
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(C, 0.);

	   	//inequalities
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[4];
		inequalities[0] = new LinearMultivariateRealFunction(new double[]{ 1., 0.}, -3.);
		inequalities[1] = new LinearMultivariateRealFunction(new double[]{-1., 0.},  0.);
		inequalities[2] = new LinearMultivariateRealFunction(new double[]{ 0., 1.}, -3.);
		inequalities[3] = new LinearMultivariateRealFunction(new double[]{ 0.,-1.},  0.);
		
		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setInteriorPointMethod(JOptimizer.BARRIER_METHOD);//select the barrier interior-point method
		or.setF0(objectiveFunction);
		or.setFi(inequalities);
		or.setTolerance(1.E-12);
		
		//optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: BarrierMethod-2
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol: " + ArrayUtils.toString(sol));
		log.debug("value  : " + objectiveFunction.value(sol));
		assertEquals(0.0, sol[0], 0.000000000001);
		assertEquals(0.0, sol[1], 0.000000000001);
	}
	
	/**
	 * Linear objective with linear eq and ineq. 
	 */
	public void testOptimize6() throws Exception {
		log.debug("testOptimize6");

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 2, 1 }, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] { -1,  0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] {  0, -1 }, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.9, 0.1 });
		// Equality constraints
		or.setA(new double[][] { { 1, 1} });
		or.setB(new double[] { 1 });
		or.setCheckKKTSolutionAccuracy(true);
		or.setCheckProgressConditions(true);
		or.setToleranceInnerStep(JOptimizer.DEFAULT_TOLERANCE_INNER_STEP * 10);
		or.setTolerance(JOptimizer.DEFAULT_TOLERANCE / 10);

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(0.,   sol[0], 0.0000001);
		assertEquals(1.,   sol[1], 0.0000001);
		assertEquals(1.,   value,  0.0000001);
	}
	
	/**
	 * Linear objective with quadratic ineq 
	 * and without initial point.
	 */
	public void testOptimize7() throws Exception {
		log.debug("testOptimize7");

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 1, 1 }, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 1);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		
		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-Math.sqrt(2),   value,  0.000001);
		assertEquals(-Math.sqrt(2)/2, sol[0], 0.000001);
		assertEquals(-Math.sqrt(2)/2, sol[1], 0.000001);
	}
	
	/**
	 * Linear objective with quadratic ineq 
	 * and with infeasible initial point.
	 * min(t) s.t.
	 * x^2 <t
	 */
	public void testOptimize7b() throws Exception {
		log.debug("testOptimize7b");

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 0, 1 }, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		double[][] PMatrix = new double[][]{{2,0},{0,0}};
		double[] qVector = new double[]{0,-1};
		inequalities[0] = new PSDQuadraticMultivariateRealFunction(PMatrix, qVector, 0, true);
		//inequalities[1] = new LinearMultivariateRealFunction(new double[]{0, -1}, 0);
		
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//or.setNotFeasibleInitialPoint(new double[]{-1, 0.9999999});//this fails, the KKT system for the Phase1 problem is singular
		or.setNotFeasibleInitialPoint(new double[]{-1, 1.0000001});
		
		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(0., sol[0], 0.000001);
		assertEquals(0., sol[1], 0.000001);
		assertEquals(0., value,  0.000001);
	}
	
	/**
	 * Linear objective, quadratically constrained.
	 * It simulates the type of optimization occurring in feasibility searching
	 * in a problem with constraints:
	 * x^2 < 1
	 */
	public void testQCQuadraticProgramming() throws Exception {
		log.debug("testQCQuadraticProgramming");
		
		// Objective function (linear (x,s)->s)
		double[] C0 = new double[] { 0, 1 };
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(C0, 0);

		//inequalities x^2 < 1 + s
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		double[][] P1 = new double[][] { { 2., 0. },{ 0., 0. }};
		double[] C1 = new double[] { 0, -1 };
		inequalities[0] = new QuadraticMultivariateRealFunction(P1, C1, -1);
		
		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 2, 5});
		
		//optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol: " + ArrayUtils.toString(sol));
		log.debug("value  : " + objectiveFunction.value(sol));
		assertEquals( 0., sol[0], 0.000001);
		assertEquals(-1., sol[1], 0.000001);
  }

	/**
	 * Exponential objective with quadratic ineq. 
	 * f0 = exp[z^2], z=(x-1, y-2) 
	 * f1 = x^2+y^2<=3^2
	 */
	public void testOptimize8() throws Exception {
		log.debug("testOptimize8");
		StrictlyConvexMultivariateRealFunction objectiveFunction = new StrictlyConvexMultivariateRealFunction() {

			public double value(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				return Math.exp(Z.zDotProduct(Z));
			}

			public double[] gradient(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				return Z.assign(Mult.mult(2 * Math.exp(Z.zDotProduct(Z)))).toArray();
			}

			public double[][] hessian(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				double d = Math.exp(Z.zDotProduct(Z));
				DoubleMatrix2D ID = F2.identity(2);
				DoubleMatrix2D ret = ALG.multOuter(Z, Z, null).assign(ID, Functions.plus).assign(Mult.mult(2 * d));
				return ret.toArray();
			}

			public int getDim() {
				return 2;
			}
		};

		// Inquality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 3);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.2, 0.2 });

		// optimization
		BarrierFunction bf = new LogarithmicBarrier(inequalities, 2);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(1., sol[0], 0.000001);
		assertEquals(2., sol[1], 0.000001);
	}
	
	/**
	 * Very simple Semidefinite programming.
	 * Represents the dim=1 QCQP 
	 * min x^2
	 * 
	 * viewed as a dim=2 SDP.
	 */
	public void testSimpleSDP() throws Exception {
		log.debug("testSimpleSDP");
		
		// Objective function (variables (x,t), dim = 2)
		double[] f0 = new double[]{0,1};
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(f0, 0);
		
		//constraint in the form (A.x+b)T.(A.x+b) - c.x - d - t < 0
		double[][] A = new double[][]{{1,0}};
		double[] b = new double[] { 0 };
		double[] c = new double[] { 0,1 };
		double d = 0;
		
		//matrix F0 for SDP
		double[][] F0 = new double[][]{{1    ,b[0] },
						               {b[0] ,d    }};
		//matrices Fi for SDP
		double[][] F1 =  new double[][]{{0        ,A[0][0]},
										{A[0][0] ,c[0]   }};
		double[][] F2 =  new double[][]{{0        ,A[0][1]},
										{A[0][1] ,c[1]   }};
		
		double[][] GMatrix = new Array2DRowRealMatrix(F0).scalarMultiply(-1).getData();
		List<double[][]> FiMatrixList = new ArrayList<double[][]>();
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F1).scalarMultiply(-1).getData());
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F2).scalarMultiply(-1).getData());
		
		//optimization request
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//or.setInitialPoint(new double[] { 0.25, 0.1});
		//or.setNotFeasibleInitialPoint(new double[] { -1, -1});
		or.setCheckKKTSolutionAccuracy(true);
		
		//optimization
		BarrierFunction bf = new SDPLogarithmicBarrier(FiMatrixList, GMatrix);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
        
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0., sol[0], 0.0000001);
		assertEquals(0., sol[1], 0.0000001);
  }
	
	/**
	 * Semidefinite programming.
	 * dim=2 QCQP viewed as a dim=3 SDP.
	 */
	public void testSemidefiniteProgramming() throws Exception {
		log.debug("testSemidefiniteProgramming");
		
		// START SNIPPET: SDProgramming-2
		
		// Objective function (variables (x,y,t), dim = 3)
		double[] c = new double[]{0,0,1};
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(c, 0);
		
		//constraint in the form (A0.x+b0)T.(A0.x+b0) - c0.x - d0 - t < 0
		double[][] A0 = new double[][]{{-Math.sqrt(21./50.),  0.             , 0},
				                       {-Math.sqrt(2)/5.   , -1./Math.sqrt(2), 0}};
		double[] b0 = new double[] { 0, 0, 0 };
		double[] c0 = new double[] { 0, 0, 1 };
		double d0 = 0;
		
		//constraint (this is a circle) in the form (A1.x+b1)T.(A1.x+b1) - c1.x - d1 < 0
		double[][] A1 = new double[][]{{1,0,0},
				                       {0,1,0}};
		double[] b1 = new double[] { 2, 2, 0 };
		double[] c1 = new double[] { 0, 0, 0 };
		double d1 = Math.pow(1.75, 2);
		
		//matrix G for SDP
		double[][] G = new double[][]{{1     ,0     ,b0[0] ,0     ,0     ,0},
				                       {0     ,1     ,b0[1] ,0     ,0     ,0},
				                       {b0[0] ,b0[1] ,d0    ,0     ,0     ,0},
				                       {0     ,0     ,0     ,1     ,0     ,b1[0]},
				                       {0     ,0     ,0     ,0     ,1     ,b1[1]},
				                       {0     ,0     ,0     ,b1[0] ,b1[1] ,d1}};
		//matrices Fi for SDP
		double[][] F1 =  new double[][]{{0        ,0        ,A0[0][0] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][0] ,0        ,0        ,0},
						                {A0[0][0] ,A0[1][0] ,c0[0]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][0]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][0]},
						                {0        ,0        ,0        ,A1[0][0] ,A1[1][0] ,c1[0]}};
		double[][] F2 =  new double[][]{{0        ,0        ,A0[0][1] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][1] ,0        ,0        ,0},
						                {A0[0][1] ,A0[1][1] ,c0[1]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][1]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][1]},
						                {0        ,0        ,0        ,A1[0][1] ,A1[1][1] ,c1[1]}};
		double[][] F3 =  new double[][]{{0        ,0        ,A0[0][2] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][2] ,0        ,0        ,0},
						                {A0[0][2] ,A0[1][2] ,c0[2]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][2]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][2]},
						                {0        ,0        ,0        ,A1[0][2] ,A1[1][2] ,c1[2]}};
		
		double[][] GMatrix = new Array2DRowRealMatrix(G).scalarMultiply(-1).getData();
		List<double[][]> FiMatrixList = new ArrayList<double[][]>();
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F1).scalarMultiply(-1).getData());
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F2).scalarMultiply(-1).getData());
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F3).scalarMultiply(-1).getData());
		
		//optimization request
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//or.setInitialPoint(new double[] { -0.8, -0.8, 10});
		
		//optimization
		BarrierFunction bf = new SDPLogarithmicBarrier(FiMatrixList, GMatrix);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: SDProgramming-2
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
        
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(-2 + 1.75/Math.sqrt(2), sol[0], 0.000001);//-0,7625631
		assertEquals(-2 + 1.75/Math.sqrt(2), sol[1], 0.000001);//-0,7625631
		assertEquals(0.814103544571        , sol[2], 0.000001);
  }
	
	/**
	 * Second-order cone programming on the Lorentz cone.
	 * Submitted 20/11/2012 by Jerry Pratt.
	 */
	public void testSOCPLorentz() throws Exception {
 		log.debug("testSOCPLorentz");
 		
 		// Objective function
 		double[] c = new double[] {1.0, 1.0, 0.0};
 		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(c, 0);
 		
 		//equalities
		double[][] A = new double[][]{{0,0,1}};
		double[] b = new double[]{1};

		List<SOCPConstraintParameters> socpConstraintParametersList = new ArrayList<SOCPLogarithmicBarrier.SOCPConstraintParameters>();
 		SOCPLogarithmicBarrier barrierFunction = new SOCPLogarithmicBarrier(socpConstraintParametersList, 3);

//       second order cone constraint in the form ||A1.x+b1||<=c1.x+d1,
 		double[][] A1 = new double[][] {{1,0,0},{0,1,0}};
 		double[] b1 = new double[] { 0, 0 };
 		double[] c1 = new double[] { 0, 0, 1 };
 		double d1 = 0;
 		SOCPConstraintParameters constraintParams1 = barrierFunction.new SOCPConstraintParameters(A1, b1, c1, d1);
 		socpConstraintParametersList.add(socpConstraintParametersList.size(), constraintParams1);
 		
 		//optimization problem
 		OptimizationRequest or = new OptimizationRequest();
 		or.setF0(objectiveFunction);
 		or.setA(A);
		or.setB(b);
		//or.setInitialPoint(new double[] { 0.5, 0.5, 1});
		//or.setNotFeasibleInitialPoint(new double[] { 0.5, 0.5, 1});
 		//or.setCheckKKTSolutionAccuracy(true);
		or.setCheckProgressConditions(true);
 		
 		//optimization
 		BarrierMethod opt = new BarrierMethod(barrierFunction);
 		opt.setOptimizationRequest(or);
 		int returnCode = opt.optimize();
 		
 	    if(returnCode==OptimizationResponse.FAILED){
 			fail();
 		}
 		
 		OptimizationResponse response = opt.getOptimizationResponse();
 		double[] sol = response.getSolution();
 		log.debug("sol: " + ArrayUtils.toString(sol));
 		log.debug("value  : " + objectiveFunction.value(sol));
 		assertEquals(-Math.sqrt(2)/2, sol[0], 1e-5);
 		assertEquals(-Math.sqrt(2)/2, sol[1], 1e-5);
 		assertEquals(1., sol[2], 1e-5);
  } 
	
	/**
	 * Second-order cone programming on the Lorentz cone
	 * with additional inequality constraint
	 */
	public void testSOCPLorentzIneq() throws Exception   {
	      log.debug("testSOCPLorentzIneq");

	      double[] minimizeF = new double[] {-1.0, -1.0, 0.0};
	      LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(minimizeF, 0.0);

	      List<SOCPConstraintParameters> socpConstraintParametersList = new ArrayList<SOCPLogarithmicBarrier.SOCPConstraintParameters>();
	   		SOCPLogarithmicBarrier barrierFunction = new SOCPLogarithmicBarrier(socpConstraintParametersList, 3);

	   		//second order cone constraint in the form ||A1.x+b1||<=c1.x+d1 (Lorentz cone)
	   		double[][] A1 = new double[][] {{1,0,0},{0,1,0}};
	   		double[] b1 = new double[] { 0, 0 };
	   		double[] c1 = new double[] { 0, 0, 1 };
	   		double d1 = 0;
	   		SOCPConstraintParameters constraintParams1 = barrierFunction.new SOCPConstraintParameters(A1, b1, c1, d1);
	   		socpConstraintParametersList.add(socpConstraintParametersList.size(), constraintParams1);
	   		
	   	  //second order cone constraint in the form ||A1.x+b1||<=c1.x+d1 (z < Sqrt[18])
	   		double[][] A2 = new double[][] {{0,0,1}};
	   		double[] b2 = new double[] { 0 };
	   		double[] c2 = new double[] { 0, 0, 0 };
	   		double d2 = Math.sqrt(18);
	   		SOCPConstraintParameters constraintParams2 = barrierFunction.new SOCPConstraintParameters(A2, b2, c2, d2);
	   		socpConstraintParametersList.add(socpConstraintParametersList.size(), constraintParams2);
	   		
	   	//optimization problem
	   		OptimizationRequest or = new OptimizationRequest();
	   		or.setF0(objectiveFunction);
	   		//or.setInitialPoint(new double[] { 0, 0, 1});
	   		
	   		//optimization
	   		BarrierMethod opt = new BarrierMethod(barrierFunction);
	   		opt.setOptimizationRequest(or);
	   		int returnCode = opt.optimize();
	   		
	   	  if(returnCode==OptimizationResponse.FAILED){
	   			fail();
	   		}
	   		
	   		OptimizationResponse response = opt.getOptimizationResponse();
	   		double[] sol = response.getSolution();
	   		log.debug("sol: " + ArrayUtils.toString(sol));
	   		log.debug("value  : " + objectiveFunction.value(sol));
	   		assertEquals(3.0, sol[0], 1e-5);
	      assertEquals(3.0, sol[1], 1e-5);
	      assertEquals(Math.sqrt(18.0), sol[2], 1e-5);
	   }
	
	/**
	 * Second-order cone programming in 2D.
	 */
	public void testSOConeProgramming2D() throws Exception {
		log.debug("testSOConeProgramming2D");
		
		// START SNIPPET: SOConeProgramming-2
		
		// Objective function (plane)
		double[] c = new double[] { -1., -1. };
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(c, 6);
		
		//equalities
		double[][] A = new double[][]{{1./4.,-1.}};
		double[] b = new double[]{0};

		List<SOCPConstraintParameters> socpConstraintParametersList = new ArrayList<SOCPLogarithmicBarrier.SOCPConstraintParameters>();
		SOCPLogarithmicBarrier barrierFunction = new SOCPLogarithmicBarrier(socpConstraintParametersList, 2);

//      second order cone constraint in the form ||A1x+b1||<=c1.x+d1,
		double[][] A1 = new double[][] {{ 0, 1. }};
		double[] b1 = new double[] { 0 };
		double[] c1 = new double[] { 1./3., 0. };
		double d1 = 1./3.;
		SOCPConstraintParameters constraintParams1 = barrierFunction.new SOCPConstraintParameters(A1, b1, c1, d1);
		socpConstraintParametersList.add(socpConstraintParametersList.size(), constraintParams1);
		
//      second order cone constraint in the form ||A2x+b2||<=c2.x+d2,
		double[][] A2 = new double[][] {{ 0, 1. }};
		double[] b2 = new double[] { 0};
		double[] c2 = new double[] { -1./2., 0};
		double d2 = 1;
		SOCPConstraintParameters constraintParams2 = barrierFunction.new SOCPConstraintParameters(A2, b2, c2, d2);
		socpConstraintParametersList.add(socpConstraintParametersList.size(), constraintParams2);
        
		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0., 0.});
		or.setA(A);
		or.setB(b);
		or.setCheckProgressConditions(true);
		
		//optimization
		BarrierMethod opt = new BarrierMethod(barrierFunction);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
	  // END SNIPPET: SOConeProgramming-2
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol: " + ArrayUtils.toString(sol));
		log.debug("value  : " + objectiveFunction.value(sol));
		assertEquals(4./3., sol[0], 0.0000001);
		assertEquals(1./3., sol[1], 0.0000001);
  }
}
