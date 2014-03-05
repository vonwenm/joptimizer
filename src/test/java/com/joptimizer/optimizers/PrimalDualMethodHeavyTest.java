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

import java.util.Arrays;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class PrimalDualMethodHeavyTest extends TestCase {

	private int dim = 10;
	private long seed = 7654321L;
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Quadratic objective with linear eq and ineq.
	 */
	public void testOptimize() throws Exception {
		log.debug("testOptimize");

		// Objective function
		DoubleMatrix2D P = Utils.randomValuesPositiveMatrix(dim, dim, -0.5, 0.5, seed);
		DoubleMatrix1D q = Utils.randomValuesMatrix(1, dim, -0.5, 0.5, seed).viewRow(0);

		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		// equalities
		double[][] AEMatrix = new double[1][dim];
		Arrays.fill(AEMatrix[0], 1.);
		double[] BEVector = new double[] { 1 };

		// inequalities
		double[][] AIMatrix = new double[dim][dim];
		for (int i = 0; i < dim; i++) {
			AIMatrix[i][i] = -1;
		}
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[dim];
		for (int i = 0; i < dim; i++) {
			inequalities[i] = new LinearMultivariateRealFunction(AIMatrix[i], 0);
		}

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		double[] ip = new double[dim];
		Arrays.fill(ip, 1. / dim);
		or.setInitialPoint(ip);
		or.setA(AEMatrix);
		or.setB(BEVector);
		or.setFi(inequalities);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if (returnCode == OptimizationResponse.FAILED) {
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
	}
}
