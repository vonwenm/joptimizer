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
package com.joptimizer.util;

import java.util.List;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskyFactorizationTest extends TestCase {
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testInvert1() throws Exception {
		log.debug("testInvert1");
		double[][] QData = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		RealMatrix Q = MatrixUtils.createRealMatrix(QData);

		CholeskyFactorization myc = new CholeskyFactorization(QData);
		RealMatrix L = new Array2DRowRealMatrix(myc.getL());
		RealMatrix LT = new Array2DRowRealMatrix(myc.getLT());
		log.debug("L: " + L);
		log.debug("LT: " + LT);
		log.debug("L.LT: " + L.multiply(LT));
		log.debug("LT.L: " + LT.multiply(L));
		
		// check Q = L.LT
		double norm = L.multiply(LT).subtract(Q).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
		
		RealMatrix LInv = new SingularValueDecomposition(L).getSolver().getInverse();
		log.debug("LInv: " + ArrayUtils.toString(LInv.getData()));
		RealMatrix LInvT = LInv.transpose();
		log.debug("LInvT: " + ArrayUtils.toString(LInvT.getData()));
		RealMatrix LTInv = new SingularValueDecomposition(LT).getSolver().getInverse();
		log.debug("LTInv: " + ArrayUtils.toString(LTInv.getData()));
		RealMatrix LTInvT = LTInv.transpose();
		log.debug("LTInvT: " + ArrayUtils.toString(LTInvT.getData()));
		log.debug("LInv.LInvT: " + ArrayUtils.toString(LInv.multiply(LInvT).getData()));
		log.debug("LTInv.LTInvT: " + ArrayUtils.toString(LTInv.multiply(LTInvT).getData()));
		
		RealMatrix Id = MatrixUtils.createRealIdentityMatrix(Q.getRowDimension());
		//check Q.(LTInv * LInv) = 1
		norm = Q.multiply(LTInv.multiply(LInv)).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 5.E-15);
		
		// check Q.QInv = 1
		RealMatrix QInv = MatrixUtils.createRealMatrix(myc.getInverse());
		norm = Q.multiply(QInv).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
		
		//check eigenvalues
		double det1 = Utils.calculateDeterminant(QData, QData.length);
		double det2 = 1;
		List<Double> eigenvalues = myc.getEigenvalues();
		for(double ev : eigenvalues){
			det2 = det2 * ev; 
		}
		log.debug("det1: " + det1);
		log.debug("det2: " + det2);
		assertEquals(det1, det2, 1.E-13);
	}
}
