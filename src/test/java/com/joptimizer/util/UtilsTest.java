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

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class UtilsTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDummy() {
		log.debug("testDummy");
	}

	public void testNorm() {
		log.debug("testNorm");
		double[] d = new double[] { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1 };
		DoubleMatrix1D v = F1.make(d);
		double n0 = v.zDotProduct(v);
		//double n1 = ALG.norm1(v);
		double n2 = ALG.norm2(v);
		log.debug("n0: " + n0);
		//log.debug("n1: " + n1);
		log.debug("n2: " + n2);
		assertEquals(n0, n2);
	}
}
