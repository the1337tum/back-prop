	//============================================================================
	// Name        : back-prop.cpp
	// Copyright   : BSDNew
	// Description : A simple backprop program
	//
	//============================================================================


public class BackProp {
	// Edges are initialised to values in the range +/- 0.3
	double BIAS = 0.0;
	double RATE = 0.9;
	
	// a flip-flop array; to extend the propagation to multiple hidden layers
	double[][] delta = new double[2][];
	
	// Edges are linked to the target, not the origin:
	//  ...  ...
	// 	e\   e\
	//	e-n  e-n
	//	e/   e/
	//  ...  ...
	double[] i;    	// input value			    	(immutable)
	double[][] ih; 	// input-hidden edge weights	(mutable)
	double[] h;    	// hidden activation			(mutable)
	double[][] ho; 	// hidden-output edge weights	(mutable)
	double[] o;    	// output activation			(mutable)
	
	double[] t;		// target value 				(immutable)

	// fuck sigmoids, they're too complicated.
	private double compress(double value) {
		return Math.tan(value);
	}
	
	private double decompress(double value) {
		return 1.0 - value * value;
	}
	
	// n = node, e = edge
	BackProp(int i_len, int h_len, int o_len) {
		// node activations
		i = new double[i_len];
		for (int n = 0; n < i_len; n++)
			i[n] = 1;
		h = new double[h_len];
		for (int n = 0; n < h_len; n++)
			h[n] = 1;
		o = new double[o_len];
		for (int n = 0; n < o_len; n++)
			o[n] = 1;
		
		// edge activations
		ih = new double[i_len][h_len];
		for (int n = 0; n < i_len; n++)
			for (int e = 0; e < h_len; e++)
				ih[n][e] = -0.3 + Math.random()*0.6; // +/- 0.3
		ho = new double[h_len][o_len];
		for (int n = 0; n < h_len; n++)
			for (int e = 0; e < o_len; e++)
				ih[n][e] = -0.3 + Math.random()*0.6; // +/- 0.3
	}

	// delta[0] = output delta
	// delta[1] = hidden delta
	// Check FIXME values against final page of lecture notes. Then add momentum.
	double push_back() {
		double variance = 0.0;

		// Output error
		for (int n = 0; n < o.length; n++) {
			double ov = decompress(o[n]);
			delta[0][n] = (t[n] - ov) * ov * (1 - ov); // FIXME

			for (int e = 0; e < ho.length; e++)
				ho[n][e] += RATE * delta[0][n] * o[n]; // FIXME
			h[n] += delta[0][n];

			variance += (t[n] - o[n]) * (t[n] - o[n]); // FIXME
		}
		
		// Propagated error
		for (int n = 0; n < h.length; n++) {
			for (int e = 0; e < ih[n].length; e++)
				delta[1][n] += delta[0][n] * ih[n][e]; // FIXME
			double hv = decompress(h[n]);
			delta[1][n] *= RATE * hv * (1 - hv);	   // FIXME

			for (int e = 0; e < ih[n].length; e++)
				ih[n][e] += delta[1][n] * i[n];			// FIXME
		}
		
		return 1/2 * variance;							// FIXME
	}

    void push_forward() {
    	// hidden activations
    	int sum = 0;
    	for (int n = 0; n < ih.length; n++) {
    		for (int e = 0; e < ih[0].length; e++)
    			sum += i[n] + ih[e][n];
    		h[n] = compress(sum);
    	}
    	// output activations
    	sum = 0;
    	for (int n = 0; n < ho.length; n++) {
    		for (int e = 0; e < ho[0].length; e++)
    			sum += h[n] + ho[e][n];
    		o[n] = compress(sum);
    	}
	}
    
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.printf("Usage: BackProp filename\n" +
						  "\t   Space deliniates nodes" +
						  "\t   New-lines deliniate input data");
	}
}
