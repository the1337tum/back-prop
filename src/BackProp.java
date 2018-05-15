	//============================================================================
	// Name        : back-prop.cpp
	// Copyright   : BSDNew
	// Description : A simple backprop program
	//============================================================================
        //http://mochajl.readthedocs.io/en/latest/user-guide/neuron.html

// example test data:
// http://www.cs.otago.ac.nz/cosc420/data.html

// text to read
// http://neuralnetworksanddeeplearning.com/

// Remove and restructure delta code: apply error directly to nodes
public class BackProp {
	// Edges are initialised to values in the range +/- 0.3
	double BIAS = 0.0;
	double RATE = 0.6;
	
	// a flip-flop array; to extend the propagation to multiple hidden layers
	double[][] delta = new double[2][];
	
	// Edges are linked to the target, not the origin:
	//  ...
	//   /e
	//  n-e
	//   \e
	//  ... 
	double[] i;    	// input value                  (immutable)
	double[][] ih; 	// input-hidden edge weights    (mutable)
	double[] h;    	// hidden activation            (mutable)
	double[][] ho; 	// hidden-output edge weights	(mutable)
	double[] o;    	// output activation            (mutable)
	
	double[] t;	// target value                 (immutable)

	// fuck sigmoids, they're too complicated.
	private double compress(double value) {
		return Math.tan(value);
	}
	
	// n = node, e = edge
	BackProp(int i_len, int h_len, int o_len) {
		// node activations
		i = new double[i_len];
		for (int n = 0; n < i_len; n++)
			i[n] = 1.0;
		h = new double[h_len];
		for (int n = 0; n < h_len; n++)
			h[n] = 1.0;
		o = new double[o_len];
		for (int n = 0; n < o_len; n++)
			o[n] = 1.0;
		
		// edge activations
		ih = new double[i_len][h_len];
		for (int n = 0; n < i_len; n++)
			for (int e = 0; e < h_len; e++)
				ih[n][e] = -0.3 + Math.random()*0.6; // +/- 0.3
		ho = new double[h_len][o_len];
		for (int n = 0; n < h_len; n++)
			for (int e = 0; e < o_len; e++)
				ho[n][e] = -0.3 + Math.random()*0.6; // +/- 0.3
		
		delta[0] = new double[o_len];
		delta[1] = new double[h_len];
	}

	// delta[0] = output delta
	// delta[1] = hidden delta
	// edges += or just = ?
	double back_prop() {
		double variance = 0.0;
		
		// output error
		for (int n = 0; n < o.length; n++) {
			delta[0][n] = (t[n] - o[n]) * o[n] * (1 - o[n]);
			variance += 0.5 * (t[n] - o[n]) * (t[n] - o[n]);
		}
		
		// hidden-output error
		for (int n = 0; n < h.length; n++) {
			for (int e = 0; e < o.length; e++)
				ho[n][e] = RATE * delta[0][e] * h[n];
		}
		
		// hidden error - TODO: apply new hidden node value
		for (int n = 0; n < h.length; n++) {
			delta[1][n] = 0;
			for (int e = 0; e < o.length; e++)
				delta[1][n] += delta[0][e] * ho[n][e]; // should this be delta[1][n] += delta[0][n]?
			delta[1][n] *= h[n] * (1 - h[n]);
		}
		
		// input-hidden error
		for (int n = 0; n < i.length; n++) {
			for (int e = 0; e < h.length; e++)
				ih[n][e] = RATE * delta[1][n] * i[n];
		}
		
		return variance;
	}

    // TODO: Hidden node should have a hidden weight - recode and test.
    void push_forward() {
    	// hidden activations
    	int sum = 0;
    	for (int e = 0; e < h.length; e++) {
    		for (int n = 0; n < i.length; n++)
    			sum += i[n] + ih[n][e];   // ih[h_len][i_len]
    		h[e] = compress(sum);
    	}
    	// output activations
    	sum = 0;
    	for (int e = 0; e < o.length; e++) {
    		for (int n = 0; n < h.length; n++)
    			sum += h[n] + ho[n][e]; 
    		o[e] = compress(sum);
    	}
	}
    
    void train(double[][][] patterns, int cycles) {
    	for (int epoch = 0; epoch < cycles; epoch++) {
        	double error = 0.0;
    		for (int p = 0; p < patterns.length; p++) {
    			i = patterns[p][0];
    			t = patterns[p][1];
    			push_forward();
    			error = back_prop();
    		}
    	}
    }
    
    void test(double[][][] patterns) {
		for (int p = 0; p < patterns.length; p++) {
			i = patterns[p][0];
			t = patterns[p][1];
			push_forward();
			back_prop();
			
			for (int n = 0; n < i.length; n++)
				System.out.print(i[n] + " ");
			System.out.print("-> ");
			for (int n = 0; n < o.length; n++)
				System.out.println(o[n] + " ");
		}
    }
    
	public static void main(String[] args) {

		double[][][] patterns = {
				{{0,0}, {0}},
				{{0,1}, {1}},
				{{1,0}, {1}},
				{{1,1}, {0}}
		};
		
		BackProp n = new BackProp(2,2,1);
		n.train(patterns, 1000);
		n.test(patterns);
		
	}
}
