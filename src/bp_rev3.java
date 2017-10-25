	//============================================================================
	// Name        : back-prop.cpp
	// Copyright   : BSDNew
	// Description : A simple backprop program
	//============================================================================
        // http://mochajl.readthedocs.io/en/latest/user-guide/neuron.html
	// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
	// https://www.youtube.com/watch?v=IHZwWFHWa-w
	// https://stats.stackexchange.com/questions/185071/can-neural-network-e-g-convolutional-neural-network-have-negative-weights
	// http://neuralnetworksanddeeplearning.com/chap2.html
	// https://www.youtube.com/watch?v=L_PByyJ9g-I

public class BackProp {
	// Edges are initialised to values in the range +/- 0.3
	double BIAS = 0.0;
	double RATE = 0.6;
	
	// a flip-flop array; to extend the propagation to multiple hidden layers
	double[][] delta = new double[2][];
	
	// Edges are linked to the target, not the origin:
	//  ...
	//  e \
	//  e -n
	//  e /
	//  ... 
	double[] i;    	// input value                  (immutable)
	double[][] ih; 	// input-hidden edge weights    (mutable)
	double[] h;    	// hidden activation            (mutable)
	double[][] ho; 	// hidden-output edge weights	(mutable)
	double[] o;    	// output activation            (mutable)
	
	double[] h_bias;// hidden bias			(mutable)
	double[] o_bias;// output bias                  (mutable)

	double[] t;	// target value                 (immutable)


	private double activate(double value) {
		return 1.0 / (1.0 + Math.exp(-value));
		// ReLU Activation
		// return Math.max(0.001 * value, value);
	}

	private double derivative(double value) {
		return value * (1.0 - value);
		// ReLU Derivative
		// if (value > 0) 
		//	return 1;
		// else
		//	return 0.001;
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

		// bias activations
		h_bias = new double[h_len];
		for (int n = 0; n < h_len; n++)
			h_bias[n] = -10.5 + Math.random() * 20;
		o_bias = new double[o_len];
		for (int n = 0; n < o_len; n++)
			o_bias[n] = -10.5 + Math.random() * 20;

		// edge activations
		ih = new double[i_len][h_len];
		for (int n = 0; n < i_len; n++)
			for (int e = 0; e < h_len; e++)
				ih[n][e] = -10 + Math.random() * 20;
		ho = new double[h_len][o_len];
		for (int n = 0; n < h_len; n++)
			for (int e = 0; e < o_len; e++)
				ho[n][e] = -10 + Math.random() * 20;
		
		delta[0] = new double[o_len];
		delta[1] = new double[h_len];
	}

	// delta[0] = output delta
	// delta[1] = hidden delta
	double back_prop() {
		// output error
		for (int n = 0; n < o.length; n++) {
			delta[0][n] = -(t[n] - o[n]) * derivative(o[n]);
			o_bias[n] -= RATE * delta[n][0];
			// System.out.println(o[n]);
			// System.out.println(o_bias[n]);
		}

		// hidden-output error
		for (int n = 0; n < h.length; n++) {
			for (int e = 0; e < o.length; e++) {
				ho[n][e] -= RATE * delta[0][e] * h[n];
				System.out.println(ho[n][e]);
			}
		}
		
		// hidden error
		double sum = 0;
		for (int e = 0; e < h.length; e++) {
			for (int n = 0; n < o.length; n++) {
				sum += delta[0][n] * ho[e][n];
			}
			delta[1][e] = sum * derivative(h[e]);
		}
		
		// input-hidden error
		for (int n = 0; n < i.length; n++) {
			h_bias[n] -= RATE * delta[1][n];
			for (int e = 0; e < h.length; e++) {
				ih[n][e] -= RATE * delta[1][e] * i[n];
			}
		}
		return 0.0;
	}

    void push_forward() {
    	// hidden activations
    	double sum = 0;
    	for (int e = 0; e < h.length; e++) {
    		for (int n = 0; n < i.length; n++) {
    			sum += i[n] * ih[n][e];
		}
		h[e] = activate(sum + h_bias[e]);
    	}
    	// output activations
    	sum = 0;
    	for (int e = 0; e < o.length; e++) {
    		for (int n = 0; n < h.length; n++) {
			sum += h[n] * ho[n][e];
		}
    		o[e] = activate(sum + o_bias[e]);
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
		
		BackProp n = new BackProp(2,3,1);
		n.train(patterns, 10000);
		n.test(patterns);
		
	}
}
