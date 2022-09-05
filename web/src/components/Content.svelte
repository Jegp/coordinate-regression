<content>
	<div class="head">
		<video class="background" autoplay loop>
			<source src="2209_event_video.mp4" type="video/mp4" />
			<source src="2209_event_video.webm" type="video/webm" />
		</video>
		<div class="hover">
			<h1>Coordinate regression for event-based vision</h1>
			<div class="box">
				<div class="author">
					Authors: <a href="https://jepedersen.dk">Jens E. Pedersen</a>, J. P.
					Romero. B, & J. Conradt
					<br />
					Contact: <i class="fa fa-envelope" />
					<a href="mailto:jeped@kth.se">jeped@kth.se</a>
					- <i class="fab fa-twitter" />
					<a href="https://twitter.com/jensegholm">@jensegholm</a>
					- <i class="fab fa-github" />
					<a href="https://github.com/jegp/coordinate-regression"
						>github.com/jegp/coordinate-regression</a
					>
				</div>
				<br />
				We present a novel method to predict translation-invariant spatial coordinates
				from sparse, event-based vision (EBV) signals using a fully-spiking convolutional
				neural network (SCNN).
			</div>

			<div class="box">
				<h2>The problem: predicting coordinates with sparse events</h2>
				<p>
					The video playing in the background illustrates the sparseness of
					event-based vision (EBV) cameras. Most pixels are completely empty
					(~95%), because EBV cameras work by detecting <b>
						luminosity changes over time</b
					>. For conventional artificial neural networks this is a challenge.
					Partly because the signal is sparse and
					<i>requires integration over time</i>
					(recurrence) to form coherent "pictures" of objects, and partly because
					conventional hardware struggles to keep up with EBV cameras sending more
					than 20M events per second.
				</p>
				<p>
					Our work here focuses on coordinate regression for event-based data,
					aiming to
					<b>predict center coordinates of geometric shapes</b> using
					<b>spiking neural neural networks</b>, designed for
					<b>neuromorphic hardware</b>.
				</p>

				<h2>Results: LIF beats ReLU</h2>
				<p>
					In our setting, a biologically inspired spiking neural network (SNN)
					with receptive fields (RF) outperform conventional convolutional
					artificial neural network (ANN). The plot below shows the pixel-wise
					error for predictions against unseen validation data for four
					different models.
				</p>
				<img src="pred_horiz.png" />
			</div>
		</div>
	</div>

	<div class="block">
		<div class="section">
			<h2>Why spiking neural networks?</h2>
			<p>
				Spiking neural networks (SNNs) and neuromorphic hardware are inherently
				parallel, asynchronous, low-energy devices that promise accelerated
				compute capacities for event-based machine learning systems. This work
				addresses coordinate regression using an inherently asynchronous and
				parallel spiking architecture that is both compatible with neuromorphic
				hardware, and, we hope, <b
					>a step towards more useful neuromorphic algorithms</b
				>.
			</p>
		</div>

		<div class="section">
			<h2>The task: event-based dataset</h2>
			<p>
				We construted a dataset of geometric shapes (circles, triangles,
				squares) with center coordinate labels. To provide realistic, structured
				noise, we superimposed the sparse shapes on office-like scene renderings
				from the
				<a href="https://neurorobotics.net/">NeuroRobotics Platform (NRP)</a>.
				The shapes are sparsely sampled from a Bernouilli distribution (p=0.8)
				and are moving around with a brownian motion. The task is for any
				network to recognize the shapes and predict their center as accurately
				as possible.
			</p>
			<p>
				We chose the geometric shapes as a means to control parameters such as
				shape velocity, event density, and shape complexity. For instance, the
				current shapes have a radius of 80 pixels, which is too large for any
				single kernel to learn. The convolutional layer is, therefore, forced to
				specialize on partial features to correctly solve the task.
			</p>
			<center>
				<img src="fig1_color.png" style="width: 30%;" />
				<img src="fig1_circle.png" style="width: 30%;" />
			</center>
			<p>
				In total, the dataset contains 2000 videos of 60 frames each (resembling 1ms of events) with a resolution of 640x480.
			</p>
		</div>
		<div class="section">
			<h2>Neural network architecture</h2>
			<p>
				We use a classical convolutional model consisting of two convolutional
				layers, followed by an inverse convolution for a slight upsampling. We
				implemented a differentiable coordinate transform (DCT) layer to
				transform the pixel-wise activations into a 2-dimensional coordinate.
				Each convolution is interspersed with non-linear activations, batch
				normalization and dropout (p=0.1).
			</p>

			<center>
				<img src="net.png" style="width: 80%;" />
			</center>
		</div>

		<div class="section">
			<h2>Translation-invariant receptive fields</h2>
			<p>
				Similar to conventional convolutional systems, we can define receptive
				field kernels for spiking neural networks. Importantly, we wish to
				retain translation-invariant properties to capture the moving shapes
				over time, as neatly illustrates in the work by <a
					href="https://www.sciencedirect.com/science/article/pii/S2405844021000025"
					>[Lindeberg 2021]</a
				>.
			</p>
			<center>
				<img src="fig_rf.png" style="width:60%;" />
			</center>
			<p>
				Preconfiguring the receptive field kernels for the spiking neural
				networks significantly reduces training time, memory consumption, and
				ability for the network to generalize, as seen in the loss curves above.
			</p>
		</div>

		<div class="section">
			<h2>Training and validation</h2>
			<p>
				We constructed and trained four independent networks over the same
				architecture
				</p>
			<ol>

				<li>a non-spiking network where ReLU units constitute the
				nonlinearities (ANN)</li>
				<li>a non-spiking network with custom receptive field kernels (ANN-RF)</li>
				<li>a spiking version, with three leaky integrate-and-fire (LIF)
				nonlinearities feeding a final, non-spiking leaky integrator (SNN)</li>
				<li>a
				spiking version, resembling (3), but where custom receptive field
				kernels promote translation-invariant feature recognition (SNN-RF)</li>
			</ol>
			<p>
				The networks were trained with backpropagation-through-time using a regular
				l<sup>2</sup>-loss via the differentiable coordinate transform (DCT)
				method (presented futher below). The models are tested on unseen
				validation data (20% of the total training data).
			</p>
		</div>

		<div class="section">
			<h2>Prediction errors and performance</h2>
			<center>
				<img src="fig_loss2.png" style="width: 80%;" />
			</center>
			<p>
				The receptive field version of the SNN outperforms even the artificial
				neural network. If we further explore the prediction errors of the
				models (sampled over the entire validation dataset), the performance
				benefit becomes clearer: the predicted coordinates from the receptive
				field model is significantly closer to the actual, labelled coordinates.
			</p>
			<center>
				<img src="pred_horiz.png" />
			</center>
		</div>

		<div class="section">
			<h2>Future work</h2>
			<p>
				This is still work in progress and more work is needed to generalize the results.
				However, there are already a few extensions that would immediately be interesting to explore
			</p>
			<ol>
				<li>
					The gaussian structure of the output predictions can be exploited to
					further increase prediction accuracy.
				</li>
				<li>
					The current shapes are quite dense (Bernouilli p=0.8), such that the artificial networks
					are able to converge to the shapes in the given frames. We wish to
					explore the sparseness of the shapes (lower the Bernouilli
					distribution of the shapes) while exploring the temporal process of
					recurrent artificial (non-spiking) networks.
				</li>
				<li>
					We currently focus on translation-invariance. We wish to extend our
					method to more complex shapes that require both scale- and
					rotation-invariance.
				</li>
			</ol>
		</div>
		<div class="box author">
			<h2>Acknowledgements</h2>
			Thank you to all the people in the
			<a href="https://neurocomputing.systems">Neurocomputing Systems Lab</a>.
			at
			<a href="https://kth.se">KTH Royal Institute of Technology</a>, where this
			work was done. We graciously acknowledge the funding we received from the
			<a href="https://www.humanbrainproject.eu/">Human Brain Project</a>. We
			also owe a debt of gratitude to the
			<a href="https://www.aicentre.dk/">Copenhagen AI Pioneer Centre</a>.
			<br />
			<center>
				<a class="borderless" href="https://neurocomputing.systems"
					><img src="ncs.png" /></a
				>
				<a class="borderless" href="https://humanbrainproject.eu"
					><img src="hbp.png" /></a
				>
				<a class="borderless" href="https://kth.se"><img src="kth.png" /></a>
			</center>
		</div>
	</div>
</content>

<style>
	@font-face {
		font-family: "Lato";
		src: url("Lato-Regular.ttf") format("ttf");
	}
	content {
		position: relative;
		margin: 0 auto;
		font-family: "Lato";
		font-size: 120%;
	}
	h1 {
		font-size: 300%;
		text-align: center;
		margin-top: 0;
	}
	.hover {
		position: relative;
		position: relative;
		min-height: 100vh;
	}
	.background {
		width: 100vw;
		position: relative;
		z-index: -1000;
		margin: -12% 0 -60% 0;
	}
	.box {
		min-width: 360px;
		max-width: 60vw;
		margin: 1em auto;
		background-color: rgba(255, 255, 255, 0.85);
		border: 2px solid #888;
		border-radius: 1cm;
		padding: 1em;
	}
	.author {
		color: #444;
	}
	.author a {
		color: #333;
	}
	.author img {
		margin: 1em;
		height: 6cm;
		display: inline-block;
	}
	.section img {
		display: inline-block;
	}
	.borderless {
		border-bottom: 0;
		text-decoration: none;
	}
	.block {
		border-top: 3px solid #282828;
		background-color: #fff;
		margin: 0 auto;
	}
	.section {
		min-width: 360px;
		width: 60vw;
		margin: 0 auto;
	}
	.head {
		margin: 0;
	}
</style>
