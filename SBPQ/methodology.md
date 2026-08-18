\subsection{Problem Formulation}
\begin{figure}[tb]
    \centering
    \vspace{-10pt}
    \includegraphics[width=\linewidth]{figures/teaser.png}
    \vspace{-1pt}
    \caption{Structure Aware Probabilistics Quantization.}
    \vspace{-1pt}
    \label{fig:teaser}
\end{figure}
Our method is shown in \Cref{fig:teaser}. Consider a pretrained neural network $f(\mathbf{X}; \mathbf{\theta})$ with $L$ layers, input $\mathbf{X} \in \mathcal{X}$, and the full-precision parameters $\mathbf{\theta}$. We represent the network as a composition of $B$ quantizable blocks,
\begin{equation}
    f(\mathbf{X}; \mathbf{\theta})=g_L \circ g_{L-1} \circ \cdots \circ g_1(\mathbf{X}),
\end{equation}
where each block $g_{b}$ corresponds to a structural unit of the network, \eg a transformer or residual block. Let $\mathcal{D}_{cal} = \{\mathbf{X}_i\}_{i=1}^N$ denote a calibration set of $N$ samples. The goal of post-training quantization is to construct a low-precision network$f_q(\mathbf{X}; \mathbf{\theta}, \mathbf{S})$ that preserves the behavior of the full-precision model, where $\mathbf{S}=\{\mathbf{S}_b\}_{b=1}^B$ denotes the collection of quantization step sizes in the network, and $\mathbf{S}_b$ contains the step sizes associated with the layers or channels within block $b$.

Instead of treating quantization as a deterministic transformation, we adopt a probabilistic formulation and model it as a stochastic process. For each block $b$, let $\mathbf{Z}_b$ and $\hat{\mathbf{Z}}_b$ denote the pre-quantization tensor and its quantized counterpart, respectively. We approximate quantization using an additive noise model~\cite{liu2025improving}:
\begin{equation}
    \hat{\mathbf{Z}}_b = \mathbf{Z}_b + \boldsymbol{\epsilon}_b, \quad \boldsymbol{\epsilon}_b \sim \mathbf{U}(-\mathbf{S}_b/2\,  | \,\mathbf{S}_b/2 ),
    \label{eq:quantization}
\end{equation}
where the noise is a uniform distribution parameterized by the quantization step sizes. However, unlike ~\cite{liu2025improving}, we model block-wise quantization stochasticity to balance between the flexibility of the prior design and the granularity of optimization. Admittedly, modeling $\mathbf{S}$ for every weight, channel, or layer would give maximum flexibility to the prior design and the finest granularity to the optimization. However, this would significantly increase the difficulty of optimization due to the large number of variables to inference on. Also, for the prior design in PFMs, the granularity would be overly small as the layer sensitivity has been analyzed mainly in groups of layers in neural operators~\cite{behroozi2025sensitivity,helwig2023group}. 

Under Equation \ref{eq:quantization}, the output of the quantized network is a random variable induced by quantization noise across all blocks. Let $\mathbf{Y}_i=f(\mathbf{X}_i;\mathbf{\theta})$ denote the full-precision output and $\hat{\mathbf{Y}}_i$ denote the stochastic output of the quantized model. We define a likelihood that measures how well the quantized model matches the original:
\begin{equation}
p(\mathbf{Y}_i \mid \mathbf{X}_i, \mathbf{S})
=
\int
p(\mathbf{Y}_i \mid \mathbf{X}_i, \boldsymbol{\epsilon}, \mathbf{S})
\, p(\boldsymbol{\epsilon} \mid \mathbf{S})
\, d\boldsymbol{\epsilon}.
\label{eq:lik}
\end{equation}
Note that Equation \ref{eq:lik} only considers the final output, \ie network-wise likelihood. Alternatively, $\mathbf{Y}_i$ could be more fine-grained, \eg including intermediate layer outputs or block outputs, \ie layer-wise likelihood or block-wise likelihood. In general, we find the network-wise likelihood gives the best degree of freedom when it comes to inference on the step size $\mathbf{S}$ across different PFMs data.

We further impose a prior distribution over the quantization step sizes, $\mathbf{S} \sim p(\mathbf{S})$, allowing the incorporation of inductive biases such as low-precision preference~\cite{liu2025improving} or structural regularity. The posterior distribution of the quantization parameters is therefore:
\begin{equation}
    p(\mathbf{S} \mid \mathcal{D}_{\mathrm{cal}})
\propto
p(\mathbf{S})
\prod_{i=1}^{N}
p(\mathbf{Y}_i \mid \mathbf{X}_i, \mathbf{S}).
\end{equation}

After defining the likelihood and the prior, it is theoretically possible to do a full Bayesian inference on $\mathbf{S}$ and conduct prediction via \eg Bayesian Model Averaging~\cite{hoeting1999bayesian}, especially when our compressed models are small, as shown later. However, since PFM is large and the dimension of the parameter space is huge, the number of models required during Bayesian Model Averaging is also large. Therefore, we leave this to future research. In this work, we estimate the quantization parameters via maximum a posteriori (MAP) inference:
\begin{equation}
    \mathbf{S}^*
=
\arg\max_{\mathbf{S} \in \mathcal{S}}
\left[
\log p(\mathbf{S})
+
\sum_{i=1}^{N}
\log p(\mathbf{Y}_i \mid \mathbf{X}_i, \mathbf{S})
\right].
\end{equation}

\subsection{Structure-Aware Prior}
We define a structure-aware prior over the quantization step sizes $\mathbf{S}$ to encourage coordinated precision allocation across the network. To this end, we organize the layers into blocks based on three types of prior knowledge. First, existing research shows that some layers, structures in neural operators are extremely sensitive to perturbation~\cite{yvinec2023safer,behroozi2025sensitivity}. Although this kind of sensitivity has not been systematically studied in PFMs, they still provide important information as PFMs are often composed of primitive neural operators~\cite{behroozi2025sensitivity,helwig2023group}. Next, we can compute the numerical sensitivity of layers and channels and group them accordingly, as explained later. Although the sensitivity can only be computed on the calibration data, it is still a useful complement to the knowledge above. Last but not the least, different PFMs have vastly different architectures where some layers are naturally clustered, \eg transformer blocks~\cite{he2023simplifying}. We also group such layers into blocks. Below, we introduce the general formulation of our block-wise prior. We include the detailed grouping information in Appendix~\ref{app:grouping} regarding the specific PFMs in experiments.


For block $b$, layer $l$, and channel $c$, let $S_{b,l,c}$ denote the corresponding step size. Following prior study~\cite{liu2025improving}, we represent the effective bitwidth by $log_2 \frac{R_{b,l,c}}{S_{b,l,c}}$, where $R_{b,l,c}$ denotes the quantization range and is computed on the calibration data using the percentile clipping method described in Appendix~\ref{app:percentile_clipping}. Rather than directly regularizing step sizes, we impose a truncated Gaussian prior over the effective bitwidths centered at a target precision $B^*_{b}$:
\begin{equation}
    \label{eq:prior0}
    \log p(\mathbf{S}_{b,l,c}) =
    -\sum_{b,l,c}\frac{\left(\log_2 \frac{R_{b,l,c}}{S_{b,l,c}} - B^*_{b}\right)^2}{2\sigma_{b}^2}.
\end{equation}
This prior encourages channels within the same block to remain close to a shared target precision while allowing controlled deviations through the variance $\sigma_{b}^2$. Operating in bitwidth space provides a direct and interpretable mechanism for block-level precision control. Also, we truncate the Gaussian when the deviation is too large, \eg when the learned bitwidth is overly larger than $B^*_{b}$, so the compression target will not be achieved.

\paragraph{Sensitivity-Guided Variance}
Despite that Equation \ref{eq:prior0} enforces channels within the same block to share a common target bit width, channels and layers can differ substantially in their influence on the final loss, making uniform regularization suboptimal. To account for this heterogeneity, we parameterize the prior variance using channel sensitivity:
\begin{equation}
    \sigma_{b,l,c}^2 = \sigma_b^2 \left( 1 + \alpha \, \mathrm{\phi}_{b,l,c} \right),
    \label{eq:adaptiveVar}
\end{equation}
where $\sigma_b^2$ is the block-level base variance and $\alpha$ controls the adaptation strength. The term $\mathrm{\phi}_{b,l,c}$ measures the sensitivity of channel $c$ to the full objective.

Following prior work~\cite{hsu2022language}, we estimate channel sensitivity, computed as the squared gradient of the loss with respect to block activations. Let $Z_{i,b,l,c,h,w}$ denote the activation at spatial location $(h,w)$ of channel $c$ in layer $l$ of block $b$ for sample $i$. We define
\begin{equation}
\label{eq:adaptive_prior}
\mathrm{\phi}_{b,l,c}
=
\frac{1}{NHW}
\sum_{i=1}^{N}
\sum_{h,w}
\left(
\frac{\partial \mathcal{L}}{\partial Z_{b,i,c,h,w}}
\right)^2,
\end{equation}
where $N$ is the number of calibration samples and $H, W$ denote the spatial resolution.


With this design, more sensitive channels receive larger variances, reducing the penalty for deviating from the target precision and allowing higher precision when needed. Less sensitive channels remain more strongly regularized toward the target, enabling more aggressive compression.

For the preservation of physical fidelity, we augment the pre-task loss $\mathcal{L}_{task}$ with a new Sobolev loss $\mathcal{L}_{sob}$, \ie $\mathcal{L} = \mathcal{L}_{task} + \mathcal{L}_{sob}$. Let $v$ be the reference field and $\hat{v}$ the model prediction over spatial domain $\Omega$. The loss is computed by the integral of their finite-difference derivatives at points $x \in \Omega$:

\begin{equation}
\label{eq:sobo_loss}
\mathcal{L}_{\mathrm{Sob}}
=
\lambda_{\alpha}
\sum_{|\alpha|=0}^{p} \frac{1}{|\Omega|} \sum_{x\in \Omega}
\left\|D^{\alpha}v(x) - D^{\alpha}\hat{v}(x)\right\|_1,
\end{equation}
where $D^{\alpha}$ denotes the finite-difference approximation of the spatial derivative indexed by the multi-index $\alpha$, $p$ is the highest derivative order included. The coefficient $\lambda_{\alpha}$ weights the contribution of each derivative order. In practice, the derivative order depends on the PDE family. We set $p=1$ for incompressible Navier--Stokes, compressible Euler and wave equations.

Substituting Equation \ref{eq:adaptiveVar} into Equation \ref{eq:prior0}, yielding
\begin{equation}
\log p(\mathbf{S}_{b,l,c})
= -\sum_{b,l,c}
\frac{
\left(
\log_2 \frac{R_{b,l,c}}{S_{b,l,c}} - B_b^*
\right)^2
}{
2\sigma_b^2
\left(
1 + \alpha \, \mathrm{\phi}_{b,l,c}^{\mathrm{Sob}}
\right)
}.
\end{equation}



\subsection{Overall Objective}
We optimize the quantization step sizes $\mathbf{S}=\{S_{b,l,c}\}$ by maximum a posteriori (MAP) inference, combining a network-wise likelihood with the proposed structure-aware prior. The objective is 
\begin{equation}
\max_{\{S_{b,l,c}\}}
\Bigg[
\sum_{i=1}^{N}
\log
\left(
\sum_{j=1}^{M}
p\big(
\mathbf{Z}_{i}^{fp}
\mid
\mathbf{S} \odot \nu^{(j)}, X_i
\big)
\right)
-
\sum_{b,l,c}
\frac{
\left(
\log_2 \frac{R_{b,l,c}}{S_{b,l,c}} - B_b^*
\right)^2
}{
2\sigma_b^2
\left(
1 + \alpha \, \phi_{b,l,c}^{\mathrm{Sob}}
\right)
}
\Bigg],
\end{equation}

where $\mathbf{S}=\{S_{b,l,c}\}$ denotes all quantization step sizes in the network, where block is $b$, layer is $l$ and channel is $c$. $\mathbf{Z}_{i}^{fp}$ denotes the full-precision activation of the network for the $i$-th sample, and $j$ indexes the Monte Carlo samples. Specifically, a reparameterization trick same as PPQ~\cite{liu2025improving} is employed by introducing an auxiliary random variable $\nu^{(j)}$ sampled from a uniform distribution, $\nu^{(j)} \sim \mathcal{U}(-\tfrac{1}{2}, \tfrac{1}{2})$. The quantization perturbation is then expressed as $\mathbf{S} \odot \nu^{(j)}$, such that the injected noise is distributed within $\left[-\tfrac{\mathbf{S}}{2}, \tfrac{\mathbf{S}}{2}\right]$. In this way, the stochasticity induced by quantization can be estimated through Monte Carlo sampling, with $j$ indicating the $j$-th sample of $\nu$.

The likelihood term is modeled using the Gaussian relaxation introduced above, namely
\begin{equation}
\mathbf{Z}_{i}^{\mathrm{fp}}
\mid
\mathbf{S},\nu^{(j)},X_i
\sim
\mathcal{N}
\big(
\mathbf{Z}_{i}^{\mathrm{quant}}(\mathbf{S} \odot \nu^{(j)}),\;
\eta I
\big),
\end{equation}
which implies
\begin{equation}
p\big(
\mathbf{Z}_{i}^{\mathrm{fp}}
\mid
\mathbf{S} \odot \nu^{(j)}, X_i
\big)
\propto
\exp\!\left(
-\frac{1}{2\eta}
\left\|
\mathbf{Z}_{i}^{\mathrm{fp}}
-
\mathbf{Z}_{i}^{\mathrm{quant}}(\mathbf{S} \odot \nu^{(j)})
\right\|^2
\right),
\end{equation}
where $\mathbf{Z}_{i}^{\mathrm{quant}}$ denotes the activation of the network for the $i$-th sample after quantization is applied. Therefore, the likelihood encourages the quantized network output to remain close to the corresponding clean full-precision output.






\subsection{Beta Distribution}
\label{subsec:beta_distribution}

The Gaussian prior introduced above provides a symmetric regularization around a target bitwidth.
However, such a symmetric prior is not always suitable for sensitivity-aware quantization.
For a structurally or physically important block, deviations towards higher bitwidths should be encouraged, since preserving more precision can reduce quantization-induced physical errors.
In contrast, for a less sensitive block, deviations towards lower bitwidths should be encouraged, since more aggressive compression is less likely to damage the model output.
Therefore, we replace the symmetric Gaussian-style prior with an asymmetric Beta-distribution prior defined in normalized bitwidth space.

For each block $b$, layer $l$, and channel $c$, let $S_{b,l,c}$ denote the learnable quantization step size, and let $R_{b,l,c}$ denote the corresponding quantization range.
Following the continuous bitwidth formulation, the effective bitwidth is defined as
\begin{equation}
B_{b,l,c}
=
\log_2
\left(
\frac{R_{b,l,c}}{S_{b,l,c}}
\right).
\label{eq:effective_bitwidth_beta}
\end{equation}
A smaller step size corresponds to a larger effective bitwidth, while a larger step size corresponds to stronger compression.
To use a Beta distribution, whose support is $[0,1]$, we normalize the effective bitwidth as
\begin{equation}
u_{b,l,c}
=
\frac{
B_{b,l,c} - B_{\min}
}{
B_{\max} - B_{\min}
},
\qquad
u_{b,l,c}\in[0,1],
\label{eq:normalized_bitwidth_beta}
\end{equation}
where $B_{\min}$ and $B_{\max}$ define the minimum and maximum bitwidths allowed during optimization.
Thus, values of $u_{b,l,c}$ close to $0$ correspond to low-precision quantization, whereas values close to $1$ correspond to high-precision quantization.

We assign a Beta prior to the normalized bitwidths within each structural block:
\begin{equation}
u_{b,l,c}
\sim
\operatorname{Beta}(a_b,b_b),
\label{eq:beta_prior_main}
\end{equation}
where the shape parameters $a_b$ and $b_b$ are shared by all channels inside block $b$.
The corresponding density is
\begin{equation}
p(u_{b,l,c})
=
\frac{
u_{b,l,c}^{a_b-1}
(1-u_{b,l,c})^{b_b-1}
}{
\mathrm{B}(a_b,b_b)
},
\label{eq:beta_density_main}
\end{equation}
where $\mathrm{B}(a_b,b_b)$ is the Beta function.

The key advantage of the Beta prior is that its shape can encode asymmetric precision preferences.
If $a_b>b_b$, the density places more probability mass near the high-bitwidth region, encouraging the optimizer to preserve more precision in block $b$.
If $a_b<b_b$, the density places more probability mass near the low-bitwidth region, encouraging stronger compression.
If $a_b=b_b$, the prior is symmetric and does not prefer either side.
Therefore, the desired qualitative behavior is
\begin{align}
\text{high block sensitivity}
&\quad \Longrightarrow \quad
a_b>b_b
\quad \Longrightarrow \quad
\text{higher preferred bitwidth},
\label{eq:beta_sensitive_behavior}
\\
\text{low block sensitivity}
&\quad \Longrightarrow \quad
a_b<b_b
\quad \Longrightarrow \quad
\text{lower preferred bitwidth}.
\label{eq:beta_insensitive_behavior}
\end{align}

However, a sensitivity score alone only determines the desired direction of the prior; it does not uniquely determine the pair $(a_b,b_b)$.
This is because the Beta parameters control two distinct properties.
The mean
\begin{equation}
\mu_b
=
\mathbb{E}[u_{b,l,c}]
=
\frac{a_b}{a_b+b_b}
\label{eq:beta_mean_main}
\end{equation}
determines the preferred location in normalized bitwidth space, while the concentration
\begin{equation}
\kappa_b
=
a_b+b_b
\label{eq:beta_concentration_main}
\end{equation}
determines how strongly the distribution is concentrated around this preference.
For example, two Beta distributions may have the same mean but different concentration, resulting in different prior strengths.
To avoid introducing an additional degree of freedom for every block, we fix the concentration to a global hyperparameter:
\begin{equation}
a_b+b_b=\kappa,
\label{eq:fixed_concentration}
\end{equation}
where $\kappa>0$ controls the overall strength of the Beta prior.
With this design, block sensitivity is used only to determine the prior mean $\mu_b$, while $\kappa$ is shared across all blocks.

We next describe how the block-wise sensitivity is mapped to the Beta mean.
Let $\phi_b^{\mathrm{Sob}}$ denote the Sobolev-aware sensitivity score of block $b$.
We first normalize the sensitivity scores across all structural blocks:
\begin{equation}
r_b
=
\frac{
\phi_b^{\mathrm{Sob}}-\min_j \phi_j^{\mathrm{Sob}}
}{
\max_j \phi_j^{\mathrm{Sob}}-\min_j \phi_j^{\mathrm{Sob}}+\varepsilon
},
\label{eq:beta_sensitivity_normalization}
\end{equation}
where $\varepsilon>0$ is a small constant for numerical stability.
The normalized score $r_b\in[0,1]$ indicates the relative sensitivity of block $b$.


To construct a balanced precision allocation, the different sizes of the structural blocks must also be considered.
If every block is treated equally, a block containing only a few weights contributes as much to the centering operation as a block containing millions of weights.
This preserves the average preferred bitwidth across blocks, but it does not preserve the average preferred bitwidth across model weights.
Since the storage cost of a block depends on both its bitwidth and its number of weights, we introduce a parameter-weighted centering strategy.

Let $n_b$ denote the number of quantized weights contained in block $b$.
The parameter-weighted mean of the normalized sensitivities is defined as
\begin{equation}
\bar r_{\mathrm{w}}
=
\frac{
\sum_{b=1}^{N_{\mathrm{block}}}
n_b r_b
}{
\sum_{b=1}^{N_{\mathrm{block}}}
n_b
},
\label{eq:weighted_mean_sensitivity}
\end{equation}
where:

\begin{itemize}
    \item $n_b$ is the number of quantized weights in block $b$;
    \item $r_b$ is the normalized sensitivity of block $b$;
    \item $N_{\mathrm{block}}$ is the total number of structural blocks;
    \item $\bar r_{\mathrm{w}}$ is the parameter-weighted average sensitivity.
\end{itemize}

A large block therefore contributes more strongly to the reference sensitivity than a small block.
This is appropriate when the objective is to control the total model storage, because changing the bitwidth of a large block affects many more weights.

The weighted relative sensitivity is then defined as
\begin{equation}
d_b
=
\frac{
r_b-\bar r_{\mathrm{w}}
}{
\max_j
\left|
r_j-\bar r_{\mathrm{w}}
\right|
+
\varepsilon_{\mathrm{rel}}
},
\label{eq:weighted_centered_relative_sensitivity}
\end{equation}
where $\varepsilon_{\mathrm{rel}}>0$ is a small scale-aware constant used only to avoid division by zero.

The numerator determines whether block $b$ is more or less sensitive than the parameter-weighted average:
\begin{align}
d_b>0
&\quad \Longrightarrow \quad
\text{block $b$ is more sensitive than the weighted average},
\\
d_b<0
&\quad \Longrightarrow \quad
\text{block $b$ is less sensitive than the weighted average}.
\end{align}

The denominator rescales the largest absolute deviation to approximately one, so that
\begin{equation}
d_b\in[-1,1].
\end{equation}

Ignoring the small numerical-stability term $\varepsilon_{\mathrm{rel}}$, the weighted deviations satisfy
\begin{equation}
\sum_{b=1}^{N_{\mathrm{block}}}
n_b d_b
=
0.
\label{eq:weighted_relative_sensitivity_balance}
\end{equation}

This is different from ordinary block-wise centering, which only guarantees that the unweighted block contributions cancel.
Equation~\eqref{eq:weighted_relative_sensitivity_balance} instead guarantees that the positive and negative shifts cancel after accounting for the number of weights in each block.

For example, consider two blocks with
\begin{equation}
n_1=999,
\qquad
n_2=1.
\end{equation}
Treating the two blocks equally could assign
\begin{equation}
d_1=1,
\qquad
d_2=-1,
\end{equation}
which appears balanced at the block level because $d_1+d_2=0$.
However, the parameter-weighted shift is
\begin{equation}
999d_1+d_2
=
999-1
=
998,
\end{equation}
so the total precision allocation is strongly shifted upward.

Under parameter-weighted centering, the relative scores instead satisfy
\begin{equation}
999d_1+d_2=0.
\end{equation}
For instance,
\begin{equation}
d_1=0.001,
\qquad
d_2=-0.999,
\end{equation}
gives
\begin{equation}
999(0.001)+1(-0.999)=0.
\end{equation}
Thus, a small upward shift applied to a large block is balanced by a larger downward shift applied to a very small block.

Given a reference bitwidth $B^*$, the preferred bitwidth of block $b$ is then defined as
\begin{equation}
B_b^{\mathrm{pref}}
=
B^*
+
\Delta_B d_b,
\label{eq:preferred_bitwidth_beta}
\end{equation}
where $\Delta_B>0$ is a global hyperparameter controlling the maximum sensitivity-induced shift from the reference bitwidth.

This construction gives sensitive blocks preferred bitwidths above $B^*$ and less sensitive blocks preferred bitwidths below $B^*$.
More importantly, the parameter-weighted average preferred bitwidth remains equal to the reference value:
\begin{equation}
\frac{
\sum_{b=1}^{N_{\mathrm{block}}}
n_b B_b^{\mathrm{pref}}
}{
\sum_{b=1}^{N_{\mathrm{block}}}
n_b
}
=
B^*.
\label{eq:weighted_average_preferred_bitwidth}
\end{equation}

This follows because
\begin{equation}
\begin{aligned}
\sum_{b=1}^{N_{\mathrm{block}}}
n_b B_b^{\mathrm{pref}}
&=
\sum_{b=1}^{N_{\mathrm{block}}}
n_b
\left(
B^*+\Delta_B d_b
\right)
\\
&=
B^*
\sum_{b=1}^{N_{\mathrm{block}}}
n_b
+
\Delta_B
\sum_{b=1}^{N_{\mathrm{block}}}
n_b d_b
\\
&=
B^*
\sum_{b=1}^{N_{\mathrm{block}}}
n_b.
\end{aligned}
\label{eq:weighted_bitwidth_balance_derivation}
\end{equation}

Therefore, the sensitivity-aware allocation redistributes precision between blocks without changing the intended average bitwidth per weight.