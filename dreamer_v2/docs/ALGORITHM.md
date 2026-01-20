# DreamerV2: Mastering Atari with Discrete World Models

## Visual Comparison (V1 vs V2)

```mermaid
graph TD
    subgraph DreamerV1 [Dreamer V1 (Continuous RSSM)]
        v1_h[h_t (Deterministic)] --> v1_prior[Prior: Gaussian N(μ, σ)]
        v1_img[Image x_t] --> v1_post[Posterior: Gaussian N(μ, σ)]
        v1_prior -.-> v1_kl((KL Loss))
        v1_post -.-> v1_kl
    end

    subgraph DreamerV2 [Dreamer V2 (Discrete RSSM + Discount)]
        v2_h[h_t (Deterministic GRU + LayerNorm)] --> v2_prior[Prior: Categorical (32x32)]
        v2_img[Image x_t] --> v2_post[Posterior: Categorical (32x32)]
        v2_prior -.-> v2_kl((KL Balancing))
        v2_post -.-> v2_kl
        
        v2_code[z_t (Sampled OneHot)] --> v2_disc[Discount Pred γ_t]
        v2_code --> v2_rew[Reward Pred r_t]
        v2_code --> v2_rec[Leconstruct Image x_t]
    end
```

## Key Differences in Detail

### 1. Categorical Latent States (The Core RSSM Change)
- **DreamerV1 (Gaussian)**:
  - Latent state $z_t$ was a vector of continuous values sampled from a Normal distribution.
  - Good for robots (continuous physics), bad for games (discrete logic: "I have the key or I don't").
- **DreamerV2 (Categorical)**:
  - Latent state $z_t$ is a grid of **32 categorical variables, each with 32 classes**.
  - It works like a vocabulary of concepts.
  - **Gradient**: Uses `Straight-Through Gumbel-Softmax` (hard samples in forward pass, soft gradients in backward pass).

### 2. The Discount Predictor ($\gamma$)
DreamerV2 explicitly adds a new head to the World Model: **The Discount Predictor**.
$$ \hat{\gamma}_t \sim p_\phi(\hat{\gamma}_t | h_t, z_t) $$
- **Why?** Instead of assuming fixed episode lengths or fixed discount factors, the model learns to predict *when the episode ends*.
- **Benefit:** This allows the agent to handle terminal states correctly during "dreaming" (imagination). If the model imagines a state that leads to death/termination, the predicted discount factor $\gamma$ should drop to 0, stopping the value accumulation for future steps in that trajectory.

### 3. KL Balancing (Optimization Stability)
In V1, we simply minimized $KL[Posterior || Prior]$.
In V2, we assume the Posterior (which sees the image) is the "ground truth" and the Prior should chase it.
$$ L = \alpha \cdot KL[sg(Post) || Prior] + (1-\alpha) \cdot KL[Post || sg(Prior)] $$
- **$\alpha = 0.8$**: We put 80% of the effort into pulling the Prior towards the Posterior.
- **Effect**: Prevents "Posterior Collapse" (where the posterior ignores the image and just copies the prior to minimize loss).

### 4. Model Architecture Tweaks
- **Mish / ELU**: V2 adopts these activation functions.
- **LayerNorms**: Added inside the GRU and MLPs. This is critical when using Categorical latents to prevent signal explosion/vanishing.

### 5. Actor-Critic
- **Reinforce with Entropy**: For discrete actions (Atari), V2 uses REINFORCE gradients with entropy regularization.
- **Continuous Actions (Walker)**: For DMC, we keep the Normal distribution for the Actor, just like in V1.

## Algorithm adapted for Walker (DMC)

We will use the **Discrete World Model** (Categorical latents) but keep the **Continuous Actor** (Normal output) for the Walker environment. This tests the hypothesis that discrete world models are robust representation learners even for continuous control tasks.

## Notation Decoder: $s_t$ vs $\{h_t, z_t\}$

In older papers (PlaNet) or simplified equations (Eq. 9), $s_t$ is a "black box" state. In Dreamer implementation, we split it explicitly.

$$ s_t = \text{Model State} = \{ \underbrace{h_t}_{\text{Deterministic}}, \underbrace{z_t}_{\text{Stochastic}} \} $$

```mermaid
graph TD
    subgraph "Time t-1"
        st_prev[State s_{t-1}]
        st_prev --> ht_prev[h_{t-1}]
        st_prev --> zt_prev[z_{t-1}]
        at_prev[Action a_{t-1}]
    end

    subgraph "Deterministic Path (Memory)"
        ht_prev & zt_prev & at_prev --> ORANGE_BOX[GRU / Recurrent Model]
        ORANGE_BOX --> ht[h_t: Deterministic State]
    end

    subgraph "Stochastic Path (Concept)"
        ht --> BLUE_BOX[Transition Predictor]
        BLUE_BOX --> z_prior[Prior: ᑮ_t]
        
        ht --> RED_BOX[Representation Model]
        img_obs[Image o_t] --> RED_BOX
        RED_BOX --> z_post[Posterior: z_t]
    end
    
    subgraph "Time t (The Result)"
        ht -.-> st_new[State s_t]
        z_post -.-> st_new
        
        st_new --> DECODER[Image/Reward/Discount Predictors]
    end
    
    linkStyle 4,5,6 stroke:orange,stroke-width:2px;
    linkStyle 8 stroke:blue,stroke-width:2px;
    linkStyle 10,11 stroke:red,stroke-width:2px;
```
