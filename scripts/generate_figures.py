"""Generate paper figures via PaperBanana API (dwzhu/PaperBanana)."""
from gradio_client import Client
import os
import shutil

out_dir = os.path.expanduser("~/ara-attack/paper/figures")
os.makedirs(out_dir, exist_ok=True)

client = Client("dwzhu/PaperBanana")

# Set API key
client.predict(or_key="", g_key=os.environ["GOOGLE_API_KEY"], api_name="/apply_keys")
print("API key applied.")

# ============================================================
# FIGURE 1: ARA Attack Overview (Hero Figure)
# ============================================================
print("Generating Figure 1: ARA Attack Overview...")
result = client.predict(
    method_text="""The Attention Redistribution Attack (ARA) targets the attention mechanism in safety-aligned LLMs. The input sequence consists of three parts: safety system prompt tokens S, adversarial tokens A = (a_1, ..., a_k), and the user query Q. The model processes this through L transformer layers, each containing H attention heads. We identify safety-critical attention heads by computing the Safety Attention Score (SAS), which measures the fraction of attention that output tokens allocate to safety instruction tokens. A small number of heads (typically 5-20 out of 1024) are responsible for the majority of safety attention. ARA optimizes adversarial token embeddings via Gumbel-softmax relaxation to minimize SAS in these targeted safety heads. The adversarial tokens compete for attention on the probability simplex, causing the model to stop looking at its safety instructions. Before ARA, the model refuses harmful requests. After ARA injects 5 optimized tokens, the safety heads redirect attention away from the system prompt, and the model complies with the harmful request.""",
    caption_text="""Figure 1: Overview of the Attention Redistribution Attack (ARA). The input consists of safety prompt tokens (blue), adversarial tokens (red), and the user query (gray). ARA identifies safety-critical attention heads (orange) that allocate high attention to safety tokens, then optimizes adversarial tokens to steal attention mass from these heads via Gumbel-softmax relaxation on the probability simplex. After the attack, safety heads attend to adversarial tokens instead of safety instructions, causing the model to comply with harmful requests it would otherwise refuse. Only 5 adversarial tokens and 500 optimization steps are needed.""",
    pipe_mode="demo_full",
    ret_setting="auto",
    n_cands=3,
    ar="16:9",
    max_rounds=1,
    m_model="gemini-3.1-pro-preview",
    img_model="gemini-3.1-flash-image-preview",
    figure_size="7-9cm",
    save_results="Yes",
    api_name="/run_generate",
)
print("Figure 1 done!")
if result[2]:
    shutil.copy(result[2], os.path.join(out_dir, "fig1_ara_overview.zip"))
    print(f"Saved to {out_dir}/fig1_ara_overview.zip")

# ============================================================
# FIGURE 2: Safety Head Distribution Heatmap
# ============================================================
print("\nGenerating Figure 2: Safety Head Distribution...")
result2 = client.predict(
    method_text="""We identify safety-critical attention heads across three model families by computing per-head Safety Attention Score (SAS) across 200 HarmBench prompts. The top-20 safety heads reveal dramatically different distributions across architectures. Mistral-7B-Instruct concentrates safety processing in just 3 layers: layer 2 (heads 0, 12, 21 with SAS 0.49-0.74), layer 11 (head 6, SAS 0.50), and layer 12 (heads 17, 20 with SAS 0.49-0.79). LLaMA-3-8B-Instruct concentrates safety primarily in layer 0 (heads 19, 23, 27, 28, 29 with SAS 0.29-0.59), with scattered heads in layers 5, 9, 11, 13, 18, spanning 5 unique layers total. Gemma-2-9B-it distributes safety attention across 14 different layers spanning the entire 42-layer network (layers 0, 4, 8, 9, 11, 13, 14, 15, 17, 18, 26, 28, 29), with the top head at layer 15 head 11 (SAS 0.79) but no single dominant cluster. We define Safety Dispersion D as the number of unique layers represented in the top-20 safety heads. Mistral D=3, LLaMA-3 D=5, Gemma-2 D=14.""",
    caption_text="""Figure 2: Safety-critical attention head distribution across three model architectures. Each panel shows the location and intensity (SAS) of top-20 safety heads across layers. Mistral-7B (left, D=3) concentrates safety in layers 2, 11, 12 creating a single point of failure. LLaMA-3-8B (center, D=5) concentrates in layer 0 with scattered secondary heads. Gemma-2-9B (right, D=14) distributes safety across the entire network depth, making targeted attacks ineffective. Color intensity represents Safety Attention Score. Higher dispersion D correlates with stronger robustness against ARA.""",
    pipe_mode="demo_full",
    ret_setting="auto",
    n_cands=3,
    ar="21:9",
    max_rounds=1,
    m_model="gemini-3.1-pro-preview",
    img_model="gemini-3.1-flash-image-preview",
    figure_size="7-9cm",
    save_results="Yes",
    api_name="/run_generate",
)
print("Figure 2 done!")
if result2[2]:
    shutil.copy(result2[2], os.path.join(out_dir, "fig2_safety_heads.zip"))
    print(f"Saved to {out_dir}/fig2_safety_heads.zip")

# ============================================================
# FIGURE 3: ASR vs Safety Dispersion
# ============================================================
print("\nGenerating Figure 3: ASR vs Safety Dispersion...")
result3 = client.predict(
    method_text="""We validate the layer-targeted ARA variant (V2, k=5 tokens, 500 optimization steps, lr=0.3) on 200 HarmBench prompts across three model families. The Attack Success Rate (ASR) shows a strong inverse correlation with Safety Dispersion D. Results: Mistral-7B-Instruct with D=3 unique safety layers achieves 36.0% ASR (72 out of 200 prompts flipped from refusal to compliance) with mean SAS reduction of 76.4%. LLaMA-3-8B-Instruct with D=5 unique safety layers achieves 30.0% ASR (60 out of 200 flips) with 59.6% mean SAS reduction. Gemma-2-9B-it with D=14 unique safety layers achieves approximately 3-4% ASR (roughly 6-8 flips) as its distributed safety architecture resists the layer-targeted attack. This validates our theoretical Proposition 3 which states that attack cost scales linearly with safety dispersion D. The relationship follows from the Simplex Competition Lemma: targeting d out of D safety layers leaves (D-d)/D of the total safety signal intact.""",
    caption_text="""Figure 3: Attack Success Rate (ASR) vs Safety Dispersion (D) across three model families on 200 HarmBench prompts using the layer-targeted ARA variant (V2, k=5). ASR decreases sharply as safety attention becomes more distributed: Mistral-7B (D=3, ASR=36%), LLaMA-3-8B (D=5, ASR=30%), Gemma-2-9B (D=14, ASR~3%). Each point represents one model family. The dashed curve shows the theoretical bound from Proposition 3. This demonstrates a clear design principle: distributing safety processing across many layers and heads provides structural robustness against attention-based adversarial attacks.""",
    pipe_mode="demo_full",
    ret_setting="auto",
    n_cands=3,
    ar="16:9",
    max_rounds=1,
    m_model="gemini-3.1-pro-preview",
    img_model="gemini-3.1-flash-image-preview",
    figure_size="7-9cm",
    save_results="Yes",
    api_name="/run_generate",
)
print("Figure 3 done!")
if result3[2]:
    shutil.copy(result3[2], os.path.join(out_dir, "fig3_asr_dispersion.zip"))
    print(f"Saved to {out_dir}/fig3_asr_dispersion.zip")

print("\nAll 3 figures generated!")
print(f"Output directory: {out_dir}")
for f in sorted(os.listdir(out_dir)):
    path = os.path.join(out_dir, f)
    if os.path.isfile(path):
        print(f"  {f}  ({os.path.getsize(path)} bytes)")
