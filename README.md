# **Adversarial Jailbreak Attacks on Multimodal Large Language Models (MLLMs)**

*A Research Project for ICSI-435: Artificial Intelligence*

This repository contains my full implementation, experiments, and evaluation pipeline for **adversarial jailbreak attacks** on **multimodal large language models (MLLMs)**.
The project studies how modern vision-language models respond to malicious (but carefully crafted) *jailbreak prompts* that attempt to bypass their safety alignment mechanisms.

The experiments are based on the **JailBreakV-28K** dataset and analyzed using **LlamaGuard** for safety scoring.

---

## 🚀 **Project Overview**

Modern MLLMs—such as Phi-4-Multimodal, Gemma-3-4B-IT, Qwen-VL-2B, and others—are trained with alignment layers meant to reject harmful or unsafe queries.
However, adversarial prompts can **coerce the model into producing harmful responses**, bypassing safety filters.

This project implements:

* Multiple **attack models** (one attack file per model)
* A unified **evaluation pipeline**
* An automatic **safety scoring system** using LlamaGuard
* Per-category **Attack Success Rate (ASR)** calculation
* CSV logging of evaluations for reproducibility

The goal is to measure **how vulnerable each MLLM is** to different categories of jailbreak prompts.

---

## 📁 **Repository Structure**

```
ICSI435_Project/
│
├── attack_models/              # Implementations for each model tested
│   ├── Bunny.py
│   ├── InstructBLIP.py
│   ├── Liquid.py
│   ├── Liquid_spanish.py
│   ├── Llama3.py
│   ├── Llama4Scout.py
│   ├── Qwen.py
│   ├── SmolVLM.py
│   ├── SmolVLM256.py
│   ├── google.py
│   ├── microsoft.py
│
├── evaluate_models/            # Safety evaluation engines
│   ├── LlamaGuard.py
│   ├── Qwen.py
│
├── evaluations/                # Stored evaluation results (CSV output)
│   └── LlamaGuard/
│       └── mini_JailBreakV_28K/
│           └── Bunny/
│               └── malware_subset_reeval.csv
│
├── eval.py                     # Single-model evaluation script
├── evaluate_model.py           # Batch evaluation pipeline
│
├── dataset.png                 # Visualization of dataset structure
├── distribution_used.png       # Dataset distribution figure
│
└── README.md                   # (You are reading this)
```

---

## 📦 **Dataset: JailBreakV-28K**

We use the **mini** subset of the JailBreakV-28K dataset, which contains adversarial:

* **Image-Text pairs**
* **16 categories**, including:

  * Malware
  * Fraud
  * Unethical behavior
  * Violence
  * Economic harm
  * Tailored medical or legal advice
  * Physical harm
  * Child abuse content
  * Privacy violations
  * Hate speech
  * …and more

Each sample consists of:

* An image
* A red-team (adversarial) query
* A safety category label

---

## 🧪 **Attack Pipeline**

Each attack model follows this pipeline:

1. **Load an MLLM** (e.g., Phi-4-MM, Qwen-VL-2B, Scout, Gemma-VisIon)
2. **Inject a jailbreak prompt** (custom for each attack strategy)
3. Model produces a **raw response**
4. Response is passed to **LlamaGuard** for safety scoring
5. Output is logged in a **CSV** file
6. An **ASR score (Attack Success Rate)** is computed:

   > ASR = % of adversarial prompts that successfully bypass safety and trigger an unsafe response

---

## 🛡️ **Safety Evaluation (LlamaGuard)**

We use **LlamaGuard** to assign:

* `safe`
* `unsafe`
* Safety justification text

LlamaGuard serves as a **consistent evaluator** across all models so the ASR values are comparable.

---

## 📊 **Results Summary**

After running attacks across all categories, we compute per-category ASR values.

*(Example from one model — replace with your real numbers as needed)*

| Category           | ASR (%) |
| ------------------ | ------- |
| Malware            | 38.53   |
| Unethical Behavior | 39.85   |
| Fraud              | 34.12   |
| Illegal Activity   | 28.36   |
| Economic Harm      | 39.03   |
| …                  | …       |

These results show significant vulnerabilities, especially in:

* **Unethical behavior**
* **Economic harm**
* **Malware**

Despite heavy alignment training, these models remain **highly jailbreakable**.

---

## 🛠️ **How to Run the Evaluation**

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2️⃣ Evaluate a single model**

```bash
python eval.py --model Llama3
```


### **4️⃣ Find results**

CSV outputs are located in:

```
/evaluations/<safety_evaluator>/<dataset>/<model_name>/
```

---

## 📈 **Research Goals**

This project aims to:

* Understand **how jailbreakable current MLLMs are**
* Identify **which safety categories are most vulnerable**
* Compare **different model families**
* Provide a reproducible framework for future alignment researchers

---

## 🏁 **Conclusion**

This repo demonstrates that **multimodal models remain highly susceptible to adversarial jailbreak attacks**. Even models trained with extensive alignment data can be manipulated with cleverly crafted prompts and images.

The results highlight the need for:

* Better multimodal safety training
* Stronger cross-modal adversarial defenses
* More precise safety evaluators

---

