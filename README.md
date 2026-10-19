# NVIDIA LLM Inference Workshop

## Purpose and Current Scope  

This lab experience covers 2 hands-on labs that showcase the flexibility and benefits of the NVIDIA NIM - a microservice that simplifies LLM deployments. You will learn how to build and deploy inference engines on NVIDIA GPUs and do extensive performance analysis. Some topics covered:

- What is a NIM?
- How to implelment LLMs and NIMs with Nvidia SDK toolkits, hosted API services, and NGC Docker containers.
- Exploring the the many protocols to for LLM requests and their similarities and uncover how to decide which one is the best for you!  Including: OpenAI, LangChain, ChatNvidia
- Use Nvidia's GenAIPerf code to learn how to find performance analysis and metrics of GPUs for LLMs such as how Time-to-first-token and inter-token latencies affect production deployments

The goal of the Nvidia Inference Microservice is to simplify LLM Inference with the following benefits:

- Production LLM deployment in 5 minutes
- API Standardization with OpenAI Protocol - enables no code change for current LLM deployments
- Optimized pre-built engines for most popular community models such as llama, mistral, gemma
- Automatic Triton Model Configuration and deployment, handles pre/post processing of of LLM inference and loading models onto the server
- Support for Retrieval Augmented Generation and Low Rank Adaptation deployments 
- Enterprise support with pre-validated, QA tested containers  which includes security and CVE scanning/patching and tech support.

This microservice is a combination of several of NVIDIA's technologies, mainly:
- [TRT-LLM - Package to build inference engines on NVIDIA GPUs](https://nvidia.github.io/TensorRT-LLM/architecture.html)
- [NVIDIA Triton - Flexible Inference Server that works across many use cases](https://github.com/triton-inference-server/tutorials)


This workshop is appropriate for Data Scientists/Software Engineers/ Developers with Python experience, Docker Container experience and an interest in fine tuning LLM applications. It can serve as a high-level introduction for those unfamiliar with deep learning and inferencing; more experienced data scientists can expect to leave with a good understanding of how to  apply NVIDIA's inferencing technologies to their use cases and workflows. 


### Shareable Description
[Some content here](./promo.md) may be helpful in sharing what to expect from this workshop.

## Getting Started

1. In an IDE or editor, change the NGC key to your own in the `.env` file. [Located Here](/home/starlord/all_work/nim-workshop/docker/.env) in the `./docker` folder.

2.  Start with the first notebook and follow in order.
3.  Have fun learning!


## See more information:

- [How to Run NIM locally and in Air Gap](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#serving-models-from-local-assets)
- [LORA description and Parameter-Efficient Fine-Tuning](https://docs.nvidia.com/nim/large-language-models/latest/peft.html#lora-setup-overview)
- [NIM Multi-node Deployment](https://docs.nvidia.com/nim/large-language-models/latest/multi-node-deployment.html)
- [Parameter-Efficient Fine-Tuning - NVIDIA Docs](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)
- [NIM GPU Support Matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html)

## Local Setup -- Python Environment

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/public-sector/nim-workshop.git
cd nim-workshop
git checkout main
# Create  python or conda virtual environment
python -m venv nim-workshop # one time
source nim-workshop/bin/activate
pip install -r docker/requirements.txt # one time
```

## Contributors

* Dustin Van Stee
* Nic Borensztein
* Chris Holland
* Jennifer Lombardo
* Dan Campbell

*Note: This lab began as a fork from the [Retail NIM Workshop](https://gitlab-master.nvidia.com/NVRetail/llms/nim-workshop) in Feb 2024 [Paul Hendricks, Mark Mayou].  Given the changes with NIM, the content has changed a lot so has the technical implementations.*

For related content, also see:
* [Telco SA team's NIM Workshop](https://gitlab-master.nvidia.com/sae-industry/telco/att-workshop/)
* [Chris Holland's NIM LoRA Workshop](https://gitlab-master.nvidia.com/cholland/nim-lora-workshop)