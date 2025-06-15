# Cerebra 

Cerebra is an AI orchestrator for LLM applications.

## Features

-   LLM orchestration
-   LLM agent creation
-   LLM agent execution
-   LLM agent monitoring
-   LLM agent evaluation
-   LLM agent feedback

## Installation

```bash
pip install cerebra
```

## Usage

```python
from cerebra import Cerebra

cerebra = Cerebra()

# Define available LLM's
llm_1 = cerebra.define_llm(name="llm_1", model="gpt-4o", api_key="your_api_key")

# Define orchestrator
orchestrator = cerebra.create_orchestrator(name="orchestrator", llms=[llm_1])

# Execute an LLM agent
response = orchestrator.execute(prompt="What is the capital of France?")

# The agent will return a response and a status containing information about which LLM was used and the time it took to generate the response.
print(response)
```
