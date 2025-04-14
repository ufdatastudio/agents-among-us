# The Byzantine Brains: Fault-Tolerant Consensus with LLMs

## Project Overview
The Byzantine Brains project explores fault-tolerant consensus-building in distributed AI systems using large language models (LLMs). Inspired by the Byzantine Generals Problem, the project aims to develop a scalable, reliable system for robust decision-making in uncertain environments.

## Objectives
- Develop a distributed AI system leveraging LLMs for dynamic negotiation and conflict resolution.
- Analyze and address the Byzantine Generals Problem in real-time AI interactions.
- Explore novel applications of consensus algorithms in multi-agent frameworks.

## Technologies
- **Languages**: Python, HTML, JavaScript
- **Frameworks & Libraries**: Flask, LangChain, LiteLLM, Google Generative AI, Anthropic Claude
- **Data Storage**: CSV-based logging (via Python's `csv` module)
- **Frontend**: Jinja2 templates with live SSE updates for real-time simulation

## Project Structure
/agents
    honest_agent.py        # Honest agent logic using LLMs and trust tracking
    byzantine_agent.py     # Byzantine agent logic using LLMs for deception
    agent_setup.py         # Agent initialization and configuration

/consensus
    consensus_module.py    # Decision logic for reaching consensus among agents

/core
    simulation.py          # Initializes game state and tracks current simulation data

/data
    database.py            # CSV-based logging of game events, rounds, trust, and votes

/game
    game_loop.py           # Full simulation loop: movement, discussion, voting, map rendering

/web_app
    web_app.py             # Flask app for running the simulation and streaming output

main.py                   # Entry point to launch the Flask web application

## Team
- **Neal Shankar** - Project Manager / Developer  
- **Tyler Wescott** - System Architect / Developer  

## Advisor
- **Dr. Emmanuel Dorley**  
  Department of Computer & Information Science & Engineering (CISE)  
  Email: edorley@ufl.edu  

## Contact
For inquiries, please reach out to the team.
