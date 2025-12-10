# agents/base_agent.py
from core.llm import ModelManager
from config.settings import ROOMS

class BaseAgent:
    def __init__(self, name, color, role, model_name):
        self.name = name
        self.color = color
        self.role = role
        self.model_name = model_name
        self.llm = ModelManager.get_instance()
        self.action_num = 0

    def think_and_act(self, world_view, round_num):
        """
        Main decision loop for the movement phase.
        Returns tuple: (action_type, target/destination, raw_reasoning)
        """
        raise NotImplementedError

    def participate_in_discussion(self, conversation_history, world_view):
        raise NotImplementedError

    def vote(self, world_view, candidates):
        raise NotImplementedError