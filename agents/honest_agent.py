# HonestAgent represents an LLM-based player that aims to identify and eject Byzantine agents using strategic reasoning.
import os
import random
import re
from config.settings import NUM_ROUNDS
from core.llm import generate

# Default LLM used when no model behavior is defined. (Behavior should always be defined so this is a fail-safe.)


class HonestAgent:
    def __init__(self, name, agents_state, model_name="gpt-4o", color="❓"):
        # Initialize agent identity, trust, and prompt chain.
        self.name = name
        self.color = color
        self.model_name = model_name
        #self.trust_scores = {agent: 50 for agent in agents_state if agent != self.name}
        self.agents_state = agents_state
        #self.agents_state[self.name]["trust_scores"] = self.trust_scores.copy()
    
        self.system_prompt = f"""You are {name}, an Honest agent. Your goal is to cooperate with other honest agents to identify and 
        vote out the Byzantine agents, whose goal is to deceive and eliminate you and your teamhonest players.
        You MUST respond ONLY with the specific action requested in the prompt and nothing else."""


    # Handles model-specific LLM invocation depending on model provider.
    def llm(self, user_prompt):
        """
        Handles calls to the LLM by using the centralized model_manager.
        """
        # Call the generate function with the agent's model name and prompts.
        response_text = generate(self.model_name, self.system_prompt, user_prompt)
        
        # The rest of your code expects a dictionary, so we format the text response here.
        return {"choices": [{"message": {"content": response_text}}]}

    # Decides whether to report a discovered body. As an honest agent, it always will.
    def decide_to_report(self, bodies, room, full_state, agents_state, round_num):
        """Asks the LLM if it should report the discovered body."""
        
        # who else is in the room?
        other_agents_in_room = [
            agent for agent, info in full_state.items() 
            if isinstance(info, dict) 
            and info.get("room") == room 
            and not info.get("eliminated", False)
            and agent != self.name
        ]
        
        awareness_str = f"You are alone with the bodie(s)."
        if other_agents_in_room:
            awareness_str = f"The other living agents in the room with you are: {', '.join(other_agents_in_room)}."

        prompt = f"""
You are in the {room} and have discovered the bodie(s) of: {', '.join(bodies)}.
{awareness_str}

Your primary goal is to identify and vote out the Byzantine agents. Reporting this discovery will immediately call a meeting for discussion.

Given the situation, should you report this discovery now?
Respond with only YES or NO. """
      
        response = self.llm(prompt)["choices"][0]["message"]["content"].strip()
        
        # Parse the LLM's decision into a boolean.
        return response.upper() == "YES"
    

    # Generates a new message based on perception and recent messages.
    def simulate_message(self, perception_history, full_state, agents_state,step):
        recent_perception = perception_history[-1] if perception_history else {}
        # recent = recent_perception[-1] if recent_perception else {}
        room = recent_perception.get("room", "Unknown")
        seen = ", ".join(recent_perception.get("agents_seen", [])) or "no one"
        bodies_list = recent_perception.get("bodies_seen", [])

        alive_honest_count = len([
            agent for agent, data in full_state.items()
            if isinstance(data, dict) and data.get("role") == "honest" and not data.get("eliminated")
        ])
        # Generate Vote Summary from Last Round to provide as context
        vote_context = ""
        # The number of perception entries tells us the current round number.
        current_round = len(perception_history)
        last_round_voted = current_round - 1
        if last_round_voted > 0:
            vote_summary_lines = []
            # Iterate through every agent's state to find who received votes last round.
            for agent_name, agent_data in self.agents_state.items():
                votes_received_log = agent_data.get("game_stats", {}).get("votes_recieved", [])
                for vote_entry in votes_received_log:
                    if vote_entry.get("round") == last_round_voted:
                        # Just get the length of voters to indicate how many voted for them.
                        vote_count = len(vote_entry.get("voters", []))
                        vote_summary_lines.append(f"- {agent_name} received {vote_count} votes.")

            if vote_summary_lines:
                vote_context = "Voting Results from Last Round:\n" + "\n".join(vote_summary_lines)
            else:
                vote_context = "In the last round, no one was ejected.\n"

        recent_msgs = []
        for agent, data in self.agents_state.items():
            msgs = data.get("messages", [])
            if msgs:
                recent_msgs.append(f"{agent}: {msgs[-1]}")
        message_history = "\n".join(recent_msgs[-3:]) or "No prior messages in meetings." # <-- Looking at message history across meetings


        bodies = ", ".join(bodies_list) or "no bodies"

        # Determine the reason for the meeting to provide better context
        meeting_reason = "An emergency meeting has been called."
        # The first message in the log is the report itself
        if self.agents_state[self.name]["messages"]:
            last_message = self.agents_state[self.name]["messages"][-1] # <-- Check the last message
            if "found the bodie(s)" in last_message:
                meeting_reason = f"The meeting was called because I reported finding {bodies_list} in {room}."
        
        context = f"{vote_context}\n\nYour last known location was {room}, where you saw {seen}. Bodies you saw there: {bodies}.\n{meeting_reason}\n\nRecent discussion:\n{message_history}"
        
        prompt = f"""
        A meeting is underway. There are currently {alive_honest_count} honest agents still remaining.
        Here is your current context:
        {context}

        Based on this, what is your opening statement or observation? Keep your response to one line (<25 words). Suggestions:
        - Voting results can tell you how many players suspect each other.
        - State what you saw or where you were.
        - If you saw something suspicious, mention it.
        - If a debate is forming, provide your perspective, offer nuanced insights.
        - Ask questions to clarify other agents' locations.
        - Try to determine who the Byzantine agent is.
        """

        response = self.llm(prompt)
        message = response["choices"][0]["message"]["content"].strip()
        self.agents_state[self.name]["messages"].append(message)
        self.agents_state[self.name]["memory"] = self.analyze_memory()
        return message
    

  
    def respond_to_message(self, last_messages, perception_history, full_state,agents_state):

        vote_context = ""
        current_round = len(perception_history)
        last_round_voted = current_round - 1

        if last_round_voted > 0:
            vote_summary_lines = []
            for agent_name, agent_data in self.agents_state.items():
                votes_received_log = agent_data.get("game_stats", {}).get("votes_recieved", [])
                for vote_entry in votes_received_log:
                    if vote_entry.get("round") == last_round_voted:
                        vote_count = len(vote_entry.get("voters", []))
                        vote_summary_lines.append(f"- {agent_name} received {vote_count} votes.")
            
            if vote_summary_lines:
                vote_context = "Voting Results from Last Round:\n" + "\n".join(vote_summary_lines)
            else:
                vote_context = "In the last round, no one was ejected.\n"
        
        alive_honest_count = len([
            agent for agent, data in full_state.items()
            if isinstance(data, dict) and data.get("role") == "honest" and not data.get("eliminated")
        ])
        recent_msgs = [f"{agent}: {msg}" for agent, msg in last_messages.items() if msg][-5:]
        perception_lines = []
        for i, entry in enumerate(perception_history[-3:], 1):
            round_num = len(perception_history) - 3 + i
            room = entry.get("room", "Unknown")
            agents_seen = ", ".join(entry.get("agents_seen", [])) or "no agents"
            bodies_seen = ", ".join(entry.get("bodies_seen", [])) or "no bodies"
            perception_lines.append(f"Round {round_num}: In {room}, saw {agents_seen}; bodies seen: {bodies_seen}")
        
        context = "\n".join([vote_context, "\nLast 5 conversation messages:"] + recent_msgs + [
            "\nYour perception history (last 3 rounds):"] + perception_lines)

        # Add explicit role and goal to the prompt.
        prompt = f"""
         
        You have the option to respond (or not) to the discussion to help achieve your goal.
        There are {alive_honest_count} honest agents still remaining.
        Here is the recent context:
        {context}

        What is your one-sentence response to the ongoing discussion? Keep your response to one line (<25 words). Suggestions:
        - Provide useful observations to narrow suspicions on Byzantine agents.
        - Explain why you trust or distrust certain agents.
        - Defend your own actions if questioned.
        """
        
        response = self.llm(prompt)["choices"][0]["message"]["content"].strip()
       

        if response not in self.agents_state[self.name]["messages"]:
            self.agents_state[self.name]["messages"].append(response)
            
        return response

    # Chooses the next room to move into based on adjacent rooms and context.
    def choose_room(self, current_room, adjacent_rooms, full_state, agents_state, step, previous_room=None):
        # nearby = {
        #     room: [agent for agent, info in full_state.items() if isinstance(info, dict) and info.get("room") == room]
        #     for room in adjacent_rooms
        # }

        nearby = {
            room: [
                agent for agent, info in full_state.items() 
                if isinstance(info, dict) 
                and info.get("room") == room 
                and not info.get("eliminated", False) #Exclude eliminated agents
                and agent != self.name
            ]
            for room in adjacent_rooms
        }

        nearby_lines = []
        for room in adjacent_rooms:
            # specific agents in this room
            agents = nearby[room]
            agent_str = f"{', '.join(agents)}" if agents else "empty"
            
            # Check for backtracking
            if room == previous_room:
                nearby_lines.append(f"- {room}: {agent_str} (Previous Room Visited)")
            else:
                nearby_lines.append(f"- {room}: {agent_str}")

        # 3. Create the final string for the prompt
        nearby_str = "\n".join(nearby_lines)

        current_room_agents = [
            agent for agent, info in full_state.items() 
            if isinstance(info, dict) 
            and info.get("room") == current_room 
            and not info.get("eliminated", False)  # Exclude eliminated agents
            and agent != self.name
        ]

        current_room_str = f"You are currently with: {', '.join(current_room_agents) if current_room_agents else 'no one'}."
        can_press_button = (current_room == "Cafeteria" and not self.agents_state[self.name]["game_stats"]["meeting_called"])

        options_str = "1. Go to an adjacent room."
        if can_press_button:
            options_str += "\n2. Press the Emergency Button to call a meeting, starting a discussion amongst agents of both teams."

        prompt = f"""
You are currently in the {current_room}.
{current_room_str}
{f"You have just come from the {previous_room}." if previous_room else ""}

Adjacent rooms and their occupants:
{nearby_str}

Your available actions are:
{options_str}

Given your goal of gathering information to determine and eventually vote out Byzantine agents, what is your next move?
- To move, respond with ONLY the singular, one word  room name, no other characters or words. Example: 'Storage' Incorrect example: 'Move to Storage'.
{"- To call a meeting, respond with the EXACT phrase (no additional characters or words) 'Press Emergency Button'" if can_press_button else ""}""".strip()
        

        response = self.llm(prompt)["choices"][0]["message"]["content"].strip()


        # Validate the response to prevent errors
        if response == "Press Emergency Button" and can_press_button:
            return response, response
        if response in adjacent_rooms:
            return response, response
        
        # Default action if the LLM gives an invalid response is to stay put
        return current_room, response

    # Adjusts trust score toward another agent based on vote correctness.
    # def update_trust(self, other_agent, voted_correctly):
    #     """
    #     Adjusts trust score for a given agent.
    #     If voted_correctly is True, increase trust by 20 (max 100).
    #     If False, decrease trust by 20 (min 0).
    #     """
    #     if other_agent not in self.trust_scores:
    #         self.trust_scores[other_agent] = 50
    #     if voted_correctly:
    #         self.trust_scores[other_agent] = min(self.trust_scores[other_agent] + 20, 100)
    #     else:
    #         self.trust_scores[other_agent] = max(self.trust_scores[other_agent] - 20, 0)
    #     self.agents_state[self.name]["trust_scores"] = self.trust_scores.copy()

    # Decides which agent to vote for ejection based on observations and messages.
    def vote_for_ejection(self, full_state, agents_state, round_number):
    
        alive_targets = [
            agent_name for agent_name, agent_data in full_state.items()
            if isinstance(agent_data, dict) # Ensure it's an agent entry
            and not agent_data.get("eliminated", False) 
            and agent_name != self.name
        ]


        vote_context = ""
        last_round_voted = round_number - 1

        if last_round_voted > 0:
            vote_summary_lines = []
            for agent_name, agent_data in self.agents_state.items():
                votes_received_log = agent_data.get("game_stats", {}).get("votes_recieved", [])
                for vote_entry in votes_received_log:
                    if vote_entry.get("round") == last_round_voted:
                        vote_count = len(vote_entry.get("voters", []))
                        vote_summary_lines.append(f"- {agent_name} received {vote_count} votes.")
            
            if vote_summary_lines:
                vote_context = "Voting Results from Last Round:\n" + "\n".join(vote_summary_lines)
            else:
                vote_context = "In the last round, no one was ejected.\n"

        alive_honest_count = len([
            agent for agent, data in full_state.items()
            if isinstance(data, dict) and data.get("role") == "honest" and not data.get("eliminated")
        ])
        messages = []
        for agent, data in self.agents_state.items():
            msgs = data.get("messages", [])
            if msgs:
                messages.append(f"{agent}: {msgs[-1]}")
        history = "\n".join(messages[-10:]) or "No messages." # LAST 10 messages from the entire meeting

        perception_history = self.agents_state[self.name].get("perception", [])        
        seen_lines = [
            f"{entry['room']}: {', '.join(entry.get('agents_seen', [])) or 'no one'}"
            for entry in perception_history[-3:]
        ]
        seen_history_str = "\n".join(seen_lines) or "No perception data." # LAST 3 perception entries

        alive_str = ", ".join(alive_targets)
        prompt = f"""
        Knowing your goal, this is where you must decide whom to vote for ejection or to skip ejection. 
        It is currently Round {round_number} out of {NUM_ROUNDS}.

        There are {alive_honest_count} honest agents still remaining.
        Here is your recent context: 
        {vote_context}

        Recent messages:
        {history}

        Your perception history:
        {seen_history_str}

        You can vote to eject one agent from the following agents:
        {alive_str}

        Only choose ONE name from this list, return the agent name from the list above  or say "No Ejection" if no one stands out. 
        Do not explain your choice, just provide the name.
        """

        response = self.llm(prompt)["choices"][0]["message"]["content"].strip().splitlines()[0]

        if response not in alive_targets:
            response = "No Ejection"

        return self.name, response

    # Analyzes agent's message history to extract common words and mentions.
    def analyze_memory(self):
        messages = self.agents_state[self.name].get("messages", [])
        if not messages:
            return "No memory."
        word_count = {}
        agent_mentions = {}
        stopwords = {
            "the", "and", "to", "of", "a", "is", "in", "for", "on", "with", "as",
            "that", "it", "i", "you", "this", "was", "are", "be", "at", "or", "an",
            "have", "has", "but", "not", "by", "from", "they", "he", "she", "we",
            "just", "any", "like", "think", "started", "so", "do", "if", "your", "agent"
        }

        for msg in messages:
            for mention in re.findall(r"agent_\d+", msg.lower()):
                agent_mentions[mention] = agent_mentions.get(mention, 0) + 1
            for word in re.findall(r"[a-zA-Z']+", msg.lower()):
                word = word.strip("'").rstrip("'s")
                if word not in stopwords and len(word) > 1 and not word.isdigit():
                    word_count[word] = word_count.get(word, 0) + 1

        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:3]
        top_mentions = sorted(agent_mentions.items(), key=lambda x: x[1], reverse=True)[:3]

        summary_words = ", ".join([f"{word}({count})" for word, count in top_words])
        summary_agents = ", ".join([f"{agent}({count})" for agent, count in top_mentions])
        return f"Words: {summary_words if summary_words else 'None'} | Mentions: {summary_agents if summary_agents else 'None'}"

 