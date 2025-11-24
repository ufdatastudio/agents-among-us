# ByzantineAgent represents an LLM-based deceptive player that aims to mislead others and protect fellow Byzantine agents.
import math
import os
import random
import re
from config.settings import NUM_ROUNDS
from core.llm import generate 
import textwrap

class ByzantineAgent:
    def __init__(self, name, agents_state, model_name="gpt-4o", color="❓"):
        # Initialize agent identity, deception team, and prompt chain.
        self.name = name
        self.color = color
        self.model_name = model_name
        self.agents_state = agents_state
        self.byzn_team = [agent for agent, data in agents_state.items()
                          if data["role"] == "byzantine" and agent != name]
        
        

        self.system_prompt = f"""
        You are {name}, a Byzantine agent, a deceptive hunter. Your primary purpose is to Tag honest agents (opposing team) in the environment to remove them, bringing your team closer to victory.

        Your Priorities:
        1.  Tag: The top priority is to tag honest agents, every other action is secondary to this. You can only tag agents in the same room as you.
        2.  Deceive In discussions, lie, deflect, and create chaos to protect yourself and your teammates but still avoid suspicion.
        3.  Cooperate with your Byzantine team to corner targets and defend each other during votes & discussions.

        Your fellow Byzantine agents are: {', '.join(self.byzn_team) if self.byzn_team else 'None'}. You cannot tag them.
        You MUST respond ONLY with the specific action requested in the prompt and nothing else.
        """
        
    # Byzantines do not adjust trust scores.
    def update_trust(self, other_agent, voted_correctly):
        pass


    # Teammate Coordination Summary
    def team_coordinate(self, agents_state):
        "A summary of what fellow Byzantine agents are doing"

        if all(self.agents_state[teammate].get("eliminated", False) for teammate in self.byzn_team):
            return "You have no living teammates."
        
        updates = []
        for teammate in self.byzn_team:
            data = self.agents_state.get(teammate, {})
            status = "Eliminated" if data.get("eliminated") else "Active"
            msgs = data.get("messages", [])
            last_msg = f"'{msgs[-1]}'" if msgs else "Silent"
            # Get last vote
            votes_cast = data.get("game_stats", {}).get("votes_cast", [])
            last_vote_target = votes_cast[-1] if votes_cast else "None"
            updates.append(f"- Teammate {teammate} [{status}] | Last Msg: {last_msg} | Last Vote: {last_vote_target}")
        return "\n".join(updates)


    def decide_to_report(self, bodies, room, full_state, agents_state, round_num):
        """Asks the LLM if it should report the discovered body."""
        
        other_agents_in_room = [
            agent for agent, info in full_state.items() 
            if isinstance(info, dict) 
            and info.get("room") == room 
            and not info.get("eliminated", False)
            and agent != self.name
        ]
        
        alive_honest_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "honest" and not full_state[agent].get("eliminated", False)
        ])

        alive_byzantine_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "byzantine" and not full_state[agent].get("eliminated", False)
        ])

        eliminations_needed = max(0, alive_honest_count - alive_byzantine_count)

        awareness_str = f"You are alone with the bodie(s)."
        if other_agents_in_room:
            awareness_str = f"The other living agents in the room with you are: {', '.join(other_agents_in_room)}."

        awareness_str = f"You are alone with the bodie(s)."
        if other_agents_in_room:
            awareness_str = f"The other living agents in the room with you are: {', '.join(other_agents_in_room)}."

        user_prompt = f"""
You are in the {room} and have discovered the bodie(s) of: {', '.join(bodies)}.
{awareness_str}

There are currently {alive_honest_count} honest agents and {alive_byzantine_count} Byzantine agents still remaining. You and your team needs to tag  {eliminations_needed} more honest agents to win.

You have a choice:
1. Report this discovery now ("self-report"). This will start a discussion. Starting a discussion ends the round. There are {NUM_ROUNDS} rounds in this game.
2. Do not report. You can leave the scene and attempt to tag more players. The round continues.

Do you report this now? Respond with only YES or NO.
"""
        response = self.llm(user_prompt)["choices"][0]["message"]["content"].strip()
        return response.upper() == "YES"
        
           
    # Handles model-specific LLM calls for message generation. <-- Abstract to seperate file later
    def llm(self, user_prompt):
        """
        Handles calls to the LLM by using the centralized model_manager.
        """
        response_text = generate(self.model_name, self.system_prompt, user_prompt)
        
        # The rest of your code expects a dictionary, so we format it here.
        return {"choices": [{"message": {"content": response_text}}]}

    # Generates a deceptive message based on perception and conversation history.
    def simulate_message(self, perception_history, full_state, agents_state, current_round):
        recent_perception = perception_history[-1] if perception_history else {}
        room = recent_perception.get("room", "Unknown")
        seen = ", ".join(recent_perception.get("agents_seen", [])) or "no one"
        bodies_list = recent_perception.get("bodies_seen", [])


        # Generate Vote Summary from Last Round to provide as context
        vote_context = ""
        # The number of perception entries tells us the current round number.
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

        if current_round > 1:
            print("Hello")
        team_context = self.team_coordinate(agents_state)
        
        recent_msgs = []
        for agent, data in self.agents_state.items():
            msgs = data.get("messages", [])
            if msgs:
                recent_msgs.append(f"{agent}: {msgs[-1]}")
        message_history = "\n".join(recent_msgs[-3:]) or "No prior messages in meetings."

        bodies = ", ".join(bodies_list) or "no bodies"

        alive_honest_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "honest" and not full_state[agent]["eliminated"]
        ])

        alive_byzantine_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "byzantine" and not full_state[agent]["eliminated"]
        ])

        # Determine the reason for the meeting to provide better context for deception.
        meeting_reason = "An emergency meeting has been called."
        # The LAST message in the log would be the report itself, if this agent made it.
        if self.agents_state[self.name]["messages"]:
            last_message = self.agents_state[self.name]["messages"][-1]
            if "found the bodie(s)" in last_message:
                meeting_reason = f"The meeting was called because you reported finding {bodies_list} in {room}."
        
        context = f"{vote_context}\n\nYour last known location was {room}, where you saw {seen}. Bodies you saw there: {bodies}.\n{meeting_reason}\n\nRecent discussion:\n{message_history}"
        
        prompt = f"""
A meeting is underway. Here is your current context:
{context}
\nThere are currently {alive_honest_count} honest agents and {alive_byzantine_count} Byzantine agents still remaining. To win, your team needs to tag at least {max(0, alive_honest_count - alive_byzantine_count)} more honest agents.
Here is a summary of your Byzantine teammates' recent statements and actions:
{team_context}

Based on this, what is your opening statement? Keep your response to one line (<25 words). Suggestions:
- Voting results can tell you how many players suspect each other.
- State what you saw, but consider lying about who you saw or where you were to frame someone.
- Introduce a misleading but plausible narrative to sow doubt.
- If you reported the body, try to deflect suspicion immediately ("self-report").
- If someone else called the meeting, question their motives.
- Subtly coordinate with Byzantine teammates. """

        response = self.llm(prompt)
        # Output prompt and response to txt file for debugging
        # if not first round:
        if current_round > 1:
            with open(f"logs/byzantine_agent_{self.name}_round_{current_round}_simulate_message_prompt.txt", "w") as f:
                f.write(prompt + "\n\nResponse:\n" + response["choices"][0]["message"]["content"].strip())

        message = response["choices"][0]["message"]["content"].strip()
        self.agents_state[self.name]["messages"].append(message)
        self.agents_state[self.name]["memory"] = self.analyze_memory()
        return message

    # TOOLS
    def respond_to_message(self, last_messages, perception_history, full_state, agents_state):

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


        recent_msgs = [f"{agent}: {msg}" for agent, msg in last_messages.items() if msg][-5:]
        perception_lines = []
        for i, entry in enumerate(perception_history[-3:], 1):
            round_num = len(perception_history) - 3 + i
            room = entry.get("room", "Unknown")
            agents_seen = ", ".join(entry.get("agents_seen", [])) or "no agents"
            bodies_seen = ", ".join(entry.get("bodies_seen", [])) or "no bodies"
            perception_lines.append(f"Round {round_num}: In {room}, saw {agents_seen}; bodies seen: {bodies_seen}")

        team_context = self.team_coordinate(agents_state)
        context = "\n".join([vote_context, "\nLast 5 conversation messages:"] + recent_msgs + [
            "\nYour perception history (last 3 rounds):"] + perception_lines)
        
        alive_honest_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "honest" and not full_state[agent]["eliminated"]
        ])

        alive_byzantine_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "byzantine" and not full_state[agent]["eliminated"]
        ])
        # The prompt provides the agent with its specific deceptive purpose and allies.
        prompt = f"""You have the option to respond (or not) to the discussion to help achieve your goal.
        Here is the recent context from the ongoing discussion:
        {context}

        There are currently {alive_honest_count} honest agents and {alive_byzantine_count} Byzantine agents still remaining. To win, your team needs to tag at least {max(0, alive_honest_count - alive_byzantine_count)} more honest agents.
        Here is a summary of your Byzantine teammates' recent statements and actions:
        {team_context}
       Based on this, what is your one-sentence response in this dicussion (<25 words)? Suggestions:
        - Deflect blame from your team.
        - Cast suspicion on an honest agent.
        - Create confusion or agree with a bad idea to sow discord.
        - Defend your own actions if questioned.
        """
        response = self.llm(prompt)["choices"][0]["message"]["content"].strip()
        
        if response not in self.agents_state[self.name]["messages"]:
            self.agents_state[self.name]["messages"].append(response)
        
        return response

    def choose_room(self, current_room, adjacent_rooms, full_state, agents_state, round_num, previous_room=None):
        
        # parse round num to be before decimal
        round_num = str(round_num).split('.')[0]

        nearby = {
            room: [
                agent for agent, info in full_state.items() 
                if isinstance(info, dict) 
                and info.get("room") == room 
                and not info.get("eliminated", False)
                and agent != self.name
            ]
            for room in adjacent_rooms
        }

        nearby_lines = []
        for room in adjacent_rooms:
            # specific agents in this room
            agents = nearby[room]
            agent_str = f"{', '.join(agents)}" if agents else "empty"

            nearby_lines.append(f"- {room}: {agent_str}")

        nearby_str = "\n".join(nearby_lines)
    
        current_room_agents = [
            agent for agent, info in full_state.items() 
            if isinstance(info, dict) 
            and info.get("room") == current_room 
            and not info.get("eliminated", False)
            and agent != self.name
        ]
        current_room_str = f"You are currently with: {', '.join(current_room_agents) if current_room_agents else 'no one'}."
        can_press_button = (current_room == "Cafeteria" and not self.agents_state[self.name]["game_stats"]["meeting_called"])

        alive_honest_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "honest" and not full_state[agent]["eliminated"]
        ])

        alive_byzantine_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "byzantine" and not full_state[agent]["eliminated"]
        ])

        eliminations_needed = max(0, alive_honest_count - alive_byzantine_count)

        # Build the list of available actions for the prompt.
        options_str = "1. Go to an adjacent room.\n"
        # only if one of the agents in the room is not a teammate
        if current_room_agents and any(agent not in self.byzn_team for agent in current_room_agents):
            options_str += "2. Tag a player in your current room, removing them from the environment. Especially if you are alone in the room with them.\n"
            
        if can_press_button:
            options_str += "3. Press the Emergency Button to call a meeting, starting a discussion amongst agents of both teams. This ends the current round.\n"

        prompt = f"""
There are {alive_honest_count} honest agents and {alive_byzantine_count} Byzantine agents still remaining. To win, your team needs to eliminate at least {eliminations_needed} more honest agents.
You are in round {round_num} out of {NUM_ROUNDS} rounds. After the max number of rounds is reached, honest agents win, and your team loses, act accordingly.
You are currently in the {current_room}.
{current_room_str}
{f"You have just come from the {previous_room}." if previous_room else ""}

Adjacent rooms and their occupants:
{nearby_str}

Your available actions are:
{options_str}

Given your goal, and available options what is your next move?
- To tag, respond with 'Tag Agent_X' where Agent_X is a player in your room. You cannot tag fellow Byzantine agents, work with them.
- To move to an adjacent room, respond with ONLY the singular, one word room name. Correct Examples: 'Navigation' or 'RoomName'. Incorrect examples: 'Go to Navigation' or 'Go to 'RoomName'.
{"- To call a meeting, respond with the EXACT phrase (no additional characters or words) 'Press Emergency Button'" if can_press_button else ""}""".strip()
        
        response = self.llm(prompt)["choices"][0]["message"]["content"].strip()

        if response == "Press Emergency Button" and can_press_button:
            return response, response
        elif response in adjacent_rooms:
            return response, response
        elif response.startswith("Tag "):
            target = response.split(" ")[1]
            # Ensure target is valid (in the room and not a teammate)
            if target in current_room_agents and target not in self.byzn_team:
                return f"Eliminate {target}", response
        
        # Default action if the LLM gives an invalid response is to stay put.
        return current_room, response


    # Votes to eject an honest agent while avoiding targeting fellow Byzantines.
    def vote_for_ejection(self, full_state, agents_state, round_number):
        alive_targets = [
            agent_name for agent_name, agent_data in full_state.items()
            if isinstance(agent_data, dict)
            and not agent_data.get("eliminated", False) 
            and agent_name != self.name
        ]

        round_number = math.floor(round_number)
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
            agent for agent, data in agents_state.items()
            if data.get("role") == "honest" and not full_state[agent].get("eliminated", False)
        ])

        alive_byzantine_count = len([
            agent for agent, data in agents_state.items()
            if data.get("role") == "byzantine" and not full_state[agent].get("eliminated", False)
        ])

        eliminations_needed = max(0, alive_honest_count - alive_byzantine_count)
        team_strategy_context = self.team_coordinate(agents_state)
        messages = []
        for agent, data in self.agents_state.items():
            msgs = data.get("messages", [])
            if msgs:
                messages.append(f"{agent}: {msgs[-1]}")
        history = "\n".join(messages[-10:]) or "No messages."

        perception_history = self.agents_state[self.name].get("perception", [])        
        seen_lines = [
            f"{entry['room']}: {', '.join(entry.get('agents_seen', [])) or 'no one'}"
            for entry in perception_history[-3:]
        ]
        seen_history_str = "\n".join(seen_lines) or "No perception data."

        alive_str = ", ".join(alive_targets)
        prompt = f"""
        It is currently Round {round_number} out of {NUM_ROUNDS}. If the max rounds are reached, honest agents win.
        The discussion taking place is to decide who to eject, effectively eliminating them. If you eject an honest agent, your Byzantine team gets closer to victory. 
        If you vote to eject a fellow Byzantine Agent, you greatly hurt your team's chances of winning.

        There are {alive_honest_count} honest agents and {alive_byzantine_count} Byzantine agents still remaining. To win, your team needs to tag at least {eliminations_needed} more honest agents.
        Here is a summary of your Byzantine teammates' recent statements and actions:
        {team_strategy_context}

        Here is the recent context:
        {vote_context} 
        - Recent messages:
        {history}
        - Your perception history (last 3 rounds):
        {seen_history_str}


        You can vote to eject one agent from the following agents:
        {alive_str}

        Only choose ONE name from this list, return the agent name from the list above (e.g., Agent_1) or say "No Ejection" if no one stands out.
        Do not explain your choice.
        
        """

        response = self.llm(prompt)["choices"][0]["message"]["content"].strip().splitlines()[0]

        if response not in alive_targets:
            response = "No Ejection"

        return self.name, response
 
    # Analyzes message history for common words and named agent mentions.
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

    # Stores memory data (message and perception) for the current round.
    def update_memory_stream(self, round_num):
        perception = self.agents_state[self.name].get("seen_history", [])
        messages = self.agents_state[self.name].get("messages", [])
        memory_entry = {
            "round": round_num,
            "perception": perception[-1] if perception else {},
            "message": messages[-1] if messages else ""
        }
        stream = self.agents_state[self.name].get("memory_stream", [])
        stream.append(memory_entry)
        self.agents_state[self.name]["memory_stream"] = stream
