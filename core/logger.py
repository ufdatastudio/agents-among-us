# core/logger.py
import os
import shutil
import csv
import json

class LogManager:
    def __init__(self, game_id, agents, scenario_name=None):
        """
        agents: List of Agent objects (needed to categorize into Byz/Honest folders)
        """
        self.game_id = game_id
        if scenario_name:
            self.base_dir = os.path.join("logs", scenario_name, f"Game_{game_id}")
        else:
            self.base_dir = os.path.join("logs", f"Game_{game_id}")
        
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir)

        self.paths = {
            "round_results": os.path.join(self.base_dir, "roundResults.log"),
            "stats": os.path.join(self.base_dir, "stats.log"),
            "agents": {},
            "discussion": os.path.join(self.base_dir, "discussion.log"),
            "stats_csv": os.path.join(self.base_dir, "stats.csv"),
            "discussion_chat": os.path.join(self.base_dir, "discussion_chat.csv"),
            # Private chain-of-thought only — never read by agents/observers.
            "thought": os.path.join(self.base_dir, "thought.log"),
        }

        # Create Root Logs
        self._create_file(self.paths["round_results"], "=== Round Results Log ===\n")
        self._create_file(self.paths["discussion"], "=== Discussion Log ===\n")
        self._create_file(self.paths["thought"], "")
        self._init_discussion_chat_csv()

        # Create Agent Directories based on Role
        for agent in agents:
            role_dir = "Byz" if agent.role == "byzantine" else "Honest"
            agent_dir = os.path.join(self.base_dir, role_dir, agent.name)
            os.makedirs(agent_dir, exist_ok=True)

            # Action Log
            action_log_path = os.path.join(agent_dir, "action.log")
            self._create_file(action_log_path, f"=== Action Log for {agent.name} ===\n")
            
            # Vote Log
            vote_log_path = os.path.join(agent_dir, "vote.log")
            self._create_file(vote_log_path, f"=== Voting History for {agent.name} ===\n")

            self.paths["agents"][agent.name] = {
                "action": action_log_path,
                "vote": vote_log_path
            }

    def _create_file(self, path, initial_content=""):
        with open(path, "w", encoding="utf-8") as f:
            f.write(initial_content)
    
    def _init_discussion_chat_csv(self):
        """Initialize discussion_chat.csv with headers"""
        with open(self.paths["discussion_chat"], "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["discussion_num", "reason", "agent_num", "model", "role", "message"])

    def log_discussion_chat(self, discussion_num, reason, agent_name, model_name, role, message):
        """Log a discussion chat message to CSV"""
        try:
            # Extract agent number from Agent_0 -> 0
            agent_num = agent_name.replace("Agent_", "") if agent_name.startswith("Agent_") else agent_name
            role_label = "Byzantine" if role == "byzantine" else "Honest"
            
            with open(self.paths["discussion_chat"], "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([discussion_num, reason, agent_num, model_name, role_label, message])
        except Exception as e:
            print(f"Error logging discussion chat: {e}")

    def log_thought(
        self,
        round,
        phase,
        tick,
        agent,
        role,
        model,
        think,
        output,
        had_tags,
        parse_ok,
    ):
        """Append one private thought record to thought.log (JSONL).

        Writes only to thought.log — never touches discussion/action/vote/stats
        logs that agents or the observer pipeline read from.
        """
        record = {
            "round": round,
            "phase": phase,
            "tick": tick,
            "agent": agent,
            "role": role,
            "model": model,
            "think": think if think is not None else "",
            "output": output if output is not None else "",
            "had_tags": bool(had_tags),
            "parse_ok": bool(parse_ok),
        }
        try:
            with open(self.paths["thought"], "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error logging thought: {e}")

    def write_log(self, log_type, agent_name=None, content=""):
        """
        log_type: 'agent', 'discussion', 'results'
        """
        if log_type == 'agent' and agent_name:
            path = self.paths["agents"].get(agent_name)
            path = path.get("action") 
            if path:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                    
        elif log_type == 'discussion':
            # Write to BOTH discussion logs so everyone sees the same public chat
            with open(self.paths["discussion"], "a", encoding="utf-8") as f:
                f.write(content + "\n")

        elif log_type == 'vote' and agent_name:
            path = self.paths["agents"].get(agent_name)
            path = path.get("vote") 
            with open(path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
  
        elif log_type == 'results':
            with open(self.paths["round_results"], "a", encoding="utf-8") as f:
                f.write(content + "\n")
        elif log_type == 'debug':
            debug_log_path = os.path.join(self.base_dir, "debug.log")
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(content + "\n")

    def export_stats(self, agents_data):
        """
        Exports the 'stats' dictionary of every agent to a CSV file.
        agents_data: The self.state.world_data["agents"] dictionary.
        """
        if not agents_data:
            return

        first_agent = list(agents_data.values())[0]
        stats_keys = list(first_agent["stats"].keys())
        fieldnames = ["agent_name"] + stats_keys

        print(f"Exporting Game Stats to: {self.paths['stats_csv']}")

        try:
            with open(self.paths["stats_csv"], "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for agent_name, data in agents_data.items():
                    row = data["stats"].copy()
                    row["agent_name"] = agent_name
                    writer.writerow(row)
        except Exception as e:
            print(f"Error exporting CSV stats: {e}")

    def get_agent_log_path(self, agent_name):
        return self.paths["agents"][agent_name]["action"]
    
    def get_discussion_log_path(self, agent_role):
        return self.paths["discussion"]

    def get_results_log_path(self):
        return self.paths["round_results"]