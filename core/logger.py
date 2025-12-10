# core/logger.py
import os
import shutil
import csv

class LogManager:
    def __init__(self, game_id, agents):
        """
        agents: List of Agent objects (needed to categorize into Byz/Honest folders)
        """
        self.game_id = game_id
        self.base_dir = os.path.join("logs", f"Game_{game_id}")
        
        # Clean/Create Directory
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir)

        # File Paths
        self.paths = {
            "round_results": os.path.join(self.base_dir, "roundResults.log"),
            "stats": os.path.join(self.base_dir, "stats.log"),
            "agents": {},
            "discussion": os.path.join(self.base_dir, "discussion.log"),
            "stats": os.path.join(self.base_dir, "stats.csv"),
        }

        # Create Root Logs
        self._create_file(self.paths["round_results"], "=== Round Results Log ===\n")
        self._create_file(self.paths["discussion"], "=== Discussion Log ===\n")

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

        # extract one agent to get headers
        first_agent = list(agents_data.values())[0]
        stats_keys = list(first_agent["stats"].keys())
        # Add 'Agent_Name' as the first column
        fieldnames = ["agent_name"] + stats_keys

        print(f"Exporting Game Stats to: {self.paths['stats']}")

        try:
            with open(self.paths["stats"], "w", newline='', encoding="utf-8") as csvfile:
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