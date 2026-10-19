import os
import yaml
from typing import Dict, Any, Tuple

class AttackMapper:
    """
    ATT&CK-informed behavioral mapping layer.
    Maps dataset categories (e.g., PortScan, FTP-BruteForce, Infilteration, Bot)
    to MITRE ATT&CK tactics and numeric stage indices.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "attack_mapping.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.mappings = self.config.get("mappings", {})
        self.stage_names = self.config.get("stage_names", {})
        
    def map_label(self, raw_label: str) -> Tuple[str, str, int]:
        """
        Maps a raw dataset label string to (Tactic Name, Tactic ID, Stage Index).
        """
        cleaned = str(raw_label).strip()
        info = self.mappings.get(cleaned)
        
        if info is None:
            # Check for partial matches or fallback to Benign
            for key, val in self.mappings.items():
                if key.lower() in cleaned.lower():
                    info = val
                    break
                    
        if info is None:
            info = {
                "tactic": "Unknown / Unclassified",
                "tactic_id": "TA0000",
                "stage_index": 0
            }
            
        return info.get("tactic", "Unknown"), info.get("tactic_id", "TA0000"), info.get("stage_index", 0)

    def get_stage_name(self, stage_index: int) -> str:
        return self.stage_names.get(stage_index, f"Stage {stage_index}")

    def num_stages(self) -> int:
        return len(self.stage_names)
