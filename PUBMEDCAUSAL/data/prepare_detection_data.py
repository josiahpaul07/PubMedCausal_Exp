"""
Data Preparation for Causal Detection Task
Uses train_30k.json and test_30k.json as-is
"""

import json
import os
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path


class DetectionDataPreparator:
    """Prepares data for causal detection (binary classification)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "train_30k.json"
        self.test_file = self.data_dir / "test_30k.json"
        
    def has_causality(self, item: Dict) -> int:
        """Check if sentence contains any cause-effect relationship"""
        # Check all 16 possible cause-effect pairs
        for i in range(1, 17):
            cause_key = f"Cause {i}"
            effect_key = f"Effect {i}"
            
            if cause_key in item and effect_key in item:
                if item[cause_key] and item[cause_key].strip():
                    return 1
                if item[effect_key] and item[effect_key].strip():
                    return 1
        return 0
    
    def prepare_dataset(self, filepath: str) -> List[Dict]:
        """Prepare detection dataset from JSON file"""
        print(f"\nProcessing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prepared_data = []
        for item in data:
            has_causal = self.has_causality(item)
            
            prepared_item = {
                's/n': item.get('s/n', ''),
                'sentence': item.get('Sentence', ''),
                'label': has_causal  # 1 = has causality, 0 = no causality
            }
            prepared_data.append(prepared_item)
        
        # Statistics
        total = len(prepared_data)
        causal = sum(1 for x in prepared_data if x['label'] == 1)
        non_causal = total - causal
        
        print(f"Total sentences: {total}")
        print(f"Causal sentences: {causal} ({causal/total*100:.2f}%)")
        print(f"Non-causal sentences: {non_causal} ({non_causal/total*100:.2f}%)")
        
        return prepared_data
    
    def save_prepared_data(self, output_dir: str):
        """Prepare and save train/test splits for detection"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare train data
        train_data = self.prepare_dataset(str(self.train_file))
        train_output = output_dir / "detection_train.json"
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {train_output}")
        
        # Prepare test data
        test_data = self.prepare_dataset(str(self.test_file))
        test_output = output_dir / "detection_test.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {test_output}")
        
        return train_data, test_data


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare detection data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train_30k.json and test_30k.json')
    parser.add_argument('--output_dir', type=str, default='./data/prepared',
                       help='Output directory for prepared data')
    
    args = parser.parse_args()
    
    preparator = DetectionDataPreparator(args.data_dir)
    preparator.save_prepared_data(args.output_dir)
    
    print("\n? Detection data preparation complete!")


if __name__ == "__main__":
    main()