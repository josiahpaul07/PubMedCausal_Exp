"""
Data Preparation for Causal Extraction Task
Implements Extraction_strat:
1. Combine train_30k + test_30k
2. Filter for causal sentences only
3. Split into X (multiple pairs) and Y (single pair)
4. Create 3 experimental splits: X-only, Y-only, Combined
"""

import json
import random
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


class ExtractionDataPreparator:
    """Prepares data for causal extraction with multiple experimental setups"""
    
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "train_30k.json"
        self.test_file = self.data_dir / "test_30k.json"
        self.seed = seed
        random.seed(seed)
        
    def count_causal_pairs(self, item: Dict) -> int:
        """Count number of valid cause-effect pairs in a sentence"""
        count = 0
        for i in range(1, 17):
            cause_key = f"Cause {i}"
            effect_key = f"Effect {i}"
            
            # Valid pair if both cause AND effect are non-empty
            if (cause_key in item and effect_key in item and
                item[cause_key] and item[cause_key].strip() and
                item[effect_key] and item[effect_key].strip()):
                count += 1
        return count
    
    def extract_causal_info(self, item: Dict) -> Dict:
        """Extract all causal information from an item"""
        extracted = {
            's/n': item.get('s/n', ''),
            'sentence': item.get('Sentence', ''),
            'pairs': []
        }
        
        for i in range(1, 17):
            cause_key = f"Cause {i}"
            effect_key = f"Effect {i}"
            sententiality_key = f"Sententiality {i}"
            causality_key = f"Causality {i}"
            
            if (cause_key in item and effect_key in item and
                item[cause_key] and item[cause_key].strip() and
                item[effect_key] and item[effect_key].strip()):
                
                pair = {
                    'cause': item[cause_key].strip(),
                    'effect': item[effect_key].strip(),
                    'sententiality': item.get(sententiality_key, '').strip(),
                    'causality': item.get(causality_key, '').strip()
                }
                extracted['pairs'].append(pair)
        
        return extracted
    
    def load_and_combine(self) -> List[Dict]:
        """Load and combine train_30k + test_30k, filter for causal sentences"""
        print("Loading datasets...")
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Combine
        combined = train_data + test_data
        print(f"Combined dataset size: {len(combined)}")
        
        # Filter for causal sentences only (at least 1 pair)
        causal_data = []
        for item in combined:
            num_pairs = self.count_causal_pairs(item)
            if num_pairs > 0:
                extracted = self.extract_causal_info(item)
                extracted['num_pairs'] = num_pairs
                causal_data.append(extracted)
        
        print(f"Causal sentences (with at least 1 pair): {len(causal_data)}")
        return causal_data
    
    def split_X_Y(self, causal_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split into X (multiple pairs) and Y (single pair)
        X: sentences with 2+ cause-effect pairs
        Y: sentences with exactly 1 cause-effect pair
        """
        X = []  # Multiple pairs
        Y = []  # Single pair
        
        for item in causal_data:
            if item['num_pairs'] >= 2:
                X.append(item)
            elif item['num_pairs'] == 1:
                Y.append(item)
        
        print(f"\nX (Multiple pairs): {len(X)}")
        print(f"Y (Single pair): {len(Y)}")
        
        # Shuffle
        random.shuffle(X)
        random.shuffle(Y)
        
        return X, Y
    
    def create_splits(self, X: List[Dict], Y: List[Dict]) -> Dict:
        """
        Create 3 experimental splits:
        1. X-only: X_train (50%) / X_test (50%)
        2. Y-only: Y_train (50%) / Y_test (50%)
        3. Combined: (X_train + Y_train) / (X_test + Y_test)
        """
        # Split X into train/test (50/50)
        X_mid = len(X) // 2
        X_train = X[:X_mid]
        X_test = X[X_mid:]
        
        # Split Y into train/test (50/50)
        Y_mid = len(Y) // 2
        Y_train = Y[:Y_mid]
        Y_test = Y[Y_mid:]
        
        # Create combined
        combined_train = X_train + Y_train
        combined_test = X_test + Y_test
        
        # Shuffle combined
        random.shuffle(combined_train)
        random.shuffle(combined_test)
        
        splits = {
            'X_only': {
                'train': X_train,
                'test': X_test
            },
            'Y_only': {
                'train': Y_train,
                'test': Y_test
            },
            'combined': {
                'train': combined_train,
                'test': combined_test
            }
        }
        
        print("\n=== Split Statistics ===")
        print(f"X-only: Train={len(X_train)}, Test={len(X_test)}")
        print(f"Y-only: Train={len(Y_train)}, Test={len(Y_test)}")
        print(f"Combined: Train={len(combined_train)}, Test={len(combined_test)}")
        
        return splits
    
    def save_splits(self, splits: Dict, output_dir: str):
        """Save all experimental splits"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, data in splits.items():
            split_dir = output_dir / f"extraction_{split_name}"
            split_dir.mkdir(exist_ok=True)
            
            # Save train
            train_path = split_dir / "train.json"
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(data['train'], f, indent=2, ensure_ascii=False)
            
            # Save test
            test_path = split_dir / "test.json"
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(data['test'], f, indent=2, ensure_ascii=False)
            
            print(f"? Saved {split_name}: {train_path.parent}")
    
    def prepare_all(self, output_dir: str):
        """Main pipeline: load, split, and save"""
        # Step 1: Load and combine
        causal_data = self.load_and_combine()
        
        # Step 2: Split into X and Y
        X, Y = self.split_X_Y(causal_data)
        
        # Step 3: Create 3 experimental splits
        splits = self.create_splits(X, Y)
        
        # Step 4: Save
        self.save_splits(splits, output_dir)
        
        return splits


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare extraction data with Extraction_strat')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train_30k.json and test_30k.json')
    parser.add_argument('--output_dir', type=str, default='./data/prepared',
                       help='Output directory for prepared data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    preparator = ExtractionDataPreparator(args.data_dir, seed=args.seed)
    preparator.prepare_all(args.output_dir)
    
    print("\n? Extraction data preparation complete!")


if __name__ == "__main__":
    main()